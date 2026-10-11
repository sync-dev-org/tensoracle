"""Report every engine declaration mismatch at the shared opening boundary."""

import hashlib
from pathlib import Path

import pytest
import tensorrt as trt

import tensoracle
from test_build import _weights
from test_execution import _open_sum


class Engine:
    num_io_tensors = 2

    def __init__(self):
        self.names = ("features", "result")
        self.shapes = {name: (3, 2) for name in self.names}
        self.modes = {"features": trt.TensorIOMode.INPUT, "result": trt.TensorIOMode.OUTPUT}
        self.dtypes = {name: trt.float32 for name in self.names}
        self.profile = ((1, 2), (3, 2), (5, 2))
        self.optional_queries = []

    def get_tensor_name(self, index):
        return self.names[index]

    def get_tensor_mode(self, name):
        return self.modes[name]

    def get_tensor_dtype(self, name):
        return self.dtypes[name]

    def get_tensor_shape(self, name):
        return self.shapes[name]

    def get_tensor_profile_shape(self, name, index):
        return self.profile

    def get_tensor_location(self, name):
        self.optional_queries.append(("location", name))
        return trt.TensorLocation.DEVICE

    def get_tensor_format(self, name):
        self.optional_queries.append(("format", name))
        return trt.TensorFormat.LINEAR


def _cached_open(tmp_path, monkeypatch, engine, declaration, io):
    import tensoracle._cache as cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weights = tmp_path / "weights.safetensors"
    _weights(weights)
    conditions = cache._conditions("checked", "1", (weights,), (), declaration)
    plan = b"external TensorRT plan"
    header = cache._json_bytes(
        {"conditions": conditions, "plan_sha256": hashlib.sha256(plan).hexdigest()}
    )
    digest = hashlib.sha256(cache._json_bytes(conditions)).hexdigest()
    path = tmp_path / "cache" / "tensoracle" / f"{digest}.engine"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cache._MAGIC + len(header).to_bytes(8, "big") + header + plan)

    class Runtime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, data):
            assert data == plan
            return engine

    monkeypatch.setattr(trt, "Runtime", Runtime)
    return tensoracle.open_engine(
        lambda _: pytest.fail("cached graph called"),
        declaration,
        name="checked",
        graph_version="1",
        weights=(weights,),
        io=io,
    )


@pytest.mark.req("REQ-TENSORACLE-002")
@pytest.mark.parametrize(
    "field",
    [
        "name",
        "mode",
        "dtype",
        "shape",
        "profile",
        "location",
        "format",
        "build_inputs",
        "fixed_profile",
    ],
)
def test_each_io_mismatch_reports_tensor_field_observed_and_expected(tmp_path, monkeypatch, field):
    """Every opening mismatch identifies its tensor, field, observed value, and expected value."""
    engine = Engine()
    inputs = {"features": tensoracle.TensorSpec(trt.float32, (3, 2))}
    outputs = {"result": tensoracle.TensorSpec(trt.float32, (3, 2))}
    declaration = tensoracle.EngineBuild("fp32", {"features": tensoracle.InputShape.fixed((3, 2))})
    name = "features"
    if field == "name":
        inputs = {"renamed": inputs["features"]}
        name = "renamed"
        observed, expected = "features", "renamed"
    elif field == "mode":
        inputs, outputs = outputs, inputs
        observed, expected = "OUTPUT", "INPUT"
        name = "result"
    elif field == "dtype":
        engine.dtypes["features"] = trt.float16
        observed, expected = "HALF", "FLOAT"
    elif field == "shape":
        engine.shapes["features"] = (4, 2)
        observed, expected = "(4, 2)", "(3, 2)"
    elif field == "profile":
        engine.shapes["features"] = (-1, 2)
        inputs["features"] = tensoracle.TensorSpec(trt.float32, (-1, 2))
        declaration = tensoracle.EngineBuild(
            "fp32", {"features": tensoracle.InputShape.ranged((1, 2), (3, 2), (6, 2))}
        )
        observed, expected = "(5, 2)", "(6, 2)"
    elif field == "location":
        inputs["features"] = tensoracle.TensorSpec(
            trt.float32, (3, 2), location=trt.TensorLocation.HOST
        )
        observed, expected = "DEVICE", "HOST"
    elif field == "format":
        inputs["features"] = tensoracle.TensorSpec(
            trt.float32, (3, 2), format=trt.TensorFormat.CHW4
        )
        observed, expected = "LINEAR", "CHW4"
    elif field == "build_inputs":
        declaration = tensoracle.EngineBuild("fp32", {"other": tensoracle.InputShape.fixed((3, 2))})
        name = "other"
        field = "name"
        observed, expected = "features", "other"
    else:
        declaration = tensoracle.EngineBuild(
            "fp32", {"features": tensoracle.InputShape.fixed((4, 2))}
        )
        field = "profile"
        observed, expected = "(3, 2)", "(4, 2)"
    with pytest.raises(tensoracle.EngineIOError) as caught:
        _cached_open(
            tmp_path, monkeypatch, engine, declaration, tensoracle.EngineIO(inputs, outputs)
        )
    message = str(caught.value)
    assert name in message and field in message
    assert "observed" in message and "expected" in message
    assert observed in message and expected in message
    assert str(tmp_path / "cache") in message


@pytest.mark.req("REQ-TENSORACLE-002")
def test_unspecified_location_and_format_are_not_queried(tmp_path, monkeypatch):
    """Omitting optional location and format preserves the existing opening checks."""
    engine = Engine()
    declaration = tensoracle.EngineBuild("fp32", {"features": tensoracle.InputShape.fixed((3, 2))})
    io = tensoracle.EngineIO(
        {"features": tensoracle.TensorSpec(trt.float32, (3, 2))},
        {"result": tensoracle.TensorSpec(trt.float32, (3, 2))},
    )
    assert _cached_open(tmp_path, monkeypatch, engine, declaration, io).engine is engine
    assert engine.optional_queries == []


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-002")
def test_device_linear_declarations_open_and_mismatches_are_rejected(tmp_path, monkeypatch):
    """A real engine opens with matching or omitted optional declarations and rejects differences."""
    handle = _open_sum(tmp_path, monkeypatch)
    profile = tensoracle.InputShape.ranged((1, 2), (3, 2), (5, 2))
    declaration = tensoracle.EngineBuild(
        "fp32",
        {"features": profile, "extra": profile},
        tensoracle.BuilderSettings(optimization_level=0),
    )
    for location, format, accepted in (
        (None, None, True),
        (trt.TensorLocation.DEVICE, trt.TensorFormat.LINEAR, True),
        (trt.TensorLocation.HOST, trt.TensorFormat.LINEAR, False),
        (trt.TensorLocation.DEVICE, trt.TensorFormat.CHW4, False),
    ):
        spec = tensoracle.TensorSpec(trt.float32, (-1, 2), location=location, format=format)

        def open_checked(spec=spec):
            return tensoracle.open_engine(
                lambda _: pytest.fail("cache rebuilt"),
                declaration,
                name="sum",
                graph_version="1",
                weights=(Path(tmp_path / "weights.safetensors"),),
                io=tensoracle.EngineIO({"features": spec, "extra": spec}, {"result": spec}),
            )

        if accepted:
            assert open_checked().path == handle.path
        else:
            with pytest.raises(tensoracle.EngineIOError):
                open_checked()
