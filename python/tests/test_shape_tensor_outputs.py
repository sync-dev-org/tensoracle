"""Warn about shape tensor outputs before unsafe TensorRT shape inference."""

import hashlib
import json
import subprocess
import sys
import warnings

import numpy as np
import pytest
import tensorrt as trt
from safetensors.numpy import save_file

import tensoracle
from test_execution import Context, Engine, _handle


def _graph(network, *, output_kind="shape"):
    features = network.add_input("features", trt.float32, (3, 2))
    result = network.add_identity(features).get_output(0)
    result.name = "result"
    network.mark_output(result)
    if output_kind != "ordinary":
        dimensions = network.add_shape(features).get_output(0)
        dimensions.name = "dimensions"
        if output_kind == "shape":
            assert network.mark_output_for_shapes(dimensions)
        else:
            network.mark_output(dimensions)


def _declaration():
    return tensoracle.EngineBuild(
        {"features": tensoracle.InputShape.fixed((3, 2))},
        tensoracle.BuilderSettings(optimization_level=0),
    )


def _io(output_kind="shape"):
    outputs = {"result": tensoracle.TensorSpec(trt.float32, (3, 2))}
    if output_kind != "ordinary":
        outputs["dimensions"] = tensoracle.TensorSpec(trt.int64, (2,))
    return tensoracle.EngineIO({"features": tensoracle.TensorSpec(trt.float32, (3, 2))}, outputs)


@pytest.fixture
def weights(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = tmp_path / "weights.safetensors"
    save_file({"unused": np.zeros(1, dtype=np.float32)}, path)
    return path


def _open(weights, *, output_kind="shape", graph=None, force_rebuild=False):
    return tensoracle.open_engine(
        graph if graph is not None else lambda network: _graph(network, output_kind=output_kind),
        _declaration(),
        name="shape-output",
        graph_version="1",
        weights=(weights,),
        io=_io(output_kind),
        force_rebuild=force_rebuild,
    )


def _read_file(path):
    data = path.read_bytes()
    offset = len(b"TENSORACLE\x01")
    header_end = offset + 8 + int.from_bytes(data[offset : offset + 8], "big")
    return json.loads(data[offset + 8 : header_end]), data[header_end:]


def _write_file(path, header, plan):
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(b"TENSORACLE\x01" + len(encoded).to_bytes(8, "big") + encoded + plan)


def _assert_warning(message):
    assert "dimensions" in message
    assert "shape tensor" in message
    assert "create_execution()" in message
    assert "cannot" in message
    assert "process" in message and "crash" in message


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-012")
@pytest.mark.req("SP-TENSORACLE-002")
def test_shape_output_build_warns_once_and_returns_plan():
    """A shape tensor output warns about the execution crash while returning plan bytes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = tensoracle.build_serialized_engine(_graph, _declaration())
    assert isinstance(plan, bytes) and plan
    assert len(caught) == 1
    assert caught[0].category is tensoracle.ShapeTensorOutputWarning
    assert issubclass(caught[0].category, UserWarning)
    assert "ShapeTensorOutputWarning" in tensoracle.__all__
    _assert_warning(str(caught[0].message))


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.req("SP-TENSORACLE-012")
@pytest.mark.req("SP-TENSORACLE-002")
def test_shape_output_cache_records_names_without_changing_conditions(weights):
    """Shape output names are saved outside conditions and do not change the cache identity."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handle = _open(weights)
    assert len(caught) == 1
    assert caught[0].category is tensoracle.ShapeTensorOutputWarning
    _assert_warning(str(caught[0].message))
    header, plan = _read_file(handle.path)
    assert header["shape_tensor_outputs"] == ["dimensions"]
    assert "shape_tensor_outputs" not in header["conditions"]
    assert header["plan_sha256"] == hashlib.sha256(plan).hexdigest()
    digest = hashlib.sha256(
        json.dumps(header["conditions"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert handle.path.name == f"{digest}.engine"
    ordinary = _open(weights, output_kind="ordinary", force_rebuild=True)
    ordinary_header, _ = _read_file(ordinary.path)
    assert ordinary.path == handle.path
    assert ordinary_header["conditions"] == header["conditions"]


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-012")
@pytest.mark.req("SP-TENSORACLE-002")
@pytest.mark.parametrize("entry", ["create_execution", "EngineExecution"])
def test_saved_shape_output_warns_before_inference_in_another_process(weights, entry):
    """A later process can turn the execution warning into an exception before unsafe inference."""
    category = tensoracle.ShapeTensorOutputWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category)
        handle = _open(weights)
    child = """
import json
import sys
import warnings
from pathlib import Path
import tensorrt as trt
import tensoracle

def forbidden(network):
    raise AssertionError('saved engine was rebuilt')

handle = tensoracle.open_engine(
    forbidden,
    tensoracle.EngineBuild({'features': tensoracle.InputShape.fixed((3, 2))},
                         tensoracle.BuilderSettings(optimization_level=0)),
    name='shape-output', graph_version='1', weights=(Path(sys.argv[1]),),
    io=tensoracle.EngineIO({'features': tensoracle.TensorSpec(trt.float32, (3, 2))},
                          {'result': tensoracle.TensorSpec(trt.float32, (3, 2)),
                           'dimensions': tensoracle.TensorSpec(trt.int64, (2,))}),
)
warnings.filterwarnings('error', category=tensoracle.ShapeTensorOutputWarning)
try:
    if sys.argv[2] == 'create_execution':
        handle.create_execution()
    else:
        tensoracle.EngineExecution(handle)
except tensoracle.ShapeTensorOutputWarning as warning:
    print(json.dumps({'path': str(handle.path), 'message': str(warning)}))
else:
    raise AssertionError('execution did not raise the warning')
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(weights), entry],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["path"] == str(handle.path)
    _assert_warning(observed["message"])


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-012")
@pytest.mark.req("SP-TENSORACLE-002")
@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.parametrize("entry", ["build_serialized_engine", "open_engine"])
def test_shape_output_warning_as_error_stops_build_without_cache_files(weights, entry):
    """Turning the build warning into an error leaves no engine or temporary engine file."""
    category = tensoracle.ShapeTensorOutputWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=category)
        with pytest.raises(category) as caught:
            if entry == "build_serialized_engine":
                tensoracle.build_serialized_engine(_graph, _declaration())
            else:
                _open(weights)
    _assert_warning(str(caught.value))
    cache_directory = weights.parent / "cache" / "tensoracle"
    assert all(path.name.startswith("weight-") for path in cache_directory.glob("*"))


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.req("SP-TENSORACLE-012")
def test_saved_engine_without_shape_output_metadata_is_reused_without_warning(weights):
    """An older header loads without rebuilding or warning when creating an execution unit."""
    first = _open(weights, output_kind="ordinary")
    header, plan = _read_file(first.path)
    header.pop("shape_tensor_outputs", None)
    _write_file(first.path, header, plan)
    legacy_bytes = first.path.read_bytes()

    def forbidden(network):
        pytest.fail("saved engine was rebuilt")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reopened = _open(weights, output_kind="ordinary", graph=forbidden)
        with reopened.create_execution() as execution:
            assert execution.get_tensor_shape("result") == (3, 2)
    assert caught == []
    assert reopened.path == first.path
    assert reopened.path.read_bytes() == legacy_bytes


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-012")
@pytest.mark.parametrize("output_kind", ["ordinary", "shape-layer"])
def test_ordinary_outputs_including_shape_layer_outputs_never_warn(weights, output_kind):
    """Ordinary marked outputs stay supported even when their values come from a shape layer."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert tensoracle.build_serialized_engine(
            lambda network: _graph(network, output_kind=output_kind), _declaration()
        )
        handle = _open(weights, output_kind=output_kind)
        reopened = _open(weights, output_kind=output_kind)
        for checked in (handle, reopened):
            with checked.create_execution() as execution:
                assert execution.get_tensor_shape("result") == (3, 2)
                if output_kind == "shape-layer":
                    assert execution.get_tensor_shape("dimensions") == (2,)
    assert caught == []


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-011")
@pytest.mark.req("SP-TENSORACLE-002")
@pytest.mark.parametrize("metadata", [None, "dimensions", {}, [1], ["dimensions", False]])
def test_invalid_shape_output_metadata_is_a_corrupt_engine(weights, metadata):
    """A present shape output record must be a list of tensor names to load a saved engine."""
    handle = _open(weights, output_kind="ordinary")
    header, plan = _read_file(handle.path)
    header["shape_tensor_outputs"] = metadata
    _write_file(handle.path, header, plan)
    with pytest.raises(tensoracle.CorruptEngineError, match="header"):
        _open(weights, output_kind="ordinary")


@pytest.mark.req("SP-TENSORACLE-012")
@pytest.mark.req("SP-TENSORACLE-002")
def test_execution_warning_as_error_releases_context_before_inference(monkeypatch):
    """A warning raised as an error releases the new context without probing output shapes."""
    category = tensoracle.ShapeTensorOutputWarning
    engine = Engine(fixed=True)
    monkeypatch.setattr(
        Context, "infer_shapes", lambda context: pytest.fail("inference was called")
    )
    handle = _handle(engine)
    object.__setattr__(handle, "_shape_tensor_outputs", ("dimensions",))
    execution = tensoracle.EngineExecution.__new__(tensoracle.EngineExecution)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=category)
        with pytest.raises(category) as caught:
            execution.__init__(handle)
    _assert_warning(str(caught.value))
    assert len(engine.contexts) == 1
    assert engine.latest is None
    with pytest.raises(RuntimeError, match="closed"):
        execution.get_tensor_shape("result")
    assert execution._handle is None
