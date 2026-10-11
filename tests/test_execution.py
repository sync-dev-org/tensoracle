"""Run checked engines with caller-owned device buffers and streams."""

import builtins
import weakref
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
import tensorrt as trt
from cuda.bindings import runtime as cudart
from safetensors.numpy import save_file

import tensoracle
from test_build import _check_cuda


class Context:
    """Stand in for the external TensorRT execution context, never CUDA."""

    def __init__(self, engine):
        self.shapes = dict(engine.shapes)
        self.calls = []
        self.fail_shape = None
        self.fail_address = None
        self.fail_enqueue = False
        self.unresolved = engine.unresolved

    def set_input_shape(self, name, shape):
        self.calls.append(("shape", name, shape))
        if self.fail_shape is not None and self.fail_shape[0] == name:
            if isinstance(self.fail_shape[1], Exception):
                raise self.fail_shape[1]
            return self.fail_shape[1]
        self.shapes[name] = tuple(shape)
        return True

    def infer_shapes(self):
        self.calls.append(("infer",))
        return []

    def get_tensor_shape(self, name):
        self.calls.append(("query", name))
        if name == "result":
            return (-1, 2) if self.unresolved else self.shapes["features"]
        return self.shapes[name]

    def set_tensor_address(self, name, address):
        self.calls.append(("address", name, address))
        return name != self.fail_address

    def execute_async_v3(self, stream):
        self.calls.append(("enqueue", stream))
        return not self.fail_enqueue


class Engine:
    """Only the TensorRT boundary is substituted in CPU tests."""

    def __init__(self, *, fixed=False, point=False):
        shape = (3, 2) if fixed else (-1, 2)
        self.shapes = {"features": shape, "extra": shape, "result": shape}
        self.profile = ((3, 2),) * 3 if point else ((1, 2), (3, 2), (5, 2))
        self.contexts = []
        self.host = None
        self.shape_input = None
        self.unresolved = False
        self.no_context = False

    @property
    def num_io_tensors(self):
        return len(self.shapes)

    def get_tensor_name(self, index):
        return tuple(self.shapes)[index]

    def get_tensor_mode(self, name):
        return trt.TensorIOMode.OUTPUT if name == "result" else trt.TensorIOMode.INPUT

    def get_tensor_dtype(self, name):
        return trt.float32

    def get_tensor_shape(self, name):
        return self.shapes[name]

    def get_tensor_profile_shape(self, name, profile):
        assert profile == 0
        return self.profile

    def get_tensor_location(self, name):
        return trt.TensorLocation.HOST if name == self.host else trt.TensorLocation.DEVICE

    def is_shape_inference_io(self, name):
        return name == self.shape_input

    def create_execution_context(self):
        if self.no_context:
            return None
        context = Context(self)
        self.contexts.append(weakref.ref(context))
        return context

    @property
    def latest(self):
        return self.contexts[-1]()


def _handle(engine):
    return tensoracle.EngineHandle(engine, object(), Path("checked.engine"))


def _ready(engine=None):
    engine = engine or Engine()
    execution = _handle(engine).create_execution()
    execution.set_input_shapes({"features": (3, 2), "extra": (3, 2)})
    engine.latest.calls.clear()
    return execution, engine


@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.parametrize(
    ("shapes", "name", "value"),
    [
        ({"features": (3, 2)}, "extra", "missing"),
        ({"features": (3, 2), "extra": (3, 2), "other": (3, 2)}, "other", "(3, 2)"),
        ({"features": (3, 2), "extra": (6, 2)}, "extra", "(6, 2)"),
        ({"features": (3, 2), "extra": (3, 3)}, "extra", "(3, 3)"),
        ({"features": (3, 2), "extra": (3,)}, "extra", "(3,)"),
        ({"features": (3, 2), "extra": (True, 2)}, "extra", "True"),
        ({"features": (3, 2), "extra": (3.0, 2)}, "extra", "3.0"),
    ],
)
def test_invalid_input_shapes_are_rejected_before_setting_any_tensor(shapes, name, value):
    """Missing, extra, and out-of-range input shapes identify the input before any setting."""
    execution, engine = _ready()
    with pytest.raises(ValueError) as caught:
        execution.set_input_shapes(shapes)
    assert name in str(caught.value) and value in str(caught.value)
    assert engine.latest.calls == []


@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.parametrize("failure", [False, RuntimeError("TensorRT failure")])
def test_partial_shape_failure_blocks_execution_until_all_shapes_are_set(failure):
    """A partial shape failure requires a complete successful setting before reuse."""
    execution, engine = _ready()
    context = engine.latest
    context.fail_shape = ("extra", failure)
    with pytest.raises(RuntimeError, match="extra.*4"):
        execution.set_input_shapes({"features": (4, 2), "extra": (4, 2)})
    assert ("shape", "features", (4, 2)) in context.calls
    context.calls.clear()
    with pytest.raises(RuntimeError, match="shape"):
        execution.enqueue({"features": 1, "extra": 2, "result": 3}, stream=0)
    with pytest.raises(RuntimeError, match="shape"):
        execution.get_tensor_shape("result")
    assert context.calls == []
    context.fail_shape = None
    execution.set_input_shapes({"features": (2, 2), "extra": (2, 2)})
    assert execution.get_tensor_shape("result") == (2, 2)
    execution.enqueue({"features": 1, "extra": 2, "result": 3}, stream=0)
    assert ("enqueue", 0) in context.calls


@pytest.mark.req("REQ-TENSORACLE-018")
def test_ranged_engine_requires_shapes_before_querying_or_enqueuing():
    """A ranged engine requires the caller's input shapes before output sizing or enqueue."""
    engine = Engine()
    execution = _handle(engine).create_execution()
    engine.latest.calls.clear()
    with pytest.raises(RuntimeError, match="shape"):
        execution.get_tensor_shape("result")
    with pytest.raises(RuntimeError, match="shape"):
        execution.enqueue({"features": 1, "extra": 2, "result": 3}, stream=0)
    assert engine.latest.calls == []


@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.parametrize("address", [0, -1, True, False, 1.5, "1", None])
def test_invalid_addresses_are_rejected_before_tensor_settings(address):
    """Every device address must be a positive integer excluding bool before TensorRT runs."""
    execution, engine = _ready()
    with pytest.raises(ValueError) as caught:
        execution.enqueue({"features": 1, "extra": 2, "result": address}, stream=0)
    assert "result" in str(caught.value) and repr(address) in str(caught.value)
    assert engine.latest.calls == []


@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.parametrize("stream", [-1, True, False, 1.5, "0", None])
def test_invalid_streams_are_rejected_before_tensor_settings(stream):
    """The stream must be a nonnegative integer excluding bool before TensorRT runs."""
    execution, engine = _ready()
    with pytest.raises(ValueError) as caught:
        execution.enqueue({"features": 1, "extra": 2, "result": 3}, stream=stream)
    assert "stream" in str(caught.value) and repr(stream) in str(caught.value)
    assert engine.latest.calls == []


@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.parametrize(
    ("addresses", "name", "value"),
    [
        ({"features": 1, "extra": 2}, "result", "missing"),
        ({"features": 1, "extra": 2, "result": 3, "other": 4}, "other", "4"),
    ],
)
def test_address_names_must_match_every_engine_tensor(addresses, name, value):
    """Missing and extra addresses identify the tensor before any address is set."""
    execution, engine = _ready()
    with pytest.raises(ValueError) as caught:
        execution.enqueue(addresses, stream=0)
    assert name in str(caught.value) and value in str(caught.value)
    assert engine.latest.calls == []


@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.parametrize("stage", ["address", "enqueue"])
def test_false_address_setting_or_enqueue_is_reported_as_failure(stage):
    """TensorRT's rejection of an address or enqueue cannot be reported as success."""
    execution, engine = _ready()
    context = engine.latest
    context.fail_address = "result" if stage == "address" else None
    context.fail_enqueue = stage == "enqueue"
    with pytest.raises(RuntimeError, match=stage):
        execution.enqueue({"features": 1, "extra": 2, "result": 3}, stream=7)
    if stage == "address":
        assert not any(call[0] == "enqueue" for call in context.calls)


@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.parametrize(
    ("kind", "name", "reason"),
    [
        ("host", "result", "HOST"),
        ("shape", "features", "shape tensor"),
        ("data", "result", "data-dependent"),
    ],
)
def test_unsupported_engine_is_rejected_when_execution_is_created(kind, name, reason):
    """Host tensors, external shape values, and data-dependent outputs identify their rejection."""
    engine = Engine()
    if kind == "host":
        engine.host = name
    elif kind == "shape":
        engine.shape_input = name
    else:
        engine.unresolved = True
    with pytest.raises(ValueError) as caught:
        _handle(engine).create_execution()
    assert name in str(caught.value) and reason in str(caught.value)


@pytest.mark.req("REQ-TENSORACLE-018")
def test_context_creation_failure_is_reported():
    """Failure to create a TensorRT execution context raises a useful error."""
    engine = Engine()
    engine.no_context = True
    with pytest.raises(RuntimeError, match="context"):
        _handle(engine).create_execution()


@pytest.mark.req("REQ-TENSORACLE-018")
def test_unresolved_output_after_shape_setting_blocks_enqueue():
    """An unresolved output dimension prevents address setting and execution."""
    execution, engine = _ready()
    engine.latest.unresolved = True
    with pytest.raises(RuntimeError, match="result.*shape"):
        execution.set_input_shapes({"features": (4, 2), "extra": (4, 2)})
    engine.latest.calls.clear()
    with pytest.raises(RuntimeError, match="shape"):
        execution.enqueue({"features": 1, "extra": 2, "result": 3}, stream=0)
    assert engine.latest.calls == []


@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.parametrize("manager", [False, True])
def test_closed_execution_rejects_operations_without_touching_tensorrt(manager):
    """Close and context exit are idempotent and reject further operations before TensorRT."""
    execution, engine = _ready()
    context = engine.latest
    if manager:
        with execution as active:
            assert active is execution
    else:
        execution.close()
    execution.close()
    for operation in (
        lambda: execution.set_input_shapes({"features": (3, 2), "extra": (3, 2)}),
        lambda: execution.enqueue({"features": 1, "extra": 2, "result": 3}, stream=0),
        lambda: execution.get_tensor_shape("result"),
        lambda: execution.get_tensor_dtype("result"),
        lambda: execution.__enter__(),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    assert context.calls == []


@pytest.mark.req("REQ-TENSORACLE-018")
def test_close_releases_context_before_handle_without_waiting_for_gc():
    """Close releases the execution context then its owning handle with no ownership cycle."""
    engine = Engine(fixed=True)
    handle = _handle(engine)
    execution = handle.create_execution()
    assert isinstance(execution, tensoracle.EngineExecution)
    released = []
    context_ref = engine.contexts[-1]
    handle_ref = weakref.ref(handle)
    weakref.finalize(context_ref(), released.append, "context")
    weakref.finalize(handle, released.append, "handle")
    del handle
    assert context_ref() is not None and handle_ref() is not None
    assert not hasattr(execution, "context")
    assert not hasattr(execution, "execution_context")
    execution.close()
    assert context_ref() is None and handle_ref() is None
    assert released == ["context", "handle"]


@pytest.mark.req("REQ-TENSORACLE-018")
def test_execution_does_not_import_or_call_a_cuda_runtime(monkeypatch):
    """Creation, shape setting, enqueue, and close use TensorRT without CUDA runtime calls."""
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"cuda", "cupy", "torch"}
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("tensoracle called a CUDA runtime API")

    for name in (
        "cudaMalloc",
        "cudaFree",
        "cudaMemcpy",
        "cudaMemcpyAsync",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
    ):
        monkeypatch.setattr(cudart, name, forbidden)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    execution, engine = _ready()
    assert execution.get_tensor_dtype("result") == trt.float32
    execution.enqueue({"features": 1, "extra": 2, "result": 3}, stream=0)
    assert ("enqueue", 0) in engine.latest.calls
    execution.close()


def _open_sum(tmp_path, monkeypatch, *, fixed=False, point=False, internal_shape=False):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights.safetensors"
    save_file({"bias": np.array([0.25, -0.5], dtype=np.float32)}, weights)
    shape = (3, 2) if fixed else (-1, 2)
    profile = (
        tensoracle.InputShape.fixed((3, 2))
        if fixed or point
        else tensoracle.InputShape.ranged((1, 2), (3, 2), (5, 2))
    )

    def graph(network):
        features = network.add_input("features", trt.float32, shape)
        extra = network.add_input("extra", trt.float32, shape)
        result = network.add_elementwise(features, extra, trt.ElementWiseOperation.SUM).get_output(
            0
        )
        if internal_shape:
            dimensions = network.add_shape(features).get_output(0)
            shuffle = network.add_shuffle(result)
            shuffle.set_input(1, dimensions)
            result = shuffle.get_output(0)
        result.name = "result"
        network.mark_output(result)

    return tensoracle.open_engine(
        graph,
        tensoracle.EngineBuild(
            "fp32",
            {"features": profile, "extra": profile},
            tensoracle.BuilderSettings(optimization_level=0),
        ),
        name="sum",
        graph_version="1",
        weights=(weights,),
        io=tensoracle.EngineIO(
            {
                name: tensoracle.TensorSpec(trt.float32, (3, 2) if point else shape)
                for name in ("features", "extra")
            },
            {"result": tensoracle.TensorSpec(trt.float32, (3, 2) if point else shape)},
        ),
    )


def _run_sum(execution, rows, *, stream=None):
    left = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2) / 4
    right = np.full((rows, 2), -0.25, dtype=np.float32)
    assert execution.get_tensor_shape("features") == left.shape
    assert execution.get_tensor_shape("extra") == right.shape
    assert execution.get_tensor_shape("result") == left.shape
    assert execution.get_tensor_dtype("result") == trt.float32
    output = np.empty(
        execution.get_tensor_shape("result"), dtype=trt.nptype(execution.get_tensor_dtype("result"))
    )
    with ExitStack() as resources:
        if stream is None:
            stream = _check_cuda(cudart.cudaStreamCreate())
            resources.callback(lambda: _check_cuda(cudart.cudaStreamDestroy(stream)))
        pointers = {}
        for name, array in (("features", left), ("extra", right), ("result", output)):
            pointer = _check_cuda(cudart.cudaMalloc(array.nbytes))
            resources.callback(lambda pointer=pointer: _check_cuda(cudart.cudaFree(pointer)))
            pointers[name] = int(pointer)
            if name != "result":
                _check_cuda(
                    cudart.cudaMemcpyAsync(
                        pointer,
                        array.ctypes.data,
                        array.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    )
                )
        execution.enqueue(pointers, stream=int(stream))
        _check_cuda(
            cudart.cudaMemcpyAsync(
                output.ctypes.data,
                pointers["result"],
                output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            )
        )
        _check_cuda(cudart.cudaStreamSynchronize(stream))
    np.testing.assert_allclose(output, left + right, rtol=0, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.parametrize("point", [False, True])
def test_fixed_or_point_profile_executes_without_caller_shape_setting(tmp_path, monkeypatch, point):
    """Fixed and single-point profiles expose concrete sizes and compute on a caller's stream."""
    handle = _open_sum(tmp_path, monkeypatch, fixed=not point, point=point)
    with handle.create_execution() as execution:
        _run_sum(execution, 3)


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-018")
@pytest.mark.req("REQ-TENSORACLE-005")
def test_multiple_ranged_inputs_resize_outputs_and_keep_shapes(tmp_path, monkeypatch):
    """All ranged inputs change together at min, opt, max, and intermediate sizes before enqueue."""
    handle = _open_sum(tmp_path, monkeypatch, internal_shape=True)
    with handle.create_execution() as execution:
        for rows in (1, 3, 5, 4):
            execution.set_input_shapes({"features": (rows, 2), "extra": (rows, 2)})
            _run_sum(execution, rows)
        _run_sum(execution, 4)


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-018")
def test_same_handle_creates_independent_execution_units(tmp_path, monkeypatch):
    """Two execution units keep different input shapes and run on distinct caller streams."""
    handle = _open_sum(tmp_path, monkeypatch)
    with ExitStack() as resources:
        first = resources.enter_context(handle.create_execution())
        second = resources.enter_context(handle.create_execution())
        streams = [_check_cuda(cudart.cudaStreamCreate()) for _ in range(2)]
        assert int(streams[0]) != int(streams[1])
        for stream in streams:
            resources.callback(lambda stream=stream: _check_cuda(cudart.cudaStreamDestroy(stream)))
        first.set_input_shapes({"features": (2, 2), "extra": (2, 2)})
        second.set_input_shapes({"features": (5, 2), "extra": (5, 2)})
        _run_sum(first, 2, stream=streams[0])
        _run_sum(second, 5, stream=streams[1])
        _run_sum(first, 2, stream=streams[0])


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-018")
def test_host_shape_input_engine_is_rejected_at_creation(tmp_path):
    """A real host-resident external shape input identifies the tensor at creation."""
    _check_cuda(cudart.cudaFree(0))
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    features = network.add_input("features", trt.float32, (3, 2))
    dimensions = network.add_input("dimensions", trt.int32, (2,))
    dimensions.location = trt.TensorLocation.HOST
    reshape = network.add_shuffle(features)
    reshape.set_input(1, dimensions)
    output = reshape.get_output(0)
    output.name = "result"
    network.mark_output(output)
    profile = builder.create_optimization_profile()
    profile.set_shape_input("dimensions", (3, 2), (3, 2), (3, 2))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 0
    config.add_optimization_profile(profile)
    # Shape-value profiles are outside tensoracle's build API; construct this
    # unsupported engine at the external TensorRT boundary for the rejection test.
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(plan))
    assert engine is not None
    assert engine.is_shape_inference_io("dimensions")
    assert engine.get_tensor_location("dimensions") == trt.TensorLocation.HOST
    handle = tensoracle.EngineHandle(engine, runtime, tmp_path / "shape.engine")
    with pytest.raises(ValueError, match="dimensions.*HOST"):
        handle.create_execution()


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-018")
def test_data_dependent_output_is_rejected_before_enqueue(tmp_path, monkeypatch):
    """A real NonZero output is rejected at execution creation because its size depends on data."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights.safetensors"
    save_file({"bias": np.array([0.25], dtype=np.float32)}, weights)

    def graph(network):
        features = network.add_input("features", trt.float32, (3, 2))
        output = network.add_non_zero(features).get_output(0)
        output.name = "indices"
        network.mark_output(output)

    handle = tensoracle.open_engine(
        graph,
        tensoracle.EngineBuild(
            "fp32",
            {"features": tensoracle.InputShape.fixed((3, 2))},
            tensoracle.BuilderSettings(optimization_level=0),
        ),
        name="nonzero",
        graph_version="1",
        weights=(weights,),
        io=tensoracle.EngineIO(
            {"features": tensoracle.TensorSpec(trt.float32, (3, 2))},
            {"indices": tensoracle.TensorSpec(trt.int32, (2, -1))},
        ),
    )
    with pytest.raises(ValueError, match="indices.*data-dependent"):
        handle.create_execution()


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-018")
def test_cupy_buffers_execute_on_default_and_nondefault_streams(tmp_path, monkeypatch):
    """CuPy array addresses and both kinds of CuPy streams produce an independent NumPy sum."""
    import cupy as cp

    handle = _open_sum(tmp_path, monkeypatch)
    with handle.create_execution() as execution:
        execution.set_input_shapes({"features": (4, 2), "extra": (4, 2)})
        for stream in (cp.cuda.Stream.null, cp.cuda.Stream(non_blocking=True)):
            with stream:
                left_host = np.arange(8, dtype=np.float32).reshape(4, 2) / 4
                right_host = np.full((4, 2), -0.25, dtype=np.float32)
                left, right = cp.asarray(left_host), cp.asarray(right_host)
                output = cp.empty(
                    execution.get_tensor_shape("result"),
                    dtype=trt.nptype(execution.get_tensor_dtype("result")),
                )
                execution.enqueue(
                    {"features": left.data.ptr, "extra": right.data.ptr, "result": output.data.ptr},
                    stream=stream.ptr,
                )
                stream.synchronize()
                np.testing.assert_allclose(
                    cp.asnumpy(output), left_host + right_host, rtol=0, atol=1e-6
                )
