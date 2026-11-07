"""Save compact engines and separate them by CUDA device compatibility."""

import ctypes
import time
from types import SimpleNamespace

import numpy as np
import pytest
import tensorrt as trt
from safetensors.numpy import save_file

import tensoracle
import tensoracle._cache as cache


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-009")
def test_compact_saved_engine_executes_and_reopens_without_calling_graph(tmp_path, monkeypatch):
    """A LeakyReLU engine stays below 1 MiB and computes correctly before and after reopening."""
    import cupy as cp

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights.safetensors"
    save_file({"slope": np.array([0.125], dtype=np.float16)}, weights)
    shape = (1, 64, 135, 240)
    graph_calls = []

    def graph(network):
        graph_calls.append(1)
        features = network.add_input("features", trt.float16, shape)
        layer = network.add_activation(features, trt.ActivationType.LEAKY_RELU)
        layer.alpha = 0.125
        result = layer.get_output(0)
        result.name = "result"
        network.mark_output(result)

    def open_engine(callback):
        return tensoracle.open_engine(
            callback,
            tensoracle.EngineBuild({"features": tensoracle.InputShape.fixed(shape)}),
            name="compact-leaky-relu",
            graph_version="1",
            weights=(weights,),
            io=tensoracle.EngineIO(
                {"features": tensoracle.TensorSpec(trt.float16, shape)},
                {"result": tensoracle.TensorSpec(trt.float16, shape)},
            ),
        )

    started = time.perf_counter()
    first = open_engine(graph)
    elapsed = time.perf_counter() - started
    size = first.path.stat().st_size
    print(f"LeakyReLU saved engine: {size} bytes, build/open: {elapsed:.6f} s")
    assert graph_calls == [1]
    assert size < 1024 * 1024

    def forbidden(network):
        pytest.fail("saved engine invoked the graph callback")

    reopened = open_engine(forbidden)
    assert reopened.path == first.path
    assert reopened.path.stat().st_size == size
    assert graph_calls == [1]
    host = np.linspace(-1, 1, np.prod(shape), dtype=np.float32).astype(np.float16).reshape(shape)
    expected = np.where(host >= 0, host, host * np.float16(0.125))
    value = cp.asarray(host)
    for handle in (first, reopened):
        output = cp.asarray(np.full_like(host, np.nan))
        with handle.create_execution() as execution:
            execution.enqueue(
                {"features": value.data.ptr, "result": output.data.ptr},
                stream=cp.cuda.get_current_stream().ptr,
            )
            np.testing.assert_array_equal(cp.asnumpy(output), expected)


@pytest.fixture
def cuda_driver(monkeypatch):
    properties = {
        "name": "NVIDIA RTX A6000",
        "total_memory": 49140 * 1024 * 1024,
        75: 8,  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        76: 6,  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        37: 384,  # CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
        38: 6 * 1024 * 1024,  # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
        8: 48 * 1024,  # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
        97: 99 * 1024,  # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        81: 100 * 1024,  # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        14: 512,  # CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT
        16: 84,  # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        18: 0,  # CU_DEVICE_ATTRIBUTE_INTEGRATED
        32: 0,  # CU_DEVICE_ATTRIBUTE_ECC_ENABLED
    }
    state = SimpleNamespace(
        device=0,
        context_status=0,
        devices=[properties.copy(), properties.copy()],
        failure=None,
        queries=[],
    )

    def initialize(flags):
        return 0

    def current_device(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = state.device
        return state.context_status

    def get_device(pointer, ordinal):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = ordinal
        return 0

    def get_name(buffer, length, device):
        buffer.value = state.devices[device]["name"].encode()
        return 0

    def get_attribute(pointer, attribute, device):
        state.queries.append(device)
        if state.failure == attribute:
            return 1
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = state.devices[device][attribute]
        return 0

    def total_memory(pointer, device):
        state.queries.append(device)
        if state.failure == "total_memory":
            return 1
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t))[0] = state.devices[device][
            "total_memory"
        ]
        return 0

    driver = SimpleNamespace(
        cuInit=initialize,
        cuCtxGetDevice=current_device,
        cuDeviceGet=get_device,
        cuDeviceGetName=get_name,
        cuDeviceGetAttribute=get_attribute,
        cuDeviceTotalMem_v2=total_memory,
    )
    original_find = ctypes.util.find_library
    original_load = ctypes.CDLL
    monkeypatch.setattr(
        ctypes.util,
        "find_library",
        lambda name: "test-cuda" if name == "cuda" else original_find(name),
    )
    monkeypatch.setattr(
        ctypes, "CDLL", lambda name: driver if name == "test-cuda" else original_load(name)
    )
    return state


@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.parametrize(
    "property,changed",
    [
        ("name", "Another GPU"),
        ("total_memory", 46068 * 1024 * 1024),
        (75, 9),
        (76, 9),
        (37, 256),
        (38, 4 * 1024 * 1024),
        (8, 64 * 1024),
        (97, 96 * 1024),
        (81, 128 * 1024),
        (14, 256),
        (16, 80),
        (18, 1),
        (32, 1),
    ],
    ids=[
        "name",
        "total-memory",
        "compute-major",
        "compute-minor",
        "memory-bus",
        "l2-cache",
        "shared-per-block",
        "shared-per-block-optin",
        "shared-per-sm",
        "texture-alignment",
        "sm-count",
        "integrated",
        "ecc-enabled",
    ],
)
def test_changed_gpu_property_builds_a_separate_saved_engine(
    tmp_path, monkeypatch, cuda_driver, property, changed
):
    """Different device properties select separate engines; matching devices reuse an engine."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    builds = []

    def graph(network):
        builds.append(cuda_driver.device)

    builder = SimpleNamespace(
        create_network=lambda flags: SimpleNamespace(num_inputs=0, num_outputs=0),
        create_builder_config=lambda: SimpleNamespace(
            clear_flag=lambda flag: None,
            set_memory_pool_limit=lambda pool, limit: None,
            create_timing_cache=lambda data: object(),
            set_timing_cache=lambda timing, ignore_mismatch: True,
        ),
        build_serialized_network=lambda network, config: b"serialized engine",
    )
    monkeypatch.setattr(trt, "Builder", lambda logger: builder)
    monkeypatch.setattr(
        trt,
        "Runtime",
        lambda logger: SimpleNamespace(
            deserialize_cuda_engine=lambda plan: SimpleNamespace(num_io_tensors=0)
        ),
    )
    weights = tmp_path / "weights"
    weights.write_bytes(b"same weights")

    def open_engine():
        return tensoracle.open_engine(
            graph,
            tensoracle.EngineBuild({}),
            name="encoder",
            graph_version="1",
            weights=(weights,),
            io=tensoracle.EngineIO({}, {}),
        )

    first = open_engine()
    cuda_driver.device = 1
    assert open_engine().path == first.path
    assert builds == [0]
    cuda_driver.devices[1][property] = changed
    second = open_engine()
    assert second.path != first.path
    assert builds == [0, 1]
    assert first.path.is_file() and second.path.is_file()
    cuda_driver.device = 0
    assert open_engine().path == first.path
    assert builds == [0, 1]


@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.req("SP-TENSORACLE-011")
@pytest.mark.parametrize("failure", [32, "total_memory"], ids=["ecc", "total-memory"])
def test_unreadable_gpu_property_cannot_select_a_saved_engine(
    tmp_path, monkeypatch, cuda_driver, failure
):
    """Failed device queries stop cache selection instead of sharing incomplete conditions."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    cuda_driver.failure = failure
    with pytest.raises(RuntimeError, match="CUDA device.*lookup failed"):
        cache._conditions("encoder", "1", (weights,), (), tensoracle.EngineBuild({}))


@pytest.mark.req("SP-TENSORACLE-009")
def test_without_a_current_cuda_context_properties_come_from_device_zero(
    tmp_path, monkeypatch, cuda_driver
):
    """With no current CUDA context, every device property belongs to the default device."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    cuda_driver.device = 1
    cuda_driver.context_status = 201
    cache._conditions("encoder", "1", (weights,), (), tensoracle.EngineBuild({}))
    assert cuda_driver.queries and set(cuda_driver.queries) == {0}
