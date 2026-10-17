"""Separate saved engines by the CUDA device properties used for compatibility."""

import ctypes
from types import SimpleNamespace

import pytest
import tensorrt as trt

import tensoracle
import tensoracle._cache as cache


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

    def serialize(network, config):
        builds.append(cuda_driver.device)
        return b"serialized engine"

    builder = SimpleNamespace(
        create_network=lambda flags: SimpleNamespace(num_inputs=0, num_outputs=0),
        create_builder_config=lambda: SimpleNamespace(
            clear_flag=lambda flag: None,
            set_memory_pool_limit=lambda pool, limit: None,
        ),
        build_serialized_network=serialize,
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
            lambda network: None,
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
