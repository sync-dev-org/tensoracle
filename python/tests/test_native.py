"""The Python binding calls the shared C++ library without a CUDA device."""

import ctypes
from importlib.metadata import distribution

import pytest

from native_build import sdk_version


@pytest.mark.req("SP-TENSORACLE-053")
@pytest.mark.req("SP-TENSORACLE-056")
def test_native_versions_are_available_without_initializing_cuda():
    """The binding reports tensoracle and TensorRT versions as three integers."""
    from tensoracle import _native

    assert _native.version() == sdk_version()
    tensorrt = _native.tensorrt_version()
    assert len(tensorrt) == 3
    assert all(isinstance(value, int) for value in tensorrt)
    assert tensorrt[0] == 11
    assert tensorrt[1] >= 2
    library = ctypes.CDLL(
        str(distribution("tensorrt-cu13-libs").locate_file("tensorrt_libs/libnvinfer.so.11"))
    )
    library.getInferLibVersion.argtypes = []
    library.getInferLibVersion.restype = ctypes.c_int32
    assert tensorrt[0] * 10000 + tensorrt[1] * 100 + tensorrt[2] == library.getInferLibVersion()
