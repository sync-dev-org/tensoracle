"""Release unused host memory when the process allocator supports trimming."""

import ctypes
import ctypes.util


def release_host_memory() -> None:
    """Return free glibc heap pages without requiring libc on other platforms."""
    try:
        library = ctypes.util.find_library("c")
        if library is None:
            return
        trim = ctypes.CDLL(library).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (OSError, AttributeError):
        pass
