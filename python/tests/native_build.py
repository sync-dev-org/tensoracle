"""Build native test fixtures using the host's CMake toolchain and artifact names."""

import subprocess
from pathlib import Path

NATIVE = Path(__file__).resolve().parent / "native"
PROJECT = NATIVE.parents[1]
ROOT = PROJECT.parent if PROJECT.name == "python" else PROJECT


def _build(work, definitions):
    build = work / "build"
    commands = [
        [
            "cmake",
            "-S",
            str(NATIVE),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            *(f"-D{name}={value}" for name, value in definitions.items()),
        ],
        ["cmake", "--build", str(build), "--config", "Release"],
    ]
    for command in commands:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
    return Path((build / "artifact-Release.txt").read_text(encoding="utf-8").strip())


def build_native(work, name, sources, *, shared=False, includes=(), tools=None):
    """Build a library or executable, optionally with tensoracle's cubin helper."""
    definitions = {
        "TEST_TARGET": name,
        "TEST_SHARED": "ON" if shared else "OFF",
        "TEST_SOURCES": ";".join(path.as_posix() for path in sources),
        "TEST_INCLUDE_DIRS": ";".join(path.as_posix() for path in includes),
    }
    if tools is not None:
        sdk, cuda = tools
        definitions.update(
            TENSORACLE_CPP_ROOT=(ROOT / "cpp").as_posix(),
            TENSORRT_ROOT=sdk.as_posix(),
            CUDAToolkit_ROOT=cuda.as_posix(),
        )
    return _build(work, definitions)


def build_library(work, name, source):
    """Compile a C fixture whose functions and constants declare TEST_EXPORT."""
    directory = work / name
    directory.mkdir()
    path = directory / f"{name}.c"
    path.write_text('#include "export.h"\n' + source, encoding="utf-8")
    return build_native(directory, name, (path,), shared=True)


def build_cubin(work, source, tools, sm):
    """Ask CMake to locate nvcc and compile a cubin for the test GPU."""
    _, cuda = tools
    return _build(
        work,
        {
            "TEST_CUBIN_SOURCE": source.as_posix(),
            "TEST_CUDA_ARCH": sm,
            "CUDAToolkit_ROOT": cuda.as_posix(),
        },
    )
