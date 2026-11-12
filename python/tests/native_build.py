"""Build native test fixtures using the host's CMake toolchain and artifact names."""

import json
import os
import re
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

NATIVE = Path(__file__).resolve().parent / "native"
PROJECT = NATIVE.parents[1]
ROOT = PROJECT.parent if PROJECT.name == "python" else PROJECT


@lru_cache
def tensorrt_root():
    """Locate the SDK in the same prepared, locked Conan graph as native builds."""
    profile = Path(os.environ.get("TENSORACLE_CONAN_PROFILE", "linux-gcc13"))
    if not profile.is_absolute():
        profile = ROOT / "cpp/conan/profiles" / profile
    command = [
        "conan",
        "graph",
        "info",
        str(ROOT / "cpp"),
        "--lockfile",
        str(ROOT / "cpp/conan.lock"),
        "--no-remote",
        "--format=json",
        "--profile:host",
        str(profile),
        "--profile:build",
        str(profile),
    ]
    cache_args = []
    if cache := os.environ.get("TENSORACLE_CONAN_CACHE"):
        cache_args = ["--core-conf", f"core.cache:storage_path={cache}"]
        command.extend(cache_args)
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, (
        result.stdout
        + result.stderr
        + "\nPrepare dependencies: cmake -P cpp/conan/prepare-tests.cmake"
    )
    node = next(
        node
        for node in json.loads(result.stdout)["graph"]["nodes"].values()
        if node.get("name") == "tensorrt"
    )
    assert node["binary"] == "Cache", "Prepare dependencies: cmake -P cpp/conan/prepare-tests.cmake"
    reference = f"{node['ref']}:{node['package_id']}#{node['prev']}"
    path = subprocess.run(
        ["conan", "cache", "path", reference, *cache_args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert path.returncode == 0, path.stdout + path.stderr
    root = Path(path.stdout.strip())
    assert (root / "include/NvInfer.h").is_file()
    return root


def sdk_version():
    """Read the expected library version from the public SDK header."""
    header = (ROOT / "cpp/include/tensoracle/tensoracle.h").read_text(encoding="utf-8")
    return tuple(
        int(re.search(rf"^#define TENSORACLE_VERSION_{part} (\d+)$", header, re.MULTILINE)[1])
        for part in ("MAJOR", "MINOR", "PATCH")
    )


def cuda_root():
    """Use an explicit test root or the installed PyPI CUDA header directory."""
    if root := os.environ.get("TENSORACLE_CUDA_ROOT"):
        return Path(root)
    try:
        return Path(distribution("nvidia-cuda-runtime").locate_file("nvidia/cu13"))
    except PackageNotFoundError:
        return None


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
