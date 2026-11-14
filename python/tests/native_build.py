"""Build native test fixtures using the host's CMake toolchain and artifact names."""

import json
import os
import re
import runpy
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

NATIVE = Path(__file__).resolve().parent / "native"
PROJECT = NATIVE.parents[1]
ROOT = PROJECT.parent if PROJECT.name == "python" else PROJECT


def build_jobs():
    return os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "8")


@lru_cache
def preparation():
    return runpy.run_path(str(ROOT / "cpp/conan/prepare.py"))


def prepare_native(build):
    """Give a native fixture its own offline toolchain without editing user presets."""
    output = Path(build) / "conan"
    preparation()["install"](output, wheel=True, offline=True)
    return f"-DCMAKE_TOOLCHAIN_FILE={output / 'conan_toolchain.cmake'}"


def tool_command(command, environment=None):
    """Windows subprocess needs an absolute executable from the activated PATH."""
    command = list(map(str, command))
    if command[0] in ("cmake", "ctest", "ninja"):
        environment = os.environ if environment is None else environment
        executable = shutil.which(command[0], path=environment.get("PATH"))
        assert executable, f"Conan did not supply {command[0]}"
        command[0] = executable
    return command


def python_code(source):
    """Native Windows test children register the prepared SDK before loading bindings."""
    if os.name == "nt":
        source = (
            "import os\n"
            f"_test_sdk_directory = os.add_dll_directory({str(tensorrt_root() / 'bin')!r})\n"
            + source
        )
    return source


def tensorrt_recipe():
    """Read the embedded recipe or the exact recipe exported during preparation."""
    embedded = ROOT / "cpp/conan/recipes/tensorrt"
    if embedded.is_dir():
        return embedded
    lock = json.loads((ROOT / "cpp/conan.lock").read_text())
    reference = next(ref.split("%")[0] for ref in lock["requires"] if ref.startswith("tensorrt/"))
    cache = os.environ.get("TENSORACLE_CONAN_CACHE")
    cache_args = ["--core-conf", f"core.cache:storage_path={cache}"] if cache else []
    return Path(
        subprocess.check_output(
            ["conan", "cache", "path", reference, *cache_args], text=True
        ).strip()
    )


@lru_cache
def _dependency_root(name):
    """Locate a package in the same prepared, locked Conan graph as native builds."""
    command = [
        "conan",
        "graph",
        "info",
        str(ROOT / "cpp"),
        "--lockfile",
        str(ROOT / "cpp/conan.lock"),
        "--no-remote",
        "--format=json",
        *preparation()["profile_arguments"](),
    ]
    cache_args = []
    if cache := os.environ.get("TENSORACLE_CONAN_CACHE"):
        cache_args = ["--core-conf", f"core.cache:storage_path={cache}"]
        command.extend(cache_args)
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, (
        result.stdout + result.stderr + "\nPrepare dependencies: python cpp/conan/prepare-tests.py"
    )
    node = next(
        node
        for node in json.loads(result.stdout)["graph"]["nodes"].values()
        if node.get("name") == name
    )
    assert node["binary"] == "Cache", "Prepare dependencies: python cpp/conan/prepare-tests.py"
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
    return root


def tensorrt_root():
    """Locate the prepared TensorRT SDK without importing a runtime wheel."""
    root = _dependency_root("tensorrt")
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
    """Use an explicit test root or the prepared Conan CUDA SDK."""
    if root := os.environ.get("TENSORACLE_CUDA_ROOT"):
        return Path(root)
    root = _dependency_root("cuda-toolkit")
    assert (root / "include/cuda.h").is_file()
    return root


def _build(work, definitions):
    build = work / "build"
    commands = [
        [
            "cmake",
            "-S",
            str(NATIVE),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            *(f"-D{name}={value}" for name, value in definitions.items()),
        ],
        ["cmake", "--build", str(build), "--config", "Release"],
    ]
    for command in commands:
        result = subprocess.run(
            tool_command(command), check=False, capture_output=True, text=True, timeout=45
        )
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
