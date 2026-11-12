"""Source builds can embed tensoracle through the installed CMake target."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from native_build import ROOT


def run(*command, **kwargs):
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=180, **kwargs
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def static_sdk(tmp_path_factory):
    work = tmp_path_factory.mktemp("static-sdk")
    build = work / "library"
    environment = {**os.environ, "TENSORACLE_CONAN_OFFLINE": "ON"}
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("LD_PRELOAD", None)
    run(
        "cmake",
        "-S",
        str(ROOT / "cpp"),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DBUILD_TESTING=OFF",
        env=environment,
    )
    run("cmake", "--build", str(build), "--parallel", "8", env=environment)
    prefix = work / "prefix"
    run("cmake", "--install", str(build), "--prefix", str(prefix), env=environment)
    relocated = work / "relocated"
    prefix.rename(relocated)
    return work, build, relocated, environment


@pytest.mark.req("SP-TENSORACLE-074")
def test_static_archive_contains_only_tensoracle_objects(static_sdk):
    """The installed archive defines tensoracle operations without bundling its dependencies."""
    _, _, prefix, _ = static_sdk
    archives = list(prefix.rglob("libtensoracle.a"))
    assert len(archives) == 1, "BUILD_SHARED_LIBS=OFF must install libtensoracle.a"
    assert not list(prefix.rglob("libtensoracle.so*"))
    symbols = run("nm", "--defined-only", str(archives[0]))
    assert re.search(r"\bT tensoracle_open_engine$", symbols, re.MULTILINE)
    assert not re.search(r"\b(?:mz_|miniz_|tdefl_|tinfl_|cuda|__cuda)\w*$", symbols, re.MULTILINE)
    members = run("ar", "t", str(archives[0])).splitlines()
    assert members and all(member.endswith(".cpp.o") for member in members)
    for config in prefix.rglob("*.cmake"):
        text = config.read_text()
        assert str(ROOT) not in text
        assert "/.conan2/" not in text
        assert "tensoracle_json" not in text
        assert "tensoracle_miniz" not in text


@pytest.mark.req("SP-TENSORACLE-074")
@pytest.mark.parametrize("language,compiler", [("c", "cc"), ("c++", "c++")])
def test_static_windows_headers_do_not_import_or_export_dlls(tmp_path, language, compiler):
    """The static definition makes the public C declarations usable without DLL attributes."""
    source = tmp_path / "declarations.h"
    source.write_text("#include <tensoracle/builder.h>\n")
    run(
        compiler,
        "-x",
        language,
        "-fsyntax-only",
        "-Werror",
        "-D_WIN32",
        "-DTENSORACLE_STATIC_LIBRARY",
        "-I",
        str(ROOT / "cpp/include"),
        str(source),
    )


@pytest.fixture(scope="module")
def static_consumers(static_sdk):
    work, library, prefix, environment = static_sdk
    source = work / "source"
    shutil.copytree(ROOT / "cpp/tests/static_consumer", source)
    build = work / "consumers"
    run(
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_TOOLCHAIN_FILE={library / 'conan/conan_toolchain.cmake'}",
        env=environment,
    )
    run("cmake", "--build", str(build), "--parallel", "8", env=environment)
    # The consumer keeps the installed SDK and resolved dependencies only.
    shutil.rmtree(source)
    return work, build, environment


@pytest.mark.req("SP-TENSORACLE-073")
@pytest.mark.req("SP-TENSORACLE-074")
def test_c_and_cpp_consumers_embed_the_installed_static_library(static_consumers):
    """Both languages link the same installed target without a tensoracle shared dependency."""
    _, build, environment = static_consumers
    for name in ("consumer_c", "consumer_cpp"):
        executable = build / name
        assert "libtensoracle.so" not in run("readelf", "-d", str(executable))
        run(str(executable), env=environment)


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-054")
@pytest.mark.req("SP-TENSORACLE-073")
@pytest.mark.req("SP-TENSORACLE-074")
def test_static_c_and_cpp_consumers_build_execute_and_reuse_engines(static_consumers):
    """Each embedded consumer builds on the host TensorRT, executes, then reuses its engine."""
    import numpy as np
    from safetensors.numpy import save_file

    work, build, environment = static_consumers
    weights = work / "bias.safetensors"
    save_file({"bias": np.array([0.5, -2, 3.25, -0.75], dtype=np.float32)}, weights)
    import tensorrt_libs

    runtime = Path(tensorrt_libs.__file__).parent
    environment = {
        **environment,
        "LD_LIBRARY_PATH": str(runtime),
        "XDG_CACHE_HOME": str(work / "cache"),
    }
    for name in ("consumer_c", "consumer_cpp"):
        for builds in (1, 0):
            run(str(build / name), str(weights), str(builds), env=environment)
