"""Configure native dependencies through the repository's Conan bootstrap."""

import os
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from native_build import PROJECT, ROOT


def configure(tmp_path, *options, environment=None):
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        f'include("{ROOT / "cpp/conan/bootstrap.cmake"}")\n'
        "project(dependency_probe LANGUAGES C CXX)\n"
        "include(CheckCXXSourceCompiles)\n"
        'check_cxx_source_compiles("#include <string>\\n'
        "static_assert(__GNUC__ == 13);\\n"
        "static_assert(_GLIBCXX_USE_CXX11_ABI == 1);\\n"
        'int main() {}" ABI_MATCHES)\n'
        "find_package(nlohmann_json CONFIG REQUIRED)\n"
        "find_package(miniz CONFIG REQUIRED)\n"
        "find_package(Catch2 CONFIG REQUIRED)\n"
        "find_package(TensorRT 11.2.1.2 EXACT CONFIG REQUIRED)\n"
        'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/sdk-includes.txt"\n'
        '  CONTENT "$<TARGET_PROPERTY:TensorRT::nvinfer,INTERFACE_INCLUDE_DIRECTORIES>")\n'
    )
    return subprocess.run(
        ["cmake", "-S", str(source), "-B", str(tmp_path / "build"), "-G", "Ninja", *options],
        env={**os.environ, **(environment or {})},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize("executable", ["missing", "wrong-version"])
def test_unavailable_or_different_conan_shows_the_pinned_install_command(tmp_path, executable):
    """Configure rejects an absent or different CLI and tells the developer how to install it."""
    cli = tmp_path / executable
    if executable == "wrong-version":
        cli.write_text("#!/bin/sh\nprintf 'Conan version 0.0.0\\n'\n")
        cli.chmod(0o755)
    requires = tomllib.loads((PROJECT / "pyproject.toml").read_text())["build-system"]["requires"]
    pinned = next(value for value in requires if value.startswith("conan=="))

    result = configure(tmp_path, f"-DTENSORACLE_CONAN_EXECUTABLE={cli}")

    assert result.returncode != 0
    assert f"uv tool install {pinned}" in result.stdout + result.stderr
    if executable == "wrong-version":
        assert "Conan version 0.0.0" in result.stdout + result.stderr


def test_unprepared_native_tests_fail_with_the_preparation_command(tmp_path):
    """An empty cache cannot trigger downloads, builds, or exports during test configure."""
    home = tmp_path / "home"
    storage = tmp_path / "packages"

    result = configure(
        tmp_path,
        "-DTENSORACLE_CONAN_OFFLINE=ON",
        f"-DTENSORACLE_CONAN_CACHE={storage}",
        environment={"CONAN_HOME": str(home)},
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "cmake" in output and "-P" in output and "prepare-tests.cmake" in output
    assert str(storage) in output
    assert "Exporting recipe" not in output
    assert not list(storage.glob("*/e/conanfile.py"))


def test_prepared_native_tests_configure_without_remotes_or_exports(tmp_path):
    """Prepared recipes supply the native libraries and TensorRT SDK without a remote."""
    assert shutil.which("conan"), "install the pinned Conan CLI before preparing tests"

    result = configure(tmp_path, "-DTENSORACLE_CONAN_OFFLINE=ON")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exporting recipe" not in result.stdout + result.stderr
    cache = (tmp_path / "build/CMakeCache.txt").read_text()
    assert "ABI_MATCHES:INTERNAL=1" in cache
    includes = (tmp_path / "build/sdk-includes.txt").read_text()
    assert includes
    assert any((Path(path) / "NvInferVersion.h").is_file() for path in includes.split(";"))


def test_native_tests_accept_a_different_system_cmake_without_changing_the_lock(tmp_path):
    """A different system CMake version leaves the locked native libraries usable offline."""
    original_cmake = shutil.which("cmake")
    tools = tmp_path / "tools"
    tools.mkdir()
    cmake = tools / "cmake"
    cmake.write_text(
        '#!/bin/sh\nif [ "$1" = "--version" ]; then\n'
        "  printf 'cmake version 3.28.4\\n'\n"
        f'else\n  exec {shlex.quote(original_cmake)} "$@"\nfi\n'
    )
    cmake.chmod(0o755)
    lock = ROOT / "cpp/conan.lock"
    before = lock.read_bytes()

    result = configure(
        tmp_path,
        "-DTENSORACLE_CONAN_OFFLINE=ON",
        environment={"PATH": str(tools) + os.pathsep + os.environ["PATH"]},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert lock.read_bytes() == before
