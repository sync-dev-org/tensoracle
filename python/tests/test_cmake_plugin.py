"""The CMake plugin helper locates the platform's standard TensorRT library."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.req("SP-TENSORACLE-015")
@pytest.mark.parametrize("windows", [False, True], ids=["linux-sdk", "windows-sdk"])
def test_plugin_helper_finds_standard_tensorrt_library(tmp_path, windows):
    """The SDK root alone selects nvinfer on Linux and nvinfer_11.lib on Windows."""
    project = Path(__file__).resolve().parents[1]
    root = project.parent if project.name == "python" else project
    sdk = tmp_path / "TensorRT SDK"
    (sdk / "include").mkdir(parents=True)
    (sdk / "include/NvInfer.h").touch()
    (sdk / "lib").mkdir()
    library = sdk / "lib" / ("nvinfer_11.lib" if windows else "libnvinfer.so")
    library.touch()
    # The CUDA SDK is external to this configure-only check; no compiler/linker runs on its stub.
    (tmp_path / "FindCUDAToolkit.cmake").write_text(
        "add_library(CUDA::cuda_driver INTERFACE IMPORTED)\n", encoding="utf-8"
    )
    (tmp_path / "plugin.cpp").touch()
    (tmp_path / "bundle.cpp").touch()
    platform = (
        'set(WIN32 TRUE)\nset(CMAKE_FIND_LIBRARY_PREFIXES "")\n'
        'set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")\n'
        if windows
        else 'set(WIN32 FALSE)\nset(CMAKE_FIND_LIBRARY_PREFIXES "lib")\n'
        'set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")\n'
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(check_sdk LANGUAGES CXX)\n"
        + platform
        + 'list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")\n'
        + f'include("{root.as_posix()}/cpp/cmake/TensoracleCubinPlugin.cmake")\n'
        + "tensoracle_add_cubin_plugin(TARGET check_sdk\n"
        '  TENSORRT_ROOT "${SDK_ROOT}" BUNDLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}"\n'
        '  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/plugin.cpp")\n'
        'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/library.txt" "${TENSORACLE_NVINFER_LIBRARY}")\n',
        encoding="utf-8",
    )
    build = tmp_path / "build with spaces"
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build), f"-DSDK_ROOT={sdk.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path((build / "library.txt").read_text(encoding="utf-8")) == library
