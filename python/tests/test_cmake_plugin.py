"""The CMake plugin helper locates the platform's standard TensorRT library."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.req("SP-TENSORACLE-015")
@pytest.mark.req("SP-TENSORACLE-038")
@pytest.mark.parametrize("windows", [False, True], ids=["linux-sdk", "windows-sdk"])
@pytest.mark.parametrize("root_in_environment", [False, True], ids=["cmake-root", "env-root"])
def test_plugin_helper_finds_standard_tensorrt_library(tmp_path, windows, root_in_environment):
    """The plugin finds TensorRT and wheel CUDA inputs without a complete OS toolkit."""
    project = Path(__file__).resolve().parents[1]
    root = project.parent if project.name == "python" else project
    sdk = tmp_path / "TensorRT SDK"
    (sdk / "include").mkdir(parents=True)
    (sdk / "include/NvInfer.h").touch()
    (sdk / "lib").mkdir()
    library = sdk / "lib" / ("nvinfer_11.lib" if windows else "libnvinfer.so")
    library.touch()
    cuda = tmp_path / "wheel site-packages/nvidia/cu13"
    (cuda / "include").mkdir(parents=True)
    (cuda / "include/cuda.h").touch()
    driver = cuda / ("lib/x64/cuda.lib" if windows else "lib/libcuda.so")
    driver.parent.mkdir(parents=True)
    driver.touch()
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
        # A consumer's existing CUDA target must not conflict with tensoracle's helper.
        + "add_library(CUDA::cuda_driver INTERFACE IMPORTED)\n"
        + f'include("{root.as_posix()}/cpp/cmake/TensoracleCubinPlugin.cmake")\n'
        + "tensoracle_add_cubin_plugin(TARGET check_sdk\n"
        '  TENSORRT_ROOT "${SDK_ROOT}" BUNDLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}"\n'
        '  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/plugin.cpp")\n'
        'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/library.txt" "${TENSORACLE_NVINFER_LIBRARY}")\n'
        "get_target_property(dependencies check_sdk LINK_LIBRARIES)\n"
        "foreach(dependency IN LISTS dependencies)\n"
        '  if(TARGET "${dependency}")\n'
        '    get_target_property(location "${dependency}" IMPORTED_LOCATION)\n'
        '    get_target_property(headers "${dependency}" INTERFACE_INCLUDE_DIRECTORIES)\n'
        "    if(location)\n"
        '      file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cuda.txt" "${location}\\n${headers}")\n'
        "    endif()\n"
        "  endif()\n"
        "endforeach()\n",
        encoding="utf-8",
    )
    build = tmp_path / "build with spaces"
    environment = dict(os.environ)
    arguments = []
    if root_in_environment:
        environment["CUDAToolkit_ROOT"] = str(cuda)
    else:
        environment.pop("CUDAToolkit_ROOT", None)
        arguments.append(f"-DCUDAToolkit_ROOT={cuda.as_posix()}")
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DSDK_ROOT={sdk.as_posix()}",
            *arguments,
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path((build / "library.txt").read_text(encoding="utf-8")) == library
    location, headers = (build / "cuda.txt").read_text(encoding="utf-8").splitlines()
    # Linux may prefer the installed driver; Windows uses the wheel's import library.
    assert Path(location) == driver if windows else Path(location).name == "libcuda.so"
    assert str(cuda / "include") in headers
    assert location in result.stdout
    assert not any(
        line.startswith("_tensoracle_") and ":" in line
        for line in (build / "CMakeCache.txt").read_text().splitlines()
    )


@pytest.mark.req("SP-TENSORACLE-038")
def test_plugin_build_reports_missing_cuda_inputs(tmp_path):
    """Missing CUDA headers identify the searched root and the PyPI root configuration."""
    project = Path(__file__).resolve().parents[1]
    root = project.parent if project.name == "python" else project
    sdk = tmp_path / "sdk"
    (sdk / "include").mkdir(parents=True)
    (sdk / "include/NvInfer.h").touch()
    missing = tmp_path / "absent-cuda"
    build = tmp_path / "build"
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(root / "cpp"),
            "-B",
            str(build),
            f"-DTENSORRT_ROOT={sdk}",
            f"-DCUDAToolkit_ROOT={missing}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert str(missing / "include") in message
    assert "cuda.h" in message
    assert "nvidia/cu13" in message
    assert "CUDAToolkit_ROOT" in message
