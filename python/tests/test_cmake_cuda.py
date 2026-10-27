"""Consumers can inspect and link the same CUDA inputs as the cubin plugin helper."""

import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent if PROJECT.name == "python" else PROJECT
HELPER = ROOT / "cpp/cmake/TensoracleCubinPlugin.cmake"
FIND_CUDA = "tensoracle_find_cuda(OUT_INCLUDE_DIR cuda_include OUT_DRIVER_LIBRARY cuda_driver)\n"
ADD_PLUGIN = (
    "tensoracle_add_cubin_plugin(TARGET plugin\n"
    '  TENSORRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/sdk"\n'
    '  BUNDLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}"\n'
    '  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/plugin.cpp")\n'
)


def configure(source, build_name, *, cuda, root_input="cmake-root"):
    environment = dict(os.environ)
    environment.pop("CUDAToolkit_ROOT", None)
    arguments = []
    if root_input == "env-root":
        environment["CUDAToolkit_ROOT"] = str(cuda)
    elif root_input == "default-root":
        arguments.append(f"-DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT={cuda.as_posix()}")
    else:
        # The CMake variable takes precedence over an environment root.
        environment["CUDAToolkit_ROOT"] = str(source / "unused-cuda")
        arguments.append(f"-DCUDAToolkit_ROOT={cuda.as_posix()}")
    return subprocess.run(
        ["cmake", "-S", str(source), "-B", str(source / build_name), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )


def platform_settings(windows):
    if windows:
        return (
            "set(WIN32 TRUE)\n"
            'set(CMAKE_FIND_LIBRARY_PREFIXES "")\n'
            'set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")\n'
        )
    return (
        "set(WIN32 FALSE)\n"
        'set(CMAKE_FIND_LIBRARY_PREFIXES "lib")\n'
        'set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")\n'
    )


@pytest.mark.req("SP-TENSORACLE-040")
@pytest.mark.parametrize("windows", [False, True], ids=["linux", "windows"])
@pytest.mark.parametrize("root_input", ["cmake-root", "env-root", "default-root"])
@pytest.mark.parametrize("plugin_first", [False, True], ids=["paths-first", "plugin-first"])
def test_public_cuda_paths_match_plugin_and_linked_consumer(
    tmp_path, windows, root_input, plugin_first
):
    """CUDA paths match the plugin and linked consumers, including repeated discovery."""
    cuda = tmp_path / "wheel site-packages/nvidia/cu13"
    headers = cuda / "include"
    headers.mkdir(parents=True)
    (headers / "cuda.h").touch()
    driver = cuda / ("lib/x64/cuda.lib" if windows else "lib/libcuda.so")
    driver.parent.mkdir(parents=True)
    driver.touch()
    (tmp_path / "sdk/include").mkdir(parents=True)
    (tmp_path / "sdk/include/NvInfer.h").touch()
    (tmp_path / "sdk/lib").mkdir()
    (tmp_path / "sdk/lib" / ("nvinfer_11.lib" if windows else "libnvinfer.so")).touch()
    (tmp_path / "plugin.cpp").touch()
    (tmp_path / "bundle.cpp").touch()
    calls = ADD_PLUGIN + FIND_CUDA if plugin_first else FIND_CUDA + ADD_PLUGIN
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(check_cuda_paths LANGUAGES CXX)\n"
        + platform_settings(windows)
        + f'include("{HELPER.as_posix()}")\n'
        + calls
        + "tensoracle_find_cuda(OUT_DRIVER_LIBRARY repeated_driver\n"
        "  OUT_INCLUDE_DIR repeated_include)\n"
        'if(NOT "${cuda_include}" STREQUAL "${repeated_include}" OR\n'
        '   NOT "${cuda_driver}" STREQUAL "${repeated_driver}")\n'
        '  message(FATAL_ERROR "Repeated CUDA discovery changed the paths")\n'
        "endif()\n"
        "get_target_property(headers tensoracle_cuda_driver INTERFACE_INCLUDE_DIRECTORIES)\n"
        "get_target_property(driver tensoracle_cuda_driver IMPORTED_LOCATION)\n"
        "get_target_property(imported tensoracle_cuda_driver IMPORTED)\n"
        'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/public.txt"\n'
        '  "${cuda_include}\\n${cuda_driver}\\n${headers}\\n${driver}\\n${imported}")\n'
        "get_target_property(dependencies plugin LINK_LIBRARIES)\n"
        "foreach(dependency IN LISTS dependencies)\n"
        '  if(TARGET "${dependency}")\n'
        '    get_target_property(location "${dependency}" IMPORTED_LOCATION)\n'
        '    get_target_property(headers "${dependency}" INTERFACE_INCLUDE_DIRECTORIES)\n'
        "    if(location)\n"
        '      file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/plugin.txt" "${headers}\\n${location}")\n'
        "    endif()\n"
        "  endif()\n"
        "endforeach()\n"
        "add_library(consumer SHARED plugin.cpp)\n"
        "target_link_libraries(consumer PRIVATE tensoracle_cuda_driver)\n"
        "get_target_property(consumer_driver consumer LINK_LIBRARIES)\n"
        'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/consumer-$<CONFIG>.txt"\n'
        '  CONTENT "$<TARGET_PROPERTY:consumer,INCLUDE_DIRECTORIES>\\n'
        '$<TARGET_FILE:${consumer_driver}>")\n',
        encoding="utf-8",
    )
    result = configure(tmp_path, "build with spaces", cuda=cuda, root_input=root_input)
    assert result.returncode == 0, result.stdout + result.stderr
    build = tmp_path / "build with spaces"
    include, library, target_include, target_library, imported = (
        (build / "public.txt").read_text(encoding="utf-8").splitlines()
    )
    assert Path(include) == headers
    # Linux may prefer the installed driver's library over the wheel root.
    assert Path(library) == driver if windows else Path(library).name == "libcuda.so"
    assert [target_include, target_library, imported] == [include, library, "TRUE"]
    assert (build / "plugin.txt").read_text(encoding="utf-8").splitlines() == [include, library]
    for consumer in build.glob("consumer-*.txt"):
        assert consumer.read_text(encoding="utf-8").splitlines() == [include, library]
    assert list(build.glob("consumer-*.txt"))


@pytest.mark.req("SP-TENSORACLE-040")
@pytest.mark.parametrize("windows", [False, True], ids=["linux", "windows"])
@pytest.mark.parametrize("missing", ["headers", "driver"])
def test_public_cuda_discovery_reports_the_same_missing_input_as_plugin(tmp_path, windows, missing):
    """Missing CUDA headers and driver libraries report the plugin's discovery error."""
    cuda = tmp_path / "missing cuda"
    if missing == "driver":
        (cuda / "include").mkdir(parents=True)
        (cuda / "include/cuda.h").touch()
    errors = []
    for name, call in [("plugin", ADD_PLUGIN), ("public", FIND_CUDA)]:
        (tmp_path / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(missing_cuda LANGUAGES NONE)\n"
            + platform_settings(windows)
            # Isolate library lookup from any driver installed on the test host.
            + "set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)\n"
            "set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)\n"
            "set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)\n"
            "set(CMAKE_FIND_USE_CMAKE_PATH FALSE)\n"
            'set(CMAKE_IGNORE_PATH "/usr/lib/wsl/lib")\n'
            + f'include("{HELPER.as_posix()}")\n'
            + call,
            encoding="utf-8",
        )
        result = configure(tmp_path, f"build-{name}", cuda=cuda)
        assert result.returncode != 0
        message = result.stdout + result.stderr
        match = re.search(
            r"CMake Error at [^\n]*TensoracleCuda.cmake:\d+ \(message\):\n(.*?)\nCall Stack",
            message,
            re.DOTALL,
        )
        assert match, message
        error = " ".join(match.group(1).split())
        assert ("cuda.h" if missing == "headers" else "driver link library") in error
        assert cuda.as_posix() in error
        errors.append(error)
    assert errors[0] == errors[1]


@pytest.mark.req("SP-TENSORACLE-040")
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ("", "OUT_INCLUDE_DIR"),
        ("OUT_INCLUDE_DIR headers", "OUT_DRIVER_LIBRARY"),
        ("OUT_DRIVER_LIBRARY driver", "OUT_INCLUDE_DIR"),
        ("OUT_INCLUDE_DIR OUT_DRIVER_LIBRARY driver", "OUT_INCLUDE_DIR"),
        ("OUT_INCLUDE_DIR headers OUT_DRIVER_LIBRARY", "OUT_DRIVER_LIBRARY"),
        ("OUT_INCLUDE_DIR inc OUT_DRIVER_LIBRARY lib OUT_INCLUDE_DIR", "OUT_INCLUDE_DIR"),
        ("OUT_INCLUDE_DIR inc OUT_DRIVER_LIBRARY lib OUT_DRIVER_LIBRARY", "OUT_DRIVER_LIBRARY"),
        ("OUT_INCLUDE_DIR headers OUT_DRIVER_LIBRARY driver UNKNOWN value", "UNKNOWN"),
    ],
)
def test_public_cuda_discovery_rejects_invalid_output_arguments(tmp_path, arguments, expected):
    """CUDA discovery rejects missing output variable names and unknown arguments."""
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(invalid_cuda_arguments LANGUAGES NONE)\n"
        f'include("{HELPER.as_posix()}")\n'
        f"tensoracle_find_cuda({arguments})\n",
        encoding="utf-8",
    )
    result = configure(tmp_path, "build", cuda=tmp_path / "unused-cuda")
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "tensoracle_find_cuda" in message
    assert expected in message
