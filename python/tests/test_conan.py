"""Configure native dependencies using a prepared Conan toolchain."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from native_build import ROOT, prepare_native, tool_command


def configure(tmp_path, *options, environment=None, cuda=False):
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(dependency_probe LANGUAGES C CXX)\n"
        "include(CheckCXXSourceCompiles)\n"
        'check_cxx_source_compiles("#include <string>\\n'
        "#ifdef __GNUC__\\nstatic_assert(__GNUC__ == 13);\\n"
        "static_assert(_GLIBCXX_USE_CXX11_ABI == 1);\\n#endif\\n"
        'int main() {}" ABI_MATCHES)\n'
        "find_package(nlohmann_json CONFIG REQUIRED)\n"
        "find_package(miniz CONFIG REQUIRED)\n"
        "find_package(Catch2 CONFIG REQUIRED)\n"
        "find_package(TensorRT 11.2.1.2 EXACT CONFIG REQUIRED)\n"
        'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/sdk-includes.txt"\n'
        '  CONTENT "$<TARGET_PROPERTY:TensorRT::nvinfer,INTERFACE_INCLUDE_DIRECTORIES>")\n'
        + (
            f'include("{(ROOT / "cpp/cmake/TensoracleCuda.cmake").as_posix()}")\n'
            "tensoracle_find_cuda(OUT_INCLUDE_DIR headers OUT_DRIVER_LIBRARY driver)\n"
            'file(WRITE "${CMAKE_BINARY_DIR}/cuda.txt" "${CUDAToolkit_ROOT}\\n${headers}\\n${driver}")\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/probe.cpp" "#include <cuda_runtime_api.h>\\n'
            'int main() { return CUDART_VERSION < 13020; }\\n")\n'
            'add_executable(cuda_probe "${CMAKE_BINARY_DIR}/probe.cpp")\n'
            "target_link_libraries(cuda_probe PRIVATE tensoracle_cuda_driver)\n"
            if cuda
            else ""
        )
    )
    return subprocess.run(
        tool_command(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(tmp_path / "build"),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                prepare_native(tmp_path / "build"),
                *options,
            ]
        ),
        env={**os.environ, **(environment or {})},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.req("SP-TENSORACLE-063")
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


@pytest.mark.req("SP-TENSORACLE-070")
@pytest.mark.req("SP-TENSORACLE-040")
def test_prepared_cuda_build_uses_conan_headers_and_driver_without_pypi(tmp_path):
    """A prepared Conan toolchain supplies CUDA headers and a driver stub to C++ consumers."""
    result = configure(tmp_path, "-DTENSORACLE_CONAN_OFFLINE=ON", cuda=True)
    assert result.returncode == 0, result.stdout + result.stderr
    root, headers, driver = (tmp_path / "build/cuda.txt").read_text().splitlines()
    assert root and "site-packages" not in root
    assert Path(headers) == Path(root) / "include"
    assert (Path(headers) / "cuda_runtime_api.h").is_file()
    assert (Path(headers) / "crt/host_config.h").is_file()
    assert not (Path(root) / "bin/nvcc").exists()
    assert Path(driver) == Path(root) / (
        "lib/x64/cuda.lib" if os.name == "nt" else "lib64/stubs/libcuda.so"
    )
    compiled = subprocess.run(
        tool_command(["cmake", "--build", str(tmp_path / "build"), "--parallel", "2"]),
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
