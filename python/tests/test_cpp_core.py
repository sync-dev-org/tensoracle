"""The source-distributed C++ helpers preserve fields and verify cubin bytes."""

import os
import subprocess
from pathlib import Path

import pytest

from native_build import build_native


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.req("SP-TENSORACLE-021")
def test_cpp_helpers_keep_field_order_and_check_digest(tmp_path):
    """The native helper retains typed fields and computes a real SHA-256 digest."""
    sdk = os.environ.get("TENSORACLE_SDK_ROOT")
    cuda = os.environ.get("TENSORACLE_CUDA_ROOT")
    if not sdk or not cuda:
        pytest.skip("TensorRT SDK and CUDA toolkit were not supplied")
    project = Path(__file__).resolve().parents[1]
    root = project.parent if project.name == "python" else project
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(root / "cpp"),
            "-B",
            str(build),
            f"-DTENSORRT_ROOT={sdk}",
            f"-DCUDAToolkit_ROOT={cuda}",
        ],
        check=True,
        timeout=45,
    )
    subprocess.run(["cmake", "--build", str(build), "--config", "Release"], check=True, timeout=45)
    subprocess.run(
        ["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"],
        check=True,
        timeout=10,
    )


@pytest.mark.req("SP-TENSORACLE-004")
def test_cubin_launcher_rejects_launches_outside_generated_contract(tmp_path):
    """Wrong argument counts and unmet specialization rules fail before CUDA work."""
    sdk = os.environ.get("TENSORACLE_SDK_ROOT")
    cuda = os.environ.get("TENSORACLE_CUDA_ROOT")
    if not sdk or not cuda:
        pytest.skip("TensorRT SDK and CUDA toolkit were not supplied")
    project = Path(__file__).resolve().parents[1]
    root = project.parent if project.name == "python" else project
    source = tmp_path / "wrong-count.cpp"
    source.write_text(
        r"""
#include "tensoracle/cubin_plugin.hpp"
using namespace tensoracle::cubin;
int main() {
    auto const* image = find_bundle_image("sm120.dot.specialized");
    if (image == nullptr) return 2;
    Kernel kernel(*image);
    CUdeviceptr x = 0x1000, weight = 0x2000, output = 0x3000;
    int rows = 16;
    void* arguments[] = {&x, &weight, &output, &rows};
    if (kernel.launch(1, 128, arguments, 3, nullptr) != CUDA_ERROR_INVALID_VALUE) return 1;
    rows = 3;
    if (kernel.launch(1, 128, arguments, 4, nullptr) != CUDA_ERROR_INVALID_VALUE) return 3;
    rows = 16;
    x = 0x1004;
    if (kernel.launch(1, 128, arguments, 4, nullptr) != CUDA_ERROR_INVALID_VALUE) return 4;
    x = 0x1000;
    if (!kernel.accepts(arguments, 4)) return 5;
    auto const* plain_image = find_bundle_image("sm120.dot.unspecialized");
    if (plain_image == nullptr) return 6;
    Kernel plain(*plain_image);
    rows = 3;
    if (!plain.accepts(arguments, 4)) return 7;
    return 0;
}
""",
        encoding="utf-8",
    )
    executable = build_native(
        tmp_path,
        "wrong-count",
        (source, root / "examples/cubin_plugin/bundle/bundle.cpp"),
        tools=(Path(sdk), Path(cuda)),
    )
    subprocess.run([str(executable)], check=True, timeout=10)
