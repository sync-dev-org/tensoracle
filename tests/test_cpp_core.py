"""The source-distributed C++ helpers preserve fields and verify cubin bytes."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.req("REQ-TENSORACLE-003")
def test_cpp_helpers_keep_field_order_and_check_digest(tmp_path):
    """The native helper retains typed fields and computes a real SHA-256 digest."""
    sdk = os.environ.get("TENSORACLE_SDK_ROOT")
    cuda = os.environ.get("TENSORACLE_CUDA_ROOT")
    if not sdk or not cuda:
        pytest.skip("TensorRT SDK and CUDA toolkit were not supplied")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "check.cpp"
    source.write_text(
        r"""
#include "tensoracle/cubin_plugin.hpp"
#include <cstring>
using namespace tensoracle::cubin;
using namespace nvinfer1;
int main() {
    if (sha256("abc", 3) !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") return 1;
    float weight[2] = {1.25f, 2.5f};
    int version = 7;
    PluginField input[] = {
        {"weight", weight, PluginFieldType::kFLOAT32, 2},
        {"version", &version, PluginFieldType::kINT32, 1},
    };
    PluginFieldCollection source{2, input};
    Fields fields;
    if (!Fields::parse(&source, fields)) return 2;
    auto const* saved = fields.collection();
    if (saved->nbFields != 2 ||
        std::strcmp(saved->fields[0].name, "weight") != 0 ||
        std::strcmp(saved->fields[1].name, "version") != 0 ||
        saved->fields[0].type != PluginFieldType::kFLOAT32 ||
        saved->fields[0].length != 2 ||
        std::memcmp(saved->fields[0].data, weight, sizeof(weight)) != 0) return 3;
    weight[0] = 9.0f;
    if (std::memcmp(saved->fields[0].data, weight, sizeof(weight)) == 0) return 4;
    Fields copy = fields;
    if (std::memcmp(copy.collection()->fields[0].data, saved->fields[0].data,
                    sizeof(weight)) != 0) return 5;
    return 0;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "check"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wno-deprecated-declarations",
            f"-I{root / 'cpp/include'}",
            f"-I{Path(sdk) / 'include'}",
            f"-I{Path(cuda) / 'include'}",
            str(source),
            str(root / "cpp/src/cubin_plugin.cpp"),
            f"-L{Path(cuda) / 'lib64/stubs'}",
            "-lcuda",
            "-o",
            str(executable),
        ],
        check=True,
        timeout=45,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


@pytest.mark.req("REQ-TENSORACLE-003")
def test_cubin_launcher_rejects_launches_outside_generated_contract(tmp_path):
    """Wrong argument counts and unmet specialization rules fail before CUDA work."""
    sdk = os.environ.get("TENSORACLE_SDK_ROOT")
    cuda = os.environ.get("TENSORACLE_CUDA_ROOT")
    if not sdk or not cuda:
        pytest.skip("TensorRT SDK and CUDA toolkit were not supplied")
    root = Path(__file__).resolve().parents[1]
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
    executable = tmp_path / "wrong-count"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wno-deprecated-declarations",
            f"-I{root / 'cpp/include'}",
            f"-I{Path(sdk) / 'include'}",
            f"-I{Path(cuda) / 'include'}",
            str(source),
            str(root / "cpp/src/cubin_plugin.cpp"),
            str(root / "examples/cubin_plugin/bundle/bundle.cpp"),
            f"-L{Path(cuda) / 'lib64/stubs'}",
            "-lcuda",
            "-o",
            str(executable),
        ],
        check=True,
        timeout=45,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
