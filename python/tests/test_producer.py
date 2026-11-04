"""The C++ CUDA producer uses PyPI tools without loading a GPU runtime."""

import json
import os
import subprocess
import sys
import sysconfig
from dataclasses import replace
from importlib.metadata import requires
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tensoracle.bundle import inspect_bundle
from tensoracle.producer import KernelSpec, compile_bundle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'extern "C" __global__ void copy_kernel(const float* x, float* y, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]*FACTOR; }'
SPEC = KernelSpec(
    "copy", SOURCE, (("x", "*fp32"), ("y", "*fp32"), ("n", "i32")), {"FACTOR": 2}, "copy_kernel"
)


@pytest.fixture
def producer_tools():
    root = Path(sysconfig.get_path("purelib")) / "nvidia/cu13"
    paths = {
        "ptxas": root / "bin/ptxas",
        "cuobjdump": root / "bin/cuobjdump",
        "nvrtc": root / "lib/libnvrtc.so.13",
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("requires tensoracle[producer] with CUDA 13.2 NVRTC and tools")
    return paths


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.req("SP-TENSORACLE-037")
@pytest.mark.req("SP-TENSORACLE-038")
@pytest.mark.req("SP-TENSORACLE-058")
@pytest.mark.parametrize("explicit_tools", [False, True], ids=["wheel-defaults", "explicit-paths"])
def test_example_producer_builds_three_architectures_without_torch(
    tmp_path, producer_tools, explicit_tools
):
    """CUDA C++ example kernels produce all SMs with ordered public arguments and no scratch."""
    output = tmp_path / "bundle"
    source = """
import sys
from pathlib import Path
from tensoracle.producer import compile_bundle
from examples.cubin_plugin.kernels import kernel_specs
assert 'torch' not in sys.modules
args = dict(zip(('ptxas', 'cuobjdump', 'nvrtc'), map(Path, sys.argv[2:])))
compile_bundle(kernel_specs(), Path(sys.argv[1]), **args)
assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment["CUDA_HOME"] = str(tmp_path / "unused-toolkit")
    arguments = (
        [str(producer_tools[name]) for name in ("ptxas", "cuobjdump", "nvrtc")]
        if explicit_tools
        else []
    )
    subprocess.run(
        [sys.executable, "-c", source, str(output), *arguments],
        env=environment,
        cwd=ROOT,
        check=True,
        timeout=120,
    )
    manifest = json.loads((output / "bundle.json").read_text())
    assert manifest["schema_version"] == 2
    assert {entry["target_sm"] for entry in manifest["entries"]} == {"sm86", "sm89", "sm120"}
    assert "triton" not in manifest["producer"]
    assert Requirement("ptxas>=13.2,<14").specifier.contains(manifest["producer"]["ptxas"])
    assert inspect_bundle(output) == manifest["bundle_sha256"]
    for entry in manifest["entries"]:
        assert "hidden_trailing_parameters" not in entry
        assert "global_scratch_bytes" not in entry["resources"]
        assert "profile_scratch_bytes" not in entry["resources"]
        assert entry["resources"]["registers_per_thread"] > 0
    committed = ROOT / "examples/cubin_plugin/bundle"
    assert {p.relative_to(output) for p in output.rglob("*") if p.is_file()} == {
        p.relative_to(committed) for p in committed.rglob("*") if p.is_file()
    }


@pytest.mark.req("SP-TENSORACLE-038")
@pytest.mark.req("SP-TENSORACLE-058")
def test_package_declares_cubin_tools_only_in_producer_extra():
    """Only the producer extra supplies CUDA 13.2 NVRTC, ptxas and cuobjdump, with no Triton."""
    expected = {
        "nvidia-cuda-nvrtc",
        "nvidia-cuda-nvcc",
        "nvidia-cuda-cuobjdump",
        "nvidia-cuda-crt",
        "nvidia-nvvm",
    }
    dependencies = [Requirement(item) for item in requires("tensoracle") or []]
    producer = [
        item
        for item in dependencies
        if item.marker is not None and item.marker.evaluate({"extra": "producer"})
    ]
    assert {item.name for item in producer} == expected
    assert all(str(item.specifier) == "==13.2.*" for item in producer)
    assert all(
        not item.marker.evaluate({"extra": ""}) and not item.marker.evaluate({"extra": "convert"})
        for item in producer
    )


@pytest.mark.req("SP-TENSORACLE-037")
@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("12.9.86", False),
        ("13.1.115", False),
        ("13.2.0", True),
        ("13.3.73", True),
        ("14.0.10", False),
    ],
)
def test_producer_checks_ptxas_range_before_writing(tmp_path, version, supported):
    """The actual C++ subprocess path checks ptxas before validating kernels or creating output."""
    tool = tmp_path / "ptxas"
    tool.write_text(f"#!/bin/sh\nprintf 'ptxas V{version}\\n'\n")
    tool.chmod(0o755)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError) as error:
        compile_bundle((), output, ptxas=tool, cuda_root=tmp_path)
    if supported:
        assert "kernel names" in str(error.value)
    else:
        assert all(text in str(error.value) for text in ("ptxas", ">=13.2", "<14", version))
    assert not output.exists()


@pytest.mark.req("SP-TENSORACLE-005")
def test_utility_imports_leave_producer_and_triton_unloaded():
    """Utility imports load neither the CUDA producer nor Triton."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tensoracle,sys; from tensoracle import compare_outputs; assert 'tensoracle.producer' not in sys.modules; assert not any(n=='triton' or n.startswith('triton.') for n in sys.modules)",
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.req("SP-TENSORACLE-038")
@pytest.mark.parametrize("configured", [False, True])
def test_producer_uses_wheel_tools_without_cuda_home(tmp_path, monkeypatch, configured):
    """C++ tool discovery uses the supplied wheel root independently of CUDA_HOME."""
    tools = tmp_path / "nvidia/cu13/bin"
    tools.mkdir(parents=True)
    ptxas = tools / "ptxas"
    ptxas.write_text("#!/bin/sh\nprintf 'ptxas V13.2.0\\n'\n")
    ptxas.chmod(0o755)
    if configured:
        monkeypatch.setenv("CUDA_HOME", str(tmp_path / "host-toolkit"))
    else:
        monkeypatch.delenv("CUDA_HOME", raising=False)
    with pytest.raises(ValueError, match="kernel names"):
        compile_bundle((), tmp_path / "bundle", cuda_root=tools.parent)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.req("SP-TENSORACLE-058")
def test_producer_rejects_python_kernel_functions(tmp_path):
    """The CUDA C++ producer cannot accept a Python or Triton callable as source."""
    with pytest.raises((TypeError, RuntimeError)):
        compile_bundle((replace(SPEC, source=lambda: None),), tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.req("SP-TENSORACLE-038")
@pytest.mark.parametrize("tool", ["ptxas", "cuobjdump", "nvrtc"])
def test_missing_wheel_tool_reports_package_path_and_configuration(tmp_path, producer_tools, tool):
    """Missing C++ tool paths identify the supplying wheel and the producer extra."""
    kwargs = dict(producer_tools)
    kwargs[tool] = tmp_path / f"missing-{tool}"
    with pytest.raises(ImportError) as error:
        compile_bundle((SPEC,), tmp_path / "bundle", **kwargs)
    expected = {
        "ptxas": "nvidia-cuda-nvcc",
        "cuobjdump": "nvidia-cuda-cuobjdump",
        "nvrtc": "nvidia-cuda-nvrtc",
    }
    assert expected[tool] in str(error.value)
    assert f"missing-{tool}" in str(error.value)
    assert "tensoracle[producer]" in str(error.value)
    assert f"{tool}=" in str(error.value)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.req("SP-TENSORACLE-058")
@pytest.mark.parametrize(
    ("change", "text"),
    [
        ({"signature": (("x", "i64"),)}, "argument"),
        ({"signature": (("x", "*fp32"), ("y", "*fp32"))}, "layout"),
        ({"source": "not CUDA C++"}, "NVRTC"),
        ({"constexprs": {"BAD-NAME": 1}}, "constant"),
    ],
)
def test_invalid_cuda_source_signature_and_constants_leave_no_output(
    tmp_path, producer_tools, change, text
):
    """Invalid CUDA input and declared argument layouts fail in C++ before publishing any files."""
    with pytest.raises(ValueError, match=text):
        compile_bundle((replace(SPEC, **change),), tmp_path / "bundle", **producer_tools)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.req("SP-TENSORACLE-058")
def test_declared_scalar_type_must_match_cuda_entry(tmp_path, producer_tools):
    """A float declaration cannot describe the CUDA entry's integer scalar even when sizes match."""
    spec = replace(SPEC, signature=(("x", "*fp32"), ("y", "*fp32"), ("n", "fp32")))
    with pytest.raises(ValueError, match="type"):
        compile_bundle((spec,), tmp_path / "bundle", **producer_tools)
    assert not (tmp_path / "bundle").exists()
