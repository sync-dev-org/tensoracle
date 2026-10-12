"""The producer extra supplies portable cubin tools without inference dependencies."""

import importlib.util
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, requires
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tensoracle.bundle import inspect_bundle


@pytest.fixture
def producer_tools():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("cubin production requires tensoracle[producer] (Triton and NVIDIA tools)")
    tools = {}
    for name, package, relative in (
        ("ptxas", "nvidia-cuda-nvcc-cu12", "nvidia/cuda_nvcc/bin/ptxas"),
        ("cuobjdump", "nvidia-cuda-cuobjdump", "nvidia/cu13/bin/cuobjdump"),
    ):
        try:
            path = Path(distribution(package).locate_file(relative))
        except PackageNotFoundError:
            pytest.skip(f"cubin production requires tensoracle[producer] ({name} wheel missing)")
        if not path.is_file():
            pytest.skip(f"cubin production requires tensoracle[producer] ({name} missing)")
        tools[name] = path
    return tools


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.req("SP-TENSORACLE-015")
@pytest.mark.parametrize("explicit_tools", [False, True], ids=["wheel-defaults", "explicit-paths"])
def test_example_producer_builds_three_architectures_without_torch(
    tmp_path, producer_tools, explicit_tools
):
    """Default wheel tools and explicit paths reproduce every bundled file without torch."""
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "bundle"
    source = """
import importlib.util
import sys
from pathlib import Path
from tensoracle.producer import compile_bundle
from examples.cubin_plugin.kernels import kernel_specs
assert importlib.util.find_spec('torch') is None
tools = {}
if len(sys.argv) > 2:
    tools = {'ptxas': Path(sys.argv[2]), 'cuobjdump': Path(sys.argv[3])}
compile_bundle(kernel_specs(), Path(sys.argv[1]), **tools)
assert 'torch' not in sys.modules
"""
    arguments = (
        [str(producer_tools[name]) for name in ("ptxas", "cuobjdump")] if explicit_tools else []
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root)
    environment["TRITON_CACHE_DIR"] = str(tmp_path / "triton-cache")
    subprocess.run(
        [sys.executable, "-c", source, str(output), *arguments],
        check=True,
        cwd=root,
        env=environment,
        timeout=55,
    )
    manifest = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert {entry["target_sm"] for entry in manifest["entries"]} == {"sm86", "sm89", "sm120"}
    assert manifest["producer"]["triton"] == "3.8.0"
    assert manifest["producer"]["ptxas"] == "12.9.86"
    assert "operating_system" not in manifest["producer"]
    assert not any(Path(entry["file"]).is_absolute() for entry in manifest["entries"])
    assert inspect_bundle(output) == manifest["bundle_sha256"]
    committed = root / "examples/cubin_plugin/bundle"
    generated_files = {path.relative_to(output) for path in output.rglob("*") if path.is_file()}
    committed_files = {
        path.relative_to(committed) for path in committed.rglob("*") if path.is_file()
    }
    assert generated_files == committed_files
    for relative in committed_files:
        assert (output / relative).read_bytes() == (committed / relative).read_bytes()


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.req("SP-TENSORACLE-005")
def test_package_declares_cubin_tools_only_in_producer_extra():
    """Installed metadata reserves pinned Triton and NVIDIA tools for the producer extra."""
    expected = {
        "triton": "==3.8.0",
        "nvidia-cuda-nvcc-cu12": "==12.9.86",
        "nvidia-cuda-cuobjdump": ">=13.3.73,<13.4",
    }
    dependencies = [Requirement(item) for item in requires("tensoracle") or []]
    producer_dependencies = [item for item in dependencies if item.name in expected]
    assert {item.name for item in producer_dependencies} == expected.keys()
    assert len(producer_dependencies) == len(expected)
    for dependency in producer_dependencies:
        assert dependency.marker is not None
        assert dependency.marker.evaluate({"extra": "producer"})
        assert not dependency.marker.evaluate({"extra": ""})
        assert not dependency.marker.evaluate({"extra": "convert"})
        assert (
            dependency.specifier
            == Requirement(dependency.name + expected[dependency.name]).specifier
        )


@pytest.mark.req("SP-TENSORACLE-005")
def test_inference_imports_leave_producer_and_triton_unloaded():
    """Package, build, and reuse imports load neither the producer nor Triton."""
    source = """
import sys
import tensoracle
assert 'tensoracle.producer' not in sys.modules
assert not any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
from tensoracle import build_serialized_engine
assert 'tensoracle.producer' not in sys.modules
assert not any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
from tensoracle import open_engine
assert 'tensoracle.producer' not in sys.modules
assert not any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.req("SP-TENSORACLE-004")
def test_missing_triton_reports_producer_extra(tmp_path):
    """A call without Triton identifies the missing dependency and producer extra."""
    source = """
import importlib.abc
import sys
from pathlib import Path
from tensoracle.producer import compile_bundle

class MissingTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'triton' or fullname.startswith('triton.'):
            raise ModuleNotFoundError("No module named 'triton'", name='triton')

sys.meta_path.insert(0, MissingTriton())
compile_bundle((), Path(sys.argv[1]), ptxas=Path('unused'), cuobjdump=Path('unused'))
"""
    output = tmp_path / "bundle"
    result = subprocess.run(
        [sys.executable, "-c", source, str(output)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Triton" in result.stderr
    assert "tensoracle[producer]" in result.stderr
    assert not output.exists()


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.parametrize("tool", ["ptxas", "cuobjdump"])
@pytest.mark.parametrize("missing", ["wheel", "executable"])
def test_missing_wheel_tool_reports_producer_extra(tmp_path, tool, missing):
    """A missing wheel or executable identifies the tool and producer extra before writing."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("checking wheel tools requires tensoracle[producer] (Triton)")
    if missing == "executable":
        package = "nvidia-cuda-nvcc-cu12" if tool == "ptxas" else "nvidia-cuda-cuobjdump"
        info = tmp_path / (package.replace("-", "_") + "-0.dist-info")
        info.mkdir()
        (info / "METADATA").write_text(f"Name: {package}\nVersion: 0\n", encoding="utf-8")
    source = """
import sys
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from tensoracle.producer import compile_bundle

# Keep real imported modules, but discover distributions only in the temporary directory.
sys.path = [sys.argv[1]]
tool = sys.argv[2]
other = 'cuobjdump' if tool == 'ptxas' else 'ptxas'
compile_bundle((), Path(sys.argv[1]) / 'bundle', **{other: Path('explicit-unused')})
"""
    result = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path), tool],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert tool in result.stderr
    assert "tensoracle[producer]" in result.stderr
    assert not (tmp_path / "bundle").exists()
