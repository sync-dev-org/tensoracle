"""The producer uses host CUDA tools without adding inference dependencies."""

import importlib.util
import json
import os
import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tensoracle.bundle import inspect_bundle


@pytest.fixture
def producer_tools():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("cubin production requires tensoracle[producer] (Triton)")
    tools = {}
    toolkit = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    for name in ("ptxas", "cuobjdump"):
        path = toolkit / "bin" / name
        if not path.is_file():
            pytest.skip(f"cubin production requires CUDA toolkit tool {path}; set CUDA_HOME")
        tools[name] = path
    return tools


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.req("SP-TENSORACLE-015")
@pytest.mark.req("SP-TENSORACLE-037")
@pytest.mark.parametrize(
    "explicit_tools", [False, True], ids=["toolkit-defaults", "explicit-paths"]
)
def test_example_producer_builds_three_architectures_without_torch(
    tmp_path, producer_tools, explicit_tools
):
    """Toolkit defaults and explicit paths produce an inspected bundle without torch."""
    project = Path(__file__).resolve().parents[1]
    root = project.parent if project.name == "python" else project
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
    environment["PYTHONPATH"] = str(project / "src") + os.pathsep + str(root)
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
    assert Requirement("ptxas>=13.2,<14").specifier.contains(manifest["producer"]["ptxas"])
    assert "operating_system" not in manifest["producer"]
    assert not any(Path(entry["file"]).is_absolute() for entry in manifest["entries"])
    assert inspect_bundle(output) == manifest["bundle_sha256"]
    committed = root / "examples/cubin_plugin/bundle"
    generated_files = {path.relative_to(output) for path in output.rglob("*") if path.is_file()}
    committed_files = {
        path.relative_to(committed) for path in committed.rglob("*") if path.is_file()
    }
    assert generated_files == committed_files


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.req("SP-TENSORACLE-005")
def test_package_declares_cubin_tools_only_in_producer_extra():
    """The producer extra declares only pinned Triton; CUDA tools come from the host."""
    expected = {"triton": "==3.8.0"}
    dependencies = [Requirement(item) for item in requires("tensoracle") or []]
    producer_dependencies = [
        item
        for item in dependencies
        if item.marker is not None and item.marker.evaluate({"extra": "producer"})
    ]
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


@pytest.mark.req("SP-TENSORACLE-004")
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
def test_producer_checks_ptxas_range_before_writing(tmp_path, monkeypatch, version, supported):
    """ptxas in [13.2, 14) reaches kernel validation; other versions stop before output."""
    pytest.importorskip("triton.backends.compiler")

    from tensoracle.producer import compile_bundle

    def tool_version(command, **kwargs):
        assert command == ["explicit-ptxas", "--version"]
        return subprocess.CompletedProcess(command, 0, stdout=f"ptxas V{version}", stderr="")

    monkeypatch.setattr(subprocess, "run", tool_version)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError) as error:
        compile_bundle(
            (), output, ptxas=Path("explicit-ptxas"), cuobjdump=Path("explicit-cuobjdump")
        )
    if supported:
        assert "kernel names" in str(error.value)
    else:
        assert "ptxas" in str(error.value)
        assert ">=13.2" in str(error.value)
        assert "<14" in str(error.value)
        assert version in str(error.value)
    assert not output.exists()


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
@pytest.mark.parametrize("configured", [False, True], ids=["default-directory", "CUDA_HOME"])
def test_producer_selects_toolkit_directory(tmp_path, monkeypatch, configured):
    """ptxas comes from CUDA_HOME or /usr/local/cuda when CUDA_HOME is unset."""
    pytest.importorskip("triton.backends.compiler")
    from tensoracle.producer import compile_bundle

    if configured:
        toolkit = tmp_path / "toolkit"
        (toolkit / "bin").mkdir(parents=True)
        for name in ("ptxas", "cuobjdump"):
            (toolkit / "bin" / name).touch()
        monkeypatch.setenv("CUDA_HOME", str(toolkit))
    else:
        toolkit = Path("/usr/local/cuda")
        monkeypatch.delenv("CUDA_HOME", raising=False)
        if not all((toolkit / "bin" / name).is_file() for name in ("ptxas", "cuobjdump")):
            pytest.skip("default CUDA toolkit tools are not installed at /usr/local/cuda/bin")

    def tool_version(command, **kwargs):
        assert command == [str(toolkit / "bin/ptxas"), "--version"]
        return subprocess.CompletedProcess(command, 0, stdout="ptxas V13.2.0", stderr="")

    monkeypatch.setattr(subprocess, "run", tool_version)
    with pytest.raises(ValueError, match="kernel names"):
        compile_bundle((), tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


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
def test_missing_toolkit_tool_reports_path_and_configuration(tmp_path, monkeypatch, tool):
    """Missing toolkit tools identify the searched path, CUDA_HOME, and explicit argument."""
    pytest.importorskip("triton")
    from tensoracle.producer import compile_bundle

    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    other = "cuobjdump" if tool == "ptxas" else "ptxas"
    output = tmp_path / "bundle"
    with pytest.raises(FileNotFoundError) as error:
        compile_bundle((), output, **{other: Path("explicit-unused")})
    assert str(tmp_path / "bin" / tool) in str(error.value)
    assert "CUDA_HOME" in str(error.value)
    assert f"{tool}=" in str(error.value)
    assert not output.exists()
