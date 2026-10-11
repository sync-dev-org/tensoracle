"""Exercise the installed example wheel with cubin plugins and cached engines."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tensoracle.bundle import PluginBundleMismatchError, inspect_bundle, inspect_wheel


@pytest.fixture(scope="module")
def installed_example(tmp_path_factory):
    sdk = os.environ.get("TENSORACLE_SDK_ROOT")
    if not sdk or shutil.which("uv") is None:
        pytest.skip("TensorRT SDK and uv were not supplied for the example wheel")
    root = Path(__file__).resolve().parents[1]
    work = tmp_path_factory.mktemp("example-wheel")
    tensoracle_dist = work / "tensoracle-dist"
    example_dist = work / "example-dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tensoracle_dist)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    main_wheel = next(tensoracle_dist.glob("*.whl"))
    with zipfile.ZipFile(main_wheel) as archive:
        for name in archive.namelist():
            if name.startswith("tensoracle/cpp/"):
                archive.extract(name, work / "distributed-source")
    environment = dict(os.environ)
    cpp_root = work / "distributed-source/tensoracle/cpp"
    environment["CMAKE_ARGS"] = f"-DTENSORRT_ROOT={sdk} -DTENSORACLE_CPP_ROOT={cpp_root}"
    subprocess.run(
        [
            "uv",
            "build",
            str(root / "examples/cubin_plugin"),
            "--wheel",
            "--out-dir",
            str(example_dist),
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    python = work / "runtime/bin/python"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(work / "runtime")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    example_wheel = next(example_dist.glob("*.whl"))
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(main_wheel), str(example_wheel)],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    return {
        "root": root,
        "work": work,
        "python": python,
        "main_wheel": main_wheel,
        "example_wheel": example_wheel,
    }


def _run(example, mode, cache, *extra):
    environment = dict(os.environ)
    environment["XDG_CACHE_HOME"] = str(cache)
    environment["CUDA_HOME"] = str(example["work"] / "absent-sdk")
    environment["TENSORRT_ROOT"] = str(example["work"] / "absent-sdk")
    result = subprocess.run(
        [str(example["python"]), str(example["root"] / "tests/example_runtime.py"), mode, *extra],
        cwd=example["work"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=50,
    )
    return json.loads(result.stdout)


@pytest.mark.req("REQ-TENSORACLE-004")
@pytest.mark.req("REQ-TENSORACLE-011")
def test_distributions_ship_cpp_without_torch_or_triton(installed_example):
    """The producer and C++ sources ship while the runtime dependencies omit torch and Triton."""
    with zipfile.ZipFile(installed_example["main_wheel"]) as archive:
        names = archive.namelist()
        metadata = next(archive.read(name).decode() for name in names if name.endswith("METADATA"))
    dependencies = [
        Requirement(line.removeprefix("Requires-Dist: "))
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: ")
    ]
    for dependency in dependencies:
        if dependency.name in {"torch", "triton"}:
            assert dependency.marker is not None
            assert not dependency.marker.evaluate({"extra": ""})
    assert "tensoracle/producer.py" in names
    assert "tensoracle/cpp/include/tensoracle/cubin_plugin.hpp" in names
    assert "tensoracle/cpp/src/cubin_plugin.cpp" in names
    assert "tensoracle/cpp/cmake/TensoracleCubinPlugin.cmake" in names
    assert inspect_wheel(
        installed_example["example_wheel"],
        plugin="example_cubin_plugin/libtensoracle_example_plugin.so",
    )


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-002")
@pytest.mark.req("REQ-TENSORACLE-003")
@pytest.mark.req("REQ-TENSORACLE-004")
@pytest.mark.req("REQ-TENSORACLE-018")
def test_installed_wheel_builds_and_runs_both_plugins(installed_example):
    """A producer-free wheel builds a dynamic graph and matches its reference output."""
    result = _run(installed_example, "build", installed_example["work"] / "build-cache")
    assert [row["rows"] for row in result["rows"]] == [3, 16]
    assert all(row["outside_tolerance_count"] == 0 for row in result["rows"])
    assert result["specialized"] >= 1
    assert result["unspecialized"] >= 1


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-007")
@pytest.mark.req("REQ-TENSORACLE-003")
def test_saved_engine_reopens_without_graph_and_plugin_change_selects_new_cache(installed_example):
    """A later process reuses the plan; changing plugin bytes selects a new engine."""
    cache = installed_example["work"] / "reuse-cache"
    first = _run(installed_example, "build", cache)
    second = _run(installed_example, "reuse", cache)
    changed = _run(installed_example, "changed", cache, "--extra", str(installed_example["work"]))
    assert second["path"] == first["path"]
    assert changed["path"] != first["path"]
    assert all(row["outside_tolerance_count"] == 0 for row in second["rows"])


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-003")
def test_direct_build_registers_before_graph_and_concurrent_contexts_run(installed_example):
    """Direct builds see both creators, and two execution contexts run together."""
    cache = installed_example["work"] / "concurrent-cache"
    assert _run(installed_example, "serialized", cache)["plan_bytes"] > 0
    _run(installed_example, "build", cache)
    concurrent = _run(installed_example, "concurrent", cache)
    assert all(row["outside_tolerance_count"] == 0 for row in concurrent["rows"])
    separate = _run(installed_example, "separate-contexts", cache)
    assert all(row["outside_tolerance_count"] == 0 for row in separate["rows"])


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-002")
@pytest.mark.req("REQ-TENSORACLE-003")
def test_different_library_with_same_creator_is_rejected_before_graph(installed_example):
    """A second library with changed bytes cannot build with the first library's creator."""
    result = _run(
        installed_example,
        "duplicate-creator",
        installed_example["work"] / "duplicate-cache",
        "--extra",
        str(installed_example["work"]),
    )
    assert result == {"error_type": "PluginLibraryError", "graph_called": False}


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-009")
@pytest.mark.req("REQ-TENSORACLE-002")
def test_fp32_example_differs_from_tf32_control(installed_example):
    """The caller's IEEE dot retains a difference that its TF32 control loses."""
    values = _run(installed_example, "control", installed_example["work"] / "control-cache")
    assert abs(values["tf32"] - values["ieee_expected"]) > 0.003


@pytest.mark.req("REQ-TENSORACLE-003")
def test_wheel_library_with_other_bundle_is_rejected(installed_example, tmp_path):
    """The wheel's bundle validator detects a library claiming another manifest."""
    installed = installed_example["work"] / "runtime/lib"
    candidates = list(installed.rglob("libtensoracle_example_plugin.so"))
    assert len(candidates) == 1
    plugin = tmp_path / "changed.so"
    data = candidates[0].read_bytes()
    root = Path(__file__).resolve().parents[1] / "examples/cubin_plugin/bundle"
    digest = json.loads((root / "bundle.json").read_text())["bundle_sha256"].encode()
    assert data.count(digest) >= 1
    plugin.write_bytes(data.replace(digest, b"0" * 64, 1))
    with pytest.raises(PluginBundleMismatchError):
        inspect_bundle(root, plugin=plugin)
