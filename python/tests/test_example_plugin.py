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

from native_build import cuda_root, tensorrt_root
from tensoracle.bundle import PluginBundleMismatchError, inspect_bundle, inspect_wheel


@pytest.fixture(scope="module")
def installed_example(tmp_path_factory):
    sdk = tensorrt_root()
    if shutil.which("uv") is None:
        pytest.skip("uv was not supplied for the example wheel")
    project = Path(__file__).resolve().parents[1]
    root = project.parent if project.name == "python" else project
    work = tmp_path_factory.mktemp("example-wheel")
    tensoracle_dist = work / "tensoracle-dist"
    example_dist = work / "example-dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tensoracle_dist)],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    main_wheel = next(tensoracle_dist.glob("*.whl"))
    environment = dict(os.environ)
    constraint = work / "build-constraints.txt"
    constraint.write_text(f"tensoracle @ {main_wheel.as_uri()}\n", encoding="utf-8")
    cmake_defines = [f"TENSORRT_ROOT={Path(sdk).as_posix()}"]
    if cuda := cuda_root():
        cmake_defines.append(f"CUDAToolkit_ROOT={Path(cuda).as_posix()}")
    subprocess.run(
        [
            "uv",
            "build",
            str(root / "examples/cubin_plugin"),
            "--wheel",
            "--out-dir",
            str(example_dist),
            "--python",
            sys.executable,
            "--build-constraints",
            str(constraint),
        ],
        cwd=root,
        # Pass SDK paths to scikit-build-core without changing dependency cache keys.
        env={**environment, "SKBUILD_CMAKE_DEFINE": ";".join(cmake_defines)},
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    python = work / "runtime" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(work / "runtime")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    example_wheel = next(example_dist.glob("*.whl"))
    installed = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            f"{main_wheel}[tensorrt]",
            str(example_wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
        # An empty uv cache downloads about 3.85 GB, which takes 3850 s at 1 MB/s (8 Mbps).
        # Add the 45 s slowest measured cold install and round up to 70 minutes.
        timeout=4200,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json, sysconfig; from importlib.metadata import distribution; "
                "package = distribution('tensoracle-cubin-example'); "
                "libraries = [str(package.locate_file(p)) for p in package.files "
                "if p.parent.as_posix() == 'example_cubin_plugin' "
                "and 'tensoracle_example_plugin' in p.name "
                "and p.suffix in ('.so', '.dll', '.dylib')]; "
                "assert len(libraries) == 1, libraries; "
                "print(json.dumps({'library': libraries[0], "
                "'site_packages': sysconfig.get_path('purelib')}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    runtime = json.loads(result.stdout)
    library = Path(runtime["library"])
    return {
        "root": root,
        "project": project,
        "work": work,
        "python": python,
        "main_wheel": main_wheel,
        "example_wheel": example_wheel,
        "library": library,
        "plugin_member": library.relative_to(runtime["site_packages"]).as_posix(),
        "environment": {
            key: value
            for key, value in environment.items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"}
        },
    }


@pytest.mark.req("SP-TENSORACLE-060")
def test_installed_example_exposes_its_own_inference_binding(installed_example):
    """The example imports and exposes inference after tensoracle's Python engine API is removed."""
    result = subprocess.run(
        [
            str(installed_example["python"]),
            "-I",
            "-c",
            (
                "import sys; import example_cubin_plugin as example; assert callable(example.infer); "
                "assert 'tensoracle' not in sys.modules; assert 'tensorrt' not in sys.modules"
            ),
        ],
        cwd=installed_example["work"],
        env={**installed_example["environment"], "CUDA_VISIBLE_DEVICES": ""},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _inference_process(example, cache, plugin=None):
    result = subprocess.run(
        [
            str(example["python"]),
            "-I",
            "-c",
            """
import hashlib, json, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
import example_cubin_plugin as example

model = example.open_model(plugin=Path(sys.argv[1]) if sys.argv[1] else None)
weights = load_file(example.weights_path())
for execution, rows, capture in ((0, 16, False), (1, 3, False), (0, 32, True), (1, 7, True)):
    features = np.linspace(-1, 1, rows * 32, dtype=np.float32).reshape(rows, 32)
    actual = model.infer(features, execution=execution, capture=capture)
    expected = (features @ weights['matrix']) * weights['scale'] + weights['bias']
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert model.output_shape(execution) == [rows, 16]
    if execution == 1:
        assert model.output_shape(0) == ([16, 16] if rows == 3 else [32, 16])
    if rows == 16:
        output_digest = hashlib.sha256(actual.tobytes()).hexdigest()
path = Path(model.engine_path)
print(json.dumps({'path': str(path), 'build_calls': model.build_calls,
                  'engine_digest': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'output_digest': output_digest}))
""",
            str(plugin) if plugin is not None else "",
        ],
        cwd=example["work"],
        env={
            **example["environment"],
            "XDG_CACHE_HOME": str(cache),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.req("SP-TENSORACLE-012")
@pytest.mark.req("SP-TENSORACLE-028")
@pytest.mark.req("SP-TENSORACLE-038")
@pytest.mark.req("SP-TENSORACLE-060")
def test_cpp_plugin_inference_reuses_engines_contexts_and_captured_enqueue(installed_example):
    """C++ plugin inference and graph replay survive restart; changed library bytes rebuild."""
    cache = installed_example["work"] / "plugin-cache"
    first = _inference_process(installed_example, cache)
    assert first["build_calls"] == 1
    reused = _inference_process(installed_example, cache)
    assert reused["build_calls"] == 0
    assert reused["path"] == first["path"]
    assert reused["engine_digest"] == first["engine_digest"]
    assert reused["output_digest"] == first["output_digest"]
    plugin = installed_example["work"] / installed_example["library"].name
    plugin.write_bytes(installed_example["library"].read_bytes() + b"changed plugin bytes")
    changed = _inference_process(installed_example, cache, plugin)
    assert changed["build_calls"] == 1
    assert changed["path"] != first["path"]
    assert changed["output_digest"] == first["output_digest"]


@pytest.mark.req("SP-TENSORACLE-005")
@pytest.mark.req("SP-TENSORACLE-015")
@pytest.mark.req("SP-TENSORACLE-038")
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
        plugin=installed_example["plugin_member"],
    )


@pytest.mark.req("SP-TENSORACLE-004")
def test_wheel_library_with_other_bundle_is_rejected(installed_example, tmp_path):
    """The wheel's bundle validator detects a library claiming another manifest."""
    library = installed_example["library"]
    plugin = tmp_path / library.name
    data = library.read_bytes()
    project = Path(__file__).resolve().parents[1]
    root = (
        project.parent if project.name == "python" else project
    ) / "examples/cubin_plugin/bundle"
    digest = json.loads((root / "bundle.json").read_text())["bundle_sha256"].encode()
    assert data.count(digest) >= 1
    plugin.write_bytes(data.replace(digest, b"0" * 64, 1))
    with pytest.raises(PluginBundleMismatchError):
        inspect_bundle(root, plugin=plugin)
