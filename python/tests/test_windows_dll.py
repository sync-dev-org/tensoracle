"""Windows imports share the wheel DLL and stub generation uses the build's DLLs."""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from native_build import PROJECT, ROOT, preparation, sdk_version

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="native Windows DLL loading")


def run(command, **kwargs):
    return subprocess.run(
        list(map(str, command)), capture_output=True, text=True, check=False, timeout=240, **kwargs
    )


def succeed(command, **kwargs):
    result = run(command, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def windows_build(tmp_path_factory):
    work = tmp_path_factory.mktemp("windows-python-build")
    dependencies = work / "dependencies"
    profile = ROOT / "cpp/conan/profiles/windows-msvc194"
    graph = json.loads(
        succeed(
            [
                "conan",
                "install",
                ROOT / "cpp",
                "--lockfile",
                ROOT / "cpp/conan.lock",
                *preparation()["profile_arguments"](profile),
                "--output-folder",
                dependencies,
                "--no-remote",
                "--build=never",
                "-c",
                "tools.cmake.cmaketoolchain:user_presets=",
                "--format=json",
            ]
        )
    )
    sdk = Path(
        next(node for node in graph["graph"]["nodes"].values() if node.get("name") == "tensorrt")[
            "package_folder"
        ]
    )
    nanobind = succeed([sys.executable, "-m", "nanobind", "--cmake_dir"]).strip()
    build = work / "build"
    succeed(
        [
            "cmake",
            "-S",
            PROJECT,
            "-B",
            build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_TOOLCHAIN_FILE={dependencies / 'conan_toolchain.cmake'}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind}",
        ]
    )
    log = succeed(["cmake", "--build", build, "--parallel", "4"])
    prefix = work / "installed"
    succeed(["cmake", "--install", build, "--prefix", prefix])
    shutil.copytree(PROJECT / "src/tensoracle", prefix / "tensoracle", dirs_exist_ok=True)
    return {"work": work, "build": build, "sdk": sdk, "prefix": prefix, "log": log}


@pytest.fixture(scope="module")
def dll_consumer(windows_build):
    work = windows_build["work"] / "consumer"
    succeed(
        [
            "cmake",
            "-S",
            PROJECT / "tests/native/dll_consumer",
            "-B",
            work,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={windows_build['prefix'] / 'tensoracle/cpp'}",
            f"-DTENSORRT_ROOT={windows_build['sdk']}",
        ]
    )
    succeed(["cmake", "--build", work, "--parallel", "4"])
    module = Path((work / "artifact.txt").read_text(encoding="utf-8"))
    assert not list(module.parent.glob("tensoracle*.dll"))
    return module


@pytest.mark.req("SP-TENSORACLE-069")
@pytest.mark.req("SP-TENSORACLE-073")
def test_import_registers_the_wheel_dll_for_later_consumers(windows_build, dll_consumer):
    """A later binding finds tensoracle's DLL only after importing the owning package."""
    script = """
import gc, json, os, sys
sys.path[:0] = sys.argv[1:3]
with os.add_dll_directory(sys.argv[3]):
    try:
        import _dll_consumer
    except ImportError:
        pass
    else:
        raise AssertionError('consumer unexpectedly loaded before import tensoracle')
    import tensoracle
    assert 'tensoracle._native' not in sys.modules
    gc.collect()
    import _dll_consumer
    print(json.dumps(_dll_consumer.version()))
"""
    value = succeed(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            windows_build["prefix"],
            dll_consumer.parent,
            windows_build["sdk"] / "bin",
        ],
        cwd=windows_build["work"],
    )
    assert json.loads(value) == list(sdk_version())


@pytest.mark.req("SP-TENSORACLE-063")
@pytest.mark.req("SP-TENSORACLE-069")
def test_editable_package_registers_its_installed_dll(windows_build, dll_consumer):
    """Source Python files can load later consumers using the editable install's DLL directory."""
    script = """
import importlib.util, json, os, sys
from pathlib import Path
with os.add_dll_directory(sys.argv[4]):
    sys.path.insert(0, sys.argv[3])
    spec = importlib.util.spec_from_file_location(
        'tensoracle', sys.argv[1], submodule_search_locations=[sys.argv[2], str(Path(sys.argv[1]).parent)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules['tensoracle'] = package
    spec.loader.exec_module(package)
    assert Path(package.__file__) == Path(sys.argv[1])
    import _dll_consumer
    print(json.dumps(_dll_consumer.version()))
"""
    value = succeed(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            PROJECT / "src/tensoracle/__init__.py",
            windows_build["prefix"] / "tensoracle",
            dll_consumer.parent,
            windows_build["sdk"] / "bin",
        ],
        cwd=windows_build["work"],
    )
    assert json.loads(value) == list(sdk_version())


@pytest.mark.req("SP-TENSORACLE-057")
@pytest.mark.req("SP-TENSORACLE-062")
@pytest.mark.req("SP-TENSORACLE-069")
def test_python_install_contains_the_dll_and_matching_native_types(windows_build):
    """The installed layout supplies its own DLL, import library, headers, and native types."""
    package = windows_build["prefix"] / "tensoracle"
    for name in (
        "bin/tensoracle-2.dll",
        "lib/tensoracle.lib",
        "include/tensoracle/tensoracle.h",
        "lib/cmake/tensoracle/tensoracleConfig.cmake",
    ):
        assert (package / "cpp" / name).is_file(), name
    assert (package / "py.typed").is_file()
    assert not list(package.rglob("nvinfer*.dll"))
    stub = ast.parse((package / "_native.pyi").read_text(encoding="utf-8"))
    declared = set()
    for node in stub.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            declared.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared.add(node.target.id)
        elif isinstance(node, ast.Assign):
            declared.update(target.id for target in node.targets if isinstance(target, ast.Name))
    names = json.loads(
        succeed(
            [
                sys.executable,
                "-I",
                "-c",
                """
import json, os, sys
sys.path.insert(0, sys.argv[1])
with os.add_dll_directory(sys.argv[2]):
    from tensoracle import _native
    print(json.dumps([name for name in dir(_native) if not name.startswith('_')]))
""",
                windows_build["prefix"],
                windows_build["sdk"] / "bin",
            ],
            cwd=windows_build["work"],
        )
    )
    assert {name for name in declared if not name.startswith("_")} == set(names)


@pytest.mark.req("SP-TENSORACLE-062")
@pytest.mark.req("SP-TENSORACLE-069")
def test_stubs_load_build_dlls_without_importing_an_installed_package(windows_build, tmp_path):
    """An installed package cannot replace the built extension or run its import side effects."""
    poison = tmp_path / "tensoracle"
    poison.mkdir()
    (poison / "__init__.py").write_text("raise AssertionError('imported wheel package')\n")
    destination = tmp_path / "native.pyi"
    build = windows_build["build"]
    result = run(
        [
            sys.executable,
            PROJECT / "build_backend.py",
            windows_build["sdk"] / "bin/nvinfer_11.dll",
            build,
            destination,
            build / "cpp/tensoracle-2.dll",
        ],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert destination.read_bytes() == (build / "_native.pyi").read_bytes()
    assert "TensorRT for stub generation:" in result.stdout
    assert "tensoracle-2.dll" in result.stdout


@pytest.mark.req("SP-TENSORACLE-062")
@pytest.mark.req("SP-TENSORACLE-069")
@pytest.mark.parametrize("library", ["nvinfer_11.dll", "tensoracle-2.dll"])
def test_stubs_reject_a_dll_loaded_from_another_directory(windows_build, tmp_path, library):
    """A preloaded foreign TensorRT or tensoracle DLL prevents stub publication."""
    build = windows_build["build"]
    sdk_bin = windows_build["sdk"] / "bin"
    source = sdk_bin / library if library.startswith("nvinfer") else build / "cpp" / library
    foreign = tmp_path / library
    shutil.copyfile(source, foreign)
    destination = tmp_path / "rejected.pyi"
    script = """
import ctypes, os, runpy, sys
backend, foreign, sdk_bin, built_bin, *arguments = sys.argv[1:]
with os.add_dll_directory(sdk_bin), os.add_dll_directory(built_bin):
    loaded = ctypes.WinDLL(foreign)
    sys.argv = [backend, *arguments]
    runpy.run_path(backend, run_name='__main__')
"""
    result = run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            PROJECT / "build_backend.py",
            foreign,
            sdk_bin,
            build / "cpp",
            sdk_bin / "nvinfer_11.dll",
            build,
            destination,
            build / "cpp/tensoracle-2.dll",
        ],
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "Stub generator expected" in result.stderr, result.stdout + result.stderr
    assert foreign.as_posix() in result.stderr.replace("\\", "/")
    assert not destination.exists()
