"""Supply shared repository sources to scikit-build-core from the Python project."""

import ctypes
import json
import os
import runpy
import shutil
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

from scikit_build_core import build as native_build
from setuptools_scm import get_version

get_requires_for_build_sdist = native_build.get_requires_for_build_sdist


@contextmanager
def _prepared(config_settings):
    prepare = runpy.run_path(str(Path.cwd() / "cpp/conan/prepare.py"))
    with tempfile.TemporaryDirectory(prefix="tensoracle-wheel-") as directory:
        output = Path(directory)
        prepare["install"](output / "conan", wheel=True)
        activated = prepare["environment"](output / "conan")
        previous = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(activated)
            cmake = shutil.which("cmake", path=os.environ.get("PATH"))
            ninja = shutil.which("ninja", path=os.environ.get("PATH"))
            if not cmake or not ninja:
                raise RuntimeError("Conan did not supply CMake and Ninja")
            os.environ["CMAKE_EXECUTABLE"] = cmake
            os.environ["CMAKE_GENERATOR"] = "Ninja"
            settings = dict(config_settings or {})
            settings.update(
                {
                    "cmake.define.CMAKE_TOOLCHAIN_FILE": str(
                        output / "conan/conan_toolchain.cmake"
                    ),
                    "cmake.define.CMAKE_MAKE_PROGRAM": ninja,
                    "build-dir": str(output / "native"),
                }
            )
            yield settings
        finally:
            os.environ.clear()
            os.environ.update(previous)


def _requirements(hook, config_settings):
    with _source_tree(), _prepared(config_settings) as settings:
        return hook(settings)


def get_requires_for_build_wheel(config_settings=None):
    return _requirements(native_build.get_requires_for_build_wheel, config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _requirements(native_build.get_requires_for_build_editable, config_settings)


@contextmanager
def _source_tree(*, editable=False, include_recipes=False):
    project = Path(__file__).resolve().parent
    source_archive = (project / "PKG-INFO").is_file()
    repository = project if source_archive else project.parent
    shared = ("README.md", "LICENSE", "cpp", "docs", "examples", "mkdocs.yml", ".gitignore")
    for name in shared:
        if not (repository / name).exists():
            raise FileNotFoundError(f"Required build source is missing: {repository / name}")
    version = get_version(
        root=str(repository),
        fallback_version="0.0.0.dev0",
        version_scheme="only-version",
        local_scheme="no-local-version",
    )
    previous_cwd = Path.cwd()
    variable = "SETUPTOOLS_SCM_PRETEND_VERSION"
    previous_version = os.environ.get(variable)
    os.environ[variable] = version
    try:
        if source_archive:
            os.chdir(project)
            yield
            return
        with tempfile.TemporaryDirectory(prefix="tensoracle-build-") as directory:
            staged = Path(directory)
            for base, names in (
                (
                    project,
                    (
                        "pyproject.toml",
                        "uv.lock",
                        "build_backend.py",
                        "CMakeLists.txt",
                        "bindings.cpp",
                        "cubin_bindings.cpp",
                        "src",
                        "tests",
                    ),
                ),
                (repository, shared),
            ):
                for name in names:
                    source, destination = base / name, staged / name
                    if source.is_dir():
                        shutil.copytree(
                            source,
                            destination,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
                        )
                    else:
                        shutil.copyfile(source, destination)
            if include_recipes:
                recipe_tools = runpy.run_path(str(repository / "cpp/conan/recipes.py"))
                cache = os.environ.get("TENSORACLE_CONAN_CACHE")
                cache_args = ["--core-conf", f"core.cache:storage_path={cache}"] if cache else []
                recipe_tools["copy_recipes"](
                    repository / "cpp/conan",
                    staged / "cpp/conan/recipes",
                    cache_args=cache_args,
                )
            if editable:
                # Source redirects must survive deletion of the staging directory.
                settings = staged / "pyproject.toml"
                settings.write_text(
                    settings.read_text(encoding="utf-8").replace(
                        'wheel.packages = ["src/tensoracle"]',
                        f"wheel.packages = [{json.dumps(str(project / 'src/tensoracle'))}]",
                    ),
                    encoding="utf-8",
                )
            os.chdir(staged)
            try:
                yield
                if editable:
                    shutil.copyfile(
                        staged / "src/tensoracle/_version.py",
                        project / "src/tensoracle/_version.py",
                    )
            finally:
                os.chdir(previous_cwd)
    finally:
        os.chdir(previous_cwd)
        if previous_version is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous_version


def _call(hook, directory, config_settings=None, metadata_directory=None, *, editable=False):
    output = str(Path(directory).resolve())
    metadata = str(Path(metadata_directory).resolve()) if metadata_directory else None
    with _source_tree(
        editable=editable,
        include_recipes=hook
        in (native_build.build_sdist, native_build.build_wheel, native_build.build_editable),
    ):
        if hook == native_build.build_sdist:
            return hook(output, config_settings)
        with _prepared(config_settings) as settings:
            if hook in (native_build.build_wheel, native_build.build_editable):
                return hook(output, settings, metadata)
            return hook(output, settings)


def build_sdist(sdist_directory, config_settings=None):
    return _call(native_build.build_sdist, sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _call(native_build.build_wheel, wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _call(
        native_build.build_editable,
        wheel_directory,
        config_settings,
        metadata_directory,
        editable=True,
    )


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _call(native_build.prepare_metadata_for_build_wheel, metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _call(
        native_build.prepare_metadata_for_build_editable, metadata_directory, config_settings
    )


def _windows_loaded_libraries():
    """Get actual DLL paths, including multiple copies with the same basename."""
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    enumerate_modules = kernel.K32EnumProcessModules
    enumerate_modules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    enumerate_modules.restype = wintypes.BOOL
    module_filename = kernel.GetModuleFileNameW
    module_filename.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    module_filename.restype = wintypes.DWORD
    capacity = 256
    needed = wintypes.DWORD()
    while True:
        modules = (wintypes.HMODULE * capacity)()
        if not enumerate_modules(
            wintypes.HANDLE(-1), modules, ctypes.sizeof(modules), ctypes.byref(needed)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value <= ctypes.sizeof(modules):
            break
        capacity = needed.value // ctypes.sizeof(wintypes.HMODULE) + 1
    loaded = set()
    for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        size = 260
        while True:
            buffer = ctypes.create_unicode_buffer(size)
            length = module_filename(module, buffer, size)
            if not length:
                raise ctypes.WinError(ctypes.get_last_error())
            if length < size:
                loaded.add(Path(buffer.value).resolve())
                break
            size *= 2
    return loaded


def _generate_stubs(library, module_path, destination, tensoracle_library=None):
    """Keep stub generation on the same TensorRT binary used by the native build."""
    import hashlib
    import importlib

    expected = Path(library).resolve()
    with ExitStack() as directories:
        if sys.platform == "win32":
            if tensoracle_library is None:
                raise ValueError("Windows stub generation requires the built tensoracle DLL path")
            expected_tensoracle = Path(tensoracle_library).resolve()
            for directory in {expected.parent, expected_tensoracle.parent}:
                directories.enter_context(os.add_dll_directory(str(directory)))
            # A regular installed package would override the build's namespace
            # package. Supply the namespace explicitly without running its __init__.
            from importlib.machinery import ModuleSpec
            from importlib.util import module_from_spec

            package = module_from_spec(ModuleSpec("tensoracle", loader=None, is_package=True))
            package.__path__ = [str(Path(module_path).resolve() / "tensoracle")]
            sys.modules["tensoracle"] = package
        # Load this SONAME before importing the extension, including when the caller
        # has a different TensorRT on its ordinary library search path.
        ctypes.CDLL(str(expected))
        os.environ["NB_STUBGEN"] = "1"
        sys.path.insert(0, module_path)
        native = importlib.import_module("tensoracle._native")
        proof = {"library": str(expected)}
        if sys.platform == "win32":
            loaded = _windows_loaded_libraries()
            for name, path in (("TensorRT", expected), ("tensoracle", expected_tensoracle)):
                actual = {file for file in loaded if file.name.lower() == path.name.lower()}
                if actual != {path}:
                    raise RuntimeError(f"Stub generator expected {name} {path}; loaded {actual}")
            if Path(native.__file__).resolve().parent != Path(package.__path__[0]):
                raise RuntimeError(
                    f"Stub generator expected built native module; loaded {native.__file__}"
                )
            proof["tensoracle"] = str(expected_tensoracle)
        else:
            loaded = {
                Path(line.split()[-1]).resolve()
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/libnvinfer.so" in line
            }
            if loaded != {expected}:
                raise RuntimeError(f"Stub generator expected TensorRT {expected}; loaded {loaded}")
        with expected.open("rb") as file:
            proof["sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
        print("TensorRT for stub generation: " + json.dumps(proof), flush=True)
        sys.argv = ["nanobind.stubgen", "-m", "tensoracle._native", "-o", destination, "-P"]
        runpy.run_module("nanobind.stubgen", run_name="__main__")


if __name__ == "__main__":
    _generate_stubs(*sys.argv[1:])
