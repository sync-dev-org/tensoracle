"""Supply shared repository sources to scikit-build-core from the Python project."""

import ctypes
import json
import os
import runpy
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from scikit_build_core import build as native_build
from setuptools_scm import get_version

get_requires_for_build_editable = native_build.get_requires_for_build_editable
get_requires_for_build_sdist = native_build.get_requires_for_build_sdist
get_requires_for_build_wheel = native_build.get_requires_for_build_wheel


@contextmanager
def _source_tree(*, editable=False):
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
    with _source_tree(editable=editable):
        if hook in (native_build.build_wheel, native_build.build_editable):
            return hook(output, config_settings, metadata)
        return hook(output, config_settings)


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


def _generate_stubs(library, module_path, destination):
    """Keep stub generation on the same TensorRT binary used by the native build."""
    import hashlib
    import importlib

    expected = Path(library).resolve()
    # Load this SONAME before importing the extension, including when the caller
    # has a different TensorRT on its ordinary library search path.
    ctypes.CDLL(str(expected))
    os.environ["NB_STUBGEN"] = "1"
    sys.path.insert(0, module_path)
    importlib.import_module("tensoracle._native")
    loaded = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libnvinfer.so" in line
    }
    if loaded != {expected}:
        raise RuntimeError(f"Stub generator expected TensorRT {expected}; loaded {loaded}")
    with expected.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    print(
        "TensorRT for stub generation: "
        + json.dumps(
            {
                "library": str(expected),
                "sha256": digest,
            }
        ),
        flush=True,
    )
    sys.argv = ["nanobind.stubgen", "-m", "tensoracle._native", "-o", destination, "-P"]
    runpy.run_module("nanobind.stubgen", run_name="__main__")


if __name__ == "__main__":
    _generate_stubs(*sys.argv[1:])
