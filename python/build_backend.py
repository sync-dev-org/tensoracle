"""Build from the Python project using the repository's canonical shared sources."""

import shutil
import tempfile
import tomllib
from contextlib import contextmanager
from pathlib import Path

from hatchling import build as hatchling_build
from hatchling.builders.sdist import SdistBuilder
from hatchling.builders.wheel import WheelBuilder

get_requires_for_build_editable = hatchling_build.get_requires_for_build_editable
get_requires_for_build_sdist = hatchling_build.get_requires_for_build_sdist
get_requires_for_build_wheel = hatchling_build.get_requires_for_build_wheel


@contextmanager
def _builder(builder_class):
    project = Path(__file__).resolve().parent
    config = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    # Source archives already contain the shared files next to pyproject.toml.
    source_archive = (project / "PKG-INFO").is_file()
    repository = project if source_archive else project.parent
    shared = ("README.md", "LICENSE", "cpp", "docs", "examples", "mkdocs.yml", ".gitignore")
    for name in shared:
        if not (repository / name).exists():
            raise FileNotFoundError(f"Required build source is missing: {repository / name}")
    if source_archive:
        yield builder_class(str(project), config=config)
        return

    with tempfile.TemporaryDirectory(prefix="tensoracle-build-") as directory:
        staged = Path(directory)
        for base, names in (
            (project, ("pyproject.toml", "uv.lock", "build_backend.py", "src", "tests")),
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
        hatch = config["tool"]["hatch"]
        hatch["version"]["raw-options"]["root"] = str(repository)
        hatch["build"]["dev-mode-dirs"] = [str(project / "src")]
        yield builder_class(str(staged), config=config)


def _build(builder_class, directory, version):
    output = str(Path(directory).resolve())
    with _builder(builder_class) as builder:
        artifact = next(builder.build(directory=output, versions=[version]))
        if version == "editable":
            generated = Path(builder.root) / "src/tensoracle/_version.py"
            target = Path(__file__).resolve().parent / "src/tensoracle/_version.py"
            if generated != target:
                shutil.copyfile(generated, target)
        return Path(artifact).name


def build_sdist(sdist_directory, config_settings=None):
    return _build(SdistBuilder, sdist_directory, "standard")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build(WheelBuilder, wheel_directory, "standard")


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build(WheelBuilder, wheel_directory, "editable")


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    with _builder(WheelBuilder) as builder:
        builder.metadata.validate_fields()
        directory = Path(metadata_directory) / f"{builder.artifact_project_id}.dist-info"
        directory.mkdir(parents=True, exist_ok=True)
        metadata = builder.config.core_metadata_constructor(builder.metadata)
        (directory / "METADATA").write_text(metadata, encoding="utf-8")
        for relative in builder.metadata.core.license_files:
            target = directory / "licenses" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(builder.root) / relative, target)
        return directory.name


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings)
