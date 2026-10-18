"""Check the files and public metadata delivered through PyPI."""

import email
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import markdown
import pytest
import yaml
from packaging.requirements import Requirement

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent if PROJECT.name == "python" else PROJECT
PUBLIC_REPOSITORY = "https://github.com/sync-dev-org/tensoracle"
PAGES_WORKFLOW = ".github/workflows/pages.yml"


@pytest.fixture(scope="module")
def publication_config():
    settings = ROOT / "noren.yaml"
    if not settings.is_file() and not (ROOT / "AGENTS.md").is_file():
        pytest.skip("the public snapshot does not include development release tooling")
    assert settings.is_file(), "the development repository must include noren.yaml"
    return yaml.safe_load(settings.read_text(encoding="utf-8"))


def matching_paths(tmp_path, patterns, paths):
    worktree = tmp_path / "tree"
    if not worktree.is_dir():
        subprocess.run(
            ["git", "init", "--quiet", "--template=", "--initial-branch=main", str(worktree)],
            check=True,
        )
    rules = tmp_path / "rules"
    rules.write_text(patterns, encoding="utf-8")
    result = subprocess.run(
        [
            "git",
            "-c",
            f"core.excludesFile={rules}",
            "-c",
            "core.ignoreCase=false",
            "check-ignore",
            "--no-index",
            "--stdin",
            "-z",
        ],
        cwd=worktree,
        input="".join(path + "\0" for path in paths),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    return set(result.stdout.rstrip("\0").split("\0")) - {""}


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    output = tmp_path_factory.mktemp("tensoracle-dist")
    result = subprocess.run(
        ["uv", "build", "--out-dir", str(output)],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {"sdist": next(output.glob("*.tar.gz")), "wheel": next(output.glob("*.whl"))}


@pytest.mark.req("SP-TENSORACLE-017")
def test_sdist_contains_public_sources_without_development_files(distributions):
    """The source distribution includes code, tests, docs, and examples without private files."""
    with tarfile.open(distributions["sdist"]) as archive:
        paths = [
            PurePosixPath(*PurePosixPath(member.name).parts[1:])
            for member in archive.getmembers()
            if member.isfile()
        ]

    private = {".nf", ".github", "AGENTS.md", "CLAUDE.md", "tools", "noren.yaml", "noren-words.txt"}
    assert not [str(path) for path in paths if path.parts[0] in private]
    assert {"src", "tests", "docs", "examples", "cpp"} <= {path.parts[0] for path in paths}
    assert {"README.md", "LICENSE", "pyproject.toml"} <= {str(path) for path in paths}


def test_sdist_exclusions_match_the_public_snapshot(publication_config, tmp_path):
    """Source archives exclude development paths and the public Pages workflow."""
    config = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    exclusions = config["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    paths = (
        subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        .stdout.rstrip("\0")
        .split("\0")
    )
    paths.append(PAGES_WORKFLOW)

    assert matching_paths(tmp_path, "\n".join(exclusions), paths) == matching_paths(
        tmp_path, publication_config["exclude"], paths
    ) | {PAGES_WORKFLOW}


def test_public_snapshot_includes_only_the_pages_workflow(publication_config, tmp_path):
    """The public snapshot contains the Pages workflow and excludes other GitHub settings."""
    paths = {
        PAGES_WORKFLOW,
        ".github/workflows/other.yml",
        ".github/workflows/private/job.yml",
        ".github/ISSUE_TEMPLATE/task.md",
        ".github/dependabot.yml",
    }
    included = matching_paths(tmp_path, publication_config["include"], paths)
    excluded = matching_paths(tmp_path, publication_config["exclude"], paths)

    assert included - excluded == {PAGES_WORKFLOW}
    assert paths - {PAGES_WORKFLOW} <= excluded


@pytest.mark.req("SP-TENSORACLE-024")
def test_pages_workflow_builds_the_public_snapshot(publication_config):
    """Only the public repository builds and deploys docs with limited permissions and no secrets."""
    workflow_path = ROOT / PAGES_WORKFLOW
    assert workflow_path.is_file()
    workflow_source = workflow_path.read_text(encoding="utf-8")
    assert "secrets." not in workflow_source
    workflow = yaml.safe_load(workflow_source)
    triggers = workflow.get("on", workflow.get(True))
    assert triggers == {"push": {"branches": ["main"]}, "workflow_dispatch": None}

    declared_permissions = set()
    for scope in (workflow, *workflow["jobs"].values()):
        permissions = scope.get("permissions", {})
        assert isinstance(permissions, dict)
        declared_permissions.update(permissions.items())
    assert declared_permissions == {
        ("contents", "read"),
        ("pages", "write"),
        ("id-token", "write"),
    }

    steps = []
    for job in workflow["jobs"].values():
        assert job["if"] == "github.repository == 'sync-dev-org/tensoracle'"
        steps.extend(job["steps"])
    commands = [step["run"] for step in steps if "run" in step]
    assert commands == [
        "uv sync --locked --only-group docs",
        "uv run --no-sync mkdocs build --strict --config-file ../mkdocs.yml",
    ]
    for job in workflow["jobs"].values():
        if any("run" in step for step in job["steps"]):
            assert job["defaults"]["run"]["working-directory"] == "python"
    actions = {}
    for step in steps:
        if "uses" in step:
            action, version = step["uses"].split("@")
            assert re.fullmatch(r"[0-9a-f]{40}", version)
            actions[action] = step
    assert actions.keys() <= {
        "actions/checkout",
        "astral-sh/setup-uv",
        "actions/configure-pages",
        "actions/upload-pages-artifact",
        "actions/deploy-pages",
    }
    assert {
        "actions/configure-pages",
        "actions/upload-pages-artifact",
        "actions/deploy-pages",
    } <= actions.keys()
    assert actions["actions/upload-pages-artifact"]["with"]["path"] == "site/"
    deployment = next(job for job in workflow["jobs"].values() if "environment" in job)
    assert deployment["needs"] == "build"
    assert deployment["environment"]["name"] == "github-pages"
    deployment_permissions = deployment.get("permissions", workflow["permissions"])
    assert deployment_permissions["pages"] == "write"
    assert deployment_permissions["id-token"] == "write"


def test_docs_dependencies_are_separate_from_gpu_development_packages(publication_config):
    """The docs group installs the site builder without the GPU development packages."""
    config = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = config["dependency-groups"]
    assert "docs" in groups
    assert {Requirement(item).name for item in groups["docs"]} == {
        "mkdocs",
        "mkdocs-material",
        "mkdocstrings",
    }
    assert "python" in next(
        Requirement(item).extras
        for item in groups["docs"]
        if Requirement(item).name == "mkdocstrings"
    )
    assert {"include-group": "docs"} in groups["dev"]
    assert not {"mkdocs", "mkdocs-material", "mkdocstrings"} & {
        Requirement(item).name for item in groups["dev"] if isinstance(item, str)
    }


@pytest.mark.req("SP-TENSORACLE-016")
def test_wheel_contains_only_the_package_and_distribution_metadata(distributions):
    """The wheel delivers the Python package and C++ sources with only its install metadata."""
    with zipfile.ZipFile(distributions["wheel"]) as archive:
        paths = [PurePosixPath(name) for name in archive.namelist()]

    top_levels = {path.parts[0] for path in paths}
    assert "tensoracle" in top_levels
    assert len(top_levels) == 2
    assert all(name == "tensoracle" or name.endswith(".dist-info") for name in top_levels)
    assert PurePosixPath("tensoracle/__init__.py") in paths
    assert {"include", "src", "cmake"} <= {
        path.parts[2] for path in paths if path.parts[:2] == ("tensoracle", "cpp")
    }


@pytest.mark.req("SP-TENSORACLE-022")
def test_wheel_is_portable_python_with_uncompiled_cpp(distributions):
    """The wheel uses py3-none-any and delivers C++ source without native binaries."""
    assert distributions["wheel"].name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(distributions["wheel"]) as archive:
        paths = archive.namelist()
        wheel = next(name for name in paths if name.endswith(".dist-info/WHEEL"))
        metadata = email.message_from_bytes(archive.read(wheel))
        assert metadata["Root-Is-Purelib"] == "true"
        assert metadata.get_all("Tag") == ["py3-none-any"]
        assert not [name for name in paths if name.endswith((".so", ".pyd", ".dll", ".dylib"))]
        for directory in ("cmake", "include", "src"):
            for source in (ROOT / "cpp" / directory).rglob("*"):
                if source.is_file():
                    member = source.relative_to(ROOT / "cpp").as_posix()
                    assert archive.read(f"tensoracle/cpp/{member}") == source.read_bytes()


def test_backend_prepares_metadata_and_editable_source(tmp_path):
    """Metadata hooks keep the README and license and editable installs use the real source."""
    staging = tmp_path / "staging"
    staging.mkdir()
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "hatchling",
            "--with",
            "hatch-vcs",
            "--python",
            sys.executable,
            "python",
            "-c",
            (
                "import build_backend as b; import sys; from pathlib import Path; "
                "p = Path(sys.argv[1]); "
                "print(b.prepare_metadata_for_build_wheel(str(p / 'wheel'))); "
                "print(b.prepare_metadata_for_build_editable(str(p / 'editable')))"
            ),
            str(tmp_path),
        ],
        cwd=PROJECT,
        env={**os.environ, "TMPDIR": str(staging)},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(staging.glob("tensoracle-build-*"))
    wheel_metadata = next((tmp_path / "wheel").glob("*.dist-info/METADATA"))
    editable_metadata = next((tmp_path / "editable").glob("*.dist-info/METADATA"))
    assert wheel_metadata.read_bytes() == editable_metadata.read_bytes()
    metadata = email.message_from_bytes(wheel_metadata.read_bytes())
    assert metadata.get_payload().splitlines() == (
        (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    )
    assert metadata.get_all("License-File") == ["LICENSE"]
    for path in (wheel_metadata, editable_metadata):
        assert (path.parent / "licenses/LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    environment = tmp_path / "venv"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
    executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        ["uv", "pip", "install", "--python", str(executable), "--no-deps", "-e", str(PROJECT)],
        check=True,
        timeout=120,
    )
    result = subprocess.run(
        [str(executable), "-c", "import tensoracle; print(tensoracle.__file__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.stdout.strip()).resolve() == PROJECT / "src/tensoracle/__init__.py"
    assert (PROJECT / "src/tensoracle/_version.py").is_file()
    if PROJECT.name == "python":
        assert not (PROJECT / "README.md").exists()
        assert not (PROJECT / "LICENSE").exists()


def test_backend_removes_staging_files_after_metadata_failure(tmp_path):
    """A failed metadata request removes its temporary copy and preserves the source files."""
    repository = tmp_path / "repository"
    project = repository / "python"
    project.mkdir(parents=True)
    for name in ("README.md", "LICENSE", "cpp", "docs", "examples", "mkdocs.yml", ".gitignore"):
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, repository / name)
        else:
            shutil.copyfile(source, repository / name)
    for name in ("pyproject.toml", "uv.lock", "build_backend.py", "src", "tests"):
        source = PROJECT / name
        if source.is_dir():
            shutil.copytree(source, project / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(source, project / name)
    settings = project / "pyproject.toml"
    settings.write_text(
        settings.read_text(encoding="utf-8").replace(
            'requires-python = ">=3.12,<3.14"', 'requires-python = "invalid"'
        ),
        encoding="utf-8",
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "hatchling",
            "--with",
            "hatch-vcs",
            "--python",
            sys.executable,
            "python",
            "-c",
            "import build_backend; build_backend.prepare_metadata_for_build_wheel('metadata')",
        ],
        cwd=project,
        env={**os.environ, "TMPDIR": str(staging)},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode != 0
    assert "requires-python" in result.stderr
    assert not list(staging.glob("tensoracle-build-*"))
    assert not (project / "README.md").exists()
    assert not (project / "LICENSE").exists()


@pytest.mark.req("SP-TENSORACLE-021")
def test_sdist_rebuilds_the_complete_wheel_without_the_repository(distributions, tmp_path):
    """An extracted sdist builds the same package, metadata, and C++ sources on its own."""
    with tarfile.open(distributions["sdist"]) as archive:
        archive.extractall(tmp_path, filter="data")
    source = next(path for path in tmp_path.iterdir() if path.is_dir())
    output = tmp_path / "wheel"
    subprocess.run(["uv", "build", str(source), "--wheel", "--out-dir", str(output)], check=True)
    with (
        zipfile.ZipFile(distributions["wheel"]) as original,
        zipfile.ZipFile(next(output.glob("*.whl"))) as rebuilt,
    ):
        assert set(original.namelist()) == set(rebuilt.namelist())
        assert original.read("tensoracle/cpp/src/cubin_plugin.cpp") == rebuilt.read(
            "tensoracle/cpp/src/cubin_plugin.cpp"
        )
        name = next(name for name in original.namelist() if name.endswith(".dist-info/METADATA"))
        assert original.read(name) == rebuilt.read(name)


@pytest.mark.req("SP-TENSORACLE-016")
@pytest.mark.req("SP-TENSORACLE-017")
@pytest.mark.req("SP-TENSORACLE-018")
@pytest.mark.req("SP-TENSORACLE-015")
def test_wheel_metadata_describes_the_public_package(distributions):
    """PyPI receives the README, author, supported environments, MIT license, and public URLs."""
    with zipfile.ZipFile(distributions["wheel"]) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        raw = archive.read(name).decode("utf-8")
        metadata = email.message_from_string(raw)
        assert (
            archive.read(name.removesuffix("METADATA") + "licenses/LICENSE")
            == (ROOT / "LICENSE").read_bytes()
        )

    assert metadata.get_payload().splitlines() == (
        (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    )
    assert metadata["Description-Content-Type"] == "text/markdown"
    assert metadata["Author-email"] == "minamik <mia@sync.dev>"
    assert metadata["License-Expression"] == "MIT"
    assert metadata.get_all("License-File") == ["LICENSE"]
    assert metadata["Requires-Python"] == "<3.14,>=3.12"
    assert set(metadata.get_all("Classifier", [])) == {
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA :: 13",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries",
    }
    assert set(metadata.get_all("Project-URL", [])) == {
        "documentation, https://sync-dev-org.github.io/tensoracle/",
        f"repository, {PUBLIC_REPOSITORY}",
        f"issues, {PUBLIC_REPOSITORY}/issues",
    }
    word_list = ROOT / "noren-words.txt"
    if word_list.is_file():
        normalized_metadata = unicodedata.normalize("NFKC", raw).casefold()
        for word in word_list.read_text(encoding="utf-8").splitlines():
            if word.strip():
                assert (
                    unicodedata.normalize("NFKC", word.strip()).casefold()
                    not in normalized_metadata
                )
    assert all(
        word in metadata["Summary"].lower()
        for word in ("build", "save", "reuse", "execute", "tensorrt", "safetensors")
    )


@pytest.mark.req("SP-TENSORACLE-016")
def test_readme_links_and_images_resolve_on_pypi():
    """README links and images use absolute URLs or same-page anchors on the PyPI page."""

    class Targets(HTMLParser):
        def __init__(self):
            super().__init__()
            self.targets = []

        def handle_starttag(self, tag, attrs):
            attribute = {"a": "href", "img": "src"}.get(tag)
            if attribute:
                self.targets.extend(value for key, value in attrs if key == attribute)

    parser = Targets()
    parser.feed(
        markdown.markdown(
            (ROOT / "README.md").read_text(encoding="utf-8"), extensions=["fenced_code"]
        )
    )
    assert parser.targets
    assert not [
        target
        for target in parser.targets
        if not (target and (urlsplit(target).scheme or target.startswith("#")))
    ]
    assert {
        f"{PUBLIC_REPOSITORY}/blob/main/docs/optimization.md",
        f"{PUBLIC_REPOSITORY}/blob/main/docs/tensorrt-sdk.md",
        f"{PUBLIC_REPOSITORY}/blob/main/examples/cubin_plugin/README.md#rebuild-the-cubins",
    } <= set(parser.targets)
