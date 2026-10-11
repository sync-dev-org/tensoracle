"""Check the files and public metadata delivered through PyPI."""

import email
import runpy
import subprocess
import tarfile
import tomllib
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import markdown
import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REPOSITORY = "https://github.com/sync-dev-org/tensoracle"


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    output = tmp_path_factory.mktemp("tensoracle-dist")
    result = subprocess.run(
        ["uv", "build", "--out-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {"sdist": next(output.glob("*.tar.gz")), "wheel": next(output.glob("*.whl"))}


@pytest.mark.req("REQ-TENSORACLE-015")
def test_sdist_contains_public_sources_without_development_files(distributions):
    """The source distribution includes code, tests, docs, and examples without private files."""
    with tarfile.open(distributions["sdist"]) as archive:
        paths = [
            PurePosixPath(*PurePosixPath(member.name).parts[1:])
            for member in archive.getmembers()
            if member.isfile()
        ]

    private = {".nf", ".github", "AGENTS.md", "CLAUDE.md", "tools"}
    assert not [str(path) for path in paths if path.parts[0] in private]
    assert {"src", "tests", "docs", "examples", "cpp"} <= {path.parts[0] for path in paths}
    assert {"README.md", "LICENSE", "pyproject.toml"} <= {str(path) for path in paths}


@pytest.mark.req("REQ-TENSORACLE-015")
def test_sdist_exclusions_match_the_public_snapshot():
    """PyPI source archives and the public repository exclude the same development paths."""
    script = ROOT / "tools/mirror_public.py"
    if not script.is_file() and not (ROOT / "AGENTS.md").is_file():
        pytest.skip("the public snapshot does not include development release tooling")
    assert script.is_file(), "the development repository must include mirror_public.py"
    exclusions = runpy.run_path(str(script))["_EXCLUSIONS"]
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert set(config["tool"]["hatch"]["build"]["targets"].get("sdist", {}).get("exclude", [])) == {
        f"/{path}" for path in exclusions
    }


@pytest.mark.req("REQ-TENSORACLE-012")
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


@pytest.mark.req("REQ-TENSORACLE-012")
@pytest.mark.req("REQ-TENSORACLE-015")
@pytest.mark.req("REQ-TENSORACLE-016")
@pytest.mark.req("REQ-TENSORACLE-011")
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

    assert metadata.get_payload() == (ROOT / "README.md").read_text(encoding="utf-8")
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
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries",
    }
    assert set(metadata.get_all("Project-URL", [])) == {
        "documentation, https://sync-dev-org.github.io/tensoracle/",
        f"repository, {PUBLIC_REPOSITORY}",
        f"issues, {PUBLIC_REPOSITORY}/issues",
    }
    assert "tensoracle-dev" not in raw
    assert all(
        word in metadata["Summary"].lower()
        for word in ("build", "save", "reuse", "execute", "tensorrt", "safetensors")
    )


@pytest.mark.req("REQ-TENSORACLE-012")
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
