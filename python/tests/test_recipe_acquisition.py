"""Acquire a locked recipe through Conan without retaining download credentials."""

import hashlib
import http.server
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest

from native_build import ROOT

SCRIPT = ROOT / "cpp/conan/recipes.py"
TOKEN = "fixture-recipe-download-credential"


@pytest.fixture
def acquisition(tmp_path):
    recipe = tmp_path / "recipe"
    recipe.mkdir()
    (recipe / "conanfile.py").write_text(
        "from conan import ConanFile\nclass SDK(ConanFile):\n"
        '    name = "tensorrt"\n    version = "11.2.1.2"\n'
        '    exports = "LICENSE"\n'
    )
    (recipe / "LICENSE").write_text("fixture license\n")
    home = tmp_path / "conan-home"
    environment = {**os.environ, "CONAN_HOME": str(home), "TMPDIR": str(tmp_path / "temporary")}
    Path(environment["TMPDIR"]).mkdir()
    reference = json.loads(
        subprocess.check_output(
            ["conan", "export", str(recipe), "--format=json"], env=environment, text=True
        )
    )["reference"]
    cuda = tmp_path / "cuda-recipe"
    cuda.mkdir()
    (cuda / "conanfile.py").write_text(
        "from conan import ConanFile\nclass CUDA(ConanFile):\n"
        '    name = "cuda-toolkit"\n    exports_sources = "redist.json"\n'
    )
    (cuda / "redist.json").write_text('{"fixture": "CUDA manifest"}\n')
    cuda_reference = json.loads(
        subprocess.check_output(
            ["conan", "export", str(cuda), "--version=13.2.2", "--format=json"],
            env=environment,
            text=True,
        )
    )["reference"]
    shutil.rmtree(home)
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        archive.add(recipe, arcname="recipes/tensorrt")
        archive.add(cuda, arcname="recipes/cuda-toolkit")
    payload = archive_bytes.getvalue()
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Accept")))
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                self.send_error(401)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conan = tmp_path / "cpp/conan"
    conan.mkdir(parents=True)
    (conan.parent / "conan.lock").write_text(
        json.dumps({"requires": [reference], "build_requires": [cuda_reference]})
    )
    manifest = {
        "tag": "v0.2.0",
        "url": f"http://127.0.0.1:{server.server_port}/assets/123",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (conan / "recipes.json").write_text(json.dumps(manifest))
    tools = tmp_path / "tools"
    tools.mkdir()
    gh = tools / "gh"
    gh.write_text(f'#!/bin/sh\nprintf "%s\\n" "{TOKEN}"\n')
    gh.chmod(0o755)
    environment["PATH"] = str(tools) + os.pathsep + environment["PATH"]
    try:
        yield conan, environment, reference, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def acquire(acquisition, tmp_path):
    conan, environment, _reference, _requests = acquisition
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--conan-dir",
            str(conan),
            "--copy-to",
            str(tmp_path / "copy"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def assert_no_credential(result, tmp_path):
    assert TOKEN not in result.stdout + result.stderr
    assert not list(tmp_path.rglob("source_credentials.json"))
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.name != "gh":
            assert TOKEN.encode() not in path.read_bytes(), path


def test_downloaded_recipe_matches_the_lock_and_can_be_copied_without_credentials(
    acquisition, tmp_path
):
    """A checked archive supplies the locked recipe, and its copy needs neither gh nor a download."""
    result = acquire(acquisition, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    conan, environment, reference, requests = acquisition
    copied = tmp_path / "copy"
    assert (copied / "tensorrt/conanfile.py").is_file()
    assert (copied / "tensorrt/LICENSE").read_text() == "fixture license\n"
    assert (copied / "cuda-toolkit/conanfile.py").is_file()
    assert (
        json.loads((copied / "cuda-toolkit/redist.json").read_text())["fixture"] == "CUDA manifest"
    )
    assert requests == [("/assets/123", "application/octet-stream")]
    assert_no_credential(result, tmp_path)
    shutil.copytree(copied, conan / "recipes")
    shutil.rmtree(Path(environment["CONAN_HOME"]))
    (tmp_path / "tools/gh").unlink()
    command = [sys.executable, str(SCRIPT), "--conan-dir", str(conan)]
    second = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stdout + second.stderr
    assert len(requests) == 1
    cached = subprocess.check_output(
        ["conan", "cache", "path", reference], env=environment, text=True
    )
    assert (Path(cached.strip()) / "LICENSE").is_file()
    cuda_reference = json.loads((conan.parent / "conan.lock").read_text())["build_requires"][0]
    cached = subprocess.check_output(
        ["conan", "cache", "path", cuda_reference], env=environment, text=True
    )
    exported = subprocess.check_output(
        ["conan", "cache", "path", cuda_reference, "--folder=export_source"],
        env=environment,
        text=True,
    )
    assert (Path(exported.strip()) / "redist.json").is_file()


@pytest.mark.parametrize("changed", ["archive", "tensorrt", "cuda-toolkit"])
def test_changed_archive_or_unlocked_recipe_fails_without_retaining_credentials(
    acquisition, tmp_path, changed
):
    """Preparation rejects a mismatched archive digest or recipe revision before dependency install."""
    conan, _environment, _reference, _requests = acquisition
    if changed == "archive":
        manifest = json.loads((conan / "recipes.json").read_text())
        manifest["sha256"] = "0" * 64
        (conan / "recipes.json").write_text(json.dumps(manifest))
    else:
        lockfile = conan.parent / "conan.lock"
        lock = json.loads(lockfile.read_text())
        field = "requires" if changed == "tensorrt" else "build_requires"
        lock[field][0] = lock[field][0].split("#")[0] + "#" + "0" * 32
        lockfile.write_text(json.dumps(lock))

    # Exercise the CMake entry point as well as the downloader: rejection must
    # stop preparation before its Conan install can run.
    for name in ("install.cmake", "prepare-tests.cmake", "recipes.py"):
        shutil.copyfile(ROOT / "cpp/conan" / name, conan / name)
    shutil.copytree(ROOT / "cpp/conan/profiles", conan / "profiles")
    project = conan.parent.parent / "python"
    project.mkdir()
    (project / "pyproject.toml").write_text('[build-system]\nrequires = ["conan==2.34.0"]\n')
    result = subprocess.run(
        ["cmake", "-P", str(conan / "prepare-tests.cmake")],
        env=acquisition[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert ("sha256" if changed == "archive" else "locked") in (
        result.stdout + result.stderr
    ).lower()
    assert not (tmp_path / "copy").exists()
    assert_no_credential(result, tmp_path)


def test_release_bundle_retains_the_verified_input_without_credentials(acquisition, tmp_path):
    """The release input preserves the pinned bundle bytes without authentication files."""
    conan, environment, _reference, requests = acquisition
    destination = tmp_path / "recipe-input.tar.gz"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--conan-dir",
            str(conan),
            "--download-bundle",
            str(destination),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((conan / "recipes.json").read_text())
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == manifest["sha256"]
    with tarfile.open(destination) as archive:
        assert archive.extractfile("recipes/tensorrt/LICENSE").read() == b"fixture license\n"
    assert requests == [("/assets/123", "application/octet-stream")]
    assert_no_credential(result, tmp_path)
