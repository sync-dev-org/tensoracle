"""Build the TensorRT SDK recipe from small, verified local download inputs."""

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml

from native_build import tensorrt_recipe


@pytest.fixture
def recipe_inputs(tmp_path):
    recipe = tmp_path / "recipe"
    shutil.copytree(tensorrt_recipe(), recipe)
    data = yaml.safe_load((recipe / "conandata.yml").read_text())
    source = data["sources"]["11.2.1.2"]
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    source["headers_url"] = downloads.as_uri()
    for name in source["headers"]:
        contents = (name + "\n").encode()
        (downloads / name).write_bytes(contents)
        source["headers"][name] = hashlib.sha256(contents).hexdigest()
    license_file = downloads / "LICENSE"
    license_file.write_text("TensorRT OSS license\n")
    source["oss_license"] = {
        "url": license_file.as_uri(),
        "sha256": hashlib.sha256(license_file.read_bytes()).hexdigest(),
    }
    wheel = downloads / "tensorrt.whl"
    library = b"verified TensorRT library\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tensorrt_libs/libnvinfer.so.11", library)
        archive.writestr(source["wheel_license"], "NVIDIA TensorRT license\n")
        archive.writestr("tensorrt_libs/libnvinfer_builder_resource.so.11", b"not part of the SDK")
    source["wheel"] = {
        "url": wheel.as_uri(),
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    source["library_sha256"] = hashlib.sha256(library).hexdigest()
    profile = tmp_path / "profile"
    profile.write_text("[settings]\nos=Linux\narch=x86_64\n")
    return recipe, data, downloads, profile


def create_sdk(tmp_path, recipe_inputs):
    recipe, data, _downloads, profile = recipe_inputs
    (recipe / "conandata.yml").write_text(yaml.safe_dump(data, sort_keys=False))
    return subprocess.run(
        [
            "conan",
            "create",
            str(recipe),
            "--no-remote",
            "--format=json",
            "--profile:host",
            str(profile),
            "--profile:build",
            str(profile),
        ],
        env={**os.environ, "CONAN_HOME": str(tmp_path / "conan-home")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.req("SP-TENSORACLE-065")
def test_verified_inputs_produce_only_headers_link_library_and_licenses(tmp_path, recipe_inputs):
    """The package contains all checked headers, the link library, a relative link, and both licenses."""
    result = create_sdk(tmp_path, recipe_inputs)

    assert result.returncode == 0, result.stdout + result.stderr
    nodes = json.loads(result.stdout)["graph"]["nodes"].values()
    package = Path(next(node["package_folder"] for node in nodes if node.get("name") == "tensorrt"))
    source = recipe_inputs[1]["sources"]["11.2.1.2"]
    assert {path.name for path in (package / "include").iterdir()} == set(source["headers"])
    assert (package / "lib/libnvinfer.so.11").read_bytes() == b"verified TensorRT library\n"
    assert os.readlink(package / "lib/libnvinfer.so") == "libnvinfer.so.11"
    assert {path.name for path in (package / "lib").iterdir()} == {
        "libnvinfer.so",
        "libnvinfer.so.11",
    }
    assert (package / "licenses/LICENSE.OSS").read_text() == "TensorRT OSS license\n"
    assert (package / "licenses/LICENSE.NVIDIA").read_text() == "NVIDIA TensorRT license\n"


@pytest.mark.req("SP-TENSORACLE-065")
@pytest.mark.parametrize("changed", ["header", "wheel", "library"])
def test_changed_sdk_inputs_fail_digest_verification(tmp_path, recipe_inputs, changed):
    """A changed header, wheel, or extracted library cannot become a Conan binary package."""
    _recipe, data, downloads, _profile = recipe_inputs
    if changed == "header":
        (downloads / "NvInfer.h").write_text("different release\n")
    elif changed == "wheel":
        with (downloads / "tensorrt.whl").open("ab") as file:
            file.write(b"changed download")
    else:
        data["sources"]["11.2.1.2"]["library_sha256"] = "0" * 64

    result = create_sdk(tmp_path, recipe_inputs)

    assert result.returncode != 0
    assert "sha256" in (result.stdout + result.stderr).lower()
    assert "Packaged" not in result.stderr
