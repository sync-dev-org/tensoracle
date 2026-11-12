"""Obtain the released C++ SDK through Conan without building tensoracle."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from native_build import ROOT


@pytest.mark.req("SP-TENSORACLE-065")
@pytest.mark.parametrize("changed", [False, True])
def test_recipe_packages_the_verified_sdk_and_rejects_changed_downloads(tmp_path, changed):
    """Conan preserves the wheel-relative SDK layout and refuses a mismatched SHA256."""
    recipe = tmp_path / "recipe"
    shutil.copytree(ROOT / "cpp/conan/package", recipe)
    sdk = tmp_path / "sdk.tar.gz"
    with tarfile.open(sdk, "w:gz") as archive:
        data = b"prebuilt ABI library\n"
        member = tarfile.TarInfo("tensoracle/cpp/lib/libtensoracle.so.2.1.2")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
        member = tarfile.TarInfo("tensoracle/cpp/lib/libtensoracle.so.2")
        member.type = tarfile.SYMTYPE
        member.linkname = "libtensoracle.so.2.1.2"
        archive.addfile(member)
    (recipe / "conandata.yml").write_text(
        json.dumps(
            {
                "sources": {
                    "2.1.2": {
                        "Linux": {
                            "x86_64": {
                                "url": sdk.as_uri(),
                                "sha256": "0" * 64
                                if changed
                                else hashlib.sha256(sdk.read_bytes()).hexdigest(),
                            }
                        }
                    }
                }
            }
        )
    )
    profile = tmp_path / "profile"
    profile.write_text("[settings]\nos=Linux\narch=x86_64\n")
    # The third-party recipe is independently tested; this stand-in supplies its
    # external package boundary without downloading NVIDIA's large wheel here.
    dependency = tmp_path / "tensorrt"
    dependency.mkdir()
    (dependency / "conanfile.py").write_text(
        "from conan import ConanFile\nclass SDK(ConanFile):\n"
        '    name = "tensorrt"\n    version = "11.2.1.2"\n'
        '    package_type = "shared-library"\n    settings = "os", "arch"\n'
    )
    environment = {**os.environ, "CONAN_HOME": str(tmp_path / "conan-home")}
    common = ["--no-remote", "-pr:h", str(profile), "-pr:b", str(profile)]
    subprocess.run(
        ["conan", "create", str(dependency), *common],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    result = subprocess.run(
        ["conan", "create", str(recipe), "--format=json", *common],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if changed:
        assert result.returncode != 0
        assert "sha256" in (result.stdout + result.stderr).lower()
        assert "Packaged" not in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        nodes = json.loads(result.stdout)["graph"]["nodes"].values()
        node = next(node for node in nodes if node.get("name") == "tensoracle")
        package = Path(node["package_folder"])
        assert (package / "tensoracle/cpp/lib/libtensoracle.so.2").read_bytes() == data
        assert (
            os.readlink(package / "tensoracle/cpp/lib/libtensoracle.so.2")
            == "libtensoracle.so.2.1.2"
        )
        assert not list(package.rglob("libnvinfer*"))
        assert node["settings"] == {"os": "Linux", "arch": "x86_64"}
