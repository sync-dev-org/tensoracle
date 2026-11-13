"""Caller-resolved Conan dependencies support shared and static native installs."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from native_build import ROOT


def run(*command):
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def caller_dependencies(tmp_path_factory):
    caller = tmp_path_factory.mktemp("caller-dependencies")
    profile = Path(os.environ.get("TENSORACLE_CONAN_PROFILE", "linux-gcc13"))
    if not profile.is_absolute():
        profile = ROOT / "cpp/conan/profiles" / profile
    graph = json.loads(
        run(
            "conan",
            "install",
            str(ROOT / "cpp"),
            "--lockfile",
            str(ROOT / "cpp/conan.lock"),
            "--profile:host",
            str(profile),
            "--profile:build",
            str(profile),
            "--output-folder",
            str(caller),
            "--no-remote",
            "--build=never",
            "--format=json",
        )
    )
    # A consuming recipe generates a standard Conan toolchain and CMakeDeps;
    # it need not reproduce a custom license manifest for tensoracle.
    (caller / "tensoracle_licenses.cmake").unlink(missing_ok=True)
    expected_licenses = {}
    filenames = {"nlohmann_json": "LICENSE.MIT", "miniz": "LICENSE"}
    for node in graph["graph"]["nodes"].values():
        if node.get("name") in filenames:
            filename = filenames[node["name"]]
            expected_licenses[filename] = (
                Path(node["package_folder"]) / "licenses" / filename
            ).read_bytes()
    assert expected_licenses.keys() == {"LICENSE.MIT", "LICENSE"}
    return caller / "conan_toolchain.cmake", expected_licenses


@pytest.mark.req("SP-TENSORACLE-063")
@pytest.mark.req("SP-TENSORACLE-074")
@pytest.mark.parametrize("shared", ["ON", "OFF"])
def test_caller_conan_toolchain_builds_and_installs_sdk(tmp_path, caller_dependencies, shared):
    """Both library forms install their SDK and dependency licenses using the caller's graph."""
    toolchain, expected_licenses = caller_dependencies
    build = tmp_path / "build"
    configure = (
        "cmake",
        "-S",
        str(ROOT / "cpp"),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        f"-DBUILD_SHARED_LIBS={shared}",
        "-DBUILD_TESTING=OFF",
        f"-DTENSORACLE_CONAN_EXECUTABLE={tmp_path / 'missing-conan'}",
    )
    for _ in range(2):
        run(*configure)
    cache = (build / "CMakeCache.txt").read_text()
    entry = next(line for line in cache.splitlines() if line.startswith("CMAKE_TOOLCHAIN_FILE:"))
    assert entry.partition("=")[2] == str(toolchain)
    assert not (build / "conan").exists()
    run("cmake", "--build", str(build), "--parallel", "8")
    prefix = tmp_path / "prefix"
    run("cmake", "--install", str(build), "--prefix", str(prefix))

    library = "libtensoracle.so*" if shared == "ON" else "libtensoracle.a"
    assert list(prefix.rglob(library))
    assert (prefix / "include/tensoracle/tensoracle.h").is_file()
    assert list(prefix.rglob("tensoracleConfig.cmake"))
    assert (prefix / "licenses/nlohmann-json/LICENSE.MIT").read_bytes() == expected_licenses[
        "LICENSE.MIT"
    ]
    assert (prefix / "licenses/miniz/LICENSE").read_bytes() == expected_licenses["LICENSE"]
