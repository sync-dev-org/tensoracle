"""Prepare locked native tools before configuring any CMake project."""

import json
import os
import runpy
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from native_build import PROJECT, ROOT, build_native, preparation

SCRIPT = ROOT / "cpp/conan/prepare.py"


@pytest.mark.req("SP-TENSORACLE-063")
@pytest.mark.parametrize("executable", ["missing", "wrong-version"])
def test_unavailable_conan_reports_the_pinned_install_command_before_configure(
    tmp_path, executable
):
    """Preparation rejects an absent or different Conan CLI with its installation command."""
    cli = tmp_path / "missing"
    if executable == "wrong-version":
        source = tmp_path / "version.cpp"
        source.write_text('#include <cstdio>\nint main() { std::puts("Conan version 0.0.0"); }\n')
        cli = build_native(tmp_path / "fixture", "wrong_conan", (source,))
    pinned = next(
        value
        for value in tomllib.loads((PROJECT / "pyproject.toml").read_text())["build-system"][
            "requires"
        ]
        if value.startswith("conan==")
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "build")],
        env={**os.environ, "TENSORACLE_CONAN_EXECUTABLE": str(cli)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert f"uv tool install {pinned}" in result.stdout + result.stderr
    if executable == "wrong-version":
        assert "Conan version 0.0.0" in result.stdout + result.stderr
    assert not (tmp_path / "build/CMakeCache.txt").exists()


@pytest.mark.req("SP-TENSORACLE-033")
@pytest.mark.req("SP-TENSORACLE-063")
def test_unprepared_offline_install_reports_preparation_without_exporting(tmp_path):
    """Offline preparation cannot download, export, or build a missing dependency."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "build"), "--offline"],
        env={**os.environ, "CONAN_HOME": str(tmp_path / "empty-home")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "prepare-tests.py" in result.stdout + result.stderr
    assert str(tmp_path / "empty-home") in result.stdout + result.stderr
    assert "Exporting recipe" not in result.stdout + result.stderr
    assert not list((tmp_path / "empty-home").glob("p/*/e/conanfile.py"))


@pytest.mark.req("SP-TENSORACLE-063")
def test_tool_profile_resolves_both_contexts_without_requiring_itself(tmp_path):
    """The common tools resolve in host and build contexts without a dependency cycle."""
    prepare = preparation()
    result = subprocess.run(
        [
            prepare["cli"](),
            "lock",
            "create",
            str(ROOT / "cpp"),
            "--lockfile",
            str(ROOT / "cpp/conan.lock"),
            "--lockfile-out",
            str(tmp_path / "conan.lock"),
            "--no-remote",
            *prepare["profile_arguments"](),
            "-pr:b",
            str(ROOT / "cpp/conan/profiles/tools"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    locked = json.loads((ROOT / "cpp/conan.lock").read_text())
    resolved = json.loads((tmp_path / "conan.lock").read_text())
    for field in ("requires", "build_requires"):
        assert {ref.split("%")[0] for ref in locked[field]} == {
            ref.split("%")[0] for ref in resolved[field]
        }


@pytest.mark.req("SP-TENSORACLE-033")
@pytest.mark.req("SP-TENSORACLE-063")
def test_native_outputs_keep_usable_presets_after_switching_and_reconfiguring(tmp_path):
    """Two prepared build directories retain their own configure, build, and test presets."""
    repository = tmp_path / "repository"
    source = repository / "cpp"
    shutil.copytree(ROOT / "cpp/conan", source / "conan")
    for name in ("conanfile.py", "conan.lock"):
        shutil.copyfile(ROOT / "cpp" / name, source / name)
    (repository / "python").mkdir()
    shutil.copyfile(PROJECT / "pyproject.toml", repository / "python/pyproject.toml")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(output_switch LANGUAGES CXX)\n"
        "find_package(nlohmann_json CONFIG REQUIRED)\n"
        "add_executable(probe probe.cpp)\n"
        "target_link_libraries(probe PRIVATE nlohmann_json::nlohmann_json)\n"
        "enable_testing()\nadd_test(NAME prepared_dependency COMMAND probe)\n"
    )
    (source / "probe.cpp").write_text(
        "#include <nlohmann/json.hpp>\n"
        'int main() { return nlohmann::json{{"value", 7}}.at("value").get<int>() == 7 ? 0 : 1; }\n'
    )
    prepare = runpy.run_path(str(source / "conan/prepare.py"))
    # The equal final component also catches prefixes based only on directory basenames.
    outputs = [repository / "build/first/native", repository / "build/second/native"]
    for output in outputs:
        prepare["install"](output, offline=True)
    for output in outputs:
        environment = prepare["environment"](output)
        cmake = shutil.which("cmake", path=environment["PATH"])
        ctest = shutil.which("ctest", path=environment["PATH"])
        preset = json.loads((output / "CMakePresets.json").read_text())["configurePresets"][0][
            "name"
        ]
        for command in (
            [cmake, "--preset", preset],
            [cmake, "--preset", preset],
            [cmake, "--build", "--preset", preset, "--parallel", "4"],
            [ctest, "--preset", preset, "--parallel", "4", "--output-on-failure"],
        ):
            result = subprocess.run(
                command,
                cwd=source,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=90,
            )
            assert result.returncode == 0, result.stdout + result.stderr
        assert (
            Path(
                json.loads((output / "CMakePresets.json").read_text())["configurePresets"][0][
                    "binaryDir"
                ]
            ).resolve()
            == output.resolve()
        )
