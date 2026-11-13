"""Keep saved engines and weight records interoperable across JSON dependency versions."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from native_build import ROOT

NATIVE = Path(__file__).parent / "native/cache"
GOLDEN = Path(__file__).parent / "fixtures/cache-3.11.3.json"


def run(command, **kwargs):
    command = [str(value) for value in command]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        **kwargs,
    )
    diagnostic = result.stdout + result.stderr
    if command[:2] == ["conan", "install"]:
        diagnostic += "\nPrepare dependencies: cmake -P cpp/conan/prepare-tests.cmake"
    assert result.returncode == 0, diagnostic
    return result.stdout


def install_json_dependencies(folder, version, *, cppstd=None):
    """Resolve one locked JSON version from prepared packages without changing other inputs."""
    selected_lock = ROOT / "cpp/conan.lock"
    if version == "3.12.0":
        lock = json.loads(selected_lock.read_text())
        compatibility = json.loads((ROOT / "cpp/conan/cache-compatibility.lock").read_text())
        lock["requires"] = [
            compatibility["requires"][0] if ref.startswith("nlohmann_json/") else ref
            for ref in lock["requires"]
        ]
        selected_lock = folder / "conan.lock"
        selected_lock.write_text(json.dumps(lock))
    profile = Path(os.environ.get("TENSORACLE_CONAN_PROFILE", "linux-gcc13"))
    if not profile.is_absolute():
        profile = ROOT / "cpp/conan/profiles" / profile
    options = ["--settings:host", f"compiler.cppstd={cppstd}"] if cppstd else []
    if cache := os.environ.get("TENSORACLE_CONAN_CACHE"):
        options.extend(["--core-conf", f"core.cache:storage_path={cache}"])
    dependencies = folder / "dependencies"
    run(
        [
            "conan",
            "install",
            ROOT / "cpp",
            "--lockfile",
            selected_lock,
            "--profile:all",
            profile,
            "--no-remote",
            "--build=never",
            "--output-folder",
            dependencies,
            "--conf:all",
            "tools.build:jobs=4",
            *options,
        ]
    )
    return dependencies


@pytest.mark.req("SP-TENSORACLE-073")
@pytest.mark.req("SP-TENSORACLE-074")
@pytest.mark.parametrize("shared", ["ON", "OFF"], ids=["shared", "static"])
def test_sources_build_with_cpp20_and_json_3_12(tmp_path, shared):
    """Source consumers can build either library form with C++20 and JSON 3.12.0."""
    dependencies = install_json_dependencies(tmp_path, "3.12.0", cppstd="20")
    build = tmp_path / "build"
    run(
        [
            "cmake",
            "-S",
            ROOT / "cpp",
            "-B",
            build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_STANDARD=20",
            "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
            f"-DCMAKE_TOOLCHAIN_FILE={dependencies / 'conan_toolchain.cmake'}",
            "-DBUILD_TESTING=OFF",
            f"-DBUILD_SHARED_LIBS={shared}",
            "-DTENSORACLE_BUILD_CUBIN_HELPER=OFF",
        ]
    )
    run(["cmake", "--build", build, "--target", "tensoracle", "--parallel", "4"])


@pytest.fixture(scope="module")
def cache_versions(tmp_path_factory):
    """Use prepared Conan packages; never download or rebuild dependencies during tests."""
    work = tmp_path_factory.mktemp("cache-json-versions")
    lock_bytes = (ROOT / "cpp/conan.lock").read_bytes()
    executables = {}
    for version in ("3.11.3", "3.12.0"):
        folder = work / version
        folder.mkdir()
        dependencies = install_json_dependencies(folder, version)
        build = folder / "build"
        run(
            [
                "cmake",
                "-S",
                NATIVE,
                "-B",
                build,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DTENSORACLE_CPP_ROOT={ROOT / 'cpp'}",
                f"-DCMAKE_TOOLCHAIN_FILE={dependencies / 'conan_toolchain.cmake'}",
            ]
        )
        run(["cmake", "--build", build, "--target", "cache_compatibility", "--parallel", "4"])
        executables[version] = Path((build / "compatibility-Release.txt").read_text())
    assert (ROOT / "cpp/conan.lock").read_bytes() == lock_bytes
    return executables


@pytest.mark.req("SP-TENSORACLE-063")
def test_cache_builds_use_locked_revisions_despite_other_cached_revisions(
    tmp_path_factory, monkeypatch
):
    """Both builds keep the declared recipe revisions and every other locked dependency."""
    locked = json.loads((ROOT / "cpp/conan.lock").read_text())
    compatibility = json.loads((ROOT / "cpp/conan/cache-compatibility.lock").read_text())
    references = {
        "3.11.3": next(ref for ref in locked["requires"] if ref.startswith("nlohmann_json/")),
        "3.12.0": compatibility["requires"][0],
    }
    seen = set()
    actual_run = subprocess.run

    def conan_boundary(command, **kwargs):
        # Only the external Conan lookup is simulated. Generated lock files,
        # dependency installation, CMake, and compiled cache readers stay real.
        if command[:3] == ["conan", "lock", "create"] and "--lockfile=" in command:
            requirement = next(value for value in command if value.startswith("--requires="))
            reference = requirement.removeprefix("--requires=")
            if "#" not in reference:
                reference += "#" + "0" * 32
            path = Path(command[command.index("--lockfile-out") + 1])
            path.write_text(json.dumps({"requires": [reference]}))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["conan", "install"]:
            path = Path(command[command.index("--lockfile") + 1])
            selected = json.loads(path.read_text())
            reference = next(
                ref for ref in selected["requires"] if ref.startswith("nlohmann_json/")
            )
            version = reference.split("/")[1].split("#")[0]
            assert reference == references[version]
            for field, values in locked.items():
                if field == "requires":
                    assert [
                        ref for ref in selected[field] if not ref.startswith("nlohmann_json/")
                    ] == [ref for ref in values if not ref.startswith("nlohmann_json/")]
                else:
                    assert selected[field] == values
            seen.add(version)
        return actual_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", conan_boundary)
    cache_versions.__wrapped__(tmp_path_factory)
    assert seen == set(references)


@pytest.mark.req("SP-TENSORACLE-033")
@pytest.mark.req("SP-TENSORACLE-063")
def test_unprepared_json_compatibility_dependency_shows_the_preparation_command(tmp_path):
    """An absent 3.12.0 recipe fails offline and tells the developer how to prepare it."""
    lock = ROOT / "cpp/conan/cache-compatibility.lock"
    reference = json.loads(lock.read_text())["requires"][0]
    profile = ROOT / "cpp/conan/profiles/linux-gcc13"
    with pytest.raises(AssertionError) as failure:
        run(
            [
                "conan",
                "install",
                f"--requires={reference}",
                "--lockfile",
                lock,
                "--profile:all",
                profile,
                "--no-remote",
                "--build=never",
                "--output-folder",
                tmp_path / "dependencies",
            ],
            env={**os.environ, "CONAN_HOME": str(tmp_path / "empty-home")},
        )
    message = str(failure.value)
    assert "nlohmann_json/3.12.0" in message
    assert "Prepare dependencies: cmake -P cpp/conan/prepare-tests.cmake" in message


@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.req("SP-TENSORACLE-011")
@pytest.mark.req("SP-TENSORACLE-055")
def test_json_versions_preserve_saved_bytes_and_read_each_others_files(cache_versions, tmp_path):
    """Both versions preserve 3.11.3 filenames and headers and reuse each other's weight records."""
    for number, case in enumerate(json.loads(GOLDEN.read_text())):
        directory = tmp_path / f"{number}-重み-🧪"
        directory.mkdir()
        weight = directory / "weights"
        weight.write_bytes(b"weight bytes")
        (directory / "plugin").write_bytes(b"plugin bytes")
        caches = {}
        manifests = {}
        records = {}
        for version, executable in cache_versions.items():
            caches[version] = directory / version
            environment = {**os.environ, "XDG_CACHE_HOME": str(caches[version])}
            observed = json.loads(
                run(
                    [executable, "write", directory],
                    input=json.dumps(case["input"]),
                    env=environment,
                )
            )
            assert observed["json_version"] == {"3.11.3": 31103, "3.12.0": 31200}[version]
            assert observed["filename"] == case["filename"]
            engine = caches[version] / "tensoracle" / observed["filename"]
            assert engine.read_bytes() == case["engine"].encode("latin1")
            manifests[version] = observed["header"]
            record = next((caches[version] / "tensoracle").glob("weight-*.json"))
            records[version] = record.read_bytes()
        assert records["3.11.3"] == records["3.12.0"]
        # Equal stat attributes with different bytes distinguish record reuse from rehashing.
        before = weight.stat()
        weight.write_bytes(b"other bytes!")
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
        for reader, writer in (("3.11.3", "3.12.0"), ("3.12.0", "3.11.3")):
            observed = json.loads(
                run(
                    [cache_versions[reader], "read", directory],
                    input=json.dumps(case["input"]),
                    env={**os.environ, "XDG_CACHE_HOME": str(caches[writer])},
                )
            )
            assert observed["filename"] == case["filename"]
            assert observed["header"] == manifests[writer]
