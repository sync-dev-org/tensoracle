"""Check Windows release assets and the combined Linux/Windows release record."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release_tools():
    path = ROOT / "tools/build_distributions.py"
    if not path.is_file():
        pytest.skip("development release tools are absent from the public snapshot")
    spec = importlib.util.spec_from_file_location("build_distributions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def windows_tools():
    path = ROOT / "tools/windows_distributions.py"
    if not (ROOT / "tools/build_distributions.py").is_file():
        pytest.skip("development release tools are absent from the public snapshot")
    spec = importlib.util.spec_from_file_location("windows_distributions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windows_sdk_reuses_every_wheel_sdk_byte(release_tools, tmp_path):
    """The ZIP copies headers, DLL, import library, and CMake files without another compile."""
    wheel = tmp_path / "tensoracle-2.5.2-cp312-abi3-win_amd64.whl"
    files = {
        "tensoracle/cpp/include/tensoracle/tensoracle.h": b"public C ABI\n",
        "tensoracle/cpp/bin/tensoracle-2.dll": b"MZ-wheel-library",
        "tensoracle/cpp/lib/tensoracle.lib": b"import-library",
        "tensoracle/cpp/lib/cmake/tensoracle/tensoracleConfig.cmake": b"relative CMake\n",
        "tensoracle/cpp/licenses/miniz/LICENSE": b"miniz\n",
        "tensoracle-2.5.2.dist-info/licenses/LICENSE": b"MIT\n",
        "tensoracle/_native.pyd": b"MZ-binding",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)

    sdk = release_tools.create_cpp_sdk(wheel, tmp_path / "work", "2.5.2")

    assert sdk.name == "tensoracle-sdk-2.5.2-windows-x86_64.zip"
    with zipfile.ZipFile(sdk) as archive:
        assert set(archive.namelist()) == {
            *(name for name in files if name.startswith("tensoracle/cpp/")),
            "tensoracle/cpp/licenses/tensoracle/LICENSE",
        }
        for name, value in files.items():
            if name.startswith("tensoracle/cpp/"):
                assert archive.read(name) == value
        assert archive.read("tensoracle/cpp/licenses/tensoracle/LICENSE") == b"MIT\n"
    releases = json.loads((tmp_path / "sdk-release.json").read_text())
    assert set(releases) == {"windows"}
    assert releases["windows"]["sha256"] == hashlib.sha256(sdk.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "path", [r"C:\Users\builder\work", "C:/Users/builder/.conan2", r"D:\build\_work"]
)
def test_public_windows_files_reject_build_machine_paths(release_tools, tmp_path, path):
    """PE and CMake files cannot retain a developer's directory or Conan storage path."""
    artifact = tmp_path / "tensoracle.dll"
    artifact.write_bytes(b"MZ\0" + path.encode() + b"\0")
    with pytest.raises(AssertionError, match="build path"):
        release_tools.check_build_paths(artifact)


@pytest.mark.parametrize(
    "defect", ["dependency", "missing-export", "extra-export", "debug-runtime", "x86"]
)
def test_pe_inspection_rejects_wrong_dependencies_exports_and_architecture(
    windows_tools, tmp_path, monkeypatch, defect
):
    """Release inspection rejects foreign DLLs, ABI drift, debug CRT, and 32-bit code."""
    library = tmp_path / "tensoracle-2.dll"
    library.write_bytes(b"MZ")
    dependencies = ["KERNEL32.dll", "nvinfer_11.dll", "VCRUNTIME140.dll"]
    exports = ["tensoracle_get_version", "tensoracle_compare_outputs"]
    machine = "8664 machine (x64)"
    if defect == "dependency":
        dependencies.append("cudart64_13.dll")
    elif defect == "debug-runtime":
        dependencies.append("VCRUNTIME140D.dll")
    elif defect == "missing-export":
        exports.pop()
    elif defect == "extra-export":
        exports.append("private_implementation")
    else:
        machine = "14C machine (x86)"

    def dumpbin(command, **kwargs):
        if "/dependents" in command:
            return "\n".join("    " + name for name in dependencies)
        if "/headers" in command:
            return machine
        return "\n".join(f" {i} {i:X} 00001000 {name}" for i, name in enumerate(exports, 1))

    monkeypatch.setattr(windows_tools.subprocess, "check_output", dumpbin)
    with pytest.raises(AssertionError):
        windows_tools.inspect_pe(library, {"tensoracle_get_version", "tensoracle_compare_outputs"})


def test_pe_inspection_accepts_the_header_declared_c_abi(windows_tools, tmp_path, monkeypatch):
    """The DLL exports exactly the C header's operations and uses the supported system CRT."""
    library = tmp_path / "tensoracle-2.dll"
    library.write_bytes(b"MZ")
    responses = {
        "/headers": "8664 machine (x64)",
        "/dependents": "    KERNEL32.dll\n    nvinfer_11.dll\n    MSVCP140.dll\n",
        "/exports": "  1 0 00001000 tensoracle_get_version\n",
    }
    monkeypatch.setattr(
        windows_tools.subprocess,
        "check_output",
        lambda command, **kwargs: responses[command[2]],
    )
    record = windows_tools.inspect_pe(library, {"tensoracle_get_version"})
    assert record["exports"] == ["tensoracle_get_version"]
    assert "nvinfer_11.dll" in record["dependencies"]


def test_binding_pe_accepts_nanobind_exports(windows_tools, tmp_path, monkeypatch):
    """The Python binding can export nanobind support symbols alongside its module initializer."""
    binding = tmp_path / "_native.pyd"
    binding.write_bytes(b"MZ")
    responses = {
        "/headers": "8664 machine (x64)",
        "/dependents": "    python3.dll\n    tensoracle-2.dll\n    VCRUNTIME140.dll\n",
        "/exports": "  1 0 00001000 PyInit__native\n  2 1 00001010 ?restore@python_error@nanobind@@QEAAXXZ\n",
    }
    monkeypatch.setattr(
        windows_tools.subprocess, "check_output", lambda command, **kwargs: responses[command[2]]
    )
    assert "PyInit__native" in windows_tools.inspect_pe(binding)["exports"]


def test_combined_records_name_both_operating_systems_and_verify_inputs(release_tools, tmp_path):
    """Both SDKs have explicit OS names and share the release version and source distribution."""
    source = tmp_path / "input.tar.gz"
    source.write_bytes(b"same sdist")
    linux = {"version": "2.5.2", "url": "https://example/linux.tar.gz", "sha256": "a" * 64}
    windows = {"version": "2.5.2", "url": "https://example/windows.zip", "sha256": "b" * 64}
    (tmp_path / "sdk-release.json").write_text(json.dumps({"linux": linux}))
    manifest = {
        "source_commit": "original",
        "conan_recipe_update": {"linux": linux},
        "artifacts": {},
    }
    record = {
        "platform": "windows-x86_64",
        "sdist_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sdk_release": windows,
        "artifacts": {"tensoracle-sdk-2.5.2-windows-x86_64.zip": {"sha256": "b" * 64}},
    }
    release_tools.merge_windows_record(tmp_path, manifest, record, source)
    release = json.loads((tmp_path / "sdk-release.json").read_text())
    assert set(release) == {"linux", "windows"}
    assert release["linux"] == linux
    assert release["windows"] == windows
    assert manifest["source_commit"] == "original"
    assert manifest["windows"] == record
    assert manifest["conan_recipe_update"]["windows"] == windows
    record["sdist_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="sdist"):
        release_tools.merge_windows_record(tmp_path, manifest, record, source)
    record["sdist_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    record["sdk_release"]["version"] = "2.6.0"
    with pytest.raises(AssertionError, match="version"):
        release_tools.merge_windows_record(tmp_path, manifest, record, source)
