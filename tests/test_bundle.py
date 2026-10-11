"""Inspect a cubin product without importing a producer or using a GPU."""

import hashlib
import json
import shutil
import struct
import subprocess
import sys
import zipfile

import pytest

from tensoracle.bundle import (
    CorruptCubinError,
    CubinArchitectureError,
    MissingCubinError,
    PluginBundleMismatchError,
    bundle_sha256,
    inspect_bundle,
    inspect_wheel,
)


def _cubin(sm: int) -> bytes:
    payload = bytearray(64)
    payload[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", payload, 18, 190)
    flags = sm << 8 if sm == 120 else (sm << 16) | sm
    struct.pack_into("<I", payload, 48, flags)
    return bytes(payload)


def _bundle(tmp_path, sm=86):
    cubin = _cubin(sm)
    path = tmp_path / "cubins" / "sample.cubin"
    path.parent.mkdir()
    path.write_bytes(cubin)
    manifest = {
        "schema_version": 1,
        "producer": {"triton": "3.8.0", "ptxas": "12.9.86"},
        "entries": [
            {
                "id": "sample",
                "file": "cubins/sample.cubin",
                "target_sm": "sm86",
                "entry_symbol": "sample",
                "sha256": hashlib.sha256(cubin).hexdigest(),
                "bytes": len(cubin),
                "signature": [{"name": "x", "type": "*fp32"}],
                "hidden_trailing_parameters": [
                    {"name": "global_scratch", "parameter_bytes": 8},
                    {"name": "profile_scratch", "parameter_bytes": 8},
                ],
                "resources": {
                    "compiled_shared_memory_bytes": 0,
                    "launch_dynamic_shared_memory_bytes": 0,
                    "registers_per_thread": 8,
                    "global_scratch_bytes": 0,
                    "profile_scratch_bytes": 0,
                },
                "specialization": {"divisible_by": {}},
            }
        ],
    }
    (tmp_path / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


@pytest.mark.req("REQ-TENSORACLE-003")
def test_inspector_distinguishes_missing_corrupt_and_false_sm(tmp_path):
    """Missing bytes, damaged bytes, and an architecture claim each give their own error."""
    cubin, _ = _bundle(tmp_path)
    inspect_bundle(tmp_path)
    cubin.unlink()
    with pytest.raises(MissingCubinError):
        inspect_bundle(tmp_path)
    cubin.write_bytes(_cubin(86) + b"damage")
    with pytest.raises(CorruptCubinError):
        inspect_bundle(tmp_path)
    cubin.write_bytes(_cubin(89))
    manifest = json.loads((tmp_path / "bundle.json").read_text())
    manifest["entries"][0]["sha256"] = hashlib.sha256(cubin.read_bytes()).hexdigest()
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(CubinArchitectureError):
        inspect_bundle(tmp_path)


@pytest.mark.req("REQ-TENSORACLE-003")
def test_inspector_rejects_unsupported_launch_contract(tmp_path):
    """A nonzero scratch requirement cannot be accepted by the current launcher."""
    _bundle(tmp_path)
    manifest = json.loads((tmp_path / "bundle.json").read_text())
    manifest["entries"][0]["resources"]["global_scratch_bytes"] = 64
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scratch"):
        inspect_bundle(tmp_path)


@pytest.mark.req("REQ-TENSORACLE-003")
def test_inspector_rejects_unknown_argument_or_specialization(tmp_path):
    """An unsupported ABI type or specialization rule cannot reach a launcher."""
    _bundle(tmp_path)
    manifest_path = tmp_path / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["signature"][0]["type"] = "i64"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="argument"):
        inspect_bundle(tmp_path)
    manifest["entries"][0]["signature"][0]["type"] = "*fp32"
    manifest["entries"][0]["specialization"] = {"divisible_by": {"missing": 16}}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="specialization"):
        inspect_bundle(tmp_path)


@pytest.mark.req("REQ-TENSORACLE-003")
def test_inspector_detects_plugin_from_another_bundle(tmp_path):
    """A native plugin cannot claim a different bundle from its inspected cubins."""
    if sys.platform != "linux" or shutil.which("cc") is None:
        pytest.skip("a Linux C compiler is needed for native plugin checks")
    _bundle(tmp_path)
    source = tmp_path / "plugin.c"
    source.write_text(
        'const char tensoracle_bundle_sha256[] = "' + "0" * 64 + '";\n'
        "void *getCreators(int *count) { *count = 0; return 0; }\n",
        encoding="utf-8",
    )
    library = tmp_path / "plugin.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    with pytest.raises(PluginBundleMismatchError):
        inspect_bundle(tmp_path, plugin=library)


@pytest.mark.req("REQ-TENSORACLE-003")
def test_inspector_checks_embedded_cubin_bytes_in_plugin(tmp_path):
    """A plugin with the right bundle name cannot hide a different embedded cubin."""
    if sys.platform != "linux" or shutil.which("cc") is None:
        pytest.skip("a Linux C compiler is needed for native plugin checks")
    _bundle(tmp_path)
    manifest_path = tmp_path / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["embedded_symbol"] = "tensoracle_cubin_sample"
    manifest_path.write_text(json.dumps(manifest))
    digest = bundle_sha256(manifest)
    source = tmp_path / "plugin.c"
    source.write_text(
        f'const char tensoracle_bundle_sha256[] = "{digest}";\n'
        "const unsigned char tensoracle_cubin_sample[] = {1, 2, 3};\n"
        "void *getCreators(int *count) { *count = 0; return 0; }\n",
        encoding="utf-8",
    )
    library = tmp_path / "plugin.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    with pytest.raises(PluginBundleMismatchError, match="embedded cubin"):
        inspect_bundle(tmp_path, plugin=library)


@pytest.mark.req("REQ-TENSORACLE-003")
def test_inspector_checks_a_synthetic_wheel(tmp_path):
    """A wheel missing its declared cubin is rejected without installation or a GPU."""
    cubin, _ = _bundle(tmp_path)
    wheel = tmp_path / "example-0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(tmp_path / "bundle.json", "example/bundle.json")
        archive.write(cubin, "example/cubins/sample.cubin")
    assert inspect_wheel(wheel)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(tmp_path / "bundle.json", "example/bundle.json")
    with pytest.raises(MissingCubinError):
        inspect_wheel(wheel)
