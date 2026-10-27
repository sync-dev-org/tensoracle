"""Inspect a cubin bundle and its native plugin without a GPU or producer."""

import hashlib
import json
import struct
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from zipfile import ZipFile

from ._plugins import PluginLibraryError, exported_data


class BundleError(ValueError):
    """A cubin bundle does not match its manifest or native plugin."""


class MissingCubinError(BundleError):
    """A declared cubin is missing."""


class CorruptCubinError(BundleError):
    """A cubin's bytes differ from its declared digest or length."""


class CubinArchitectureError(BundleError):
    """A cubin's ELF architecture differs from its declared SM."""


class PluginBundleMismatchError(BundleError):
    """A native plugin was built for a different cubin bundle."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def bundle_sha256(manifest: dict) -> str:
    """Return the identity of a manifest, excluding its own digest field."""
    return hashlib.sha256(
        _canonical({k: v for k, v in manifest.items() if k != "bundle_sha256"})
    ).hexdigest()


def cubin_sm(data: bytes) -> str:
    """Read the NVIDIA architecture from a cubin ELF header."""
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise CubinArchitectureError("cubin is not a 64-bit little-endian ELF")
    if struct.unpack_from("<H", data, 18)[0] != 190:
        raise CubinArchitectureError("cubin has no NVIDIA CUDA ELF machine")
    flags = struct.unpack_from("<I", data, 48)[0]
    abi = data[8]  # e_ident[EI_ABIVERSION]
    if abi == 7:
        sm = (flags >> 16) & 0xFF
    elif abi == 8:
        sm = (flags >> 8) & 0xFF
    else:
        raise CubinArchitectureError(f"unsupported cubin ELF ABI version: {abi}")
    if sm not in (86, 89, 120):
        raise CubinArchitectureError(f"unsupported cubin architecture in ELF flags: {flags:#x}")
    return f"sm{sm}"


def _validate_entry(entry: dict) -> None:
    if entry.get("target_sm") not in ("sm86", "sm89", "sm120"):
        raise BundleError("unsupported target SM")
    hidden = entry.get("hidden_trailing_parameters")
    if hidden != [
        {"name": "global_scratch", "parameter_bytes": 8},
        {"name": "profile_scratch", "parameter_bytes": 8},
    ]:
        raise BundleError("unsupported hidden trailing parameters")
    resources = entry.get("resources")
    if not isinstance(resources, dict):
        raise BundleError("missing cubin resources")
    if resources.get("global_scratch_bytes") != 0 or resources.get("profile_scratch_bytes") != 0:
        raise BundleError("nonzero scratch is unsupported")
    for key in (
        "compiled_shared_memory_bytes",
        "launch_dynamic_shared_memory_bytes",
        "registers_per_thread",
    ):
        if type(resources.get(key)) is not int or resources[key] < 0:
            raise BundleError(f"invalid cubin resource {key}")
    if resources["registers_per_thread"] == 0 or (
        resources["launch_dynamic_shared_memory_bytes"] < resources["compiled_shared_memory_bytes"]
    ):
        raise BundleError("unsupported cubin resource range")
    signature = entry.get("signature")
    if (
        not isinstance(entry.get("entry_symbol"), str)
        or not entry["entry_symbol"]
        or not isinstance(signature, list)
    ):
        raise BundleError("missing cubin symbol or signature")
    if not 1 <= len(signature) <= 14:
        raise BundleError("unsupported public argument count")
    names = set()
    for argument in signature:
        if not isinstance(argument, dict) or not isinstance(argument.get("name"), str):
            raise BundleError("invalid public argument")
        if argument["name"] in names or argument.get("type") not in (
            "*fp16",
            "*fp32",
            "*i8",
            "*i32",
            "i32",
            "fp32",
        ):
            raise BundleError("unsupported public argument")
        names.add(argument["name"])
    specialization = entry.get("specialization")
    if not isinstance(specialization, dict) or set(specialization) != {"divisible_by"}:
        raise BundleError("unsupported specialization rule")
    divisors = specialization["divisible_by"]
    if not isinstance(divisors, dict) or any(
        name not in names
        or type(divisor) is not int
        or divisor < 1
        or divisor > (1 << 30)
        or divisor & (divisor - 1)
        for name, divisor in divisors.items()
    ):
        raise BundleError("unsupported specialization rule")
    if any(
        argument["type"] == "fp32" and divisors.get(argument["name"], 1) != 1
        for argument in signature
    ):
        raise BundleError("unsupported floating point specialization rule")


def _relative_path(name: str) -> bool:
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    return not (
        posix.is_absolute()
        or windows.drive
        or windows.root
        or ".." in posix.parts
        or ".." in windows.parts
    )


def inspect_bundle(root: Path, *, plugin: Path | None = None) -> str:
    """Verify cubin bytes, ELF SM, supported launch metadata, and plugin identity."""
    root = Path(root)
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise BundleError("unsupported bundle manifest")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise BundleError("bundle has no cubin entries")
    digest = bundle_sha256(manifest)
    if "bundle_sha256" in manifest and manifest["bundle_sha256"] != digest:
        raise BundleError("bundle manifest digest differs")
    visited = set()
    cubins = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise BundleError("invalid cubin entry")
        _validate_entry(entry)
        name = entry.get("file")
        if not isinstance(name, str) or not _relative_path(name):
            raise BundleError("invalid cubin path")
        if name in visited:
            raise BundleError("duplicate cubin path")
        visited.add(name)
        path = root / name
        if not path.is_file():
            raise MissingCubinError(f"missing cubin: {path}")
        data = path.read_bytes()
        if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get(
            "sha256"
        ):
            raise CorruptCubinError(f"cubin digest or size differs: {path}")
        actual_sm = cubin_sm(data)
        if actual_sm != entry["target_sm"]:
            raise CubinArchitectureError(
                f"cubin {path} declares {entry['target_sm']} but ELF contains {actual_sm}"
            )
        if "embedded_symbol" in entry:
            cubins[entry["embedded_symbol"]] = data
    if plugin is not None:
        try:
            library = Path(plugin).read_bytes()
            embedded = exported_data(library, "tensoracle_bundle_sha256", 65)
        except (OSError, PluginLibraryError) as error:
            raise PluginBundleMismatchError(
                f"plugin bundle identity is unavailable: {plugin}"
            ) from error
        if embedded.rstrip(b"\0") != digest.encode("ascii"):
            raise PluginBundleMismatchError(f"plugin and cubin bundle differ: {plugin}")
        for symbol, data in cubins.items():
            try:
                native = exported_data(library, symbol, len(data))
            except PluginLibraryError as error:
                raise PluginBundleMismatchError(f"plugin lacks embedded cubin {symbol}") from error
            if native != data:
                raise PluginBundleMismatchError(f"plugin embedded cubin differs: {symbol}")
    return digest


def inspect_wheel(path: Path, *, plugin: str | None = None) -> str:
    """Inspect the one cubin bundle inside a wheel without installing it."""
    with tempfile.TemporaryDirectory() as directory, ZipFile(path) as archive:
        root = Path(directory)
        for info in archive.infolist():
            if not _relative_path(info.filename):
                raise BundleError("wheel contains an invalid path")
            member = Path(info.filename)
            if info.is_dir():
                continue
            destination = root / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))
        manifests = list(root.rglob("bundle.json"))
        if len(manifests) != 1:
            raise BundleError("wheel must contain exactly one cubin bundle")
        if plugin is not None and not _relative_path(plugin):
            raise BundleError("invalid plugin path in wheel")
        library = root / plugin if plugin is not None else None
        return inspect_bundle(manifests[0].parent, plugin=library)
