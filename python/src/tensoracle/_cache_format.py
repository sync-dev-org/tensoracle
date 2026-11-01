"""Shared on-disk cache format and weight records, without TensorRT imports."""

import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO

_MAGIC = b"TENSORACLE\x01"
_WEIGHT_RECORD_VERSION = 1


class EngineCacheError(ValueError):
    """A stored engine cannot be used for this declaration."""


class CorruptEngineError(EngineCacheError):
    """A stored engine is incomplete or corrupt."""


class EngineConditionError(EngineCacheError):
    """A stored engine belongs to different build conditions."""


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _cache_directory() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache_root / "tensoracle"


def _read_header(stream: BinaryIO, path: Path, file_size: int) -> dict:
    """Read and validate only the header, leaving the stream at the plan body."""
    prefix = stream.read(len(_MAGIC) + 8)
    if not prefix.startswith(_MAGIC) or len(prefix) != len(_MAGIC) + 8:
        raise CorruptEngineError(f"incomplete TensorRT engine file: {path}")
    header_size = int.from_bytes(prefix[len(_MAGIC) :], "big")
    if header_size == 0 or len(prefix) + header_size >= file_size:
        raise CorruptEngineError(f"invalid TensorRT engine header: {path}")
    try:
        header = json.loads(stream.read(header_size))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise CorruptEngineError(f"invalid TensorRT engine header: {path}") from error
    if not isinstance(header, dict) or not isinstance(header.get("conditions"), dict):
        raise CorruptEngineError(f"invalid TensorRT engine header: {path}")
    shape_tensor_outputs = header.get("shape_tensor_outputs", [])
    if not isinstance(shape_tensor_outputs, list) or any(
        not isinstance(name, str) for name in shape_tensor_outputs
    ):
        raise CorruptEngineError(f"invalid TensorRT engine header: {path}")
    return header


def _read_weight_record(path: Path) -> dict | None:
    """Validate a digest record without accessing the original weight file."""
    try:
        record = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    if (
        not isinstance(record, dict)
        or record.keys() != {"version", "path", "size", "mtime_ns", "inode", "sha256", "checksum"}
        or type(record["version"]) is not int
        or record["version"] != _WEIGHT_RECORD_VERSION
        or not isinstance(record["path"], str)
        or any(type(record[key]) is not int for key in ("size", "mtime_ns", "inode"))
        or not isinstance(record["sha256"], str)
        or len(record["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in record["sha256"])
        or record["checksum"]
        != hashlib.sha256(
            _json_bytes({key: value for key, value in record.items() if key != "checksum"})
        ).hexdigest()
    ):
        return None
    return record
