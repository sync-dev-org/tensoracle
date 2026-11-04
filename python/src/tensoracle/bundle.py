"""Inspect cubin bundles and wheels through the C++ implementation."""

import json
from pathlib import Path

from . import _native
from ._native import (
    BundleError,
    CorruptCubinError,
    CubinArchitectureError,
    MissingCubinError,
    PluginBundleMismatchError,
)

__all__ = [
    "BundleError",
    "CorruptCubinError",
    "CubinArchitectureError",
    "MissingCubinError",
    "PluginBundleMismatchError",
    "bundle_sha256",
    "cubin_sm",
    "inspect_bundle",
    "inspect_wheel",
]


def bundle_sha256(manifest: dict) -> str:
    """Return the C++ manifest identity, excluding its own digest field."""
    return _native.bundle_sha256(json.dumps(manifest))


def cubin_sm(data: bytes) -> str:
    """Read a cubin's NVIDIA architecture without initializing CUDA."""
    return _native.cubin_sm(bytes(data))


def inspect_bundle(root: Path, *, plugin: Path | None = None) -> str:
    """Verify the manifest, cubins, and optional plugin without loading the plugin."""
    return _native.inspect_bundle(str(root), None if plugin is None else str(plugin))


def inspect_wheel(path: Path, *, plugin: str | None = None) -> str:
    """Inspect the one cubin bundle inside a wheel without installing it."""
    return _native.inspect_wheel(str(path), plugin)
