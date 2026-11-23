"""Convert PyTorch tensor checkpoints to safetensors with optional torch dependencies."""

import argparse
import os
import pickle
from collections.abc import Mapping


def pth_to_safetensors(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write a tensor mapping to safetensors without unrestricted pickle loading.

    Nested string keys are joined with ``.``. Invalid mapping keys or leaves
    raise ``TypeError``; duplicate flattened names raise ``ValueError``.
    Both are rejected before the destination is written.
    Optional ``metadata`` writes a string-to-string mapping to ``__metadata__``.
    Non-mappings and non-string keys or values raise ``TypeError``. Invalid
    UTF-8 strings or a header exceeding 100000000 bytes raise ``ValueError``
    before writing. Empty strings and embedded NUL characters are accepted.
    Omitting metadata or passing ``None`` preserves the original output bytes.
    Torch is required only when this function is called.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Checkpoint conversion requires torch; install it with pip install 'tensoracle[convert]'"
        ) from exc
    from safetensors import SafetensorError
    from safetensors.torch import save, save_file

    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("Metadata must be a mapping of strings to strings")
        normalized = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise TypeError("Metadata keys must be strings")
            if not isinstance(value, str):
                raise TypeError("Metadata values must be strings")
            key.encode("utf-8")
            value.encode("utf-8")
            normalized[key] = value
        metadata = normalized

    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(f"Cannot safely load tensor checkpoint: {source}") from exc
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping of string keys to tensors or nested mappings")

    tensors = {}
    active_mappings = set()

    def flatten(mapping: Mapping, prefix: tuple[str, ...]) -> None:
        if id(mapping) in active_mappings:
            raise ValueError("Checkpoint contains a cyclic mapping")
        active_mappings.add(id(mapping))
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError("Checkpoint mapping keys must be strings")
            parts = (*prefix, key)
            name = ".".join(parts)
            if isinstance(value, Mapping):
                flatten(value, parts)
            elif isinstance(value, torch.Tensor):
                if name in tensors:
                    raise ValueError(f"Duplicate flattened tensor name: {name}")
                # Separate shared storage and make views contiguous without changing values.
                tensors[name] = value.detach().clone(memory_format=torch.contiguous_format)
            else:
                raise TypeError(f"Checkpoint value at {name!r} must be a tensor or mapping")
        active_mappings.remove(id(mapping))

    flatten(checkpoint, ())
    if metadata is None:
        save_file(tensors, destination)
    else:
        # Serialize first so the complete padded UTF-8 header is checked before
        # opening the destination, including with older safetensors writers.
        try:
            serialized = save(tensors, metadata=metadata)
        except SafetensorError as exc:
            raise ValueError(
                f"Cannot serialize safetensors (header limit is 100000000 bytes): {exc}"
            ) from exc
        if int.from_bytes(serialized[:8], "little") > 100_000_000:
            raise ValueError("Safetensors header exceeds 100000000 bytes")
        with open(destination, "wb") as file:
            file.write(serialized)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PyTorch checkpoint to safetensors")
    parser.add_argument("source", help="input .pth checkpoint")
    parser.add_argument("destination", help="output .safetensors file")
    args = parser.parse_args()
    try:
        pth_to_safetensors(args.source, args.destination)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    _main()
