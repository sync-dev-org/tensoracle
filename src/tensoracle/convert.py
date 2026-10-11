"""Convert PyTorch tensor checkpoints to safetensors with optional torch dependencies."""

import argparse
import os
import pickle
from collections.abc import Mapping


def pth_to_safetensors(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    """Write a tensor mapping to safetensors without unrestricted pickle loading.

    Nested string keys are joined with ``.``. Invalid mapping keys or leaves
    raise ``TypeError``; duplicate flattened names raise ``ValueError``.
    Both are rejected before the destination is written.
    Torch is required only when this function is called.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Checkpoint conversion requires torch; install it with pip install 'tensoracle[convert]'"
        ) from exc
    from safetensors.torch import save_file

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
    save_file(tensors, destination)


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
