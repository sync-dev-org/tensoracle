"""A caller-owned binding that builds and executes two cubin TensorRT plugins."""

from pathlib import Path

from ._native import Model


def library_path() -> Path:
    candidates = [
        path
        for path in Path(__file__).parent.iterdir()
        if "tensoracle_example_plugin" in path.name
        and path.suffix in {".so", ".dll", ".dylib"}
    ]
    if len(candidates) != 1:
        raise FileNotFoundError("expected one native example plugin library")
    return candidates[0]


def weights_path() -> Path:
    return Path(__file__).parent / "weights.safetensors"


def open_model(*, weights=None, plugin=None, force_rebuild=False) -> Model:
    """Open the example in C++, retaining two independent execution contexts."""
    return Model(
        str(weights if weights is not None else weights_path()),
        str(plugin if plugin is not None else library_path()),
        str(Path(__file__).parent / "bundle"),
        force_rebuild,
    )


def infer(features):
    """Infer an fp32 CPU array of shape (rows,32) and return an array of shape (rows,16)."""
    return open_model().infer(features)
