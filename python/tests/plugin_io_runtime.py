"""Build one invalid plugin graph in a separate TensorRT process."""

import sys
from pathlib import Path

import tensorrt as trt

from test_plugin_io import _plan


def main():
    plugin, fault = sys.argv[1:]
    shapes = {"x": (1, 48, 3, 5), "residual": (1, 48, 3, 5)}
    if fault == "missing-input":
        shapes.pop("residual")
    elif fault == "extra-input":
        shapes["extra"] = (1, 48, 3, 5)
    elif fault == "shape":
        shapes["residual"] = (1, 48, 4, 5)
    try:
        _plan(
            Path(plugin), "Residual", shapes, dtype=trt.float32 if fault == "type" else trt.float16
        )
    except RuntimeError as error:
        assert str(error) in {
            "TensorRT rejected the plugin IO declaration",
            "TensorRT failed to build the serialized engine",
        }, str(error)
        print("declaration rejected")
    else:
        raise AssertionError("TensorRT built a network outside the plugin declaration")


if __name__ == "__main__":
    main()
