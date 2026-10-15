"""Measure build-only host memory in an isolated process with real TensorRT objects."""

import ctypes
import ctypes.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors.numpy import save_file

from tensoracle import (
    BuilderSettings,
    EngineBuild,
    EngineIO,
    InputShape,
    TensorSpec,
    build_serialized_engine,
    open_engine,
)


def _rss() -> int:
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def measure(entry: str, fail: bool, directory: Path) -> dict[str, int]:
    """Keep a successful engine alive and measure after handling a build failure."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    weights = directory / "weights.safetensors"
    save_file({"matrix": np.full((512, 512), 1 / 512, dtype=np.float32)}, str(weights))

    def graph(network):
        value = network.add_input("features", trt.float32, (1, 512))
        for _ in range(64):
            matrix = np.full((512, 512), 1 / 512, dtype=np.float32)
            constant = network.add_constant(matrix.shape, matrix).get_output(0)
            value = network.add_matrix_multiply(
                value, trt.MatrixOperation.NONE, constant, trt.MatrixOperation.NONE
            ).get_output(0)
        if not fail:
            value.name = "result"
            network.mark_output(value)

    declaration = EngineBuild(
        {"features": InputShape.fixed((1, 512))},
        BuilderSettings(optimization_level=0),
    )
    result = failure = None
    try:
        if entry == "build_serialized_engine":
            result = build_serialized_engine(graph, declaration)
        else:
            result = open_engine(
                graph,
                declaration,
                name="memory-regression",
                graph_version="1",
                weights=(weights,),
                io=EngineIO(
                    {"features": TensorSpec(trt.float32, (1, 512))},
                    {"result": TensorSpec(trt.float32, (1, 512))},
                ),
            )
    except RuntimeError as error:
        failure = error
    if fail:
        assert failure is not None and result is None
        assert str(failure) == "TensorRT failed to build the serialized engine"
        assert not list((directory / "tensoracle").glob("*.engine"))
        failure = None
    else:
        assert failure is None and result
        if entry == "open_engine":
            assert result.path.is_file()
            assert result.engine.get_tensor_shape("result") == (1, 512)

    before = _rss()
    libc.malloc_trim(0)
    after = _rss()
    return {"before": before, "after": after, "reduction": before - after}


if __name__ == "__main__":
    print(json.dumps(measure(sys.argv[1], sys.argv[2] == "failure", Path(sys.argv[3]))))
