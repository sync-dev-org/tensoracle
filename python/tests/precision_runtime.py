"""Build and execute the precision guide's linear graph, including after a restart."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from safetensors.numpy import load_file, save_file

from tensoracle import (
    BuilderSettings,
    EngineBuild,
    EngineIO,
    InputShape,
    TensorSpec,
    open_engine,
)


def sample_input():
    return np.random.default_rng(11).uniform(-1, 1, (16, 32)).astype(np.float32)


def write_weights(path):
    rng = np.random.default_rng(29)
    matrix = rng.uniform(-0.25, 0.25, (32, 16)).astype(np.float32)
    bias = rng.uniform(-0.1, 0.1, (1, 16)).astype(np.float32)
    weights = {"matrix": matrix, "bias": bias}
    # A small test fixture, not a model calibration algorithm or a package API.
    for precision, maximum in (("int8", 127), ("fp8", 448)):
        weights[f"{precision}.input_scale"] = np.array(
            [np.max(np.abs(sample_input())) / maximum], dtype=np.float32
        )
        weights[f"{precision}.weight_scale"] = np.array(
            [np.max(np.abs(matrix)) / maximum], dtype=np.float32
        )
    save_file(weights, path)


def graph(path, precision):
    def define(network):
        weights = load_file(path)

        def constant(array):
            array = np.ascontiguousarray(array, dtype=np.float32)
            return network.add_constant(array.shape, array).get_output(0)

        features = network.add_input("features", trt.float32, (16, 32))
        matrix = constant(weights["matrix"])
        bias = constant(weights["bias"])
        if precision == "bf16":
            features = network.add_cast(features, trt.bfloat16).get_output(0)
            matrix = network.add_cast(matrix, trt.bfloat16).get_output(0)
            bias = network.add_cast(bias, trt.bfloat16).get_output(0)
        elif precision in ("int8", "fp8"):
            quantized_type = trt.int8 if precision == "int8" else trt.fp8

            def qdq(tensor, key):
                scale = constant(weights[f"{precision}.{key}_scale"])
                quantized = network.add_quantize(tensor, scale, quantized_type).get_output(0)
                return network.add_dequantize(quantized, scale, trt.float32).get_output(0)

            features = qdq(features, "input")
            matrix = qdq(matrix, "weight")
        value = network.add_matrix_multiply(
            features, trt.MatrixOperation.NONE, matrix, trt.MatrixOperation.NONE
        ).get_output(0)
        value = network.add_elementwise(value, bias, trt.ElementWiseOperation.SUM).get_output(0)
        if precision == "bf16":
            value = network.add_cast(value, trt.float32).get_output(0)
        value.name = "result"
        network.mark_output(value)

    return define


def open_linear(path, precision, *, reuse=False):
    def forbidden(network):
        raise AssertionError("a saved precision engine called its graph after a process restart")

    return open_engine(
        forbidden if reuse else graph(path, precision),
        EngineBuild(
            {"features": InputShape.fixed((16, 32))},
            BuilderSettings(optimization_level=0),
        ),
        name=f"linear-{precision}",
        graph_version="1",
        weights=(path,),
        io=EngineIO(
            {"features": TensorSpec(trt.float32, (16, 32))},
            {"result": TensorSpec(trt.float32, (16, 16))},
        ),
    )


def _cuda(result):
    status, *values = result
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA runtime returned {status}")
    return values[0] if values else None


def run(handle):
    features = sample_input()
    output = np.empty((16, 16), dtype=np.float32)
    source = target = stream = None
    with handle.create_execution() as execution:
        assert execution.get_tensor_shape("result") == output.shape
        assert execution.get_tensor_dtype("result") == trt.float32
        try:
            source = _cuda(cudart.cudaMalloc(features.nbytes))
            target = _cuda(cudart.cudaMalloc(output.nbytes))
            stream = _cuda(cudart.cudaStreamCreate())
            _cuda(
                cudart.cudaMemcpyAsync(
                    source,
                    features.ctypes.data,
                    features.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )
            execution.enqueue({"features": int(source), "result": int(target)}, stream=int(stream))
            _cuda(
                cudart.cudaMemcpyAsync(
                    output.ctypes.data,
                    target,
                    output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
            )
        finally:
            if stream is not None:
                _cuda(cudart.cudaStreamSynchronize(stream))
                _cuda(cudart.cudaStreamDestroy(stream))
            if target is not None:
                _cuda(cudart.cudaFree(target))
            if source is not None:
                _cuda(cudart.cudaFree(source))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("precision", choices=("bf16", "int8", "fp8"))
    parser.add_argument("outputs", type=Path)
    args = parser.parse_args()
    outputs = {}
    paths = {}
    for precision in ("fp32", args.precision):
        handle = open_linear(args.weights, precision, reuse=True)
        paths[precision] = str(handle.path)
        outputs[precision] = run(handle)
    np.savez(args.outputs, **outputs)
    print(json.dumps(paths))


if __name__ == "__main__":
    main()
