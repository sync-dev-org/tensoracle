# Build bf16, int8, and fp8 engines

The graph function selects tensor types. Use the same `EngineBuild`, `open_engine`, and execution API for fp32, fp16, bf16, int8, and fp8. For bf16, cast tensors to `trt.bfloat16`. For int8 and fp8, place explicit quantize/dequantize (Q/DQ) layers and their scales in the graph. TensorRT 11's [`add_quantize` and `add_dequantize` operators](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/_static/operators/Quantize.html) accept the quantized type and dequantized type explicitly.

| Graph precision | Tensor placement in this example | Supported tensoracle GPU targets |
| --- | --- | --- |
| fp32 | fp32 matrix multiplication and bias, with TF32 disabled | sm86, sm89, sm120 |
| bf16 | Cast features, matrix, and bias to bf16; cast the result to fp32 | sm86, sm89, sm120 |
| int8 | Q/DQ the features and matrix with `trt.int8` and explicit scales | sm86, sm89, sm120 |
| fp8 | Q/DQ the features and matrix with `trt.fp8` (E4M3) and explicit scales | sm89, sm120 |

Do not build the fp8 graph on sm86. The graph keeps fp32 input and output buffers for every precision; the internal tensors determine its arithmetic. TensorRT chooses implementations for the typed graph, so an int8 or fp8 graph can retain higher precision operations such as the bias addition. This example does not establish a speed improvement.

## Prepare weights and scales outside tensoracle

For a real model, use an external quantization library, such as [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), to determine scales from representative data or quantization-aware training. The model package translates those results into its safetensors file and graph. tensoracle does not calculate scales, calibrate a model, or translate a quantizer's export format.

Store the fp32 weights and positive fp32 scales together in safetensors. In this example, `matrix` has shape `(32, 16)`, `bias` has shape `(1, 16)`, and each scale has shape `(1,)`. The keys `int8.input_scale`, `int8.weight_scale`, `fp8.input_scale`, and `fp8.weight_scale` distinguish the two quantization recipes. Both Q and DQ use the same scale for each tensor.

The following small fixture uses maximum absolute values to choose scales. Its inputs and weights match the GPU test; this scale calculation is only a demonstration for that fixture, not a calibration recipe for other models:

```python
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

path = Path("linear.safetensors")
features = np.random.default_rng(11).uniform(-1, 1, (16, 32)).astype(np.float32)
rng = np.random.default_rng(29)
matrix = rng.uniform(-0.25, 0.25, (32, 16)).astype(np.float32)
bias = rng.uniform(-0.1, 0.1, (1, 16)).astype(np.float32)
weights = {"matrix": matrix, "bias": bias}
for precision, maximum in (("int8", 127), ("fp8", 448)):
    weights[f"{precision}.input_scale"] = np.array(
        [np.max(np.abs(features)) / maximum], dtype=np.float32
    )
    weights[f"{precision}.weight_scale"] = np.array(
        [np.max(np.abs(matrix)) / maximum], dtype=np.float32
    )
save_file(weights, path)
```

[`safetensors.numpy` 0.8.0](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/py_src/safetensors/numpy.py) with NumPy 2.5 does not directly load bf16 or fp8 tensors into NumPy arrays: bf16 loading raises `TypeError`, and FP8 E4M3 loading raises `AttributeError`. A safetensors file can store those types, but its NumPy reader cannot represent them in this environment. Keep this example's weights and scales in fp32, then cast or quantize in TensorRT. If an existing checkpoint stores bf16 or fp8 weights, convert them with a reader that supports that dtype outside this NumPy graph path. Installing torch is not required for this example.

## Define the typed graph

This graph computes `features @ matrix + bias`. The bf16 branch casts all three operands before arithmetic and returns fp32. The int8/fp8 branch applies per-tensor Q/DQ to the features and matrix; its bias stays fp32. Keeping the stored matrix in fp32 lets TensorRT quantize it with the supplied scale during building.

```python
import tensorrt as trt
from safetensors.numpy import load_file


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
```

## Build, save, and reuse each precision

Use a distinct engine name or graph version whenever tensor types or the graph recipe change. The same weight file serves all four graphs here, so its digest cannot distinguish their precision. Scales are part of that file and therefore part of the saved engine identity.

```python
from tensoracle import (
    BuilderSettings, EngineBuild, EngineIO, InputShape, TensorSpec, open_engine,
)

declaration = EngineBuild(
    {"features": InputShape.fixed((16, 32))},
    BuilderSettings(optimization_level=0),
)
io = EngineIO(
    {"features": TensorSpec(trt.float32, (16, 32))},
    {"result": TensorSpec(trt.float32, (16, 16))},
)


def open_linear(precision):
    return open_engine(
        graph(path, precision),
        declaration,
        name=f"linear-{precision}",
        graph_version="1",
        weights=(path,),
        io=io,
    )


reference_handle = open_linear("fp32")
candidate_handle = open_linear("bf16")  # Or "int8"; "fp8" on sm89/sm120.
```

TF32 remains disabled by tensoracle, including for the fp32 reference. Engines are saved under `XDG_CACHE_HOME/tensoracle`, or `~/.cache/tensoracle`. Repeat the same call with unchanged build conditions in another process to reuse the plan without invoking the graph function. Keep the weight file and the engine names/versions stable across the restart. Follow [Execute an engine](index.md#execute-an-engine) to allocate fp32 input/output buffers and enqueue each handle with the same inputs.

## Compare outputs and choose limits

First establish the fp32 engine against fixed fp32 outputs from the official model implementation, as described in [stage 1](optimization.md#1-establish-the-fp32-reference). Then run the candidate and accepted fp32 engine with identical inputs. Pass their CPU NumPy outputs to `compare_outputs` and choose quality limits in the application:

```python
from tensoracle import compare_outputs

# candidate_output and fp32_output are the application's engine outputs.
comparison = compare_outputs(candidate_output, fp32_output, atol=0.01, rtol=0)
accepted = (
    comparison.candidate_nonfinite_count == 0
    and comparison.reference_nonfinite_count == 0
    and comparison.outside_tolerance_count == 0
    and comparison.relative_l2_error is not None
    and comparison.relative_l2_error < 0.01
)
```

These example limits belong to the small bf16 fixture. For the fixture above, a TensorRT 11.4 run on sm120 measured:

| Candidate | Maximum absolute error | Relative L2 error | Test `atol` (`rtol=0`) | Test relative L2 limit |
| --- | --- | --- | --- | --- |
| bf16 | 0.00357 | 0.00294 | 0.01 | 0.01 |
| int8 | 0.00710 | 0.00572 | 0.02 | 0.02 |
| fp8 | 0.0501 | 0.0382 | 0.12 | 0.08 |

The test allows at least twice the observed errors for differences in GPU tactics. These are not tensoracle defaults or quality guarantees for other models. Choose representative data, per-output limits, and model-specific quality measurements for the real application. The GPU test also checks that the two engines occupy separate cache entries and reproduce their outputs in another process without calling either graph.
