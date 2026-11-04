# Build bf16, int8, and fp8 engines

The model package selects precision through tensor types in its C++ TensorRT network. Use `tensoracle::builder_options` and `tensoracle::open_engine` for every precision. TF32 is disabled; the options have no precision flag. For bf16, use TensorRT's `kBF16` casts. For int8 and fp8, place explicit quantize/dequantize (Q/DQ) layers and supply scales in safetensors.

| Graph precision | Tensor placement in this example | Target GPUs |
| --- | --- | --- |
| fp32 | fp32 matrix multiplication and bias, with TF32 disabled | sm86, sm89, sm120 |
| fp16 | Cast features, matrix, and bias to `kHALF`; cast the result to fp32 | sm86, sm89, sm120 |
| bf16 | Cast features, matrix, and bias to `kBF16`; cast the result to fp32 | sm86, sm89, sm120 |
| int8 | Q/DQ features and matrix with `kINT8` and explicit scales | sm86, sm89, sm120 |
| fp8 | Q/DQ features and matrix with `kFP8` and explicit scales | sm89, sm120 |

Do not build the fp8 graph on sm86. The graph keeps fp32 input and output buffers while its internal tensor types select arithmetic. TensorRT may retain higher precision operations such as bias addition. Lower precision alone does not establish a speed improvement.

## Prepare weights and scales

Use an external quantization tool to determine scales from representative data or quantization-aware training. The model package translates those results into safetensors and its network. tensoracle does not calculate scales, calibrate models, or translate a quantizer's export format.

The example network computes `features @ matrix + bias`. Its safetensors file contains the following tensors:

| Key | Storage type | Shape |
| --- | --- | --- |
| `matrix` | float32 | `(32, 16)` |
| `bias` | float32 | `(1, 16)` |
| `int8.input_scale`, `int8.weight_scale` | positive float32 | `(1,)` each |
| `fp8.input_scale`, `fp8.weight_scale` | positive float32 | `(1,)` each |

Store weights and scales in fp32 for this example, then cast or quantize in the TensorRT network. This avoids needing a NumPy representation of bf16 or fp8. tensoracle's C++ weight view also supports those safetensors storage types when the model's network handles them directly. Weight dtype, dimensions, and interpretation remain the model package's responsibility.

## Define the C++ network

This excerpt follows the graph in `cpp/tests/builder_fixture.hpp`, used by `cpp/tests/builder.cpp`. It is network-building code for a model package; the surrounding application must supply the safetensors file, current CUDA context, buffers, and error handling. It is not a standalone inference program.

```cpp
#include <tensoracle/builder.hpp>
#include <NvInfer.h>
#include <algorithm>
#include <stdexcept>
#include <string>

void define_linear(nvinfer1::INetworkDefinition& network,
                   const tensoracle::weights_view& weights,
                   const std::string& precision) {
    auto constant = [&](const char* name) {
        auto value = weights.at(name);
        if (value.dtype != TENSORACLE_FLOAT32)
            throw std::runtime_error("this network requires float32 weights");
        nvinfer1::Dims shape{};
        shape.nbDims = static_cast<int32_t>(value.rank);
        std::copy_n(value.dimensions, value.rank, shape.d);
        return network.addConstant(
            shape, {nvinfer1::DataType::kFLOAT, value.data,
                    static_cast<int64_t>(value.element_count)})->getOutput(0);
    };
    auto* features = network.addInput("features", nvinfer1::DataType::kFLOAT,
                                      nvinfer1::Dims2(16, 32));
    auto* matrix = constant("matrix");
    auto* bias = constant("bias");
    if (precision == "fp16" || precision == "bf16") {
        auto type = precision == "fp16" ? nvinfer1::DataType::kHALF
                                         : nvinfer1::DataType::kBF16;
        features = network.addCast(*features, type)->getOutput(0);
        matrix = network.addCast(*matrix, type)->getOutput(0);
        bias = network.addCast(*bias, type)->getOutput(0);
    } else if (precision == "int8" || precision == "fp8") {
        auto type = precision == "int8" ? nvinfer1::DataType::kINT8
                                        : nvinfer1::DataType::kFP8;
        auto qdq = [&](nvinfer1::ITensor* value, const std::string& field) {
            auto* scale = constant((precision + "." + field + "_scale").c_str());
            auto* quantized = network.addQuantize(*value, *scale, type)->getOutput(0);
            return network.addDequantize(*quantized, *scale,
                                         nvinfer1::DataType::kFLOAT)->getOutput(0);
        };
        features = qdq(features, "input");
        matrix = qdq(matrix, "weight");
    }
    auto* result = network.addMatrixMultiply(
        *features, nvinfer1::MatrixOperation::kNONE,
        *matrix, nvinfer1::MatrixOperation::kNONE)->getOutput(0);
    result = network.addElementWise(*result, *bias,
                                    nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    if (precision == "fp16" || precision == "bf16")
        result = network.addCast(*result, nvinfer1::DataType::kFLOAT)->getOutput(0);
    result->setName("result");
    network.markOutput(*result);
}
```

The fp16/bf16 branch casts operands before arithmetic and returns fp32. The int8/fp8 branch applies per-tensor Q/DQ to features and matrix; bias stays fp32. Keeping the matrix in fp32 lets TensorRT quantize it during building with the declared scale. The callback borrows weight storage that tensoracle keeps alive through serialization.

## Build, save, and reuse

Use a distinct engine name or graph version whenever tensor types or the network recipe change. A shared weight file cannot distinguish precision by its digest alone; scale changes within that file do change its digest.

With `define_linear` above, the model package opens a chosen precision from C++:

```cpp
tensoracle::engine_handle open_linear(const std::string& path,
                                      const std::string& precision) {
    tensoracle::builder_options options;
    options.name = "linear-" + precision;
    options.graph_version = "1";
    options.weights = {path};
    options.inputs = {{"features", {16, 32}, {16, 32}, {16, 32}}};
    options.io = {{"features", TENSORACLE_FLOAT32, {16, 32}, true},
                  {"result", TENSORACLE_FLOAT32, {16, 16}, false}};
    options.workspace_limit = 64 * 1024 * 1024;
    options.optimization_level = 0;
    return tensoracle::open_engine(options,
        [&](nvinfer1::INetworkDefinition& network,
            const tensoracle::weights_view& weights) {
            define_linear(network, weights, precision);
        });
}
```

Call with `"fp32"` for the reference and `"fp16"`, `"bf16"`, `"int8"`, or `"fp8"` for a candidate on the appropriate GPU. These options and network operations follow the public-builder tests. They do not set application quality limits.

A repeat call with unchanged conditions reuses the saved engine and skips the callback, including in a new process. Saved engines live under `XDG_CACHE_HOME/tensoracle`, or `~/.cache/tensoracle` when unset. Follow [Execute an engine](index.md#execute-an-engine) to create an execution unit and supply fp32 input/output device buffers. This network uses input `features` with shape `(16, 32)` and output `result` with shape `(16, 16)`. The model package exposes any Python inference entry through its own binding.

For ranged batches, change the network's leading input dimension and the corresponding I/O dimensions to `-1`, give the input profile positive minimum/optimum/maximum bounds, and set the concrete shape before enqueue. `cpp/tests/builder_fixture.hpp` includes that variant.

## Compare outputs and choose limits

First establish the fp32 TensorRT engine against fixed fp32 outputs from the official implementation, as described in [stage 1](optimization.md#1-establish-the-fp32-reference). Run the candidate and accepted fp32 engine with identical inputs. Pass their CPU NumPy outputs to the retained Python utility:

```python
from tensoracle import compare_outputs

# candidate_output and fp32_output come from the model package's inference binding.
comparison = compare_outputs(candidate_output, fp32_output, atol=0.01, rtol=0)
accepted = (
    comparison.candidate_nonfinite_count == 0
    and comparison.reference_nonfinite_count == 0
    and comparison.outside_tolerance_count == 0
    and comparison.relative_l2_error is not None
    and comparison.relative_l2_error < 0.01
)
```

These limits illustrate an application decision; they are not tensoracle defaults or accuracy guarantees. Choose representative data, per-output limits, and model-specific measurements for the real model. Validate after process restarts as well as after the first build, and measure performance separately. The C++ tests contain separate cases for fp32/fp16/bf16/int8 and for fp8; the fp8 case requires sm89 or sm120. This guide does not claim every precision was executed on every target GPU.
