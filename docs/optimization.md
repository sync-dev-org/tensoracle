# Optimize a model in four stages

Use the same representative inputs, shapes, and corresponding outputs throughout this workflow. The model package defines its TensorRT network and opens and executes engines through tensoracle's C++ API, exposing inference through its own binding when Python is needed. The application runs the reference implementation, compares output arrays, and decides which measurements are acceptable.

## 1. Establish the fp32 reference

- **tensoracle tools:** Build an fp32 TensorRT engine from a C++ network whose tensor types are fp32, using `tensoracle::builder_options` for input profiles and settings and `tensoracle::open_engine` for construction and reuse. TF32 is disabled. Run the engine through its C++ execution unit, then use Python's `compare_outputs(candidate, reference, atol=..., rtol=...)` or C++'s `tensoracle::compare_outputs` on corresponding CPU outputs.
- **You provide:** Run the official PyTorch implementation in fp32 outside `tensoracle` and fix its outputs as the reference. Supply the TensorRT graph, weights, representative inputs, and an acceptance limit for each output. Keep the reference and candidate inputs and preprocessing identical.
- **Gate:** Reject non-finite outputs and any output that misses your limits. Advance only when every fp32 TensorRT output passes against its fixed PyTorch reference. Correct the graph or inputs before trying lower precision if this gate fails.

## 2. Accept lower precision by quality

- **tensoracle tools:** Build an fp16, bf16, int8, or fp8 engine using C++ input profiles and builder options. The network callback determines precision through tensor types. For int8 and fp8, place explicit quantize/dequantize layers and read scales from the supplied safetensors weight view; prepare scales outside tensoracle. Follow [Build bf16, int8, and fp8 engines](precision.md) for C++ excerpts and GPU targets (fp8 requires sm89 or sm120). With `tensoracle::open_engine`, use a different `name` or `graph_version` from the fp32 engine. Compare each lower precision output against its accepted fp32 TensorRT output. Python's `compare_outputs` reports non-finite counts, maximum absolute error, relative L2 error, cosine similarity, and, when both tolerances are supplied, counts and fractions outside tolerance.
- **You provide:** Choose limits for those model-independent measurements. Compute any model-specific measurements separately, such as PSNR for images or application-specific time-series quality measures, and choose their limits. Run both engines on the same inputs.
- **Gate:** Accept the lower precision engine only when its outputs meet all applicable limits, including the model-specific ones. Otherwise keep the accepted fp32 engine or revise the candidate and repeat this stage.

## 3. Replace a remaining bottleneck with a plugin

First check whether a combination of standard TensorRT layers can preserve the operation's meaning, shapes, control flow, and output quality. For example, a deformable convolution may be expressible through grid sampling, mask multiplication, and convolution. If a plugin bundled with TensorRT matches the required behavior, compare that option too. Use a custom cubin plugin only when the suitable alternatives still leave a bottleneck.

- **tensoracle tools:** `tensoracle.producer.compile_bundle` wraps the C++ producer to generate cubins and a manifest from CUDA C++ kernels for sm86, sm89, and sm120. Triton kernels are not accepted. The common C++ plugin helper supports declared tensor types, formats, and an input whose shape the output copies. `tensoracle.bundle.inspect_bundle` or `inspect_wheel` checks the cubins and their match with the plugin. Put the plugin library paths in C++ `tensoracle::builder_options::plugins` before calling `tensoracle::open_engine`. Follow [Build a cubin plugin](index.md#build-a-cubin-plugin), [the producer API](api/producer.md), and [the CMake build guide](cpp-python-repositories.md) for generation, tensor declarations, native builds, and inspection.
- **You provide:** Identify the bottleneck, write the kernel and model-specific plugin behavior, choose its supported shapes and types, build and package its native library, and run the resulting engines. An fp32 kernel must not use TF32.
- **Gate:** For a custom cubin plugin, the bundle and plugin inspections must pass. Reapply stage 1 to the replacement's fp32 engine against the same fixed PyTorch reference. If using a lower precision engine, reapply stage 2 against an accepted fp32 engine with the same limits and model-specific measurements. Advance only when the replacement meets those quality gates; otherwise retain the accepted implementation.

## 4. Measure speed under isolated load

- **tensoracle tools:** None for timing. `tensoracle` does not benchmark engine execution.
- **You provide:** A timing harness, representative inputs, and a speed target for the application. Measure with no load from other processing, using comparable conditions for the accepted baseline and candidate.
- **Gate:** Use the candidate only if it meets your speed target and still passes the quality gates above. Do not loosen quality limits to meet a speed target. If it misses the target, keep the accepted implementation and investigate the remaining bottleneck before trying another candidate.
