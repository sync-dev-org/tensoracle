# Optimize a model in four stages

Use the same representative inputs, shapes, and corresponding outputs throughout this workflow. The model package defines the graph and runs both its reference implementation and the TensorRT engines. `tensoracle` builds engines and compares output arrays; the application executes the engines and decides which measurements are acceptable.

## 1. Establish the fp32 reference

- **tensoracle tools:** Build an fp32 TensorRT engine with `EngineBuild(precision="fp32")` and `build_serialized_engine` or `open_engine`. TF32 is disabled for the engine. Run the engine in the application, then use `compare_outputs(candidate, reference, atol=..., rtol=...)` on corresponding CPU NumPy output arrays.
- **You provide:** Run the official PyTorch implementation in fp32 outside `tensoracle` and fix its outputs as the reference. Supply the TensorRT graph, weights, representative inputs, and an acceptance limit for each output. Keep the reference and candidate inputs and preprocessing identical.
- **Gate:** Reject non-finite outputs and any output that misses your limits. Advance only when every fp32 TensorRT output passes against its fixed PyTorch reference. Correct the graph or inputs before trying lower precision if this gate fails.

## 2. Accept lower precision by quality

- **tensoracle tools:** Build an fp16 TensorRT engine with `EngineBuild(precision="fp16")`. Use `compare_outputs` with each fp16 output as the candidate and its accepted fp32 TensorRT output as the reference. It reports non-finite counts and, for finite floating point arrays, maximum absolute error, relative L2 error, cosine similarity, and, when both tolerances are supplied, counts and fractions outside tolerance.
- **You provide:** Choose limits for those model-independent measurements. Compute any model-specific measurements separately, such as PSNR for images or application-specific time-series quality measures, and choose their limits. Run both engines on the same inputs.
- **Gate:** Accept the fp16 engine only when its outputs meet all applicable limits, including the model-specific ones. Otherwise keep the accepted fp32 engine or revise the candidate and repeat this stage.

## 3. Replace a remaining bottleneck with a plugin

First check whether a combination of standard TensorRT layers can preserve the operation's meaning, shapes, control flow, and output quality. For example, a deformable convolution may be expressible through grid sampling, mask multiplication, and convolution. If a plugin bundled with TensorRT matches the required behavior, compare that option too. Use a custom cubin plugin only when the suitable alternatives still leave a bottleneck.

- **tensoracle tools:** `tensoracle.producer.compile_bundle` generates cubins and a manifest for supported GPU generations. The shared C++ plugin helpers support a caller-built native plugin, and `tensoracle.bundle.inspect_bundle` or `inspect_wheel` checks the cubins and their match with the plugin. Pass the plugin library through `plugins=` to `build_serialized_engine` or `open_engine`. Follow [Build a cubin plugin](https://github.com/sync-dev-org/tensoracle#build-a-cubin-plugin) and the [cubin plugin example](https://github.com/sync-dev-org/tensoracle/blob/main/examples/cubin_plugin/README.md) for the generation, build, and inspection operations.
- **You provide:** Identify the bottleneck, write the kernel and model-specific plugin behavior, choose its supported shapes and types, build and package its native library, and run the resulting engines. An fp32 kernel must not use TF32.
- **Gate:** For a custom cubin plugin, the bundle and plugin inspections must pass. Reapply stage 1 to the replacement's fp32 engine against the same fixed PyTorch reference. If using fp16, reapply stage 2 against an accepted fp32 engine with the same limits and model-specific measurements. Advance only when the replacement meets those quality gates; otherwise retain the accepted implementation.

## 4. Measure speed under isolated load

- **tensoracle tools:** None for timing. `tensoracle` does not benchmark engine execution.
- **You provide:** A timing harness, representative inputs, and a speed target for the application. Measure with no load from other processing, using comparable conditions for the accepted baseline and candidate.
- **Gate:** Use the candidate only if it meets your speed target and still passes the quality gates above. Do not loosen quality limits to meet a speed target. If it misses the target, keep the accepted implementation and investigate the remaining bottleneck before trying another candidate.
