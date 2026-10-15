# Cubin plugin example

This project is the caller side of tensoracle's cubin workflow. `kernels.py` supplies five Triton entries for each of sm86, sm89, and sm120. `bundle/` contains their checked cubins, manifest, and generated C++ source. `plugin.cpp` builds two TensorRT `IPluginV3` creators with tensoracle's C++ helpers. The Python package defines the graph and includes the native library and cubins in its wheel.

The first plugin computes an fp32 32 by 16 dot product and then scales it in a second launch. Its fields mean matrix weights and a scalar multiplier. The second plugin adds a 16 element bias; its field means an additive offset. Both serialize their fields and copy parameters to the device. The input row count ranges from 1 to 32. The producer records the pointer alignment and row divisibility conditions in the generated C++ entry. The plugin asks `Kernel::accepts` whether the specialized entry accepts the current arguments, then uses the unspecialized entry when it does not. The common launcher rejects any argument count or specialization mismatch. The `dot.tf32-control` cubin is a numerical control, while the plugin uses `input_precision="ieee"`.

## Rebuild the cubins

From the tensoracle repository root, run:

```bash
uv run --project python --extra producer python -c 'from pathlib import Path; from examples.cubin_plugin.kernels import kernel_specs; from tensoracle.producer import compile_bundle; compile_bundle(kernel_specs(), Path("examples/cubin_plugin/bundle"))'
```

This installs `tensoracle[producer]` with Triton 3.8.0, ptxas 12.9.86, and cuobjdump 13.3, and recreates `bundle/` without using a GPU. `compile_bundle` finds ptxas and cuobjdump in their installed NVIDIA wheels. The producer checks the cubin ELF SM, public and hidden argument layout, register count, and zero scratch sizes before writing the manifest. Run the tests below with the producer extra alone so torch is absent; `compile_bundle` rejects an already imported torch.

Check that the regenerated files match the committed example without using a GPU or setting producer environment variables:

```bash
uv run --project python --extra producer pytest -m "not gpu" python/tests/test_producer.py
```

Without the producer extra, the compilation tests skip with an explanation of the missing tools.

## Build and inspect the wheel

Follow [Prepare a TensorRT SDK](../../docs/tensorrt-sdk.md) to obtain and verify the headers and library, then set `TENSORRT_ROOT` to that TensorRT 11 SDK containing `include/NvInfer.h` and a `lib` directory. The procedure supplies fixed inputs for SDK 11.2.1.2 and explains how to verify other TensorRT 11 releases. The example has been built against SDK 11.0.0.114, 11.1.0.106, 11.2.1.2, 11.3.0.99, and 11.4.0.106 and run on sm86, sm89, and sm120; see the [tested environments](../../README.md#tested-environments). `TENSORACLE_CPP_ROOT` may point to the `cpp` directory inside an installed tensoracle wheel; when building from this repository it defaults to `../../cpp`.

```bash
CMAKE_ARGS="-DTENSORRT_ROOT=$TENSORRT_ROOT" uv build examples/cubin_plugin --wheel
```

The example's build backend and CUDA tooling belong to this project, not to tensoracle's runtime dependencies. Before installation, inspect the wheel with `tensoracle.bundle.inspect_wheel(wheel, plugin="example_cubin_plugin/libtensoracle_example_plugin.so")`. This compares the library's embedded digest and cubin bytes with the manifest and cubin files.

Install the tensoracle wheel and this example wheel into a runtime Python environment with TensorRT 11 and `cuda-python`, but without Triton, torch, or the TensorRT SDK. `open_engine` registers the library before calling `graph` or opening a saved engine. The tests use two separate processes and an `XDG_CACHE_HOME` directory to verify that the second process reuses the saved engine without building a graph.

## Test on each GPU generation

Run from the tensoracle repository with the SDK and CUDA toolkit paths set. The GPU tests build the example wheel, install it in a temporary runtime environment, and run dynamic shapes, direct builds, saved engine reuse, simultaneous execution in two distinct CUDA contexts, and an fp32 versus TF32 control. Each GPU subprocess has a 50 second limit.

```bash
TENSORACLE_SDK_ROOT="$TENSORRT_ROOT" uv run --project python pytest -q python/tests/test_example_plugin.py
```

Record the GPU name, SM, TensorRT runtime version, test result, and wheel digest on sm86, sm89, and sm120 separately. The code and wheel contain all three cubin generations; successful execution on one generation does not establish the other two.
