# Migrate from tensoracle 1.x

tensoracle v2 moves engine construction, saving and reuse, execution, and plugin registration into a C++ library. A model package writes the network and engine-opening code in C++ and exposes its own inference function to Python, for example with nanobind. Update the model package before updating applications that use it.

## Check the platform first

v2 supplies only a Linux x86_64 `manylinux_2_28` wheel with the `cp312-abi3` tag. It includes the C++ shared library, public headers, and CMake package configuration. The package currently declares Python 3.12 and 3.13 support. TensorRT 11 and sm86, sm89, and sm120 remain the targets; fp8 requires sm89 or sm120.

Windows x86_64 users must stay on tensoracle 1.x until a later v2 minor release adds a Windows wheel. Use `tensoracle<2` in a Windows application's dependency declaration during that period. WSL2 can use the Linux wheel inside its Linux Python environment.

## Replace the Python engine API

The following Python utilities remain:

| Python entry | Purpose |
| --- | --- |
| `tensoracle.OutputComparison`, `tensoracle.compare_outputs` | Compare CPU NumPy outputs using application-chosen limits |
| `tensoracle.producer.KernelSpec`, `compile_bundle` | Compile CUDA C++ kernels through the C++ producer |
| `tensoracle.bundle` | Inspect cubin bundles and model wheels |
| `tensoracle cache list`, `tensoracle cache prune` | List and maintain saved engines through C++ cache operations |
| `tensoracle.convert.pth_to_safetensors`, `python -m tensoracle.convert` | Convert checkpoints with the optional `convert` extra |

The engine-related Python names below are removed. These are migration mappings, not v2 Python imports:

| Removed 1.x Python names | v2 model-package code |
| --- | --- |
| `open_engine`, `build_serialized_engine`, `EngineBuild`, `BuilderSettings`, `InputShape` | `tensoracle::builder_options` and `tensoracle::open_engine` in `tensoracle/builder.hpp` |
| `EngineIO`, `TensorSpec` | `tensoracle::tensor_spec` entries in `builder_options.io` |
| `EngineHandle`, `EngineExecution` | `tensoracle::engine_handle`, `tensoracle::engine_execution` in `tensoracle/engine.hpp` |
| `EngineCacheError`, `CorruptEngineError`, `EngineConditionError`, `EngineIOError` | `tensoracle::error` with C status values, including the corrupt-engine, condition-mismatch, and I/O-mismatch statuses |
| `PluginLibraryError` and Python plugin loading | `builder_options.plugins` or `tensoracle::register_libraries` in `tensoracle/plugins.hpp`; errors use `TENSORACLE_PLUGIN_LIBRARY_ERROR` |
| `ShapeTensorOutputWarning` | Shape inference outputs are rejected with an error before serialization or execution-context creation |

Python TensorRT network callbacks and capsules carrying network builders are not accepted. Move the complete operation that opens an engine into the model package's C++ code. The [C++ guide](index.md#build-and-reopen-an-engine) shows the network callback, weight access, options, and saved-engine handling.

## Build the model package's inference binding

Use [examples/wheel_consumer](https://github.com/sync-dev-org/tensoracle/tree/main/examples/wheel_consumer) as a complete separate-wheel example:

1. Locate `tensoracle/cpp` in the installed wheel, add that prefix to `CMAKE_PREFIX_PATH`, and use `find_package(tensoracle 2 CONFIG REQUIRED)`.
2. Link the model extension to `tensoracle::tensoracle`. Supply TensorRT SDK headers for the network-building code and CUDA headers and driver linking for the caller's CUDA operations. The example uses CMake 3.26 or later and C++17.
3. Define the TensorRT network in C++ using the callback's `tensoracle::weights_view`. Declare safetensors files in `builder_options.weights`; tensoracle keeps their values alive through engine serialization.
4. Open with `tensoracle::open_engine`, create an execution unit, supply device buffers and a CUDA stream, and wait before releasing those resources. Expose the model's inference function through nanobind or your chosen binding.
5. Package the model extension separately from tensoracle. Follow the example's relative RUNPATH setup so the extension finds `libtensoracle` from the installed tensoracle wheel. Do not copy tensoracle, TensorRT, or driver libraries into the model wheel.

The example's Python entry is `tensoracle_consumer.infer(weights_path, input_array)`. Its C++ implementation owns the network, engine open, CUDA context, buffers, stream, and execution. The example README supplies build and installation commands and a separate-process reuse check. Importing it does not initialize CUDA. Applications call this model-specific inference entry instead of constructing engines through tensoracle's Python API.

Reapply your quality checks after moving the network. Preserve its tensor types, input profiles, names, and output semantics. TF32 remains disabled. For CUDA Graphs, the model package must own capture and replay and follow the [warm-up and resource-lifetime rules](index.md#capture-and-replay-with-cuda-graphs).

## Rewrite kernels in CUDA C++

v2's producer accepts CUDA C++ source with an `extern "C" __global__` entry, not Triton functions. Rewrite the kernel and supply its `entry_symbol`, ordered argument `signature`, compile constants, and any divisibility constraints in `KernelSpec`. Hidden Triton parameters and scratch arguments are not part of the v2 launch ABI.

Regenerate the cubins, manifest, and embedding source with `tensoracle.producer.compile_bundle` or the C++ producer. Install `tensoracle[producer]` for the PyPI CUDA 13.2 NVRTC, ptxas, and cuobjdump tools; no GPU or OS-installed CUDA Toolkit is needed for generation. Build the native plugin against the regenerated bundle, export `getCreators`, and include its path in `builder_options.plugins`. Use `inspect_bundle` and `inspect_wheel` to verify the resulting artifacts, then rerun numerical and inference tests. See [the producer API](api/producer.md) and [plugin build guide](cpp-python-repositories.md).

## Rebuild saved engines

The first v2 open builds a new engine rather than reusing a TensorRT engine saved by 1.x. Keep the network-building callback, original safetensors, and required plugin libraries available. Allow build time and disk space for the new engine files. v2's compatibility guarantee applies within the v2 major version and does not cover 1.x files.

The cache directory is still `XDG_CACHE_HOME/tensoracle`, or `~/.cache/tensoracle` when unset. v2's `tensoracle cache list` reports 1.x files as `unrecognized`. `tensoracle cache prune` keeps them, including when pruning by maximum size. Those files count toward the total, so a size target may remain unattainable until you remove obsolete files yourself during maintenance. Keep files needed by applications still using 1.x.

Preview cleanup with `--dry-run` and run it only when no engines are being built or opened in that directory. The [cache guide](index.md#manage-saved-engines) explains recorded times, exclusions, and failures.
