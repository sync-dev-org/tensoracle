# tensoracle

Build, save, reuse, and execute TensorRT engines through a C++ library embedded in your model package. Define the TensorRT network and engine-opening code in C++, then expose inference through your own binding, such as nanobind. Python utilities provide output comparison, cubin production and inspection, cache management commands, and `.pth` checkpoint conversion.

Engine building, reuse, execution, output comparison, and cubin production do not require torch. Only the optional converter requires `tensoracle[convert]`.

Read the [v2 user guide](https://sync-dev-org.github.io/tensoracle/) and [Migrate from 1.x](https://sync-dev-org.github.io/tensoracle/migration/).

## Terms and workflows

- A **TensorRT engine** is TensorRT's executable form of a model, optimized for a GPU, precision, and TensorRT version. It can be held in memory or saved to disk.
- **Building an engine** means creating that executable form from the model package's C++ network-building function and safetensors weights.
- A **cubin plugin** is a native TensorRT plugin whose GPU kernels are supplied as compiled CUDA binary (`.cubin`) files. The model package owns its CUDA C++ kernels and plugin behavior.

| Role | Use | Additional requirements |
| --- | --- | --- |
| Inference application | Call the model package's inference binding. | NVIDIA GPU and driver; the model package manages CUDA buffers and streams. |
| Model package developer | Open and execute engines through tensoracle's C++ SDK, exposing inference through your own binding. | TensorRT 11.2 or later (below 12) headers, CMake 3.26 or later, C++17 compiler, and a binding framework when needed. |
| Cubin plugin developer | Compile CUDA C++ kernels and package a native plugin with the model. | `tensoracle[producer]`, PyPI CUDA headers and tools, TensorRT SDK, CMake, and a C++17 compiler. |

v2 supplies a Linux x86_64 `manylinux_2_28`, `cp312-abi3` wheel, including WSL2. The package currently declares Python 3.12 and 3.13 support. Target GPUs are sm86, sm89, and sm120 with TensorRT 11.2 or later (below 12); fp8 requires sm89 or sm120. Windows x86_64 users must keep using tensoracle 1.x until a later v2 minor release adds a Windows wheel.

The wheel contains the C++ public headers, `libtensoracle`, CMake package configuration, and cubin plugin helper sources. TensorRT is resolved through a wheel dependency; the NVIDIA driver is not bundled. Public C++ interfaces and saved engine files remain compatible within a major version.

## Build and reopen an engine in C++

Include `tensoracle/builder.hpp` and link `tensoracle::tensoracle` from `find_package(tensoracle 2 CONFIG REQUIRED)`. Locate the wheel's SDK with:

```bash
python -c 'from importlib.metadata import distribution; print(distribution("tensoracle").locate_file("tensoracle/cpp"))'
```

Pass that directory in `CMAKE_PREFIX_PATH`. The model package supplies a callback taking `nvinfer1::INetworkDefinition&` and `const tensoracle::weights_view&`. tensoracle reads declared safetensors and keeps their values alive through the build.

This excerpt follows the [separate consumer wheel](https://github.com/sync-dev-org/tensoracle/tree/main/examples/wheel_consumer). It assumes the example's `build_network` function, four-element float32 `bias.safetensors`, and caller-owned current CUDA context. The [user guide](https://sync-dev-org.github.io/tensoracle/#build-and-reopen-an-engine) shows the callback; the complete example supplies allocation, error handling, and nanobind integration.

```cpp
#include <tensoracle/builder.hpp>

tensoracle::builder_options options;
options.name = "wheel-consumer-add";
options.graph_version = "1";
options.weights = {"bias.safetensors"};
options.inputs = {{"input", {4}, {4}, {4}}};
options.io = {{"input", TENSORACLE_FLOAT32, {4}, true},
              {"output", TENSORACLE_FLOAT32, {4}, false}};
options.workspace_limit = 64 * 1024 * 1024;
options.optimization_level = 0;
auto engine = tensoracle::open_engine(options, build_network);
```

On a valid cache hit, `tensoracle::open_engine` skips the callback, including after a process restart. Engine identity includes weights, plugin bytes, name, graph version, input profiles, builder optimization level, GPU compatibility properties, OS, CPU architecture, and TensorRT version. Update the graph version when network code changes; use separate names or versions for different precisions. Tensor types in the network determine precision, and TF32 is disabled.

Leave `workspace_limit` unset to use TensorRT's default or supply a positive byte count. It does not enter the cache identity; set `options.force_rebuild = true` to rebuild with a changed limit. Saved files live under `XDG_CACHE_HOME/tensoracle`, or `~/.cache/tensoracle` when unset. The [user guide](https://sync-dev-org.github.io/tensoracle/#save-and-reuse-an-engine) describes integrity checks, errors, and weight digest reuse.

## Execute through the model package

The C++ model package calls `engine.create_execution()`, sets ranged input shapes together, queries output shapes and types, and enqueues every I/O device address on its own CUDA stream. Each execution unit retains the engine and owns an independent TensorRT context. The caller owns allocation, transfers, stream ordering, and completion checks. tensoracle does not synchronize enqueue or release; wait before changing shapes, freeing buffers, or closing the unit.

The [execution guide](https://sync-dev-org.github.io/tensoracle/#execute-an-engine) follows the consumer wheel's C++ API usage. The model package exposes its inference function to Python. Python applications do not construct or execute engines through tensoracle's Python utilities.

For CUDA Graphs, warm up one enqueue after each shape change before capturing on a nondefault stream. The graph fixes shapes, addresses, and execution unit; keep resources alive until work completes and the graph is destroyed. The caller creates and recreates graphs. Engines with loops, conditionals, or host-synchronizing layers or plugins cannot be captured. See the [capture guide](https://sync-dev-org.github.io/tensoracle/#capture-and-replay-with-cuda-graphs).

## Manage saved engines

```bash
tensoracle cache list
tensoracle cache prune --unused-for 30 --dry-run
tensoracle cache prune --unused-for 30
tensoracle cache prune --max-size 10GiB --dry-run
```

These commands use the same cache directory as the C++ engine opener. They inspect headers and metadata without a GPU, original weights, or plan deserialization. Listing reports conditions, size, recorded time, workspace limit, and known weight paths. Empty files are `building or leftover`; invalid headers, including 1.x engine files, are `unrecognized`. Pruning never deletes either category.

An engine file's modification time records its initial save or last successful open. It does not indicate whether inference continues. Failed loads and management commands do not change it. Timestamp write failures do not prevent a successful load; records that could not be written, or predate timestamp recording, may be stale.

Pruning requires exactly one of `--unused-for DAYS` and `--max-size SIZE`. It prints a plan and rebuild notice before deletion and separates removed, excluded, missing, and failed files in its result. `--dry-run` changes nothing. Run pruning during maintenance without concurrent builds or opens. The [management guide](https://sync-dev-org.github.io/tensoracle/#manage-saved-engines) describes sizes, selection, and exit statuses.

## Build a cubin plugin

Install `tensoracle[producer]` to generate sm86, sm89, and sm120 cubins from CUDA C++ source. The C++ producer uses NVRTC, `ptxas >=13.2,<14`, and `cuobjdump` from PyPI CUDA 13.2 wheels, without a GPU or OS-installed CUDA Toolkit. Triton kernels are not accepted. See the [producer API](https://sync-dev-org.github.io/tensoracle/api/producer/) for `KernelSpec` and `compile_bundle`.

Build the native plugin with [the wheel's C++ sources and CMake helpers](https://sync-dev-org.github.io/tensoracle/cpp-python-repositories/) and a separately prepared [TensorRT SDK](https://sync-dev-org.github.io/tensoracle/tensorrt-sdk/). Export TensorRT's `getCreators` and list all required libraries in `tensoracle::builder_options::plugins`. tensoracle inspects and registers them before construction or deserialization and hashes their bytes into the engine identity. Explicit registration is also available through `tensoracle/plugins.hpp`.

Package the plugin library and cubin bundle in your model wheel. `tensoracle.bundle.inspect_bundle` and `inspect_wheel` verify the manifest, cubin digests and actual architectures, declared argument ABI, and the plugin's embedded cubins without loading it. The manifest records NVRTC and ptxas versions and kernel source digests. The [plugin guide](https://sync-dev-org.github.io/tensoracle/#build-a-cubin-plugin) describes tensor types, formats, and the common lifecycle helper.

Follow the [four-stage optimization workflow](https://sync-dev-org.github.io/tensoracle/optimization/) and [precision guide](https://sync-dev-org.github.io/tensoracle/precision/) when validating a model or plugin.

## Convert a PyTorch checkpoint

Install the conversion extra in the environment where you prepare weights:

```bash
pip install 'tensoracle[convert]'
python -m tensoracle.convert model.pth model.safetensors
```

Build scripts can call the same converter:

```python
from tensoracle.convert import pth_to_safetensors

pth_to_safetensors("model.pth", "model.safetensors")
```

The input must be a mapping with string keys whose values are tensors or nested mappings. Nested keys are joined with `.`: `{"state_dict": {"layer.weight": tensor}}` becomes `state_dict.layer.weight`. The converter preserves each tensor's dtype, shape, and values, including shared or noncontiguous dense tensors. It rejects non-tensor metadata and duplicate flattened names before writing the destination. Weight renaming and model-specific preprocessing belong to the model package.

The converter loads weights on the CPU with `torch.load(..., weights_only=True)` and rejects checkpoint objects instead of falling back to unrestricted pickle loading. The extra requires torch 2.6 or later, which includes the fix for [CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6). Torch is imported only when conversion is called; it is absent from the package's required dependencies and is not imported by engine building or reuse. Calling the converter without torch raises `ImportError` with the installation command for `tensoracle[convert]`.

## Compare output arrays

Run the TensorRT engines in your application, then pass one CPU NumPy output array and its corresponding reference array to `compare_outputs`. The first argument is the candidate; the second is the reference. Both arrays must have the same nonempty shape and be floating point, integer, or boolean arrays of the same kind. The function does not run an engine or decide whether its output passes a quality limit.

For floating point arrays, `OutputComparison` contains the non-finite count for each side, the maximum absolute error, the relative L2 error (difference norm divided by reference norm), and cosine similarity. Supply both `atol` and `rtol` to also get the count and fraction of elements outside `abs(candidate - reference) <= atol + rtol * abs(reference)`. Choose these limits for each output in your application. Floating point calculations use float64, including comparisons between fp16 and fp32 arrays. A non-finite value on either side makes all floating point measurements `None`. A zero reference norm makes relative L2 error `None`; a zero norm on either side makes cosine similarity `None`.

First, compare a saved correct output with the output from an fp32 TensorRT engine. The application creates and fixes the correct output outside tensoracle and runs the engine itself:

```python
import numpy as np

from tensoracle import compare_outputs

correct_output = np.array([1.0, 2.0], dtype=np.float32)  # Fixed reference output
fp32_output = np.array([1.0001, 2.0002], dtype=np.float32)  # Engine output on CPU
fp32_result = compare_outputs(fp32_output, correct_output, atol=0.001, rtol=0.0)
fp32_within_tolerance = (
    fp32_result.candidate_nonfinite_count == 0
    and fp32_result.reference_nonfinite_count == 0
    and fp32_result.outside_tolerance_count == 0
)
```

Then use the fp32 TensorRT engine output as the reference for an fp16 TensorRT engine output. The application chooses limits for whole-array closeness or direction as well as any elementwise tolerance:

```python
fp16_output = np.array([1.0, 2.0], dtype=np.float16)  # Engine output on CPU
fp16_result = compare_outputs(fp16_output, fp32_output, atol=0.001, rtol=0.001)
fp16_accepted = (
    fp16_result.candidate_nonfinite_count == 0
    and fp16_result.reference_nonfinite_count == 0
    and fp16_result.outside_tolerance_count == 0
    and fp16_result.relative_l2_error is not None
    and fp16_result.relative_l2_error <= 0.01  # Application's limit
)
```

When tolerances are omitted, `outside_tolerance_count` and `outside_tolerance_fraction` are `None`. For integer and boolean arrays, use `mismatch_count` and `mismatch_fraction`; these measure exact equality instead of floating point distances. A zero mismatch count means those arrays agree. Shape, kind, or empty-array errors raise `ValueError`.
