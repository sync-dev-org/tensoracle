# tensoracle

`tensoracle` is a C++ library for building, saving, reusing, and executing TensorRT engines from caller-defined networks and safetensors weights. The model package writes its network and engine-opening code in C++ and exposes inference through its own binding, such as nanobind. tensoracle's Python API provides output comparison, cubin production and inspection, cache management commands, and checkpoint conversion.

Engine building, reuse, execution, output comparison, and cubin production do not require torch. Only the optional `.pth` converter uses torch, through `tensoracle[convert]`.

If your model package uses tensoracle 1.x, start with [Migrate from 1.x](migration.md). Follow the [four-stage optimization workflow](optimization.md) to establish output quality, evaluate lower precision and plugins, and measure speed.

## Terms

- A **TensorRT engine** is an executable form of a model optimized for a GPU, precision, and TensorRT version. It can be kept in memory or saved to disk.
- **Building an engine** means making that executable form from a model network. The model package supplies the C++ network-building function.
- A **cubin plugin** is a native TensorRT plugin that runs GPU kernels supplied as compiled CUDA binary (`.cubin`) files. The model package supplies the CUDA C++ kernels and plugin behavior.

## Choose a workflow

| Role | What to do | Build inputs and runtime responsibilities |
| --- | --- | --- |
| Inference application | Call the model package's inference binding. | Install the model package and tensoracle; provide an NVIDIA GPU and driver. The model package owns buffers and streams. |
| Model package developer | [Build and reopen an engine](#build-and-reopen-an-engine) in C++, then [execute it](#execute-an-engine). | tensoracle's C++ SDK, TensorRT 11 headers, CMake 3.26 or later, and a C++17 compiler. Add a binding framework when exposing inference to Python. |
| Cubin plugin developer | [Produce cubins and build a plugin](#build-a-cubin-plugin). | `tensoracle[producer]` for generation, TensorRT SDK headers and libraries, PyPI CUDA headers and tools, and the plugin's C++ build. |
| Output validation | Compare CPU NumPy outputs with [the comparison API](api/index.md#output-comparison). | The application runs the engines and chooses the quality limits. |

v2 supplies a Linux x86_64 `manylinux_2_28` wheel, including WSL2, with the `cp312-abi3` tag. The package currently declares Python 3.12 and 3.13 support. TensorRT 11 and GPU generations sm86, sm89, and sm120 are the targets; fp8 requires sm89 or sm120. Windows x86_64 users must use tensoracle 1.x until a later v2 minor release supplies a Windows wheel. A Linux wheel runs under WSL2, not directly in Windows Python.

The wheel includes the shared C++ library, public headers, CMake package configuration, and source-based cubin plugin helpers. It resolves TensorRT through its wheel dependency rather than bundling TensorRT or the NVIDIA driver. The C ABI uses opaque handles and structures with `struct_size`; the C++ headers add exceptions and automatic resource release. Public C++ interfaces and saved engine files remain compatible within the same major version.

### Connect a C++ model package

Locate the installed SDK without importing tensoracle:

```bash
python -c 'from importlib.metadata import distribution; print(distribution("tensoracle").locate_file("tensoracle/cpp"))'
```

Pass that directory in `CMAKE_PREFIX_PATH`. A consumer links the exported target:

```cmake
find_package(tensoracle 2 CONFIG REQUIRED)
target_link_libraries(my_model PRIVATE tensoracle::tensoracle)
```

`my_model` is your existing CMake target. Add TensorRT SDK headers to the target that defines the network; CUDA headers and driver linking are needed when the caller uses CUDA. The [separate consumer wheel](https://github.com/sync-dev-org/tensoracle/tree/main/examples/wheel_consumer) provides the complete nanobind project, including header discovery, CUDA helpers, and relative runtime library paths. Its `bindings.cpp` keeps all TensorRT construction and execution in C++; Python receives an inference function.

### Read weights without a GPU

Open a safetensors file from C++ before creating a TensorRT engine. This uses the same reader and validation as engine construction and requires neither a GPU nor CUDA initialization:

```cpp
#include <tensoracle/builder.hpp>
#include <iostream>

auto weights = tensoracle::open_weights("model.safetensors");
auto view = weights.view();
for (size_t i = 0; i < view.size(); ++i) {
    auto tensor = view.at(i);
    std::cout << tensor.name << ": " << tensor.element_count << " elements\n";
    // tensor.dtype, tensor.rank, tensor.dimensions, tensor.data and
    // tensor.byte_size describe the typed, shaped, contiguous CPU values.
}
for (size_t i = 0; i < weights.metadata_size(); ++i) {
    auto [key, value] = weights.metadata_at(i);
    std::cout << key << "=" << value << '\n';
}
```

Read a named tensor with `view.at("bias")` or a metadata value with `weights.metadata_at("format")`. Missing names and out-of-range indexes throw `tensoracle::error` with status `TENSORACLE_INVALID_ARGUMENT`. A file without `__metadata__` has zero metadata entries. Metadata keys and values are UTF-8 byte strings; the C++ operations return `std::string_view`, preserving embedded NUL bytes. Enumerate metadata by index to read keys containing NUL bytes; lookup by key accepts a NUL-terminated string.

`weights_handle` owns the opened file storage, supports moves, and releases it on destruction or `close()`. The tensor names, dimensions, values, metadata, and views remain valid until that storage is closed. Do not close it while using a borrowed view or pointer.

C callers include `tensoracle/builder.h`, call `tensoracle_weights_open`, and release the owned handle with `tensoracle_weights_destroy`. The existing `tensoracle_weights_count`, `tensoracle_weights_at`, and `tensoracle_weights_find` read its tensors. For metadata, use `tensoracle_weights_metadata_count`, `tensoracle_weights_metadata_at`, and `tensoracle_weights_metadata_find`. Initialize `tensoracle_weights_metadata_entry.struct_size` before reading an entry; its `key_size` and `value_size` are byte counts excluding the extra NUL terminators. These lengths preserve strings with embedded NUL bytes.

The metadata operations also work on the borrowed weights passed to a network callback. For multiple declared files, they return the union of metadata keys; the first declared file's value wins for repeated keys. Callback weights stay borrowed and must never be passed to `tensoracle_weights_destroy`.

### Build and reopen an engine

The following C++ excerpts follow `examples/wheel_consumer/bindings.cpp`. They are parts of the model package, not standalone programs; the complete example also supplies CUDA context selection, allocation, error handling, and binding code.

Define the network with TensorRT's C++ API and read the declared safetensors through `tensoracle::weights_view`:

```cpp
#include <tensoracle/builder.hpp>
#include <NvInfer.h>
#include <stdexcept>

void build_network(nvinfer1::INetworkDefinition& network,
                   const tensoracle::weights_view& weights) {
    auto bias = weights.at("bias");
    if (bias.dtype != TENSORACLE_FLOAT32 || bias.rank != 1 || bias.dimensions[0] != 4)
        throw std::invalid_argument("bias must be float32 with shape (4,)");
    nvinfer1::Dims shape{};
    shape.nbDims = 1;
    shape.d[0] = 4;
    auto* input = network.addInput("input", nvinfer1::DataType::kFLOAT, shape);
    auto* constant = network.addConstant(shape, {nvinfer1::DataType::kFLOAT, bias.data, 4});
    if (!input || !constant) throw std::runtime_error("cannot create network inputs");
    auto* sum = network.addElementWise(*input, *constant->getOutput(0),
                                     nvinfer1::ElementWiseOperation::kSUM);
    if (!sum) throw std::runtime_error("cannot create addition layer");
    auto* output = sum->getOutput(0);
    output->setName("output");
    network.markOutput(*output);
}
```

Prepare `bias.safetensors` outside the network callback, for example with Python:

```python
import numpy as np
from safetensors.numpy import save_file

save_file({"bias": np.array([0.5, -2.0, 3.25, -0.75], dtype=np.float32)}, "bias.safetensors")
```

With the caller's CUDA context current, open the engine from C++:

```cpp
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

Each input profile contains a name and minimum, optimum, and maximum dimensions. Equal dimensions select a fixed size. For ranged dimensions, put `-1` in the network and I/O declaration and supply positive ordered bounds. `options.io` lists all input and output names, types, and build-time shapes; its final boolean marks inputs. Optional `location` and `format` fields request additional TensorRT I/O checks.

The callback receives borrowed weights by name, type, shape, and value. tensoracle keeps their storage alive through serialization, including after the callback returns. Tensor names must be unique across the declared weight files. The callback returns `void` and reports failure with an exception; the wrapper translates it into a C ABI diagnostic.

Tensor types in the network determine precision. The builder has no precision or TF32 option: TF32 is disabled. See [bf16, int8, and fp8 engines](precision.md) for C++ casts and Q/DQ layers. `workspace_limit` is optional; leave it unset to use TensorRT's default or assign a positive byte count. `optimization_level` ranges from 0 to 5 and defaults to 3.

### Save and reuse an engine

`tensoracle::open_engine` builds and saves a missing engine and skips the network callback on a valid cache hit, including in another process. The caller still supplies the same options and callback on each open. A cache miss can build, so a runtime that must avoid building needs a matching saved engine already available.

Saved files live under `XDG_CACHE_HOME/tensoracle`, or `~/.cache/tensoracle` when the variable is unset. The identity includes the engine name, graph version, weight and plugin contents, input profiles, optimization level, disabled TF32 setting, GPU compatibility properties, OS, CPU architecture, and TensorRT version. Update `graph_version` when network code or another value outside these conditions changes the engine contents. Use separate names or versions for different precisions or model roles.

Changing only `workspace_limit` reuses a matching engine. To rebuild with a new limit, set `options.force_rebuild = true` before opening. The saved header records the build's workspace limit as reference information; cache hits do not change that value.

Weight SHA-256 records are reused across processes when the absolute path, size, modification time, and inode still match. An in-place weight edit that preserves those values can leave an old digest in use. `force_rebuild` reads weights again, refreshes their digests, and saves under the current identity. Plugin bytes are hashed on each open.

The C++ wrapper throws `tensoracle::error`. Its `status()` distinguishes `TENSORACLE_CORRUPT_ENGINE`, `TENSORACLE_ENGINE_CONDITION_MISMATCH`, and `TENSORACLE_ENGINE_IO_MISMATCH`. I/O diagnostics identify the tensor, field, observed value, expected value, and saved path. The caller chooses whether to rebuild or stop; tensoracle does not silently accept damaged or mismatching files.

### Execute an engine

Include `tensoracle/engine.hpp` (also included by `builder.hpp`). Create an execution unit, set all ranged input shapes together, query concrete output shapes and types, allocate buffers, then enqueue on the caller's stream. Fixed inputs and single-point profiles are initialized when the unit is created.

For the fixed four-element network above, this excerpt uses the CUDA Driver API and the `cuda_check` helper from the consumer example. `source` and `target` are caller-allocated device buffers of at least four floats each; `queue` is a caller-owned `CUstream` in the current context. The caller has already written the input and established its stream ordering:

```cpp
#include <tensoracle/engine.hpp>
#include <cuda.h>

auto execution = engine.create_execution();
auto output_shape = execution.get_tensor_shape("output");
auto output_type = execution.get_tensor_dtype("output");
execution.enqueue({{"input", reinterpret_cast<void*>(source)},
                   {"output", reinterpret_cast<void*>(target)}}, queue);
cuda_check(cuStreamSynchronize(queue));
```

For a ranged network, call `execution.set_input_shapes({{"features", {8, 32}}})` before querying concrete shapes or enqueuing. Supply every ranged input exactly once and omit fixed inputs. `cpp/tests/builder_fixture.hpp` demonstrates this ranged profile and execution sequence.

Each execution unit owns an independent TensorRT execution context and retains its engine. The caller owns the CUDA context, streams, correctly typed and sized buffers, transfers, and completion checks. Supply every I/O tensor exactly once with a non-null device address. A null stream selects the default stream. tensoracle does not allocate caller buffers, copy their contents, or synchronize execution. Use each unit serially and wait before changing shapes, releasing buffers, or closing the unit. Destruction and `close()` release resources without waiting.

Invalid shape arguments leave the last valid shapes intact. A partial TensorRT setting failure blocks concrete shape queries and enqueue until a complete setting succeeds. HOST tensors, external shape inputs, and data-dependent output sizes are unsupported. Shape inference outputs marked with `markOutputForShapes` are rejected before serialization or execution-context creation; ordinary shape-layer outputs remain supported.

#### Capture and replay with CUDA Graphs

One `tensoracle::engine_execution::enqueue` call can be captured and replayed. The caller owns graph creation, recreation, and destruction. `cpp/tests/execution.cpp` demonstrates capture with the CUDA Driver API.

Use a dedicated execution unit for each graph. After setting or changing input shapes, enqueue once and wait for completion before capture. Capture on a nondefault stream, keeping synchronous CUDA operations and legacy default-stream work outside capture. The graph fixes the input shapes, tensor addresses, and execution unit. Keep that unit, buffers, and plugin preparation resources alive until outstanding work finishes and the graph is destroyed. Recapture when shapes or addresses change; input contents may change at the same addresses between completed replays.

Engines with loops or conditionals, or layers or plugins that synchronize with the host during inference, cannot be captured. End an attempted capture and release graph resources before falling back to ordinary enqueue.

### Register plugin libraries

Pass all native libraries providing the graph's plugin creators in `options.plugins` before calling `tensoracle::open_engine`. tensoracle checks the `getCreators` export and registers libraries before building or deserializing. Their SHA-256 digests enter the engine identity. A missing, unreadable, incompatible, or already-loaded-but-changed library raises `tensoracle::error` with status `TENSORACLE_PLUGIN_LIBRARY_ERROR`.

For explicit registration, `tensoracle/plugins.hpp` exposes the same operation used in `cpp/tests/plugins.cpp`:

```cpp
#include <tensoracle/plugins.hpp>

auto digests = tensoracle::register_libraries({"/absolute/path/to/model_plugin.so"});
```

The path stands for a built library exporting TensorRT's `getCreators`. Loaded libraries remain registered for the process lifetime, even after the returned digests are released. Identical bytes are deduplicated. Registration can partially succeed before a later library fails. Explicit registration does not replace `options.plugins`: keep the libraries in those options so that the saved engine identity includes them. tensoracle does not verify that the supplied libraries provide every creator actually used by the graph.

### Manage saved engines

Inspect saved engines and preview or perform cleanup with the installed command:

```bash
tensoracle cache list
tensoracle cache prune --unused-for 30 --dry-run
tensoracle cache prune --unused-for 30
tensoracle cache prune --max-size 10GiB --dry-run
tensoracle cache prune --max-size 10GiB
```

The commands use the same `XDG_CACHE_HOME` / home `.cache` directory as `tensoracle::open_engine`. They read only filesystem metadata, engine headers, and saved weight digest records. They do not read or checksum the plan body, load the original weights, deserialize engines, import the TensorRT Python module, or query a GPU. Listing recognizes the header format; it does not establish that an engine can be loaded.

`cache list` prints one JSON object with `directory`, `entries`, and `total_size_bytes`. Each entry contains the file name, size in bytes, recorded time in UTC (`recorded_at`) and exact nanoseconds (`recorded_at_ns`), and the complete header conditions. This distinguishes engines with the same name and graph version but different weights or plugins. Each recognized entry also contains `workspace_limit`: a positive integer in bytes for an explicit build limit, JSON `null` when the build left the limit at TensorRT's default, or `"unknown"` when the header has no reference value. `weight_paths` associates each weight digest with paths from valid, checksum-verified `weight-*.json` records whose SHA-256 matches that digest; an empty `paths` array means unknown. These paths are references and do not establish which application uses an engine or whether the weight file still exists. Empty files have status `building or leftover`; unreadable or invalid headers have status `unrecognized` and a reason.

An engine's `.engine` modification time records its initial save or the last successful `tensoracle::open_engine` cache read. Failed loads and these management commands leave that time unchanged. A timestamp write failure, including a read-only cache, does not prevent a successful load, and a disappeared file is never recreated to record its time. The record describes the time of `tensoracle::open_engine`, not whether inference continues afterward. Files saved before timestamp recording was introduced, and files whose timestamp could not be written, may not reflect the actual last successful read.

`cache prune` requires exactly one of `--unused-for DAYS` and `--max-size SIZE`. Days are finite nonnegative numbers, may be fractional, and mean 24 hours each; only timestamps strictly earlier than the cutoff are selected. Sizes accept nonnegative bytes or the case-insensitive units `B`, `KiB`, `MiB`, `GiB`, and `TiB` (powers of 1024). Fractions such as `1.5GiB` are allowed and fractional bytes are rounded down. Size pruning selects the oldest recorded times first, breaking ties by file name, until the total `.engine` bytes would fit. Empty and unrecognized files count toward that total but are always excluded from deletion. Weight digest records are kept.

Before deletion, pruning prints a JSON `plan` containing the selected engines, excluded files, and a notice that removed engines will be rebuilt the next time they are used. It then prints a JSON `result` separating removed, excluded, failed, and already missing files, the bytes actually removed, and the remaining total. `--dry-run` shows the same targets without changing file contents or recorded times; its result also gives the projected total. Exit status is 0 for success, 1 for filesystem failures, failed deletions, or an unattainable size limit, and 2 for invalid usage. Files that disappear before removal are reported in `missing` and do not count as bytes removed. Observed file changes are reported as failed removals with a reason.

Run pruning during maintenance when no engine is being built or opened in the same cache directory. Cleanup provides no lock or guarantee of protection against concurrent builds and reads. Inspect the selection with `--dry-run` before performing cleanup. `tensoracle cache prune --help` describes the options.

Files saved by tensoracle 1.x appear as `unrecognized` and are excluded from pruning. See [Migrate from 1.x](migration.md#rebuild-saved-engines) before the first v2 run.

### Convert a PyTorch checkpoint

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

### Build a cubin plugin

Install `tensoracle[producer]` to compile caller-owned CUDA C++ kernels:

```bash
pip install 'tensoracle[producer]'
```

`tensoracle.producer.compile_bundle` wraps the C++ producer. It uses NVRTC, `ptxas >=13.2,<14`, and `cuobjdump` from PyPI CUDA 13.2 wheels, producing sm86, sm89, and sm120 cubins, a manifest, and `bundle.cpp`. It requires neither a GPU nor an OS-installed CUDA Toolkit. The [producer API](api/producer.md) describes CUDA C++ entry symbols, ordered arguments, constants, and tool overrides. Triton kernels are not accepted.

Use [the C++ sources and CMake helper in the wheel](cpp-python-repositories.md) to build the native plugin. The model package owns tensor types and formats, plugin fields, output semantics, variant selection, launch sequence, and CUDA C++ kernel behavior. tensoracle's common helper handles the TensorRT V3 lifecycle, cubin inspection and loading, and argument validation. A native library must export `getCreators`; static registration constructors are unsupported.

`tensoracle::cubin::PluginIO` is the fourth argument to `PluginBase(name, plugin_namespace, fields, io)`. It declares ordered inputs, one output, and the input whose shape the output copies. For two fp16 / HWC8 inputs with output shape taken from input 0, the declaration is:

```cpp
#include <tensoracle/cubin_plugin.hpp>

tensoracle::cubin::PluginIO residual_io() {
    using namespace tensoracle::cubin;
    tensoracle::cubin::TensorSpec half{nvinfer1::DataType::kHALF, nvinfer1::TensorFormat::kHWC8};
    return PluginIO{{half, half}, half, 0};
}
```

Implement `validate_shape(nvinfer1::Dims const* inputs, std::int32_t input_count, nvinfer1::Dims const& output) const noexcept` to validate dimensions, including wildcard `-1` dimensions during configuration. The kernel must implement the declared storage format. Omitting `io` selects one fp32 / LINEAR input and output with input 0's shape. An empty input list or invalid `shape_input_index` throws `std::invalid_argument`. For multiple outputs or computed output shapes, override the TensorRT V3 methods in your derived class.

Inspect the generated bundle and its matching plugin with `tensoracle.bundle.inspect_bundle`, or inspect the model wheel with `inspect_wheel`. Inspection checks the manifest, SHA-256 digests, actual cubin architectures, declared CUDA C++ parameter ABI, and the library's embedded cubin bytes without loading the plugin. The manifest records NVRTC and ptxas versions and kernel source digests, without host names or source paths.

## API reference

- [C++ interfaces and Python output comparison](api/index.md)
- [Cubin bundle inspection](api/bundle.md)
- [Cubin producer](api/producer.md)
