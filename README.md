# tensoracle

Build and execute TensorRT engines from graphs defined with TensorRT's Python network API. The graph function owns the model's safetensors weights and tensor types. Engine building, engine reuse, execution, and output comparison do not require torch. Only the optional `.pth` checkpoint converter requires torch, installed with `tensoracle[convert]`.

## Terms and workflows

- A **TensorRT engine** is TensorRT's executable form of a model, optimized for a GPU, precision, and TensorRT version. It can be held in memory or saved to disk.
- **Building an engine** means creating that executable form from a model graph. `tensoracle` builds from a graph function supplied by the model package.
- A **cubin plugin** is a native TensorRT plugin whose GPU kernels are supplied as compiled CUDA binary (`.cubin`) files. The model package owns the kernels and plugin behavior.

Choose the path that matches your work:

| Role | Use | Additional tools beyond the `tensoracle` package |
| --- | --- | --- |
| Inference application | [Reopen a saved engine](#use-a-saved-engine-for-inference) and enqueue it with `handle.create_execution()`. | The application supplies device buffers and CUDA streams, for example through CuPy or `cuda-python`. |
| Standard graph developer | [Define and build a graph](#define-and-build-a-standard-graph), [save and reopen](#save-and-reopen-an-engine), and [compare outputs](#compare-output-arrays). | None for standard TensorRT layers. |
| Cubin plugin developer | [Build a cubin plugin](#build-a-cubin-plugin) and package it with the model. | `tensoracle[producer]`, CUDA toolkit, TensorRT SDK, CMake, and a C++17 compiler. |

| Requirement | Inference application | Standard graph developer | Cubin plugin developer |
| --- | --- | --- | --- |
| Python 3.12 or 3.13, `numpy`, `safetensors` | Package dependencies | Package dependencies; the graph reads safetensors weights | Package dependencies in the plugin runtime |
| `tensorrt-cu13` 11.x | Open and run engines | Build and open engines | Open plugin engines |
| NVIDIA GPU and driver | Open and run engines | Build and run engines | Run and test engines; cubin production itself can run without a GPU |
| CuPy or `cuda-python` | The application chooses its buffer and stream library | The application chooses its buffer and stream library | `cuda-python` is used by the example runtime and tests, not by `tensoracle` |
| Triton 3.8.0, `ptxas` 12.9.86, `cuobjdump` 13.3 | Not needed | Not needed | Installed with `tensoracle[producer]` for cubin production |
| CUDA toolkit | Not needed | Not needed | Native plugin build |
| TensorRT 11 SDK | Not needed | Not needed | Native plugin build |
| CMake and C++17 compiler | Not needed | Not needed | Native plugin build |

The supported targets are Linux x86_64 (including WSL2), TensorRT 11, and GPU generations sm86, sm89, and sm120. The producer and native build tools are separate from the `tensoracle` runtime dependencies. Engine building, engine reuse, output comparison, and cubin production do not require torch. Converting `.pth` checkpoints requires the `convert` extra described below.

Follow the [four-stage optimization workflow](https://github.com/sync-dev-org/tensoracle/blob/main/docs/optimization.md) to establish output quality, evaluate fp16 and plugins, and measure speed.

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

## Define and build a standard graph

```python
import numpy as np
import tensorrt as trt
from safetensors.numpy import load_file

from tensoracle import BuilderSettings, EngineBuild, InputShape, build_serialized_engine


def graph(network: trt.INetworkDefinition) -> None:
    weights = load_file("model.safetensors")
    features = network.add_input("features", trt.float32, (1, -1, 2))
    bias = np.ascontiguousarray(weights["bias"][None, None, :], dtype=np.float32)
    constant = network.add_constant(bias.shape, bias).get_output(0)
    result = network.add_elementwise(features, constant, trt.ElementWiseOperation.SUM)
    output = result.get_output(0)
    output.name = "result"
    network.mark_output(output)


declaration = EngineBuild(
    precision="fp32",
    inputs={"features": InputShape.ranged((1, 1, 2), (1, 64, 2), (1, 448, 2))},
    settings=BuilderSettings(workspace_limit=1 << 30, optimization_level=3),
)
plan: bytes = build_serialized_engine(graph, declaration)
```

Use `InputShape.fixed((1, 64, 2))` for one input size; the graph may declare that size directly or use `-1` for a dimension fixed by the profile. For a range, put `-1` in each varying dimension of the graph input and declare its minimum, optimum, and maximum sizes. Every graph input needs a matching entry in `EngineBuild.inputs`.

Set `precision="fp16"` to declare an fp16 engine. The graph function chooses each tensor's actual type, including any fp32 tensors inside an fp16 graph. TF32 is always disabled; `BuilderSettings(tf32=True)` raises `ValueError`. The function returns serialized plan bytes. Invalid declarations raise `ValueError`, and a failed TensorRT build raises `RuntimeError`.

## Save and reopen an engine

`open_engine` builds on the first call and reuses the saved engine on later calls, including calls from another process. It invokes `graph` only when building. Declare the expected input and output names, TensorRT types, and build-time shapes:

```python
from pathlib import Path

from tensoracle import EngineIO, TensorSpec, open_engine

handle = open_engine(
    graph,
    declaration,
    name="example",
    graph_version="1",
    weights=(Path("model.safetensors"),),
    io=EngineIO(
        inputs={"features": TensorSpec(trt.float32, (1, -1, 2))},
        outputs={"result": TensorSpec(trt.float32, (1, -1, 2))},
    ),
)
engine = handle.engine  # Keep handle.runtime alive while using the engine.
```

Saved files live in the `tensoracle` directory under `XDG_CACHE_HOME`, or under `~/.cache` when that variable is unset. The cache identity includes the contents of all declared safetensors weight files and optional `plugins` files, the engine name, graph version, precision, input shapes, builder settings, GPU product name, OS, CPU architecture, and TensorRT version. Raise `graph_version` when graph code changes; without a version change, the previous engine is reused. Pass `force_rebuild=True` to replace a cached engine even when it is intact.

`plugins` names native TensorRT plugin libraries. Both `open_engine` and `build_serialized_engine` accept it. Each function checks that every library exports `getCreators`, loads it through TensorRT's registry, and retains it for the process before building the graph or deserializing an engine. Repeating the same library does not register it twice. `open_engine` also hashes the loaded library bytes into the engine identity; a changed library selects a different saved engine in a new process. A missing, unreadable, or incompatible library, or a library changed after loading in the same process, raises `PluginLibraryError` before a cached plan is opened. Keep the library and its cubin bundle together in the caller's wheel.

The caller must pass through `plugins=` all libraries providing the plugin creators (name, version, namespace) used by the graph and saved engine, with no missing or unnecessary libraries. For plugin libraries, tensoracle guarantees inspection and registration of the supplied files and inclusion of their SHA-256 digests in the saved engine identity. It does not verify that those files provide the creators actually used: if the graph uses a creator registered earlier by another library in the same process, tensoracle cannot detect that mismatch. Passing a different library that provides the same creator causes TensorRT to reject registration and tensoracle to raise `PluginLibraryError`.

For each weight file, `open_engine` stores its SHA-256 alongside the saved engines and reuses it across processes when the file's absolute path, size, modification time, and inode still match. If any of these change, or the stored digest cannot be used, it reads the weight file again. Plugin files are hashed on every call. This shortcut accepts a risk: changing a file in place while keeping the same size and modification time can leave the old digest in use, so an engine built from the previous weights may be returned. Pass `force_rebuild=True` to read the weights again, refresh their stored digests, and save the engine under the digest of the current contents.

`TensorSpec` also accepts optional `location=trt.TensorLocation.DEVICE` and `format=trt.TensorFormat.LINEAR`. `open_engine` checks them only when supplied, alongside names, modes, dtypes, build-time shapes, and profiles. An `EngineIOError` identifies the tensor, mismatching field, observed value, expected value, and saved file path. Execution does not take another I/O declaration.

`CorruptEngineError` reports incomplete or damaged saved files and plans that TensorRT cannot deserialize. When deserialization fails, its message asks the caller to check the required plugin libraries in `plugins=`. `EngineConditionError` reports a saved header with different build conditions. These types and `EngineIOError` derive from `EngineCacheError`. The caller chooses whether to rebuild after an error.

## Use a saved engine for inference

An inference application can call `open_engine` with the same build declaration and files used to save the engine. For example, assume the model package exports `graph`, `declaration`, and `io` matching the example above:

```python
from pathlib import Path

from my_model import declaration, graph, io
from tensoracle import open_engine

handle = open_engine(
    graph,
    declaration,
    name="example",
    graph_version="1",
    weights=(Path("model.safetensors"),),
    io=io,
)
```

`my_model` stands for the application's own model package. It must provide the graph function, build declaration, and expected `EngineIO` even when the saved engine is reused. On a valid cache hit, `open_engine` does not call `graph`. If the engine is absent, it builds one, so an inference environment that must avoid building needs a matching saved engine already in the user cache. `open_engine` returns a TensorRT engine and its owning runtime; `open_engine` itself does not execute the engine or manage GPU buffers.

The execution flow is: open the engine, create an execution unit, set all ranged input shapes together, read concrete output shapes and dtypes, allocate output buffers, then enqueue the device addresses on a caller-owned stream. Each `EngineExecution` keeps the handle alive and owns an independent TensorRT context. Fixed inputs and single-point profiles are initialized when the unit is created. Ranged inputs retain their shapes until the next successful `set_input_shapes` call. A partial TensorRT setting failure blocks queries for concrete shapes and enqueue until a complete setting succeeds.

### Keep data on the GPU with CuPy

Install CuPy in the application when using this example. For the `features` / `result` graph above, the arrays stay on the device until the caller chooses to read the result:

```python
import cupy as cp
import tensorrt as trt

with handle.create_execution() as execution:
    execution.set_input_shapes({"features": (1, 4, 2)})
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        features = cp.arange(8, dtype=cp.float32).reshape(1, 4, 2) / 4
        result = cp.empty(
            execution.get_tensor_shape("result"),
            dtype=trt.nptype(execution.get_tensor_dtype("result")),
        )
        execution.enqueue(
            {"features": features.data.ptr, "result": result.data.ptr},
            stream=stream.ptr,
        )
        stream.synchronize()  # The caller waits before closing or reading on the host.
        output = cp.asnumpy(result)
```

### Supply host copies with cuda-python

The same execution API accepts buffers allocated by `cuda-python`. The application adds allocation, copies, waiting, and release:

```python
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def cuda(result):
    status, *values = result
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA returned {status}")
    return values[0] if values else None


with handle.create_execution() as execution:
    execution.set_input_shapes({"features": (1, 4, 2)})
    features = np.arange(8, dtype=np.float32).reshape(1, 4, 2) / 4
    output = np.empty(
        execution.get_tensor_shape("result"),
        dtype=trt.nptype(execution.get_tensor_dtype("result")),
    )
    source, target, stream = None, None, None
    try:
        source = cuda(cudart.cudaMalloc(features.nbytes))
        target = cuda(cudart.cudaMalloc(output.nbytes))
        stream = cuda(cudart.cudaStreamCreate())
        cuda(cudart.cudaMemcpyAsync(
            source, features.ctypes.data, features.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream,
        ))
        execution.enqueue({"features": int(source), "result": int(target)}, stream=int(stream))
        cuda(cudart.cudaMemcpyAsync(
            output.ctypes.data, target, output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream,
        ))
    finally:
        if stream is not None:
            cuda(cudart.cudaStreamSynchronize(stream))
            cuda(cudart.cudaStreamDestroy(stream))
        if target is not None:
            cuda(cudart.cudaFree(target))
        if source is not None:
            cuda(cudart.cudaFree(source))
```

`enqueue` requires exactly one positive integer address per input and output and a nonnegative integer stream handle; bool is rejected. Stream 0 is accepted, but prefer a nondefault stream for explicit ordering. Tensoracle does not inspect the buffers: the caller must provide the actual dtype, shape, format, contiguous layout where required, and sufficient storage. CuPy and `cuda-python` are application choices, not required dependencies or extras of tensoracle.

The caller selects the CUDA device and context, orders inputs written on other streams before enqueue, and keeps buffers, streams, plugin preparation resources, and the execution unit alive until completion. Use one execution unit serially; do not call it concurrently from multiple threads. Confirm completion before changing shapes or streams, reusing buffers, or closing. `close()` and context exit release the TensorRT context before the handle, without waiting, and repeated close calls do nothing. Operations on a closed unit raise `RuntimeError` before touching TensorRT. The unit does not retain caller arrays or expose its TensorRT context.

Tensoracle neither allocates or frees device buffers, copies data, nor calls stream or device synchronization. TensorRT itself may allocate internal resources or act synchronously for some networks and plugins. Execution units reject HOST-location tensors, externally supplied shape tensor values, and data-dependent output sizes at creation. Ordinary networks that compute shapes internally from input dimensions are supported. CUDA graphs, profile switching, and caller-owned TensorRT workspace memory are outside this API.

Networks with outputs marked as shape tensors emit `tensoracle.ShapeTensorOutputWarning` during building and when creating an execution unit (`create_execution()` or `EngineExecution(handle)`). The warning names the affected tensors: `create_execution()` cannot handle these engines, and calling it crashes the process. Building continues by default. To stop either operation safely, turn this warning into an exception with `warnings.filterwarnings("error", category=tensoracle.ShapeTensorOutputWarning)` after importing `warnings` and `tensoracle`. Saved engines retain the shape output names outside the cache conditions; older files without these names are reused without an execution warning.

## Build a cubin plugin

Follow [Prepare a TensorRT SDK](https://github.com/sync-dev-org/tensoracle/blob/main/docs/tensorrt-sdk.md) to acquire and verify the headers and native library for a build SDK, including the fixed inputs for SDK 11.2.1.2 and the `TENSORRT_ROOT` setup.

Install the producer extra when you need to compile cubins from your own kernels:

```bash
pip install 'tensoracle[producer]'
```

`tensoracle.producer.compile_bundle` uses the extra's **Triton 3.8.0**, **ptxas 12.9.86**, and **cuobjdump 13.3**. It finds the NVIDIA wheel executables by default; explicit `ptxas` and `cuobjdump` paths can override them. The normal import, engine build, and engine reuse paths do not import the producer or Triton. Cubin production does not need a GPU or import torch; run it without importing torch first. The caller supplies Triton functions and `KernelSpec` records with their public argument types, compile time constants, and any divisibility assumptions. The producer writes sm86, sm89, and sm120 cubins, `bundle.json`, and a C++ translation unit that embeds exactly those cubins. The generated C++ entry records carry each public argument's order, type, count, and divisibility rule into the common launcher. It rejects an unexpected parameter ABI or nonzero scratch requirement. The included [example](https://github.com/sync-dev-org/tensoracle/blob/main/examples/cubin_plugin/README.md#rebuild-the-cubins) has the command to rebuild its cubins from a repository clone.

The wheel also contains `tensoracle/cpp/include`, `tensoracle/cpp/src`, and `tensoracle/cpp/cmake`. Pass the installed `tensoracle/cpp` directory to CMake as `TENSORACLE_CPP_ROOT` and include `TensoracleCubinPlugin.cmake`. Its `tensoracle_add_cubin_plugin` function compiles the common C++ source, the caller's plugin source, and the generated `bundle.cpp` into the caller's library. The caller implements the graph's meaning, field meanings, weight placement, launch sequence, and variant selection. The common source owns typed fields in their original order, handles `IPluginV3` lifecycle methods, checks cubin digests and resources, loads one module per CUDA context, checks the generated argument count and specialization rules, appends the two supported zero scratch pointers, and launches kernels. The caller selects a variant using `Kernel::accepts` and derives the count passed to `Kernel::launch` from its argument array. The common source releases modules and device allocations in their owning contexts. The library must export TensorRT's `getCreators`; a static registration constructor is not supported.

`tensoracle.bundle.inspect_bundle(bundle_directory, plugin=library_path)` checks the manifest, each cubin's SHA-256 and actual ELF SM, the supported hidden arguments and zero scratch sizes, and the cubin bytes embedded in the library. `inspect_wheel(wheel_path, plugin="path/inside/wheel.so")` performs the same checks on a wheel without installing it. Their distinct exceptions identify a missing cubin, corrupt cubin, false SM claim, or plugin and bundle mismatch. The manifest records Triton, ptxas, and kernel source digests without a producer host or OS path. The TensorRT versions tested with the included example are listed below.

## Tested environments

On 2026-10-10, one example wheel was built for each TensorRT SDK version without changing the example cubin bundle (digest `86816421d8f6f66a19dcf3469677d7656f33c3c5398024c5b7d7882ba98c5571`). Each SDK was assembled from NVIDIA's release headers and the `tensorrt-cu13-libs` wheel, and each wheel passed `inspect_wheel`. These five wheels were built with gcc 15.2.0, CMake 4.2.3, and CUDA toolkit 13.3.73, and the same files were run on every GPU below. The matrix covers TensorRT runtime `11.0.0.114`, `11.1.0.106`, `11.2.1.2`, `11.3.0.99`, and `11.4.0.106` against example wheels built with the same five SDK versions (25 pairs per GPU). The tables list only what was run.

| GPU | Matrix result | Python | OS / glibc / kernel / NVIDIA driver |
| --- | --- | --- | --- |
| NVIDIA RTX PRO 4000 Blackwell (sm120) | 25 of 25 pairs passed | 3.13.15 | WSL2 / glibc 2.43 / `6.18.33.2-microsoft-standard-WSL2` / 616.92 |
| NVIDIA RTX 6000 Ada Generation (sm89) | 25 of 25 pairs passed | 3.13.15 | WSL2 / glibc 2.43 / `6.18.33.2-microsoft-standard-WSL2` / 596.72 |
| NVIDIA RTX A6000 (sm86) | 25 of 25 pairs passed | 3.13.7 | WSL2 / glibc 2.39 / `6.18.40.1-microsoft-standard-WSL2` / 597.16 |

For every pair, the example `build`, the `reuse` of the same saved engine in a separate process without calling the graph, the TF32 `control`, and `open_engine` with `compare_outputs` without the plugin all succeeded. Both example plugins matched their reference values within tolerance, and both the specialized and unspecialized variants launched. The TF32 control was distinguishable from the IEEE reference. At the start of each run, the runner checked the actual GPU SM and the exact TensorRT version installed in the environment.

The example was also built with an older compiler and CUDA toolkit:

| Build | Runtime check | Python | OS / glibc |
| --- | --- | --- | --- |
| gcc 13.3.0 / CMake 4.2.3 / CUDA toolkit 12.9.86 / TensorRT SDK 11.2.1.2 | `build` and a separate-process `reuse` with TensorRT runtime 11.2.1.2 on sm86 succeeded with zero outputs outside tolerance and both variants launched | 3.13.7 | WSL2 / glibc 2.39 |

Each GPU subprocess had a 50-second limit. Speed was not measured. Native Linux, other kernels, and TensorRT 11 versions not listed here have not been tested.

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
