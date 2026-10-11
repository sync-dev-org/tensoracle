# tensoracle

`tensoracle` builds TensorRT engines from model graphs supplied through TensorRT's Python network API. A model package owns its graph, safetensors weights, and output quality limits. `tensoracle` supplies engine building, saved engine reuse, execution, output comparison, and tools for cubin plugins. These workflows do not require torch. Only the optional `.pth` checkpoint converter requires torch, installed with `tensoracle[convert]`.

Follow the [four-stage optimization workflow](optimization.md) to establish output quality, evaluate fp16 and plugins, and measure speed.

## Terms

- A **TensorRT engine** is an executable form of a model optimized for a GPU, precision, and TensorRT version. It can be kept in memory or saved to disk.
- **Building an engine** means making that executable form from a model graph. The graph function comes from the model package.
- A **cubin plugin** is a native TensorRT plugin that runs GPU kernels supplied as compiled CUDA binary (`.cubin`) files. The model package supplies the kernels and plugin behavior.

## Choose a workflow

| Role | What to do | Additional tools beyond the `tensoracle` package |
| --- | --- | --- |
| Inference application | Use `open_engine` to reopen a matching saved engine, then enqueue it with `handle.create_execution()`. | The application supplies buffers and streams, for example through CuPy or `cuda-python`. |
| Standard graph developer | Define a TensorRT graph, build it with `build_serialized_engine` or `open_engine`, and measure outputs with `compare_outputs`. | None for standard TensorRT layers. |
| Cubin plugin developer | Produce cubins, build a native TensorRT plugin, and package it with the model. | `tensoracle[producer]`, CUDA toolkit, TensorRT SDK, CMake, and a C++17 compiler. |

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

The supported platform is Linux x86_64 (including WSL2) with sm86, sm89, or sm120 GPUs and TensorRT 11. The producer and native build tools are separate from the `tensoracle` runtime dependencies. Engine building, engine reuse, output comparison, and cubin production do not require torch. Converting `.pth` checkpoints requires the `convert` extra.

### Convert a PyTorch checkpoint

Prepare safetensors weights before passing them to the model graph. Install the conversion extra and run:

```bash
pip install 'tensoracle[convert]'
python -m tensoracle.convert model.pth model.safetensors
```

The same operation is available to Python scripts:

```python
from tensoracle.convert import pth_to_safetensors

pth_to_safetensors("model.pth", "model.safetensors")
```

The converter accepts mappings with string keys containing tensors or nested mappings. It joins nested keys with `.`, preserves tensor dtypes, shapes, and values, and rejects non-tensor metadata or duplicate flattened names. It loads on the CPU with `weights_only=True`, without an unrestricted pickle fallback. The extra requires torch 2.6 or later, including the fix for [CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6). Only calling conversion imports torch; engine building and reuse do not. The model package supplies any weight renaming or model-specific preprocessing.

### Reopen a saved engine

The model package must provide a graph function, an `EngineBuild` declaration, and expected `EngineIO` even when the engine is already saved. For example, with those objects exported by `my_model`:

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

`my_model` represents the application's model package. On a valid cache hit, `open_engine` does not call `graph`; on a miss, it builds and saves a new engine. `open_engine` itself does not execute the engine or manage GPU buffers. Create an execution unit from its returned handle to enqueue inference. Saved engines live under `XDG_CACHE_HOME/tensoracle`, or `~/.cache/tensoracle` when `XDG_CACHE_HOME` is unset. See the [complete cache example](https://github.com/sync-dev-org/tensoracle#save-and-reopen-an-engine) for the declaration and expected input and output types.

### Execute an engine

The execution flow is: open the engine, create an execution unit, set all ranged input shapes together, read concrete output shapes and dtypes, allocate output buffers, then enqueue the device addresses on a caller-owned stream. Each `EngineExecution` keeps the handle alive and owns an independent TensorRT context. Fixed inputs and single-point profiles are initialized when the unit is created. Ranged inputs retain their shapes until the next successful `set_input_shapes` call. A partial TensorRT setting failure blocks queries for concrete shapes and enqueue until a complete setting succeeds.

#### Keep data on the GPU with CuPy

Install CuPy in the application when using this example. For the `features` / `result` graph in the [README example](https://github.com/sync-dev-org/tensoracle#define-and-build-a-standard-graph), the arrays stay on the device until the caller chooses to read the result:

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

#### Supply host copies with cuda-python

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

`TensorSpec` accepts optional `location=trt.TensorLocation.DEVICE` and `format=trt.TensorFormat.LINEAR`. `open_engine` checks these only when declared, together with tensor names, modes, dtypes, build-time shapes, and profiles. An `EngineIOError` names the tensor, field, observed value, expected value, and saved file path. The execution unit does not accept a second declaration.

### Build a standard graph

Define `graph(network)` with TensorRT's Python network API and load model weights from safetensors. Describe each graph input with `InputShape.fixed` or `InputShape.ranged`, and pass an `EngineBuild` declaration to `build_serialized_engine`. `open_engine` uses the same declaration plus a name, graph version, weight files, and `EngineIO` to save and reuse a checked engine. The supported precisions are fp32 (with TF32 disabled) and fp16. The [graph example](https://github.com/sync-dev-org/tensoracle#define-and-build-a-standard-graph) shows the exact calls.

The application runs the engine and passes a CPU NumPy candidate output and a corresponding reference output to `compare_outputs`. The application chooses the acceptance limits. See [Compare output arrays](https://github.com/sync-dev-org/tensoracle#compare-output-arrays).

### Build a cubin plugin

Install `tensoracle[producer]` when you need to compile cubins from your own kernels:

```bash
pip install 'tensoracle[producer]'
```

`tensoracle.producer.compile_bundle` uses the extra's Triton 3.8.0, `ptxas` 12.9.86, and `cuobjdump` 13.3 to produce cubins for sm86, sm89, and sm120. It finds the NVIDIA wheel executables by default and accepts explicit paths as overrides. Cubin production does not need a GPU or import torch; run it without importing torch first. Build the native plugin with a TensorRT 11 SDK, CUDA toolkit, CMake, and a C++17 compiler. Inspect the resulting bundle or wheel with `tensoracle.bundle`. The [cubin plugin example](https://github.com/sync-dev-org/tensoracle/tree/main/examples/cubin_plugin) covers rebuilding from a repository clone, packaging, and runtime use. Normal import, engine building, and engine reuse do not import the producer or Triton, and do not need the producer extra or native build tools.

## API reference

- [Engine building, cache reuse, execution, and output comparison](api/index.md)
- [Cubin bundle inspection](api/bundle.md)
- [Cubin producer](api/producer.md)
