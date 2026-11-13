# Cubin plugin example

`kernels.cu` contains caller-owned CUDA C++ kernels. `kernels.py` declares their entry symbols, ordered public arguments, integer compile constants, and specialization conditions. The C++ tensoracle producer uses the PyPI NVRTC compiler to produce PTX and the PyPI ptxas assembler to produce cubins for sm86, sm89, and sm120. It requires no GPU or OS-installed CUDA Toolkit.

The dot plugin computes an fp32 32 by 16 matrix product and scales the output in a second launch. The bias plugin adds a 16 element bias. Both use TensorRT IPluginV3 and retain serialized fields and per-context device parameters. The dot plugin selects an entry using the declared pointer alignment and row divisibility conditions, with an unspecialized entry for other inputs. The `dot.tf32-control` entry rounds its inputs to TF32 for numerical comparison; the plugin uses the fp32 entry.

## Rebuild the cubins

Install the pinned Conan CLI and prepare the native dependencies as described in [the SDK guide](../../docs/tensorrt-sdk.md). Tensoracle and this example obtain the verified TensorRT SDK through the locked Conan recipe. From the repository root:

```bash
uv sync --project python --extra producer
uv run --project python --extra producer python -c 'from pathlib import Path; from examples.cubin_plugin.kernels import kernel_specs; from tensoracle.producer import compile_bundle; compile_bundle(kernel_specs(), Path("examples/cubin_plugin/bundle"))'
```

The producer extra supplies the CUDA 13.2 series of `nvidia-cuda-nvrtc`, `nvidia-cuda-nvcc`, `nvidia-cuda-cuobjdump`, `nvidia-cuda-crt`, and `nvidia-nvvm`. The C++ implementation validates ptxas >=13.2,<14 before writing cubins or a manifest. `cuda_root` can select a PyPI `nvidia/cu13` directory; independent `nvrtc`, `ptxas`, and `cuobjdump` path overrides are also available. `CUDA_HOME` is ignored.

`bundle.json` uses schema version 2. Every entry records its CUDA C++ entry symbol, ordered argument names and types, compile constants, source and cubin SHA-256, actual SM, register count, static shared memory, dynamic shared memory, and specialization conditions. CUDA C++ entries have no hidden Triton arguments or scratch fields. `bundle.cpp` embeds the cubin bytes and supplies the matching `KernelImage` entries.

```bash
uv run --project python --extra producer pytest -m 'not gpu' python/tests/test_producer.py
```

## Build and inspect the example wheel

The example owns `bindings.cpp`, which uses tensoracle's C++ interface to open an engine, construct its network from safetensors, register the two plugins, and run inference. Python only supplies arrays and selects paths. It uses no TensorRT Python API or tensoracle Python engine API.

Build tensoracle's v2 wheel first. Its build obtains TensorRT SDK 11.2.1.2 from Conan:

```bash
uv build python --wheel --out-dir /tmp/tensoracle-wheels
```

For an unpublished v2 build, create `/tmp/tensoracle-build-constraints.txt` with the absolute file URI of that wheel. This makes the isolated example build use the same C++ SDK:

```text
tensoracle @ file:///tmp/tensoracle-wheels/<tensoracle-wheel-filename>.whl
```

Install the [host's TensorRT runtime](https://sync-dev-org.github.io/tensoracle/tensorrt-sdk/#install-the-runtime) with apt, then build and install the example and tensoracle into a new environment:

```bash
uv build examples/cubin_plugin --wheel --out-dir /tmp/example-wheels \
  --build-constraints /tmp/tensoracle-build-constraints.txt
uv venv --python 3.13 /tmp/tensoracle-example-runtime
uv pip install --python /tmp/tensoracle-example-runtime/bin/python \
  /tmp/tensoracle-wheels/*.whl /tmp/example-wheels/*.whl
```

The build requires CMake 3.26 or later and a C++17 compiler. Its Python build dependencies supply nanobind. The locked Conan recipes supply TensorRT and CUDA 13.2 headers and the driver link stub. `CUDAToolkit_ROOT` and `TENSORRT_ROOT` still accept existing verified SDK roots. The plugin links the CUDA Driver API stub and loads the installed driver at runtime. No OS CUDA Toolkit is required.

The CMake project uses `find_package(tensoracle 2 CONFIG REQUIRED)` against the installed wheel and `tensoracle_add_cubin_plugin` to compile the plugin. Its own nanobind extension links to `tensoracle::tensoracle`. The extension's RUNPATH is `$ORIGIN/../tensoracle/cpp/lib`; the example wheel contains no copy of tensoracle, TensorRT, or the CUDA driver. Importing the example does not initialize CUDA.

Inspect the example wheel with `tensoracle.bundle.inspect_wheel(wheel, plugin="example_cubin_plugin/libtensoracle_example_plugin.so")`. The C++ inspector reads the ZIP, validates the bundle and actual cubin architectures, and compares the plugin's exported digest and embedded bytes against the bundle. Inspection does not execute plugin initializers or initialize CUDA.

## Run inference

Start the installed interpreter outside the source tree, with an available GPU:

```bash
cd /tmp
/tmp/tensoracle-example-runtime/bin/python -I
```

```python
import numpy as np
from safetensors.numpy import load_file
import example_cubin_plugin as example

model = example.open_model()
x = np.linspace(-1, 1, 16 * 32, dtype=np.float32).reshape(16, 32)
output = model.infer(x)
weights = load_file(example.weights_path())
expected = (x @ weights["matrix"]) * weights["scale"] + weights["bias"]
np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
print(model.engine_path, model.build_calls)
```

`example.infer(x)` opens a model and returns its output in one call. `open_model(weights=..., plugin=..., force_rebuild=True)` can select another safetensors file or plugin library, or replace a saved engine. Reopen in another process with the same weights and `XDG_CACHE_HOME` to reuse the same saved engine; `build_calls` then reports zero.

Each model retains two independent TensorRT execution contexts. `model.infer(x, execution=1)` uses the second context, and `model.output_shape(1)` reports its output shape. Inputs must be contiguous NumPy fp32 arrays with shape `(rows,32)` and 1 through 32 rows. The C++ code owns each call's buffers and non-default stream, synchronizes before returning a CPU output, and keeps CUDA resources inside the binding.

`model.infer(x, capture=True)` warms up after setting the input shape, captures one enqueue, and replays it with new input bytes. C++ owns the graph, its buffers, and its destruction. The example's two plugins enqueue only GPU work and support capture.

## GPU verification

Bundle production, plugin compilation, import, and bundle inspection do not establish that a kernel runs correctly on a GPU. After reserving the host's GPU, run from the repository root:

```bash
TENSORACLE_CONAN_OFFLINE=ON uv run --project python pytest -q python/tests/test_example_plugin.py -m gpu
```

The test installs both wheels, checks dot and bias inference, preserves each execution context's shape, captures and replays plugin enqueue, reuses an engine in a new process, and rebuilds when plugin bytes change. Record sm86, sm89, and sm120 results separately. CPU checks use `-m 'not gpu'`.
