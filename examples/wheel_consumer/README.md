# A separate wheel using tensoracle's C++ SDK

This Linux example packages a nanobind extension independently of tensoracle.
Its C++ code reads a four-element `bias` tensor from safetensors, builds an
addition network through `tensoracle::open_engine`, and enqueues inference
through tensoracle's public execution interface. Python supplies a NumPy input
and receives a NumPy output. Importing the example does not initialize CUDA.

The build needs the tensoracle v2 wheel, the pinned Conan CLI, CMake 3.26 or
later, and a C++17 compiler. It obtains TensorRT SDK 11.2.1.2 from the repository's
recipe and lock; an explicit `TENSORRT_ROOT` remains supported. The isolated build installs nanobind, scikit-build-core,
and CUDA 13.2 headers from wheels. It uses the installed NVIDIA driver for the
CUDA Driver API. No OS CUDA Toolkit is required.

Build tensoracle's wheel first. For an unpublished v2 build, constrain the build
dependency to that wheel so that the build does not pick a published v1 package:

```bash
# Run from the repository root after installing the pinned Conan CLI.
cmake -P cpp/conan/prepare-tests.cmake
uv build python --wheel --out-dir /tmp/tensoracle-wheels
```

Create `/tmp/tensoracle-build-constraints.txt` with one line containing the
absolute file URI of the wheel just built:

```text
tensoracle @ file:///tmp/tensoracle-wheels/<tensoracle-wheel-filename>.whl
```

Then build and install the two wheels into a new environment:

```bash
uv build examples/wheel_consumer --wheel --out-dir /tmp/consumer-wheels \
  --build-constraints /tmp/tensoracle-build-constraints.txt
uv venv --python 3.13 /tmp/tensoracle-consumer-runtime
uv pip install --python /tmp/tensoracle-consumer-runtime/bin/python \
  /tmp/tensoracle-wheels/*.whl /tmp/consumer-wheels/*.whl
cd /tmp
/tmp/tensoracle-consumer-runtime/bin/python -I
```

Run this in that interpreter with an available GPU:

```python
import numpy as np
from safetensors.numpy import save_file
from tensoracle_consumer import infer

save_file({"bias": np.array([0.5, -2.0, 3.25, -0.75], dtype=np.float32)}, "bias.safetensors")
result = infer("bias.safetensors", np.array([-1.5, 4.0, 0.25, -8.0], dtype=np.float32))
print(result["output"])       # [-1.0, 2.0, 3.5, -8.75]
print(result["build_calls"])  # 1 for a new engine; 0 for an existing engine
print(result["engine_path"])
```

Repeat the inference in a new process with the same weights and `XDG_CACHE_HOME`
to reuse the saved engine without calling the network builder.

The CMake project uses `find_package(tensoracle 2 CONFIG REQUIRED)` against the
build dependency's `tensoracle/cpp` prefix and links `tensoracle::tensoracle`.
It uses the existing `tensoracle_find_cuda` helper for CUDA headers and driver
linking. It computes the relative path between the installed consumer package
and `tensoracle/cpp/lib`, giving `$ORIGIN/../tensoracle/cpp/lib` as the extension's
installed RUNPATH. `INSTALL_RPATH_USE_LINK_PATH` is disabled so that temporary
build paths and driver link directories do not enter that RUNPATH.

The consumer wheel contains no copy of `libtensoracle`, TensorRT, or the CUDA
driver. The tensoracle wheel supplies `libtensoracle.so.2`, which locates
TensorRT through its own relative RUNPATH and declared wheel dependency.
Neither importing nor running this example needs `LD_LIBRARY_PATH`.
