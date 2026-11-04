# Cubin producer API

`tensoracle.producer` wraps the C++ producer in `tensoracle/cubin.h` and `tensoracle/cubin.hpp`. It compiles caller-owned CUDA C++ source into sm86, sm89, and sm120 cubins, `bundle.json`, and `bundle.cpp`. Install `tensoracle[producer]` for NVRTC and the PyPI CUDA tools; neither a GPU nor an OS-installed CUDA Toolkit is required. Triton kernels are not accepted.

The extra declares `nvidia-cuda-nvrtc`, `nvidia-cuda-nvcc`, `nvidia-cuda-cuobjdump`, `nvidia-cuda-crt`, and `nvidia-nvvm` in the 13.2 series. `compile_bundle` obtains NVRTC from `nvidia/cu13/lib/libnvrtc.so.13`, ptxas from `nvidia/cu13/bin/ptxas`, and cuobjdump from `nvidia/cu13/bin/cuobjdump` in the Python environment. `cuda_root` selects another CUDA package root; `nvrtc`, `ptxas`, and `cuobjdump` independently override individual paths. The producer does not use `CUDA_HOME`. Missing tools report the package and searched path. ptxas outside `>=13.2,<14` is rejected before output files are written.

## Describe a CUDA C++ kernel

Each `KernelSpec` supplies CUDA C++ source, an `extern "C"` entry symbol, ordered public argument names and types, and integer compile constants. The source uses those constants as compile-time definitions. The following kernel and argument layout follow the producer test fixture in `python/tests/test_producer.py`:

```python
from pathlib import Path
from tensoracle.producer import KernelSpec, compile_bundle

source = '''extern "C" __global__ void copy_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * FACTOR;
}'''
spec = KernelSpec(
    name="copy",
    source=source,
    signature=(("x", "*fp32"), ("y", "*fp32"), ("n", "i32")),
    constexprs={"FACTOR": 2},
    entry_symbol="copy_kernel",
)
manifest = compile_bundle((spec,), Path("bundle"))
```

`divisible_by` adds argument specialization constraints when needed, and `dynamic_shared_memory_bytes` declares dynamic shared memory. The producer verifies the compiled entry's public parameter count, order, and types. CUDA C++ kernels have no hidden trailing parameters or Triton scratch arguments. The manifest uses schema version 2 and records NVRTC and ptxas versions, source digests, architecture, resource use, and parameter ABI.

Use [bundle inspection](bundle.md) before distributing a plugin, and [the CMake helper](../cpp-python-repositories.md) to build it from the generated bundle. Generation alone does not establish a kernel's numerical correctness, supported shapes, or runtime performance.

::: tensoracle.producer.KernelSpec

::: tensoracle.producer.compile_bundle
