# Cubin producer API

The following names are available from `tensoracle.producer`. Install `tensoracle[producer]` for Triton 3.8.0 and the PyPI CUDA tools; no OS-installed CUDA Toolkit is required. The extra declares `nvidia-cuda-nvcc`, `nvidia-cuda-cuobjdump`, `nvidia-cuda-crt`, and `nvidia-nvvm` in the 13.2 series.

`compile_bundle` obtains `ptxas` from `nvidia-cuda-nvcc` at `nvidia/cu13/bin/ptxas` and `cuobjdump` from `nvidia-cuda-cuobjdump` at `nvidia/cu13/bin/cuobjdump` (with `.exe` on Windows). It ignores `CUDA_HOME`. Explicit `ptxas` and `cuobjdump` paths override the defaults independently. A missing wheel or executable reports the package and searched path and explains how to install `tensoracle[producer]` or pass an explicit path. The producer rejects ptxas versions outside `>=13.2,<14` before writing cubins or a manifest.

Cubin production does not need a GPU. Triton production runs on Linux; the resulting cubins can also be embedded in a Windows plugin DLL. Normal imports, engine building, and engine reuse do not import the producer or Triton.

::: tensoracle.producer.KernelSpec

::: tensoracle.producer.compile_bundle
