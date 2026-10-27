# Cubin producer API

The following names are available from `tensoracle.producer`. Install `tensoracle[producer]` for Triton 3.8.0 and provide a host CUDA toolkit with `ptxas >=13.2,<14` and `cuobjdump`. `compile_bundle` uses the toolkit directory named by `CUDA_HOME`, or `/usr/local/cuda` when `CUDA_HOME` is unset. Explicit `ptxas` and `cuobjdump` paths override the defaults independently. A missing tool reports the searched path and how to set `CUDA_HOME` or pass an explicit path. Cubin production does not need a GPU. Normal imports, engine building, and engine reuse do not import the producer or Triton.

::: tensoracle.producer.KernelSpec

::: tensoracle.producer.compile_bundle
