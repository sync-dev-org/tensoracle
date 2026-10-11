# Cubin producer API

The following names are available from `tensoracle.producer`. Install `tensoracle[producer]` to compile cubins with Triton 3.8.0, `ptxas` 12.9.86, and `cuobjdump` 13.3. `compile_bundle` finds the NVIDIA wheel executables when their paths are omitted; explicit `ptxas` and `cuobjdump` paths override the defaults. Normal imports, engine building, and engine reuse do not import the producer or Triton.

::: tensoracle.producer.KernelSpec

::: tensoracle.producer.compile_bundle
