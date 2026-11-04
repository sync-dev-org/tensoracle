# Cubin bundle inspection API

`tensoracle.bundle` wraps the C++ inspection API in `tensoracle/cubin.h` and `tensoracle/cubin.hpp`. It checks the schema version 2 manifest, cubin contents and architectures, declared public arguments, resource declarations, and optional plugin embedding without loading the plugin or initializing a GPU. CUDA C++ bundles have no hidden trailing parameters or Triton scratch declarations. Use these checks before distributing a model wheel.

::: tensoracle.bundle.inspect_bundle

::: tensoracle.bundle.inspect_wheel

::: tensoracle.bundle.bundle_sha256

::: tensoracle.bundle.cubin_sm

## Inspection errors

The following exception classes are exported by `tensoracle.bundle` from the native binding:

| Exception | Meaning |
| --- | --- |
| `BundleError` | Invalid manifest or another bundle validation failure |
| `MissingCubinError` | A referenced cubin is missing or unreadable |
| `CorruptCubinError` | Cubin contents disagree with the declared size or SHA-256 |
| `CubinArchitectureError` | Cubin ELF architecture or ABI is unsupported, or the actual SM disagrees with the manifest |
| `PluginBundleMismatchError` | The plugin's embedded bundle does not match the supplied bundle |
