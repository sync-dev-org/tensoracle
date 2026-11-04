# C++ interfaces and Python comparison API

Engine construction, saving and reuse, execution, and plugin registration use the C++ SDK. The model package calls these interfaces directly and exposes inference through its own binding. Follow the [user guide](../index.md#connect-a-c-model-package) and [separate consumer wheel](https://github.com/sync-dev-org/tensoracle/tree/main/examples/wheel_consumer) for integration.

## Public C++ headers

The installed SDK is under `tensoracle/cpp` in the wheel. Use `find_package(tensoracle 2 CONFIG REQUIRED)` and link `tensoracle::tensoracle`.

| C ABI header | C++ wrapper | Operations |
| --- | --- | --- |
| `tensoracle/tensoracle.h` | `tensoracle/tensoracle.hpp` | Version checks, status and owned diagnostics, CPU output comparison |
| `tensoracle/builder.h` | `tensoracle/builder.hpp` | Network callback, safetensors weight views, input profiles, I/O checks, `tensoracle::open_engine` |
| `tensoracle/engine.h` | `tensoracle/engine.hpp` | Engine handles, execution units, input shapes, tensor metadata, enqueue on a caller stream |
| `tensoracle/plugins.h` | `tensoracle/plugins.hpp` | Library inspection and process-lifetime registration with `tensoracle::register_libraries` |
| `tensoracle/cache.h` | `tensoracle/cache.hpp` | List saved engines, plan pruning, and apply pruning |
| `tensoracle/cubin.h` | `tensoracle/cubin.hpp` | Compile CUDA C++ kernels and inspect bundles or wheels |

`tensoracle/cubin_plugin.hpp` supplies the source-based TensorRT plugin helper. The [plugin build guide](../cpp-python-repositories.md) describes its CMake functions and source files.

The C boundary uses opaque handles, `struct_size` fields, status values, and owned diagnostics. Initialize each structure's `struct_size` to the size actually supplied. Destroy error handles with `tensoracle_error_destroy`; diagnostic text is borrowed until destruction. No C++ exception crosses this boundary. The header-only C++ wrappers release handles automatically and throw `tensoracle::error`, whose `status()` preserves the C status.

## Python utilities

Only `OutputComparison` and `compare_outputs` are exported directly from `tensoracle`. Other Python operations live in `tensoracle.producer`, `tensoracle.bundle`, and `tensoracle.convert`, or the `tensoracle cache` commands. They wrap C++ operations except for checkpoint conversion, which uses torch. See [Migrate from 1.x](../migration.md) for removed Python names and their C++ replacements.

## Output comparison

The caller runs inference and passes CPU NumPy candidate and reference arrays to the comparison function. The application chooses quality limits; this function does not run an engine or make an acceptance decision.

::: tensoracle._comparison.OutputComparison

::: tensoracle._comparison.compare_outputs
