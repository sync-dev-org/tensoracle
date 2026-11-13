# Embed a cubin plugin in a model module

A model module can link its cubin plugin into its own shared library or executable.
Register the linked creators directly with TensorRT and inspect the manifest and
cubins from memory. A separate plugin library or bundle directory is unnecessary
at runtime. The existing shared-plugin path and directory inspection APIs remain
available.

## Build the linked plugin

After [preparing the TensorRT and CUDA build inputs](tensorrt-sdk.md), use the
installed CMake helpers:

```cmake
find_package(tensoracle 2 CONFIG REQUIRED)
tensoracle_add_cubin_plugin(
    TARGET model_kernels
    TYPE STATIC
    TENSORRT_ROOT "${TENSORRT_ROOT}"
    BUNDLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bundle"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/plugin.cpp")

add_library(model SHARED model.cpp bundle_view.cpp)
target_compile_features(model PRIVATE cxx_std_17)
target_include_directories(model PRIVATE "${TENSORRT_ROOT}/include")
target_link_libraries(model PRIVATE model_kernels tensoracle::tensoracle)
```

`TYPE` accepts `SHARED`, `STATIC`, and `OBJECT`; omitting it preserves the shared
library build. Static and object targets can be linked into a shared model library
or an executable, and use position-independent code. With `TYPE OBJECT`, link the
target through `target_link_libraries` so its driver and TensorRT link dependencies
also reach the final target. The plugin helper continues to compile the supplied
plugin sources, tensoracle's common plugin implementation, and the producer's
`bundle.cpp`.

## Register creators already in the process

Include `tensoracle/plugins.hpp`. The model supplies its own creator accessor; it
does not need to export a function named `getCreators` or let TensorRT load a file.
Each pointer passed to the registration API must point to an
`nvinfer1::IPluginCreatorInterface` subobject.

```cpp
#include <NvInfer.h>
#include <tensoracle/builder.hpp>
#include <tensoracle/plugins.hpp>

// Implemented in this model's plugin sources. Creators have process lifetime.
nvinfer1::IPluginCreatorInterface* const* model_creators(int32_t& count) noexcept;

int32_t count = 0;
auto creators = model_creators(count);
std::vector<void*> pointers(creators, creators + count);
auto registration = tensoracle::register_embedded_plugins(
    reinterpret_cast<const void*>(&model_creators), pointers, "my.model");

tensoracle::builder_options options;
// Set the engine name, graph version, weight files, input profiles, and IO.
options.embedded_plugins = {registration.get()};
// tensoracle::open_engine(options, define_network);
```

The address must belong to the module containing the plugin implementation. On
Linux, tensoracle finds the owning module's file-backed segment in
`/proc/self/maps`. It opens that mapping's absolute pathname, or `/proc/self/exe`
when the plugin is linked into the executable, and checks the descriptor's device
and inode against the mapping before hashing the complete file through the same
descriptor. An address in a module's anonymous BSS is also accepted. Relative
library paths remain usable after changing the working directory. If the loaded
module cannot be matched to its current file, including after the file is replaced,
registration fails before registering creators and asks for a process restart.
Registration uses
the supplied creators through TensorRT's `registerCreator`; it does not call
`loadLibrary`. Tensoracle reads the containing file to compute its complete
SHA-256. `registration.digests()` returns that identity as a one-element vector.
Windows currently reports `TENSORACLE_PLUGIN_LIBRARY_ERROR` for this operation.

Use the same TensorRT namespace as the model's creators. Repeating registration
with the same creators and namespace is valid; a conflicting creator is an error.
The model must keep its creators and containing module alive for the process
lifetime. Destroying a registration result frees its data without deregistering
creators. An operation can register some creators before a later creator fails.

Directly registering some creators does not complete registration of their whole
library. Passing that same library to `register_libraries` still checks its
`getCreators` export and TensorRT's loader result. TensorRT may reject a library
whose creators are already registered; this is reported as a registration error.
Different modules can use the two registration APIs together.

Keep registration results alive through `open_engine`. Its `embedded_plugins`
array can contain results from multiple model modules and can be combined with
the existing `plugins` array of shared-library paths. Both contribute digests to
the existing engine cache conditions. Tensoracle checks that the containing files
still match their registered identities before selecting an engine. Rebuilding
any part of the containing library or executable changes the identity, including
changes unrelated to the plugin, and causes a separate engine to be built.

The equivalent C function is `tensoracle_register_embedded_plugins` in
`tensoracle/plugins.h`. It returns a `tensoracle_plugin_libraries` handle, read
with the existing count/digest operations and freed with the existing destroy
operation. The trailing `embedded_plugins` and `embedded_plugin_count` fields of
`tensoracle_builder_options` accept a pointer array of these handles. Initialize
`struct_size`; older callers that omit these fields continue to use their
existing behavior. The saved engine file format is unchanged.

## Inspect an embedded manifest and cubins

Embed the complete `bundle.json` as bytes in the model and provide a named view
of every cubin referenced by its entries. The producer's `bundle.cpp` already
contains the cubin arrays under each entry's `embedded_symbol`. The model's build
can reference those symbols and use each entry's `file` and `bytes` for the
descriptors. The model owns the manifest and descriptor definitions; tensoracle
does not require a particular resource-embedding mechanism.

```cpp
#include <tensoracle/cubin.hpp>

// Definitions belong to the model's generated resource sources.
extern const char model_manifest[];
extern const size_t model_manifest_size; // Excludes a terminating NUL.
extern const tensoracle_bundle_file* const model_cubins[];
extern const size_t model_cubin_count;

auto inspected = tensoracle::inspect_embedded_bundle(
    {model_manifest, model_manifest_size}, model_cubins, model_cubin_count);
auto digest = inspected.sha256();
```

Each `tensoracle_bundle_file` has `struct_size`, the relative `name` from the
manifest, a `data` pointer, and a byte `size`. Its name must be unique in the
supplied array. All input is borrowed only for the inspection call, and the
returned result owns its manifest and digest. Inspection opens no files, loads
no plugin, and initializes no CUDA state. It uses the same manifest, digest, size,
argument, resource, and ELF architecture checks as `tensoracle_inspect_bundle`.
A missing cubin returns `TENSORACLE_MISSING_CUBIN`; damaged bytes return
`TENSORACLE_CORRUPT_CUBIN`.

C callers use `tensoracle_inspect_embedded_bundle` and the existing
`tensoracle_bundle_result_get` / `tensoracle_bundle_result_destroy` operations.
The original `tensoracle_inspect_bundle` and `tensoracle_inspect_wheel` operations
retain their directory and wheel behavior. Python's public API is unchanged;
the model exposes its own inference binding.
