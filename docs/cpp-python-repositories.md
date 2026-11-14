# Use tensoracle's C++ SDK and plugin sources

The v2 wheel supplies a shared C++ library, public headers, CMake package configuration, and source-based cubin plugin helpers. Your model package defines the TensorRT network and engine-opening code in C++, links the shared library, and exposes inference through its own binding. For custom plugins it also supplies CUDA C++ kernels and model-specific plugin behavior.

Starting with v2.6.0, releases provide `cp312-abi3` wheels for Linux x86_64 (`manylinux_2_28`, including WSL2) and Windows x86_64 (`win_amd64`). Both platforms use the same version and release tag. Native applications can also use the matching [prebuilt C and C++ SDK archives](cpp-sdk.md).

## C++ SDK in the wheel

The tensoracle wheel includes the following common files and Linux library:

```text
tensoracle/cpp/
  include/tensoracle/tensoracle.h
  include/tensoracle/tensoracle.hpp
  include/tensoracle/builder.h
  include/tensoracle/builder.hpp
  include/tensoracle/engine.h
  include/tensoracle/engine.hpp
  include/tensoracle/cubin_plugin.hpp
  lib/libtensoracle.so.2
  lib/cmake/tensoracle/tensoracleConfig.cmake
  src/cubin_plugin.cpp
  cmake/TensoracleCubinPlugin.cmake
  cmake/TensoracleCuda.cmake
```

The Windows wheel has the same header, CMake, and source layout, with `bin/tensoracle-2.dll` and `lib/tensoracle.lib` in place of the Linux shared library. `tensoracle.lib` is the DLL's import library. The SDK also includes plugin, cache, and cubin C ABI headers and C++ wrappers. Your plugin's native library is compiled separately and belongs to your model package. TensorRT and NVIDIA driver libraries are not copied into the tensoracle wheel.

Locate `tensoracle/cpp` with the metadata command below and add it to `CMAKE_PREFIX_PATH`. Link a model target with `find_package(tensoracle 2 CONFIG REQUIRED)` and `target_link_libraries(my_model PRIVATE tensoracle::tensoracle)`. The [user guide](index.md#connect-a-c-model-package) and [separate consumer wheel](https://github.com/sync-dev-org/tensoracle/tree/main/examples/wheel_consumer) show the integration and Linux relative runtime library lookup. On Windows, import `tensoracle` before your model's native binding: it registers the installed wheel's `cpp/bin` with `os.add_dll_directory`. Your model package must also arrange [TensorRT's DLL lookup](tensorrt-sdk.md#windows-x86_64). The C ABI and header-only C++ wrappers are listed in [the API reference](api/index.md).

## Prepare the build inputs

Use CMake 3.26 or later for the model package SDK integration and a C++17 compiler. The source-based cubin plugin helper alone requires CMake 3.20 or later. Follow [Prepare a TensorRT SDK](tensorrt-sdk.md) to obtain and verify the TensorRT 11.2 or later (below 12) headers and link library and prepare the Conan CUDA headers and driver link inputs on Linux or Windows; no OS-installed CUDA Toolkit is required. Windows builds use MSVC's dynamic Release runtime (`/MD`).

Generate a cubin bundle from CUDA C++ source with the [cubin producer](api/producer.md). The bundle directory must contain `bundle.cpp`, `bundle.json`, and the cubin files. Follow [Build a cubin plugin](index.md#build-a-cubin-plugin) for the model-specific plugin interface and tensor declarations.

## Use the packaged CMake helper

In the Python environment where tensoracle is installed, locate its C++ source directory:

```bash
python -c 'from importlib.metadata import distribution; print(distribution("tensoracle").locate_file("tensoracle/cpp"))'
```

Pass that directory as `TENSORACLE_CPP_ROOT` to your plugin's CMake project. Include the helper and give it your plugin source, verified SDK root, and generated bundle directory:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_model_plugin LANGUAGES CXX)

include("${TENSORACLE_CPP_ROOT}/cmake/TensoracleCubinPlugin.cmake")
tensoracle_add_cubin_plugin(
    TARGET my_model_plugin
    TENSORRT_ROOT "${TENSORRT_ROOT}"
    BUNDLE_ROOT "${TENSORACLE_BUNDLE_ROOT}"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/plugin.cpp")
```

`tensoracle_add_cubin_plugin` builds a shared library from your `plugin.cpp`, tensoracle's common `cubin_plugin.cpp`, and the bundle's generated `bundle.cpp`. It adds the required include directories and links TensorRT and the CUDA driver.

For a Bash build, replace the absolute paths below with your plugin project, installed tensoracle sources, verified TensorRT SDK, Conan CUDA root, and generated bundle:

```bash
cmake -S /absolute/path/to/plugin-project -B build/plugin \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORACLE_CPP_ROOT=/absolute/path/to/tensoracle/cpp \
  -DTENSORRT_ROOT=/absolute/path/to/TensorRT-SDK \
  -DCUDAToolkit_ROOT=/absolute/path/to/conan-cuda-toolkit \
  -DTENSORACLE_BUNDLE_ROOT=/absolute/path/to/bundle
cmake --build build/plugin --config Release
```

The native library must export TensorRT's `getCreators`. To use the resolved CUDA inputs in another target, call `tensoracle_find_cuda(OUT_INCLUDE_DIR cuda_include OUT_DRIVER_LIBRARY cuda_driver)` after including `TensoracleCuda.cmake` or `TensoracleCubinPlugin.cmake`. Both keywords name variables set in the calling scope. The public IMPORTED target `tensoracle_cuda_driver` carries the same include directory and driver library. For example:

```cmake
tensoracle_find_cuda(
    OUT_INCLUDE_DIR cuda_include
    OUT_DRIVER_LIBRARY cuda_driver)
target_link_libraries(my_model PRIVATE tensoracle_cuda_driver)
```

`my_model` stands for your existing CMake target. The function uses the same `CUDAToolkit_ROOT` variable, environment setting, and default locations as `tensoracle_add_cubin_plugin`, with the same missing-input errors.

## Package and inspect the plugin

Package the native library and its generated bundle together in your model wheel. Use the [bundle inspection API](api/bundle.md) to check the bundle and its match with the compiled plugin before distribution. Put required library paths in C++ `builder_options.plugins` before opening the model's engine; test construction, separate-process reuse, inference, and output quality through the model's binding. The [optimization workflow](optimization.md) describes those quality gates.
