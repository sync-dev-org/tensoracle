# Build plugins with tensoracle's C++ sources

Tensoracle supplies the shared C++ implementation and CMake helpers for cubin plugins. Your model package supplies the kernels and model-specific plugin behavior, compiles the native plugin, and packages it for the application.

## C++ sources in the wheel

The tensoracle wheel includes:

```text
tensoracle/cpp/
  include/tensoracle/cubin_plugin.hpp
  src/cubin_plugin.cpp
  cmake/TensoracleCubinPlugin.cmake
  cmake/TensoracleCuda.cmake
```

The wheel contains no native binaries and uses the `py3-none-any` tag. Your plugin's native library is compiled separately and belongs to your model package.

## Prepare the build inputs

Use CMake 3.20 or later and a C++17 compiler. Follow [Prepare a TensorRT SDK](tensorrt-sdk.md) to obtain and verify the TensorRT 11 headers and library and prepare the PyPI CUDA headers and driver link library. That guide covers Linux, WSL2, and [Windows](tensorrt-sdk.md#windows-x86_64); no OS-installed CUDA Toolkit is required.

Generate a cubin bundle with the [cubin producer](api/producer.md). The bundle directory must contain `bundle.cpp`, `bundle.json`, and the cubin files. Cubin production runs on Linux; the same bundle can be used to build a Windows plugin DLL. Follow [Build a cubin plugin](index.md#build-a-cubin-plugin) for the model-specific plugin interface and tensor declarations.

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

For a Bash build, replace the absolute paths below with your plugin project, installed tensoracle sources, verified SDK, PyPI CUDA root, and generated bundle:

```bash
cmake -S /absolute/path/to/plugin-project -B build/plugin \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORACLE_CPP_ROOT=/absolute/path/to/tensoracle/cpp \
  -DTENSORRT_ROOT=/absolute/path/to/TensorRT-SDK \
  -DCUDAToolkit_ROOT=/absolute/path/to/site-packages/nvidia/cu13 \
  -DTENSORACLE_BUNDLE_ROOT=/absolute/path/to/bundle
cmake --build build/plugin --config Release
```

The [plugin guide](https://github.com/sync-dev-org/tensoracle#build-a-cubin-plugin) documents the required `getCreators` export and the optional `tensoracle_find_cuda` function and `tensoracle_cuda_driver` target for consumers that also need the resolved CUDA inputs.

## Package and inspect the plugin

Follow the [cubin plugin example](https://github.com/sync-dev-org/tensoracle/blob/main/examples/cubin_plugin/README.md#build-and-inspect-the-wheel) for packaging the native library and bundle in your model wheel and testing it in a runtime environment. Use the [bundle inspection API](api/bundle.md) to check the bundle and its match with the compiled plugin before distribution.
