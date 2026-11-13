# Use the prebuilt C and C++ SDK

Each Linux x86_64 release attaches `tensoracle-sdk-X.Y.Z-linux-x86_64.tar.gz` to the
corresponding [public release](https://github.com/sync-dev-org/tensoracle/releases).
The archive contains the shared C ABI library, public C and C++ headers, the CMake
package and plugin helpers, their source files, and MIT and third-party licenses.
It does not contain TensorRT. The release asset requires no authentication.
Python applications continue to install tensoracle through PyPI.
Release SDK archives and Python wheels contain the shared library only. To
embed tensoracle statically, build it from source; static libraries are not
distributed.
The tensoracle Conan package recipe is maintained separately and is not included
in this repository.

## Use the SDK with CMake

Download the archive from the public release and check its SHA256 against the
expected value for that release. For version 2.2.1, the download URL is
`https://github.com/sync-dev-org/tensoracle/releases/download/v2.2.1/tensoracle-sdk-2.2.1-linux-x86_64.tar.gz`
and the SHA256 is
`f0d30002ec53262ac8eb1d2b9e57e1c8e83fad21b1c9ff2c665ae38afaa30391`.
Extract the verified archive with its relative layout and symlinks intact:

```bash
tar -xzf tensoracle-sdk-X.Y.Z-linux-x86_64.tar.gz -C /path/to/sdk
```

Point CMake at the extracted `tensoracle/cpp` directory and use the SDK's own
relocatable CMake package:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/sdk/tensoracle/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

```cmake
find_package(tensoracle 2 CONFIG REQUIRED)
add_executable(application main.cpp)
target_link_libraries(application PRIVATE tensoracle::tensoracle)
```

The SDK supports Linux x86_64 and shared libraries only. Every application needs
the TensorRT 11 link library at build time. The CMake package supplies its
link-time search path through `tensoracle::tensoracle` automatically: it uses an existing `TensorRT::nvinfer`
target or discovers that target through `find_package(TensorRT CONFIG)`, including
Conan's CMakeDeps files. Otherwise it searches the system library locations. For
a TensorRT installation outside those locations, pass
`-DTENSORRT_ROOT=/path/to/TensorRT` or
`-DTENSORACLE_TENSORRT_LIBRARY=/path/to/libnvinfer.so.11` to CMake. Configuration
stops with these options if the library is missing. No build-time
`LD_LIBRARY_PATH` is needed. The SDK inside a Python wheel can also find the
separate TensorRT wheel installed beside it.

Code that constructs TensorRT networks, or includes
`tensoracle/cubin_plugin.hpp`, additionally needs the TensorRT 11.2 headers; see
[Prepare a TensorRT SDK](tensorrt-sdk.md). The `tensoracle_find_cuda`,
`tensoracle_cuda_driver`, and `tensoracle_add_cubin_plugin` helpers are available
after `find_package(tensoracle)`. Plugin builds use Conan
`cuda-toolkit/13.2.2` (`compiler=False`) for CUDA headers and the driver link stub.
Pass its `user.cuda:root` through `CUDAToolkit_ROOT` in the consumer toolchain,
as shown in the SDK guide. Existing explicit CUDA SDK roots remain supported;
the link stub is not part of the application runtime.

## Embed tensoracle from source

Source builds select a shared library by default (`BUILD_SHARED_LIBS=ON`). After
[preparing the build dependencies](https://github.com/sync-dev-org/tensoracle/blob/main/cpp/README.md), select a static library
with `BUILD_SHARED_LIBS=OFF`:

```bash
cmake -S cpp -B build/static -G Ninja -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF
cmake --build build/static --parallel 8
cmake --install build/static --prefix /path/to/prefix
```

The installed package keeps the same public headers and
`tensoracle::tensoracle` target. The archive contains only tensoracle's objects;
miniz and the CUDA driver remain separate link dependencies. The CMake package
finds these dependencies, TensorRT's shared library, and the platform thread
and dynamic-loader libraries for the final link. The current library uses the
CUDA Driver API and does not use the CUDA runtime. If a future version uses
the CUDA runtime, its static parts must also be supplied to the final link.
TensorRT remains a host-supplied shared runtime.

Configure a consumer with the prepared Conan toolchain so it can resolve the
same miniz and CUDA build inputs. For example, from the tensoracle source root:

```bash
cmake -S /path/to/application -B build/application \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/build/static/conan/conan_toolchain.cmake" \
  -DCMAKE_PREFIX_PATH=/path/to/prefix
cmake --build build/application --parallel 8
```

The consumer uses the same `find_package(tensoracle)` and
`target_link_libraries(... tensoracle::tensoracle)` calls shown above. The
static target supplies `TENSORACLE_STATIC_LIBRARY`, which disables DLL import
and export attributes in the public headers. Source-based cubin plugin helpers
remain available after `find_package(tensoracle)`. A source-building Conan
recipe selects this form with `from_source=True` and `shared=False`; the
prebuilt SDK route supports shared libraries only.

A process must contain exactly one tensoracle implementation, used by all its
code. An artifact that embeds tensoracle must be its only owner in that
process. Do not load another artifact embedding tensoracle or a tensoracle
shared library into the same process. Python processes use the wheel's shared
library. Relink an artifact that embeds tensoracle when updating the embedded
tensoracle version.

## Install the host runtime and relocate a prefix

The execution host needs TensorRT 11.2 or later, below 12, compatible with its
NVIDIA driver and GPU. Install the full runtime, including the resources needed
by the TensorRT builder, using
[NVIDIA's installation instructions](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html).
The TensorRT build SDK provides build inputs; a single link library does not
provide everything needed to build an engine on the execution host.

When assembling a native application prefix, copy tensoracle's package content
with its relative layout intact:

```text
prefix/
  bin/application
  tensoracle/cpp/
    include/tensoracle/
    lib/libtensoracle.so -> libtensoracle.so.2.Y.Z
    lib/libtensoracle.so.2 -> libtensoracle.so.2.Y.Z
    lib/libtensoracle.so.2.Y.Z
    lib/cmake/tensoracle/
    cmake/
    src/
    licenses/
```

Keep relative symlinks when copying. An executable in `bin/` can use
`$ORIGIN/../tensoracle/cpp/lib` as its install RUNPATH. Each application library
must resolve its own direct dependencies relative to its actual installation
location. Move the complete prefix together.

The native SDK's `libtensoracle.so.2` uses `$ORIGIN` as RUNPATH and has no
TensorRT-specific search path. The host loader finds `libnvinfer.so.11` through
its normal configuration. If TensorRT is installed outside the system loader's
configured locations, configure that loader or set `LD_LIBRARY_PATH` to the
runtime directory. Do not copy TensorRT or driver libraries into tensoracle's
SDK or the application prefix. The Python wheel uses the host runtime too;
it also checks the sibling TensorRT wheel when the optional
`tensoracle[tensorrt]` extra is installed. See [runtime installation](tensorrt-sdk.md#install-the-runtime).

## Package an application that embeds tensoracle

An application that ships tensoracle relies on the host's TensorRT at run time:

- A Linux deb lists the TensorRT runtime package in `Depends` with a lower bound
  only. Use tensoracle's minimum TensorRT version, 11.2, as that bound.
- Do not list the CUDA Toolkit as a dependency. The NVIDIA driver is a host
  prerequisite.
- A Windows installer checks for the TensorRT DLLs during installation and stops
  if they are not found.
