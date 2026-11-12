# Use the prebuilt C and C++ SDK

Each Linux x86_64 release attaches `tensoracle-sdk-X.Y.Z-linux-x86_64.tar.gz` to the
corresponding [public release](https://github.com/sync-dev-org/tensoracle/releases).
The archive contains the shared C ABI library, public C and C++ headers, the CMake
package and plugin helpers, their source files, and MIT and third-party licenses.
It does not contain TensorRT. The release asset requires no authentication.
Python applications continue to install tensoracle through PyPI.

## Use the SDK with CMake

Download the archive, check its SHA256 against the value you recorded for that
release, and extract it with its relative layout and symlinks intact:

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

The SDK supports Linux x86_64 and shared libraries only. Code that constructs
TensorRT networks, or includes `tensoracle/cubin_plugin.hpp`, also needs the
TensorRT 11.2 headers and link library at build time; see
[Prepare a TensorRT SDK](tensorrt-sdk.md). The `tensoracle_find_cuda`,
`tensoracle_cuda_driver`, and `tensoracle_add_cubin_plugin` helpers are available
after `find_package(tensoracle)`.

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
SDK or the application prefix. The Python wheel retains its existing SDK and
its relative lookup of the separate TensorRT wheel.

## Package an application that embeds tensoracle

An application that ships tensoracle relies on the host's TensorRT at run time:

- A Linux deb lists the TensorRT runtime package in `Depends` with a lower bound
  only. Use tensoracle's minimum TensorRT version, 11.2, as that bound.
- Do not list the CUDA Toolkit as a dependency. The NVIDIA driver is a host
  prerequisite.
- A Windows installer checks for the TensorRT DLLs during installation and stops
  if they are not found.
