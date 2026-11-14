# Use the prebuilt C and C++ SDK

Starting with v2.6.0, each release attaches SDK archives for Linux x86_64 and
Windows x86_64 to the same
[public release](https://github.com/sync-dev-org/tensoracle/releases), using the
same version and tag:

| Platform | Release asset |
| --- | --- |
| Linux x86_64 (including WSL2) | `tensoracle-sdk-X.Y.Z-linux-x86_64.tar.gz` |
| Windows x86_64 | `tensoracle-sdk-X.Y.Z-windows-x86_64.zip` |

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
expected value for that release. Both archives extract under `tensoracle/cpp/`.
The Windows SDK copies its wheel's `tensoracle/cpp/` files without changing their
bytes and adds the MIT license.

### Linux x86_64

The download URL is
`https://github.com/sync-dev-org/tensoracle/releases/download/vX.Y.Z/tensoracle-sdk-X.Y.Z-linux-x86_64.tar.gz`.
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

### Windows x86_64

Choose a release from v2.6.0 onward. In PowerShell, replace `X.Y.Z` with its
version, download the ZIP, and compare its SHA256 with the expected release value
before extracting it:

```powershell
$version = "X.Y.Z"
$archive = "tensoracle-sdk-$version-windows-x86_64.zip"
Invoke-WebRequest "https://github.com/sync-dev-org/tensoracle/releases/download/v$version/$archive" -OutFile $archive
Get-FileHash -Algorithm SHA256 $archive
Expand-Archive -Path $archive -DestinationPath C:\path\to\sdk
```

The ZIP contains `tensoracle/cpp/bin/tensoracle-2.dll`, its import library
`tensoracle/cpp/lib/tensoracle.lib`, and the common `include/`,
`lib/cmake/tensoracle/`, `cmake/`, `src/`, and `licenses/` directories. The
prebuilt DLL uses MSVC's dynamic Release runtime (`/MD`).

From an x64 MSVC development shell with CMake and Ninja available, configure
your application against the extracted SDK and a separately prepared TensorRT
SDK containing `lib/nvinfer_11.lib`:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -DCMAKE_PREFIX_PATH=C:/path/to/sdk/tensoracle/cpp -DTENSORRT_ROOT=C:/path/to/TensorRT-SDK
cmake --build build
```

Both platforms use the same CMake calls:

```cmake
find_package(tensoracle 2 CONFIG REQUIRED)
add_executable(application main.cpp)
target_link_libraries(application PRIVATE tensoracle::tensoracle)
```

Every application needs the TensorRT 11 link library at build time
(`libnvinfer.so.11` on Linux or `nvinfer_11.lib` on Windows).
The CMake package supplies its
link-time search path through `tensoracle::tensoracle` automatically: it uses an existing `TensorRT::nvinfer`
target or discovers that target through `find_package(TensorRT CONFIG)`, including
Conan's CMakeDeps files. Otherwise it searches the system library locations. For
a TensorRT installation outside those locations, pass
`-DTENSORRT_ROOT=/path/to/TensorRT` or
`-DTENSORACLE_TENSORRT_LIBRARY=/path/to/libnvinfer.so.11` on Linux, or
`-DTENSORACLE_TENSORRT_LIBRARY=C:/path/to/TensorRT-SDK/lib/nvinfer_11.lib` on
Windows, to CMake. Configuration
stops with these options if the library is missing. No build-time
`LD_LIBRARY_PATH` is needed. On Linux, the SDK inside a Python wheel can also
find the separate TensorRT wheel installed beside it.
On Windows, the TensorRT runtime wheel's DLL does not replace the build-time
import library; supply the SDK's `nvinfer_11.lib`.

Code that constructs TensorRT networks, or includes
`tensoracle/cubin_plugin.hpp`, additionally needs the TensorRT 11.2 headers; see
[Prepare a TensorRT SDK](tensorrt-sdk.md). The `tensoracle_find_cuda`,
`tensoracle_cuda_driver`, and `tensoracle_add_cubin_plugin` helpers are available
after `find_package(tensoracle)`. Plugin builds use Conan
`cuda-toolkit/13.2.2` (`compiler=False`) for CUDA headers and the Linux driver
link stub or Windows `lib/x64/cuda.lib` import library.
Pass its `user.cuda:root` through `CUDAToolkit_ROOT` in the consumer toolchain,
as shown in the SDK guide. Existing explicit CUDA SDK roots remain supported;
the link stub is not part of the application runtime.

## Embed tensoracle from source

To embed a cubin plugin in a model library or executable, see
[Embedded cubin plugins](embedded-plugins.md). This works with either the shared
tensoracle SDK or a source-built static tensoracle library.

Source builds select a shared library by default (`BUILD_SHARED_LIBS=ON`). After
[preparing the build dependencies](https://github.com/sync-dev-org/tensoracle/blob/main/cpp/README.md), select a static library
with `BUILD_SHARED_LIBS=OFF`:

```bash
python3 cpp/conan/prepare.py --output build/cpp
. ./build/cpp/conanbuild.sh
cmake --preset conan-release -S cpp -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF
cmake --build build/cpp --parallel 8
cmake --install build/cpp --prefix /path/to/prefix
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
  -DCMAKE_TOOLCHAIN_FILE="$PWD/build/cpp/conan_toolchain.cmake" \
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

### Linux application prefix

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

### Windows application prefix

Keep the extracted SDK layout when placing it beside your application:

```text
prefix/
  bin/application.exe
  tensoracle/cpp/
    bin/tensoracle-2.dll
    lib/tensoracle.lib
    include/tensoracle/
    lib/cmake/tensoracle/
    cmake/
    src/
    licenses/
```

Link through `tensoracle::tensoracle`; `tensoracle.lib` is a build input and
`tensoracle-2.dll` is loaded at runtime. For a native executable, put the SDK's
`bin` directory and the host's full TensorRT runtime DLL directory on the
process's `PATH` before launch. For example, from cmd after moving the complete
prefix:

```bat
set "PATH=C:\path\to\prefix\tensoracle\cpp\bin;C:\path\to\TensorRT\bin;%PATH%"
C:\path\to\prefix\bin\application.exe
```

The TensorRT directory must contain `nvinfer_11.dll` and the full runtime's
builder resources and dependencies. The build-time `TENSORRT_ROOT` setting
does not configure runtime DLL lookup. TensorRT and NVIDIA driver DLLs stay on
the host rather than being copied into the SDK or application prefix.

Python model bindings use the tensoracle wheel's DLL. Import `tensoracle`
before your model's native binding so its `os.add_dll_directory` registration
is active. Register the host's TensorRT DLL directory separately, or import
`tensorrt` first when using the optional extra; see the
[Windows Python runtime examples](tensorrt-sdk.md#windows-x86_64).
Use one tensoracle implementation per process, as described above.

## Package an application that embeds tensoracle

An application that ships tensoracle relies on the host's TensorRT at run time:

- A Linux deb lists the TensorRT runtime package in `Depends` with a lower bound
  only. Use tensoracle's minimum TensorRT version, 11.2, as that bound.
- Do not list the CUDA Toolkit as a dependency. The NVIDIA driver is a host
  prerequisite.
- A Windows installer checks for the TensorRT DLLs during installation and stops
  if they are not found.
