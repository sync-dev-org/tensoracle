# Prepare a TensorRT SDK

Tensoracle uses a locked Conan recipe for the TensorRT **11.2.1.2** SDK on Linux x86_64 and Windows x86_64. It supplies 13 headers, the platform's link library and runtime libraries, builder resources, and the TensorRT OSS and NVIDIA license texts. WSL2 uses the Linux SDK. Starting with v2.6.0, tensoracle supplies wheels and prebuilt C++ SDK archives for both operating systems with the same version and release tag.

| TensorRT build input | Linux x86_64 | Windows x86_64 |
| --- | --- | --- |
| Link library | `lib/libnvinfer.so.11`, with a relative `lib/libnvinfer.so` symlink | `lib/nvinfer_11.lib` |
| Runtime library | `lib/libnvinfer.so.11` | `bin/nvinfer_11.dll` |
| Builder resources | `lib/libnvinfer_builder_resource_*.so.*` | `bin/nvinfer_builder_resource_*_11.dll` |

Native applications can obtain tensoracle itself from its [public prebuilt SDK archive](cpp-sdk.md),
verify that release's SHA256, and configure CMake with `CMAKE_PREFIX_PATH`.
The tensoracle Conan package recipe is maintained separately and is not included in this repository.

The TensorRT SDK supplies build inputs. Tensoracle's standalone C++ build, Python wheel build, native test preparation, and source distribution build all use `cpp/conanfile.py` and `cpp/conan.lock`. They do not require `TENSORACLE_SDK_ROOT` or a manually assembled TensorRT installation. CUDA build headers and the Linux driver link stub or Windows import library come from Conan `cuda-toolkit/13.2.2` with `compiler=False`; an OS-installed CUDA Toolkit is not required. The cubin producer continues to obtain its runtime compiler, ptxas, and cuobjdump from PyPI.

## Fixed and verified inputs

The source distribution used as a release build input includes the exact locked recipe and its input manifest at `cpp/conan/recipes/tensorrt/conandata.yml`. Published releases provide wheels and SDK archives, without a source distribution. The manifest records:

- The 13 header filenames and individual SHA256 values from TensorRT OSS commit `1dade062a4e796c14ab6b3f32461ad694ec58951`.
- The exact NVIDIA package-index wheels `tensorrt_cu13_libs-11.2.1.2-py3-none-manylinux_2_28_x86_64.whl` and `tensorrt_cu13_libs-11.2.1.2-py3-none-win_amd64.whl`, with their SHA256 values.
- The SHA256 of each extracted runtime and builder resource library, including Linux's `tensorrt_libs/libnvinfer.so.11` and Windows' `tensorrt_libs/nvinfer_11.dll`.
- The OSS license URL and SHA256, and the NVIDIA license member in the verified wheel.

The recipe verifies every downloaded header and the complete wheel for the selected OS before extracting and verifying the libraries. On Windows it generates `nvinfer_11.lib` from the checked DLL's named exports using MSVC's x64 librarian. A failed download or digest check stops Conan without producing a package. The SDK contains the checked headers, runtime and builder resource libraries, link inputs, and licenses. These TensorRT files are build inputs and are not copied into tensoracle's wheel or prebuilt C++ SDK. The source distribution used by release builds includes the recipe, manifest, lock, profiles, and both license texts.

## Build tensoracle

Use the repository's supported C++ compiler. The shared Conan tool profile supplies CMake 3.31.12 and Ninja 1.13.2; neither tool needs a host installation. Install the Conan CLI version pinned in `python/pyproject.toml`, then prepare the native test dependencies from the repository root:

```bash
uv tool install "$(python3 -c 'import tomllib; from pathlib import Path; print(next(r for r in tomllib.loads(Path("python/pyproject.toml").read_text())["build-system"]["requires"] if r.startswith("conan==")))')"
python3 cpp/conan/prepare-tests.py
```

Preparation obtains the locked TensorRT and CUDA recipes and checks their revisions before resolving native dependencies. Miniz 3.1.1, nlohmann/json, and Catch2 come from ConanCenter. Release builds embed both recipes, the CUDA redist manifest, and the TensorRT license texts in their source distribution input. The TensorRT input wheel is a large download; Conan keeps the assembled SDK for subsequent builds. See [Building the C++ library](https://github.com/sync-dev-org/tensoracle/blob/main/cpp/README.md) for the standalone build.

The commands above use Bash on Linux or WSL2. For a native Windows build, follow
the cmd commands in the C++ README; its Windows profile selects MSVC's dynamic
Release runtime (`/MD`) and the Windows TensorRT and CUDA inputs.

For Python development, run from `python/`:

```bash
uv sync
TENSORACLE_CONAN_OFFLINE=ON uv run pytest
```

The default development group selects `tensoracle[tensorrt]`, so `uv sync` and `uv run pytest` keep the TensorRT runtime wheel available for tests. Use `uv sync --all-extras` when also testing checkpoint conversion and cubin production.

Isolated wheel builds install their Python build requirements and use the same Conan recipe and lock. TensorRT is not a Python build requirement. During stub generation, the build imports the extension with the exact Conan SDK library and records the loaded library's path and SHA256. The installed wheel contains no TensorRT library and requires its pip runtime only when the `tensorrt` extra is selected.

## Install the runtime

For native Windows Python, follow [Windows x86_64](#windows-x86_64). For native
C++ applications, also follow the [SDK runtime and prefix guide](cpp-sdk.md#install-the-host-runtime-and-relocate-a-prefix).

Linux and WSL2 applications use a TensorRT runtime installed by apt on their execution host. Follow [NVIDIA's Debian installation instructions](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-debian.html) to configure the CUDA network repository for your distribution, then run:

```bash
sudo apt-get update
sudo apt-get install libnvinfer11
pip install tensoracle
python -c 'from tensoracle import _native; print(_native.tensorrt_version())'
```

Use TensorRT 11.2 or later, below 12, including the builder resources needed for engine construction. The wheel is built with TensorRT SDK 11.2.1.2. `apt` resolves the runtime's CUDA library dependencies; a full CUDA Toolkit and TensorRT's Python bindings are not required to execute tensoracle. The host must provide a compatible NVIDIA driver.

To obtain TensorRT through pip instead, select the optional runtime:

```bash
pip install 'tensoracle[tensorrt]'
```

That extra requires `tensorrt-cu13>=11.2,<12`. On Linux, the wheel's relative RUNPATH checks the sibling `tensorrt_libs` directory when present. Without it, the host loader finds apt's `libnvinfer.so.11` through its normal configuration; no `LD_LIBRARY_PATH` is needed for a standard apt installation. On Windows, import `tensorrt` to establish the extra's DLL lookup before using tensoracle's native utilities or model bindings. Packages that use the host runtime declare `tensoracle` without an extra. Neither the tensoracle wheel nor a model wheel should copy TensorRT or the NVIDIA driver libraries.

## Use the recipe in a C++ consumer

Consumers need TensorRT headers and a link library in addition to the
[prebuilt tensoracle SDK](cpp-sdk.md). When your build inputs include the locked
TensorRT recipe, export it before resolving the consumer graph; its revision is
recorded in `cpp/conan.lock`. A consumer requires the SDK and links the CMakeDeps target:

```python
self.requires("tensorrt/11.2.1.2")
```

```cmake
find_package(TensorRT 11.2.1.2 EXACT CONFIG REQUIRED)
target_link_libraries(model PRIVATE TensorRT::nvinfer)
```

Use the generated Conan toolchain when configuring the consumer. The recipe supports Linux x86_64 and Windows x86_64 and does not compile TensorRT itself. Its Windows package generates the import library from the verified DLL. Applications use a separately installed host runtime for execution.

You can also use a separately verified TensorRT SDK with `TENSORRT_ROOT`;
consumers of the public tensoracle SDK archive do not need a tensoracle source
distribution. See [SDK linking](cpp-sdk.md#use-the-sdk-with-cmake).

The repository's [cubin plugin example](https://github.com/sync-dev-org/tensoracle/tree/main/examples/cubin_plugin) and [separate consumer wheel](https://github.com/sync-dev-org/tensoracle/tree/main/examples/wheel_consumer) use the prepared Conan toolchain when no `TENSORRT_ROOT` is supplied. Prepare the repository's dependencies, activate `conanbuild`, and pass the toolchain to their wheel builds as shown in their README files.

The existing `TENSORRT_ROOT` argument of `tensoracle_add_cubin_plugin` remains supported. A CMake consumer can derive that root from CMakeDeps. This example uses the checked-in `bundle/` and `plugin.cpp` beside `examples/cubin_plugin/CMakeLists.txt`:

```cmake
find_package(TensorRT 11.2.1.2 EXACT CONFIG REQUIRED)
list(GET TensorRT_INCLUDE_DIRS 0 tensorrt_include)
get_filename_component(TENSORRT_ROOT "${tensorrt_include}" DIRECTORY)
tensoracle_add_cubin_plugin(
    TARGET model_plugin
    TENSORRT_ROOT "${TENSORRT_ROOT}"
    BUNDLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bundle"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/plugin.cpp")
```

The CUDA helper reads `cuda.h` and the driver link stub from the selected Conan CUDA root. On Linux it prefers `lib64/stubs/libcuda.so`, then the other SDK library locations, and falls back to the installed driver (including `/usr/lib/wsl/lib` on WSL). The stub is only a link input; execution uses the host driver's `libcuda.so.1`. Do not package the stub or add its directory to the installed RUNPATH. It reports the selected headers and driver library during configuration. Existing explicitly supplied SDK roots remain supported by the public helper.

On Windows, the same helper selects `lib/x64/cuda.lib` under the Conan CUDA
root. This import library is a build input; execution uses the NVIDIA driver's
host-installed `nvcuda.dll`.

## CUDA build inputs

The same lock selects `cuda-toolkit/13.2.2` as a Conan `tool_requires` dependency, with `compiler=False`. The release build's source distribution input includes this recipe and its verified NVIDIA input manifest at `cpp/conan/recipes/cuda-toolkit/`. Native preparation downloads and checks the CUDA runtime headers, CRT headers, and platform's driver link inputs. The generated Conan toolchain sets `CUDAToolkit_ROOT` to the recipe's `user.cuda:root`; the CMake helpers use this root without importing a CUDA Python package.

If your build input includes that recipe directory, export it once before resolving a consumer graph:

```bash
conan export cpp/conan/recipes/cuda-toolkit --version=13.2.2
```

A consumer that prepares its own Conan toolchain can use the same recipe:

```python
from conan.tools.cmake import CMakeToolchain

# In build_requirements():
self.tool_requires("cuda-toolkit/13.2.2", options={"compiler": False})

# In generate():
toolchain = CMakeToolchain(self)
toolchain.variables["CUDAToolkit_ROOT"] = self.dependencies.build["cuda-toolkit"].conf_info.get("user.cuda:root")
toolchain.generate()
```

Configure the consumer with that generated toolchain. `tensoracle_find_cuda` and `tensoracle_add_cubin_plugin` then select the same headers and driver link library. Explicit `CUDAToolkit_ROOT` CMake variables, environment roots, and existing SDK layouts remain supported. The build dependencies do not change the PyPI tools used at runtime by `tensoracle[producer]`.

## Check the SDK

For a consumer configured with the Conan toolchain, derive the SDK root from `TensorRT_INCLUDE_DIRS` as shown above. The `NV_TENSORRT_MAJOR`, `MINOR`, `PATCH`, and `BUILD` macros in `NvInferVersion.h` must evaluate to **11, 2, 1, 2**. The header defines aliases, so use the C++ preprocessor to resolve them:

```bash
c++ -E -P -x c++ -I "$TENSORRT_ROOT/include" - <<'CPP'
#include "NvInferVersion.h"
NV_TENSORRT_MAJOR
NV_TENSORRT_MINOR
NV_TENSORRT_PATCH
NV_TENSORRT_BUILD
CPP
readelf -d "$TENSORRT_ROOT/lib/libnvinfer.so.11"
```

These commands use Linux tools. The preprocessor prints `11`, `2`, `1`, and `2` on separate lines; the library's SONAME is `libnvinfer.so.11`. The SONAME alone does not distinguish TensorRT 11 releases. On Windows, resolve the same macros with MSVC's preprocessor and inspect `bin/nvinfer_11.dll` with `dumpbin /headers` and `/exports`. The DLL name likewise identifies only the major version. The manifest's full digests establish which headers and library the build uses.

## Runtime and other releases

A Conan SDK package supplies build inputs. Native and Python applications install TensorRT 11.2 or later (below 12), including the builder resources needed for engine construction, on their execution host: use apt on Linux or NVIDIA's Windows runtime installation. Python applications can instead select `tensoracle[tensorrt]`. Validate engine construction, separate-process reuse, inference, and numerical output on an NVIDIA GPU and driver. Cubin generation and bundle inspection can run without a GPU.

This recipe fixes SDK 11.2.1.2. Releases before 11.2 are outside tensoracle's supported range and cannot be used with v2. Another SDK release in the supported range requires its own reviewed header commit, wheel filename and platform, all digests, and exact four-part header version. An explicit external SDK passed to the public plugin helper remains the caller's responsibility to verify and test.

## Windows x86_64

Use tensoracle v2.6.0 or later for the native Windows x86_64
`cp312-abi3-win_amd64` wheel. The package currently declares CPython 3.12 and
3.13 support. WSL2 uses the Linux wheel inside its Linux Python environment.
See [Migrate from 1.x](migration.md#check-the-platform-first) for model-package
changes, and the [prebuilt SDK guide](cpp-sdk.md) for native C++ applications.

Install a full TensorRT 11.2 or later runtime, below 12, on the Windows host
using [NVIDIA's installation instructions](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html).
Keep its builder resources and dependencies available alongside `nvinfer_11.dll`.
Install tensoracle without an extra when using this host runtime:

```powershell
pip install "tensoracle>=2.6.0"
```

Before importing native bindings in Python, register the actual TensorRT DLL
directory and retain the registration for as long as those bindings are used:

```python
import os

tensorrt_dll_directory = os.add_dll_directory(r"C:\path\to\TensorRT\bin")
import tensoracle
from tensoracle import _native

print(_native.tensorrt_version())
```

`import tensoracle` registers the wheel's own `tensoracle/cpp/bin` directory
before its native module is loaded. A model package must import `tensoracle`
before its own native binding and must not bundle another `tensoracle-2.dll`.
The TensorRT registration above is separate from tensoracle's registration.

To use the optional TensorRT wheel instead, install the extra and import
`tensorrt` before loading native bindings:

```powershell
pip install "tensoracle[tensorrt]>=2.6.0"
python -c "import tensorrt; import tensoracle; from tensoracle import _native; print(_native.tensorrt_version())"
python -c "import tensorrt; from tensoracle._cli import main; raise SystemExit(main())" cache list
```

CLI processes also need TensorRT's DLL lookup established; the last command
shows that setup for the extra. Neither installation route bundles TensorRT or
NVIDIA driver DLLs in tensoracle. A host TensorRT installation does not require
TensorRT's Python bindings or a full CUDA Toolkit for execution.
