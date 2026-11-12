# Prepare a TensorRT SDK

Tensoracle uses a locked Conan recipe for the Linux x86_64 TensorRT **11.2.1.2** SDK. It supplies 13 headers, the link library `lib/libnvinfer.so.11`, a relative `lib/libnvinfer.so` symlink, and the TensorRT OSS and NVIDIA license texts. WSL2 uses this Linux SDK. Tensoracle v2 currently supplies Linux wheels; [Windows x86_64 users](#windows-x86_64) stay on tensoracle 1.x until a later v2 minor release adds Windows wheels.

Native applications can obtain tensoracle itself from its [public prebuilt SDK archive](cpp-sdk.md),
verify that release's SHA256, and configure CMake with `CMAKE_PREFIX_PATH`.
The tensoracle Conan package recipe is maintained separately and is not included in this repository.

The TensorRT SDK is a build input. Tensoracle's standalone C++ build, Python wheel build, native test preparation, and source distribution build all use `cpp/conanfile.py` and `cpp/conan.lock`. They do not require `TENSORACLE_SDK_ROOT` or a manually assembled TensorRT installation. CUDA build headers and the driver link stub come from Conan `cuda-toolkit/13.2.2` with `compiler=False`; an OS-installed CUDA Toolkit is not required. The cubin producer continues to obtain its runtime compiler, ptxas, and cuobjdump from PyPI.

## Fixed and verified inputs

Source distributions include the exact locked recipe and its input manifest at `cpp/conan/recipes/tensorrt/conandata.yml`. The manifest records:

- The 13 header filenames and individual SHA256 values from TensorRT OSS commit `1dade062a4e796c14ab6b3f32461ad694ec58951`.
- The exact NVIDIA package-index wheel `tensorrt_cu13_libs-11.2.1.2-py3-none-manylinux_2_28_x86_64.whl` and its SHA256.
- The SHA256 of `tensorrt_libs/libnvinfer.so.11` extracted from that wheel.
- The OSS license URL and SHA256, and the NVIDIA license member in the verified wheel.

The recipe verifies every downloaded header and the complete wheel before extracting the library. It then verifies the extracted library. A failed download or digest check stops Conan without producing a package. It copies only the checked headers, link library, relative symlink, and licenses into the SDK; builder resources and the other TensorRT libraries are not packaged. The source distribution includes the recipe, manifest, lock, profiles, and both license texts.

## Build tensoracle

Use CMake 3.26 or later and the repository's supported C++ compiler. Install the Conan CLI version pinned in `python/pyproject.toml`, then prepare the native test dependencies from the repository root:

```bash
uv tool install "$(python3 -c 'import tomllib; from pathlib import Path; print(next(r for r in tomllib.loads(Path("python/pyproject.toml").read_text())["build-system"]["requires"] if r.startswith("conan==")))')"
cmake -P cpp/conan/prepare-tests.cmake
```

Preparation obtains the locked TensorRT and CUDA recipes and checks their revisions before resolving native dependencies. Miniz 3.1.1, nlohmann/json, and Catch2 come from ConanCenter. For a build without an existing recipe cache, use a source distribution, which includes both recipes, the CUDA redist manifest, and the TensorRT license texts. The SDK wheel is a large download; Conan keeps the assembled SDK for subsequent builds. See [Building the C++ library](https://github.com/sync-dev-org/tensoracle/blob/main/cpp/README.md) for the standalone build.

For Python development, run from `python/`:

```bash
uv sync
TENSORACLE_CONAN_OFFLINE=ON uv run pytest
```

Isolated wheel builds install their Python build requirements and use the same Conan recipe and lock. TensorRT is not a Python build requirement. During stub generation, the build imports the extension with the exact Conan SDK library and records the loaded library's path and SHA256. The installed wheel requires `tensorrt-cu13>=11.2,<12` and keeps its relative TensorRT RUNPATH; it contains no TensorRT library.

## Use the recipe in a C++ consumer

For consumers that need TensorRT headers and a link library in addition to the
[prebuilt tensoracle SDK](cpp-sdk.md), export `cpp/conan/recipes/tensorrt/` from an
extracted tensoracle source distribution once before resolving a consumer graph. The recipe revision is recorded in `cpp/conan.lock`. A consumer requires the SDK and links the CMakeDeps target:

```python
self.requires("tensorrt/11.2.1.2")
```

```cmake
find_package(TensorRT 11.2.1.2 EXACT CONFIG REQUIRED)
target_link_libraries(model PRIVATE TensorRT::nvinfer)
```

Use the generated Conan toolchain when configuring the consumer. The recipe supports Linux x86_64 and does not compile TensorRT. It provides build inputs rather than an installation of TensorRT's runtime and builder resources.

The repository's [cubin plugin example](https://github.com/sync-dev-org/tensoracle/tree/main/examples/cubin_plugin) and [separate consumer wheel](https://github.com/sync-dev-org/tensoracle/tree/main/examples/wheel_consumer) obtain this SDK automatically when no `TENSORRT_ROOT` is supplied. Prepare the repository's Conan dependencies before running their builds offline.

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

## CUDA build inputs

The same lock selects `cuda-toolkit/13.2.2` as a Conan `tool_requires` dependency, with `compiler=False`. A source distribution includes this recipe and its verified NVIDIA input manifest at `cpp/conan/recipes/cuda-toolkit/`. Native preparation downloads and checks the CUDA runtime headers, CRT headers, and driver link stub. The generated Conan toolchain sets `CUDAToolkit_ROOT` to the recipe's `user.cuda:root`; the CMake helpers use this root without importing a CUDA Python package.

From an extracted source distribution, export the CUDA recipe once before resolving a consumer graph:

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

The preprocessor prints `11`, `2`, `1`, and `2` on separate lines; the library's SONAME is `libnvinfer.so.11`. The SONAME alone does not distinguish TensorRT 11 releases. The manifest's full digests establish which headers and library the build uses.

## Runtime and other releases

A Conan SDK package supplies build inputs. Native applications install TensorRT 11.2 or later (below 12), including the builder resources needed for engine construction, on their execution host using NVIDIA's instructions. Python applications obtain the runtime through tensoracle's wheel dependency. Validate engine construction, separate-process reuse, inference, and numerical output on an NVIDIA GPU and driver. Cubin generation and bundle inspection can run without a GPU.

This recipe fixes SDK 11.2.1.2. Releases before 11.2 are outside tensoracle's supported range and cannot be used with v2. Another SDK release in the supported range requires its own reviewed header commit, wheel filename and platform, all digests, and exact four-part header version. An explicit external SDK passed to the public plugin helper remains the caller's responsibility to verify and test.

## Windows x86_64

v2 has no Windows wheel yet. Keep Windows Python applications on tensoracle 1.x until a later v2 minor release supplies one. Preparing Windows TensorRT headers does not make the Linux v2 wheel installable in Windows Python. WSL2 users can use the Linux recipe. See [Migrate from 1.x](migration.md#check-the-platform-first).
