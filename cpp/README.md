# Building tensoracle's C++ library

Applications can use the [prebuilt C and C++ SDK](../docs/cpp-sdk.md) from the public
GitHub Release: verify the archive's SHA256, extract it, and configure CMake with
`CMAKE_PREFIX_PATH` pointing to `tensoracle/cpp`. The tensoracle Conan package
recipe is maintained separately and is not included in this repository.
The following commands build tensoracle from source.

The default is a shared library (`BUILD_SHARED_LIBS=ON`). To embed tensoracle
statically, build from source with `-DBUILD_SHARED_LIBS=OFF` and install the result:

```bash
cmake -S cpp -B build/static -G Ninja -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF
cmake --build build/static --parallel 8
cmake --install build/static --prefix /path/to/prefix
```

Both forms provide the same headers and CMake target, `tensoracle::tensoracle`.
The static archive contains only tensoracle's objects. Its installed CMake
package resolves miniz, the CUDA driver link stub, TensorRT, and the platform
thread and dynamic-loader libraries in the consumer's build environment. Use
the prepared Conan toolchain for those dependencies, as described in the
[SDK guide](../docs/cpp-sdk.md#embed-tensoracle-from-source). The static target
supplies `TENSORACLE_STATIC_LIBRARY` so public declarations do not import or
export a DLL. The standalone `tensoracle_cubin` archive and source-based cubin
plugin helpers remain available in source builds.

Release SDK archives and Python wheels contain the shared library only; static
libraries are not distributed. A source-building Conan recipe selects static
embedding with `from_source=True` and `shared=False`. TensorRT stays a shared
library supplied by the execution host. The current library uses the CUDA
Driver API and does not require the CUDA runtime.

A process must contain one tensoracle implementation shared by all its code.
If an application or library embeds tensoracle, that artifact must be the only
owner in the process: do not load another artifact embedding tensoracle or a
tensoracle shared library there. Python processes use the wheel's shared
library. Relink an artifact that embeds tensoracle whenever its embedded
tensoracle version is updated.

C++ builds require CMake 3.26 or later, Ninja, GCC 13 on Linux x86_64, the pinned
Conan dependencies. CUDA headers and the driver link stub come from the locked
`cuda-toolkit/13.2.2` recipe (`compiler=False`), as described in the
[SDK guide](../docs/tensorrt-sdk.md).
Python bindings are built separately through `python/`. The supported TensorRT
runtime is 11.2 or later (below 12); the build SDK is pinned to 11.2.1.2.

Install the Conan CLI at the exact version declared by `python/pyproject.toml`:

```bash
uv tool install "$(python3 -c 'import tomllib; from pathlib import Path; print(next(r for r in tomllib.loads(Path("python/pyproject.toml").read_text())["build-system"]["requires"] if r.startswith("conan==")))')"
cmake -P cpp/conan/prepare-tests.cmake
cmake -S cpp -B .nf/tmp/cpp -G Ninja
cmake --build .nf/tmp/cpp --parallel 8
ctest --test-dir .nf/tmp/cpp --output-on-failure
```

Run these commands from the repository root. Test configuration uses prepared
recipes and binaries with no remotes, package builds, or recipe exports. Prepare
again after changing a recipe, profile, or lock. Without prepared dependencies,
configuration fails and prints the preparation command. To build only the library,
pass `-DBUILD_TESTING=OFF`; dependencies can then be obtained during configuration.

CTest needs a full TensorRT runtime, including its builder resources, separately
from the Conan SDK used for compilation and linking. By default, test configuration
looks for `tensorrt_libs` under `python/.venv/lib/python*/site-packages/`, as installed
by running `uv sync` from `python/`. To use a separate runtime without a Python
environment, add `-DTENSORACLE_TEST_TENSORRT_RUNTIME=/path/to/tensorrt/lib` to the
configure command. The directory must contain `libnvinfer.so.11` and the TensorRT
builder resource libraries. Configuration stops with this option in its diagnostic
if no full runtime is available. CTest supplies the selected directory through
`LD_LIBRARY_PATH` to the tests and their child processes; no manual environment
setting is needed. Library builds with `BUILD_TESTING=OFF` do not require a test
runtime.

The C++ cubin producer test uses the CUDA compiler tools from `python/.venv`
when the producer extra is installed. Its compilation still uses the Conan
CUDA headers. Without those runtime tools, that optional test is skipped.

A source build using `add_subdirectory(cpp)` includes `cpp/conan/bootstrap.cmake`
in its top-level `CMakeLists.txt` before the first `project()`, so the parent
project also uses the generated compiler settings and dependency search paths.

`cpp/conanfile.py` and `cpp/conan.lock` select nlohmann/json 3.11.3, miniz 3.1.1,
Catch2 3.16.1, TensorRT SDK 11.2.1.2, and CUDA build inputs 13.2.2. Miniz, nlohmann/json, and Catch2
come from ConanCenter. Source distributions include the locked TensorRT recipe,
verified input manifest, and licenses in `cpp/conan/recipes/tensorrt/`, plus the
CUDA recipe and manifest in `cpp/conan/recipes/cuda-toolkit/`. The generated
toolchain supplies `CUDAToolkit_ROOT`; an explicit existing CUDA root is still accepted. TensorRT headers
and the link library come from the same digest-verified recipe for standalone,
wheel, and test builds; no `TENSORACLE_SDK_ROOT` or `TENSORRT_ROOT` is required. The default profile is `linux-gcc13`; release builds select
`linux-gcc14` for the manylinux image's compiler. Each profile selects C++17,
`libstdc++11`, and at most eight build jobs. CMake is supplied by the build
environment, including isolated Python builds.
CMake's tool requirement is represented as `cmake/system` in Conan, so the lock
does not depend on the installed CMake version. Each CMake entry point checks
the minimum version. Conan does not download or build a CMake package.

Set `CONAN_HOME` to isolate Conan's metadata. To isolate its package storage, pass
`-DTENSORACLE_CONAN_CACHE=/absolute/package/directory` to both preparation and
configuration. The equivalent environment variable also applies to Python test
fixtures. Python's editable build accepts
`-Ccmake.define.TENSORACLE_CONAN_CACHE=/absolute/package/directory`.
Run Python tests after preparation with `TENSORACLE_CONAN_OFFLINE=ON`; native
fixtures always use this mode. Conan is a build dependency: installed CMake
consumers can use the prebuilt SDK archive without Conan or its targets.

nanobind stays in Python's build requirements. C++ build inputs come from Conan;
the CUDA compiler and tools used at runtime by the cubin producer stay in PyPI
wheels. The release profile uses the CUDA toolkit in the manylinux image through
the same recipe's `system` mode; development uses verified NVIDIA redist archives. Installed Python wheels resolve
TensorRT through their existing runtime wheel dependency. Native applications
install TensorRT and its builder resources on their execution host.
