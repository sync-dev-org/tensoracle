# Building tensoracle's C++ library

Applications can use the [prebuilt C and C++ SDK](../docs/cpp-sdk.md) from the public
GitHub Release: verify the archive's SHA256, extract it, and configure CMake with
`CMAKE_PREFIX_PATH` pointing to `tensoracle/cpp`. The tensoracle Conan package
recipe is maintained separately and is not included in this repository.
The following commands build tensoracle from source.

The default is a shared library (`BUILD_SHARED_LIBS=ON`). To embed tensoracle
statically, build from source with `-DBUILD_SHARED_LIBS=OFF` and install the result:

```bash
python3 cpp/conan/prepare.py --output build/cpp
. ./build/cpp/conanbuild.sh
cd cpp
cmake --preset conan-release -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF
cmake --build --preset conan-release
cmake --install ../build/cpp --prefix /path/to/prefix
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

C++ builds use GCC 13 on Linux x86_64 or MSVC 19.4 on Windows x86_64.
Conan supplies CMake 3.31.12, Ninja 1.13.2, and the pinned native dependencies.
CUDA headers and the driver link stub come from the locked
`cuda-toolkit/13.2.2` recipe (`compiler=False`), as described in the
[SDK guide](../docs/tensorrt-sdk.md).
Python bindings are built separately through `python/`. The supported TensorRT
runtime is 11.2 or later (below 12); the build SDK is pinned to 11.2.1.2.

Install the Conan CLI at the exact version declared by `python/pyproject.toml`:

```bash
uv tool install "$(python3 -c 'import tomllib; from pathlib import Path; print(next(r for r in tomllib.loads(Path("python/pyproject.toml").read_text())["build-system"]["requires"] if r.startswith("conan==")))')"
python3 cpp/conan/prepare-tests.py
python3 cpp/conan/prepare.py --output build/cpp --offline
. ./build/cpp/conanbuild.sh
cd cpp
cmake --preset conan-release
cmake --build --preset conan-release
ctest --preset conan-release --output-on-failure -LE gpu
```

The default output `build/cpp` uses `conan-release`. Other output directories
use `conan-<directory-id>-release`, where the identifier is derived from the
complete resolved output path. `prepare.py` prints the exact preset name.
All outputs remain included, so switching output directories leaves earlier
presets usable, including directories with the same final component.
From the repository root, for example:

```bash
python3 cpp/conan/prepare.py --output build/first --offline
python3 cpp/conan/prepare.py --output build/second --offline
```

Activate the chosen output's `conanbuild.sh` (or `call conanbuild.bat` in cmd),
then run configure, build, and test from `cpp/` using the name printed for that
output in place of `conan-release`. `cmake --list-presets` lists all prepared
outputs. Reusing the same output preserves its preset name.

Run these commands from the repository root. Test configuration uses prepared
recipes and binaries with no remotes, package builds, or recipe exports. Prepare
again after changing a recipe, profile, or lock. Without prepared dependencies,
the preparation script fails and prints the preparation command. To build only the library,
pass `-DBUILD_TESTING=OFF` when configuring. CMake never installs dependencies.

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

A source build using `add_subdirectory(cpp)` or `conan create` uses the caller's
Conan graph and toolchain. Declare tensoracle's dependencies in that graph;
tensoracle does not run another preparation or replace the caller's toolchain.

`cpp/conanfile.py` and `cpp/conan.lock` select nlohmann/json 3.11.3, miniz 3.1.1,
Catch2 3.16.1, TensorRT SDK 11.2.1.2, and CUDA build inputs 13.2.2. Miniz, nlohmann/json, and Catch2
come from ConanCenter. Source distributions include the locked TensorRT recipe,
verified input manifest, and licenses in `cpp/conan/recipes/tensorrt/`, plus the
CUDA recipe and manifest in `cpp/conan/recipes/cuda-toolkit/`. The generated
toolchain supplies `CUDAToolkit_ROOT`; an explicit existing CUDA root is still accepted. TensorRT headers
and the link library come from the same digest-verified recipe for standalone,
wheel, and test builds; no `TENSORACLE_SDK_ROOT` or `TENSORRT_ROOT` is required. The default profile is `linux-gcc-toolset-13`; release builds select
`linux-gcc-toolset-14` for the manylinux image's compiler. Linux profiles select
C++17 and `libstdc++11`; the Windows profile selects
MSVC's dynamic Release runtime. Linux and wheel builds default to eight jobs;
the Windows profile retains its four-job default. Set `CMAKE_BUILD_PARALLEL_LEVEL`
to limit a run, and use `-Cbuild.tool-args=-j4` or `SKBUILD_BUILD_TOOL_ARGS=-j4`
when limiting Python builds. The preparation scripts apply the CMake job limit
to Conan as well. The common
`cpp/conan/profiles/tools` selects CMake and Ninja for consumers and dependencies;
`cpp/conan.lock` records their recipe revisions. Host installations of CMake and
Ninja are unnecessary. Preparation requires their prebuilt ConanCenter binaries
and never builds these tools from source.

For native Windows, run these commands from cmd after installing the pinned Conan CLI:

```bat
python cpp\conan\prepare-tests.py
python cpp\conan\prepare.py --output build\cpp --offline
call build\cpp\conanbuild.bat
if errorlevel 1 exit /b %errorlevel%
cd cpp
cmake --preset conan-release -DTENSORACLE_TEST_TENSORRT_RUNTIME=C:/path/to/TensorRT/lib
cmake --build --preset conan-release
ctest --preset conan-release --output-on-failure -LE gpu
```

When placing these commands in a batch file, stop after each command that fails.
The build environment includes MSVC and Windows SDK settings. CTest propagates the
selected full TensorRT runtime and the tensoracle DLL directory to child processes.
A standalone C++ build needs no Python environment after dependency preparation;
a runtime directory can be passed explicitly. Windows runtime DLLs can also be
discovered under `python/.venv/Lib/site-packages/tensorrt_libs`.

Set `CONAN_HOME` to isolate Conan's metadata and `TENSORACLE_CONAN_CACHE` to isolate
package storage. Use the same values for preparation, native builds, and wheel
builds. `TENSORACLE_CONAN_PROFILE` selects a profile by name or absolute path;
`TENSORACLE_CONAN_BUILD_PROFILE` can select a separate build profile.
The wheel backend prepares its own toolchain and tools before dependency-discovery,
metadata, wheel, or editable hooks, and leaves native user presets unchanged.
Run Python tests after preparation with `TENSORACLE_CONAN_OFFLINE=ON`; native
fixtures always use this mode. Conan is a build dependency: installed CMake
consumers can use the prebuilt SDK archive without Conan or its targets.

nanobind stays in Python's build requirements. C++ build inputs come from Conan;
the CUDA compiler and tools used at runtime by the cubin producer stay in PyPI
wheels. The release profile uses the CUDA toolkit in the manylinux image through
the same recipe's `system` mode; development uses verified NVIDIA redist archives. Installed Python wheels without the
`tensorrt` extra use the host's TensorRT runtime (installed with apt on Linux);
`tensoracle[tensorrt]` selects the optional TensorRT wheel. Native applications
install TensorRT and its builder resources on their execution host.
