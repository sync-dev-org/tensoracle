# Building tensoracle's C++ library

C++ builds require CMake 3.26 or later, Ninja, GCC 13 on Linux x86_64, the pinned
Conan dependencies, and CUDA headers and the driver link library. CUDA inputs
can come from the PyPI wheels described in the [SDK guide](../docs/tensorrt-sdk.md).
Python bindings are built separately through `python/`. The supported TensorRT
runtime is 11.2 or later (below 12); the build SDK is pinned to 11.2.1.2.

Install the Conan CLI at the exact version declared by `python/pyproject.toml`:

```bash
uv tool install "$(python3 -c 'import tomllib; from pathlib import Path; print(next(r for r in tomllib.loads(Path("python/pyproject.toml").read_text())["build-system"]["requires"] if r.startswith("conan==")))')"
cmake -P cpp/conan/prepare-tests.cmake
cmake -S cpp -B .nf/tmp/cpp -G Ninja \
  -DCUDAToolkit_ROOT=/path/to/site-packages/nvidia/cu13
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

A source build using `add_subdirectory(cpp)` includes `cpp/conan/bootstrap.cmake`
in its top-level `CMakeLists.txt` before the first `project()`, so the parent
project also uses the generated compiler settings and dependency search paths.

`cpp/conanfile.py` and `cpp/conan.lock` select nlohmann/json 3.11.3, miniz 3.0.2,
Catch2 3.16.1, and TensorRT SDK 11.2.1.2. The local recipes and verified input
information are in `cpp/conan/miniz/` and `cpp/conan/tensorrt/`. TensorRT headers
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
consumers use the wheel's SDK without Conan or its targets.

nanobind stays in Python's build requirements, CUDA components stay in the PyPI
wheels, and the build uses the Conan TensorRT SDK. Installed Python wheels resolve
TensorRT through their existing runtime wheel dependency. Native applications
install TensorRT and its builder resources on their execution host.
