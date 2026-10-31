# Build a C++ and Python repository

Use this guide to organize a repository with a standalone C++ library and a Python package, then build and inspect its Linux wheels. Keep the C++ build independent of Python, put the Python project in its own directory, and make the package build collect the files it needs from those sources. The binding and wheel examples below illustrate a package named `example`; adapt the target names and dependencies to your project.

## 1. Separate the language projects

Keep `cpp/` and `python/` at the repository root:

```text
README.md
LICENSE
docs/
cpp/
  CMakeLists.txt
  include/
  src/
  tests/
  bindings.cpp          # Optional Python binding source
python/
  pyproject.toml
  uv.lock
  CMakeLists.txt        # Only when the Python build compiles an extension
  src/
    example/
      __init__.py
  tests/
```

`cpp/` owns the C++ sources, headers, CMake targets, and CTest tests. Its standalone configuration builds and tests the C++ library without discovering Python or nanobind. The Python build can compile `bindings.cpp` separately and link it to that library.

`python/` is the Python project root. It owns package metadata, runtime dependencies, development dependencies, the lockfile, and the src-layout package. Run the project's uv commands there. Keep repository-wide documentation, the license, and examples at the repository root; keep `pyproject.toml`, Python tests, and the Python environment under `python/`.

Keep one editable copy of each source file. Configure the package build to collect C++ sources or compiled libraries into the package as appropriate. A build that uses sibling `cpp/` files must also make those files available when building from a source distribution: stage a complete source tree or arrange an equivalent archive layout, then verify a build from the extracted archive.

Tensoracle is an example of source packaging: its wheel includes the shared C++ headers, sources, and CMake helpers under `tensoracle/cpp/`, and the caller compiles its own plugin. Tensoracle's wheel contains no native binaries and uses `py3-none-any`. A repository that builds a Python extension needs the additional steps in sections 4–6.

## 2. Keep development tools in the appropriate environment

Create the Python environment at `python/.venv`, alongside `python/pyproject.toml` and `python/uv.lock`; keep the repository root free of a second `.venv`. uv uses a project-local [virtual environment](https://docs.astral.sh/uv/concepts/projects/layout/#the-project-environment). For this layout, start from the repository root and run:

```bash
cd python
uv sync
```

Add `.venv` (without a trailing slash) to `python/.gitignore`. A rule in the repository root's `.gitignore` is not enough: some build backends, such as scikit-build-core, read ignore files only from the project directory and the directories beneath it that they reach without following symbolic links, while they follow symbolic links when collecting files. When `python/.venv` is a symbolic link to an environment elsewhere, the `.gitignore` that uv writes inside the environment is not read, and an sdist build walks every file in the environment. Without the trailing slash, the rule matches `.venv` whether it is a directory or a symbolic link.

Declare Python development tools such as pytest, Ruff, and a type checker in the Python project's development dependency group. Keep runtime dependencies in the `[project]` table's `dependencies` field, optional runtime features in extras, and isolated build requirements in the `[build-system]` table's `requires` field.

Keep one CMake and one Ninja installation on the host's `PATH`, shared by standalone C++ builds and tests that invoke CMake. Do not add `cmake` or `ninja` to each project's Python development dependency group. Their installation method can follow the developer's platform and package manager.

Each CMake entry point declares its minimum supported version with [`cmake_minimum_required`](https://cmake.org/cmake/help/latest/command/cmake_minimum_required.html). The host installation must meet all projects' minimums. Tensoracle's standalone C++ project requires CMake 3.20; the binding example below uses CMake 3.26 for `Development.SABIModule`. Verify affected C++ builds before upgrading a shared host installation.

An isolated wheel build has separate requirements. Keep any CMake requirement it needs in `[build-system] requires`; that environment does not replace the host tools. [scikit-build-core](https://scikit-build-core.readthedocs.io/en/stable/configuration/index.html#cmake-and-ninja-minimum-versions) can obtain CMake or Ninja when the available versions do not satisfy its configuration.

## 3. Test each language and identify GPU tests

Run pytest from `python/` so that the Python project's configuration and environment apply:

```bash
uv run pytest
uv run pytest -m "not gpu"
```

Register a `gpu` marker in the pytest configuration and apply `@pytest.mark.gpu` to tests that need a GPU. Use strict marker checking to catch misspelled markers. The second command selects tests that can run without a GPU; those tests must also avoid GPU initialization during collection. See [pytest's marker documentation](https://docs.pytest.org/en/stable/how-to/mark.html).

Register the standalone C++ tests with CTest, using `include(CTest)` and `add_test`. From the repository root, configure and run them without activating the Python environment:

```bash
cmake -S cpp -B build/cpp -G Ninja -DBUILD_TESTING=ON
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure
ctest --test-dir build/cpp --output-on-failure -LE gpu
```

Supply any required SDK or toolchain paths at configuration time. Python independence does not remove C++ library, header, or CUDA toolkit prerequisites. For TensorRT plugins, follow [Prepare a TensorRT SDK](tensorrt-sdk.md).

Give GPU-dependent CTest tests a [`gpu` label](https://cmake.org/cmake/help/latest/prop_test/LABELS.html); the final command excludes that label. A machine without a CUDA toolkit may also need a configuration that omits GPU targets. Test C++ library behavior through CTest and Python imports, bindings, and integration through pytest. Before distributing a wheel, install it into a clean environment and test the installed package outside the source tree.

## 4. Install nanobind extensions inside the package

Keep the standalone library's CMake configuration independent of the Python bindings. The Python project's CMake configuration discovers Python and nanobind, adds the C++ library, and installs the extension into the Python package. For a C++ target named `example_core`, these fragments illustrate that arrangement.

In `python/pyproject.toml`, alongside the project's metadata:

```toml
[build-system]
requires = ["scikit-build-core", "nanobind"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
wheel.packages = ["src/example"]
wheel.py-api = "cp312"
```

In `python/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.26)
project(example_python LANGUAGES CXX)

find_package(Python 3.12 REQUIRED COMPONENTS
    Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(../cpp cpp)
set_target_properties(example_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
nanobind_add_module(_native STABLE_ABI ../cpp/bindings.cpp)
target_link_libraries(_native PRIVATE example_core)
install(TARGETS _native LIBRARY DESTINATION example)
```

The binding source declares `NB_MODULE(_native, m)`, and `python/src/example/__init__.py` imports from `._native`. The installed wheel contains `example/__init__.py` and the extension, such as `example/_native.abi3.so`. The snippets assume that `cpp/CMakeLists.txt` defines `example_core` and that the package build supplies the sibling C++ sources as described in section 1. See [nanobind packaging](https://nanobind.readthedocs.io/en/latest/packaging.html) for the binding build and installation mechanism.

For nanobind's linked build mode, [`STABLE_ABI`](https://nanobind.readthedocs.io/en/latest/api_cmake.html#command:nanobind_add_module) requires Python 3.12 or later. Pair it with `wheel.py-api = "cp312"` so that the binary and wheel metadata agree. This produces a CPython stable-ABI extension for compatible GIL-enabled Python versions from that floor, subject to the package's `requires-python`. It does not remove platform, C++ ABI, or runtime library constraints. Free-threaded Python needs its own build and compatibility decisions.

Read the three parts of the [wheel compatibility tag](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/) separately:

| Wheel contents | Example Python–ABI–platform tag |
| --- | --- |
| Python and source files, without native binaries | `py3-none-any` |
| Native libraries that do not use the Python C API | `py3-none-manylinux_2_28_x86_64` |
| A CPython stable-ABI extension with a Python 3.12 floor | `cp312-abi3-manylinux_2_28_x86_64` |
| A CPython 3.12 extension using its version-specific ABI | `cp312-cp312-manylinux_2_28_x86_64` |

These are illustrative targets, not tags to assign before inspecting the binaries. `none` describes the Python ABI; a wheel carrying a shared library still needs a platform tag. `abi3` retains a minimum Python version and does not mean every Python implementation or build is supported.

## 5. Check manylinux compatibility before uploading to PyPI

For glibc-based Linux x86_64 wheels, use a supported manylinux platform tag. PyPI's [platform validation](https://github.com/pypi/warehouse/blob/main/warehouse/utils/wheel.py) rejects the generic `linux_x86_64` tag. Under [PEP 600](https://peps.python.org/pep-0600/), `manylinux_2_28_x86_64` promises compatibility with mainstream x86_64 Linux systems using glibc 2.28 or later. It does not describe CUDA or GPU driver requirements; document those separately.

The actual minimum follows the symbols and libraries used by the built artifacts. Linking on a newer system can introduce newer glibc requirements even when the source code has not changed. The host's glibc version alone is not a measurement of the wheel's minimum, and changing the filename cannot lower that minimum. Inspect required `GLIBC_*` versions, along with `GLIBCXX_*` and `CXXABI_*` requirements for the C++ runtime.

Run [`auditwheel show`](https://github.com/pypa/auditwheel) in the build container. Extract the wheel and inspect its ELF files with [`readelf`](https://sourceware.org/binutils/docs/binutils/readelf.html). For example, after setting `WHEEL` to the exact wheel path and choosing a new extraction directory:

```bash
auditwheel show "$WHEEL"
python -m zipfile -e "$WHEEL" wheel-inspect
LC_ALL=C readelf --version-info --wide wheel-inspect/example/_native.abi3.so
LC_ALL=C readelf --dynamic --wide wheel-inspect/example/_native.abi3.so
```

Repeat the ELF checks for every shared library and executable in the wheel, including nested dependencies. Inspect the version requirements and `NEEDED`, `RPATH`, and `RUNPATH` entries. Use `LC_ALL=C` when tools or scripts rely on English output headings, and compare numeric version components rather than sorting version strings lexically. Rebuild in an appropriate environment if a required symbol exceeds the intended baseline; auditwheel cannot repair an overly new glibc or libstdc++ dependency.

### Shared libraries and search paths

Assign each shipped shared library to one package. If another wheel supplies a library, declare that package as a runtime dependency and arrange a documented runtime lookup path. Do not copy its library into the consuming wheel. RAPIDS publishes separate [Python binding](https://pypi.org/project/pylibcudf-cu12/) and [C++ library](https://pypi.org/project/libcudf-cu12/) distributions as `pylibcudf-cu12` and `libcudf-cu12`; their release files illustrate the distinction between Python ABI tags and platform tags.

`auditwheel repair` normally copies non-policy shared libraries into the wheel and changes their search paths. Review that list before repair, and exclude libraries owned by dependency wheels and libraries provided by the GPU driver. An exclusion does not establish manylinux compliance. Resolve and test those dependencies separately, and inspect the final wheel again. Also test libraries loaded through `dlopen` or `ctypes`, which auditwheel cannot discover from `NEEDED` entries alone.

Use `$ORIGIN`-relative ELF search paths, where `$ORIGIN` is the directory containing the referencing binary. For an extension whose own package contains a `lib/` subdirectory, a CMake target can use:

```cmake
set_target_properties(_native PROPERTIES
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "$ORIGIN/lib")
```

[`INSTALL_RPATH`](https://cmake.org/cmake/help/latest/prop_tgt/INSTALL_RPATH.html) controls the installed artifact; `BUILD_RPATH_USE_ORIGIN` applies to build-tree paths. Select the relative path for the actual installed layout, including dependency packages where needed, and check every library's own dependencies. Remove absolute build-machine paths, including paths introduced by imported CMake targets. Keep `$ORIGIN` literal when passing a value through a shell.

Keep NVIDIA driver libraries such as `libcuda.so.1`, `libnvidia-encode.so.1`, and `libnvcuvid.so.1` outside the wheel. The target system supplies the compatible driver. Link-time SDK stubs must also stay out of the runtime package and its search paths; NVIDIA's [Video Codec SDK instructions](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/read-me/index.html) distinguish these stubs from the installed driver. Treat CUDA runtime libraries separately, using an appropriate static runtime or declared runtime package dependencies for the chosen toolkit.

Check artifact size before upload. PyPI's [default individual file limit](https://docs.pypi.org/project-management/storage-limits/) is 100 MB; projects can have an approved increase. Installing and testing the final wheel with only its declared dependencies is still necessary after tag and ELF inspection.

## 6. Record the wheel build environment

Build release wheels in a [manylinux container](https://github.com/pypa/manylinux) for the intended architecture and compatibility floor. For example, `quay.io/pypa/manylinux_2_28_x86_64` provides a glibc 2.28 baseline. Select compiler, SDK, CUDA toolkit, and dependency versions that can build and run against that baseline. A recent compiler does not require building against a recent host glibc.

Pin the image by an immutable [image digest](https://docs.docker.com/reference/cli/docker/image/pull/#pull-an-image-by-digest-immutable-identifier), using `image@sha256:<digest>` with the actual digest of the selected image. A moving tag is useful for discovering updates but does not identify a repeatable build input. Pin any derived image as well, and record additional package installation inputs.

Keep a build record with:

- The source revision, patches, and source manifest.
- The image reference, manifest digest, image configuration digest, and architecture.
- The compiler, linker, CMake, Ninja, Python interpreter, and build backend versions.
- The CUDA toolkit and SDK versions, when used, and relevant compiler or CMake options.
- The resolved build and runtime dependencies, resulting wheel hashes, ELF inspection results, and test results.

Build from clean inputs inside that container, inspect the final artifacts there, and test their installation in a separate runtime environment. When native C++ objects cross extension boundaries, align the C++ library ABI, compiler compatibility, debug mode, symbol visibility, and nanobind ABI. Python's stable ABI does not make incompatible C++ objects interchangeable. [nanobind's compatibility requirements](https://nanobind.readthedocs.io/en/latest/faq.html#how-can-i-avoid-conflicts-with-other-projects-using-nanobind) describe those constraints.

## 7. Record and reconstruct upstream C++ sources

When development needs a local copy of an upstream C++ project, copy the selected files into a dedicated directory and keep the source inputs explicit. Record the upstream URL and immutable revision, the selected paths, the ordered local patches, and the license information. Keep a SHA-256 manifest covering every file in the resulting source tree, including files added by patches.

For example, a provenance directory can contain:

```text
upstream/library/
  PROVENANCE.md
  patches/
  MANIFEST.sha256
  fetch-and-verify.py
```

`PROVENANCE.md` identifies the upstream revision and explains the selection and patch order. `MANIFEST.sha256` records the expected relative paths and hashes after patching. The acquisition script reads these recorded inputs, retrieves the exact revision, selects the required files, applies the patches, and verifies the resulting tree. Keep the revision in one authoritative location that the script reads.

Use ordinary copied files for the development tree, rather than a Git submodule. A submodule records a commit reference instead of ordinary source files, so a publication process that copies tracked files into a snapshot does not materialize that source tree. An explicit acquisition script also makes the build independent of recursive Git checkout behavior.

Keep the copied upstream source tree out of the public repository. Publish the revision record, patches, complete file manifest, acquisition and verification script, and instructions for building and testing from the reconstructed sources. Preserve required license notices in the appropriate distributed artifacts. The public checkout must let another developer reconstruct the inputs and build the native components without access to the development copy.

Verify both file contents and the complete set of paths: missing, changed, and unexpected files must fail verification. A checksum command that only checks listed files will not detect extra files by itself. Verify before configuring or compiling, and stop on a download, patch, or manifest failure. When updating upstream, regenerate the manifest from the reviewed source selection and rerun the native build and tests. If publishing a source distribution, ensure its documented acquisition process makes the same verified inputs available for its build.
