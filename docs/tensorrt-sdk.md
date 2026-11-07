# Prepare a TensorRT SDK

Use the Linux procedure below to prepare the headers and `libnvinfer` needed to build a C++ model package or native cubin plugin on Linux x86_64, including WSL2. v2 supplies only a Linux wheel; [Windows x86_64 users](#windows-x86_64) must stay on tensoracle 1.x until a later v2 minor release adds a Windows wheel. The Linux SDK root contains 13 headers in `include/`, `lib/libnvinfer.so.11`, and a relative `lib/libnvinfer.so` symlink pointing to `libnvinfer.so.11`. CMake (3.26 or later for the v2 SDK), a C++17 compiler, and the CUDA headers from PyPI are separate build prerequisites; no OS-installed CUDA Toolkit is required. TensorRT headers are absent from the Python wheels and must be prepared separately. Normal engine reuse does not need this SDK root.

The inputs below are fixed for SDK **11.2.1.2**: headers from a release commit of [NVIDIA TensorRT OSS](https://github.com/NVIDIA/TensorRT), and the exact `tensorrt-cu13-libs` wheel from [NVIDIA's package index](https://pypi.nvidia.com/tensorrt-cu13-libs/). The wheel platform is `manylinux_2_28_x86_64`; use that file rather than an aarch64 or Windows wheel.

## 1. Choose a workspace and record the inputs

Use Bash with `curl`, `unzip`, `sha256sum`, `readelf` (from binutils), and `c++` available. Choose a new workspace outside the repository. The following example keeps the inputs and SDK under the user cache. `mkdir "$SDK_WORK"` fails if that workspace already exists; choose a new workspace or verify the existing files instead of overwriting them. Stop if any command or verification fails, and do not pass an unverified SDK to CMake.

```bash
set -e
SDK_WORK="${XDG_CACHE_HOME:-$HOME/.cache}/tensoracle-sdk-11.2.1.2"
mkdir -p "$(dirname "$SDK_WORK")"
mkdir "$SDK_WORK"
cd "$SDK_WORK"
export TENSORRT_ROOT="$SDK_WORK/sdk"
OSS_COMMIT="1dade062a4e796c14ab6b3f32461ad694ec58951"
WHEEL_NAME="tensorrt_cu13_libs-11.2.1.2-py3-none-manylinux_2_28_x86_64.whl"
mkdir -p sdk/include sdk/lib
```

Record the expected SHA-256 values in the workspace. The first entry checks the wheel, the second checks the extracted library, and the remaining 13 entries check every header. The shell expands `$WHEEL_NAME` to the exact filename selected above.

```bash
cat > SHA256SUMS <<EOF
c2b7c6c3972b986bc66b219aff79cc4de80c845c188c38d4eacd8a0643854753  $WHEEL_NAME
45c9e379255ba2b36da064fd35272ea0212eb67c4fe6bdbfb0641afddc1e9d1a  sdk/lib/libnvinfer.so.11
6b4068af7ff651a800be9907544a1766283dfcf080295a3b09feb5f1ffa614ac  sdk/include/NvInfer.h
d68f32455b5d869123b4e936243abe45a89ac489e35bb1c6aef47ac970b71872  sdk/include/NvInferImpl.h
3e10fe5b85545f2e956e6935271badf0b1c72553b698000a61d46d69c5d53030  sdk/include/NvInferLegacyDims.h
b3290404a076f3492dc627ffdfc3b1fa803ea0eccc49f23986d645a2d23a63c7  sdk/include/NvInferPlugin.h
759f354754e64bf4d1e68d7df796c155c46f8a27541bf1912ba13e79a5a26fc2  sdk/include/NvInferPluginBase.h
106692e986b690fd2b3c26f8dab5caa8027b4d3cca6ce6006c5351392f1bded4  sdk/include/NvInferPluginUtils.h
dd795beae7f612066897f8c3db2314fdfee219e7efa67e0a5781d1061dae3643  sdk/include/NvInferRuntime.h
ec38759dbf8a5538c4cd30890f1764658bc2dcec87e95ac18148e476cbe1671b  sdk/include/NvInferRuntimeBase.h
eff4c0251cb4813bbc7fe34c6b28fdb6c08734c27ae53b07974d2557bf1431be  sdk/include/NvInferRuntimeCommon.h
d4db4236da9c03e543949247a75478b5ce68fc84eb30cd15b35081b94c8d8391  sdk/include/NvInferRuntimePlugin.h
69aa045e8fd3b6a82efbf3f0be7de84816ba8a8151473afe64a0406df719b981  sdk/include/NvInferVersion.h
a36f12c8115152f4cec674ab414a3029fcde9871b6ccf33d4f95fb49f1f4fc7a  sdk/include/NvOnnxConfig.h
1443c9411e030ac35b8a549392745731515485367579ace586156806ece11d60  sdk/include/NvOnnxParser.h
EOF
```

## 2. Download and verify the wheel

Download the selected file from NVIDIA's index without installing it. Verify its SHA-256 before extracting any library:

```bash
curl --fail --location \
  "https://pypi.nvidia.com/tensorrt-cu13-libs/$WHEEL_NAME" \
  --output "$WHEEL_NAME"
head -n 1 SHA256SUMS | sha256sum --check
```

The check must report the wheel as `OK`. A different wheel version or platform requires its own expected digest.

## 3. Assemble and verify the SDK files

Fetch each listed header from `include/` at the fixed OSS commit. Extract only `tensorrt_libs/libnvinfer.so.11` from the verified wheel, then check all 15 digests:

```bash
while read -r expected relative_path; do
  case "$relative_path" in
    sdk/include/*)
      header="${relative_path#sdk/include/}"
      curl --fail --location \
        "https://raw.githubusercontent.com/NVIDIA/TensorRT/$OSS_COMMIT/include/$header" \
        --output "$relative_path"
      ;;
  esac
done < SHA256SUMS
unzip -p "$WHEEL_NAME" tensorrt_libs/libnvinfer.so.11 > sdk/lib/libnvinfer.so.11
sha256sum --check SHA256SUMS
ln -s libnvinfer.so.11 sdk/lib/libnvinfer.so
```

Every checksum must report `OK`. Keep `libnvinfer.so.11` as a regular file and use the relative symlink shown above, so the linker name resolves inside the SDK root. If a download, extraction, or check fails, discard the incomplete SDK and fetch the fixed inputs again before building.

## 4. Check the exact header version and library SONAME

The `NV_TENSORRT_MAJOR`, `NV_TENSORRT_MINOR`, `NV_TENSORRT_PATCH`, and `NV_TENSORRT_BUILD` macros in `NvInferVersion.h` must evaluate to **11, 2, 1, 2**, respectively. These headers define the `NV_TENSORRT_*` values through `TRT_*_ENTERPRISE` aliases; use the C++ preprocessor to resolve them:

```bash
c++ -E -P -x c++ -I "$TENSORRT_ROOT/include" - <<'CPP'
#include "NvInferVersion.h"
NV_TENSORRT_MAJOR
NV_TENSORRT_MINOR
NV_TENSORRT_PATCH
NV_TENSORRT_BUILD
CPP
readelf -d "$TENSORRT_ROOT/lib/libnvinfer.so.11" | grep '(SONAME)'
```

The preprocessor must print four lines containing `11`, `2`, `1`, and `2` in that order. The SONAME entry must contain `[libnvinfer.so.11]`.

The SONAME identifies only the major version. Matching `libnvinfer.so.11` does not distinguish different TensorRT 11 releases or establish that the headers and library belong together. Check the exact four-part header version and the expected SHA-256 of the wheel, extracted library, and each header to detect a mixture of releases sharing the same major version.

## 5. Pass the verified root to the model build

Keep `TENSORRT_ROOT` set to the absolute path of the verified `sdk` directory. Install the CUDA headers in the model package's build environment and locate their root without importing tensoracle:

```bash
python -m pip install 'nvidia-cuda-nvcc==13.2.*' 'nvidia-cuda-runtime==13.2.*'
export CUDAToolkit_ROOT="$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-cuda-runtime").locate_file("nvidia/cu13"))')"
export TENSORACLE_CPP_ROOT="$(python -c 'from importlib.metadata import distribution; print(distribution("tensoracle").locate_file("tensoracle/cpp"))')"
```

The installed tensoracle package must be v2 for the shared C++ SDK. Add `$TENSORACLE_CPP_ROOT` to `CMAKE_PREFIX_PATH` and link `tensoracle::tensoracle`. The target defining the network needs the SDK's `include/` directory. The [separate consumer wheel](https://github.com/sync-dev-org/tensoracle/tree/main/examples/wheel_consumer) reads this root from `TENSORACLE_SDK_ROOT`:

```bash
export TENSORACLE_SDK_ROOT="$TENSORRT_ROOT"
```

For cubin plugin builds, pass the verified SDK as `TENSORRT_ROOT` and PyPI CUDA root as `CUDAToolkit_ROOT` to `tensoracle_add_cubin_plugin`, as shown in the [CMake helper guide](cpp-python-repositories.md). The helper reads `cuda.h` from that CUDA root and links the installed NVIDIA driver's `libcuda.so`; on WSL it searches `/usr/lib/wsl/lib`. It prints the selected headers and driver library during configuration. Existing OS Toolkit roots and default locations remain supported, but an OS-installed CUDA Toolkit is not required.

The SDK is a build input. At runtime, tensoracle resolves TensorRT through its installed wheel dependency. Test the model's engine construction, separate-process reuse, inference, and numerical outputs through its own binding with an NVIDIA GPU and driver. Cubin generation and bundle inspection can run without a GPU.

## Use another TensorRT 11 release

Use the same acquisition and verification steps for another TensorRT 11 release, but first establish its release commit, exact wheel filename and platform, expected wheel and library digests, and expected digests of every required header. Check that release's exact `NV_TENSORRT_MAJOR`, `MINOR`, `PATCH`, and `BUILD` values and library SONAME. The caller must verify and record those inputs; the 11.2.1.2 values above apply only to that SDK and cannot verify another release. Build and test the caller's plugin with the selected SDK and runtime.

## Windows x86_64

v2 has no Windows wheel yet. Keep Windows Python applications on tensoracle 1.x until a later v2 minor release supplies one. Preparing Windows TensorRT headers does not make the Linux v2 wheel installable in Windows Python. WSL2 users can follow the Linux SDK procedure inside their Linux environment. See [Migrate from 1.x](migration.md#check-the-platform-first) for platform and package requirements.
