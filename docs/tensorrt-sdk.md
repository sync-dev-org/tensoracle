# Prepare a TensorRT SDK

Use the Linux procedure below to prepare the headers and `libnvinfer` needed to build a native cubin plugin on Linux x86_64, including WSL2. For Windows x86_64, use the [Windows procedure](#windows-x86_64). The Linux SDK root contains 13 headers in `include/`, `lib/libnvinfer.so.11`, and a relative `lib/libnvinfer.so` symlink pointing to `libnvinfer.so.11`. CMake, a C++17 compiler, and the CUDA headers from PyPI are separate build prerequisites; no OS-installed CUDA Toolkit is required. TensorRT headers are absent from the Python wheels and must be prepared separately. Normal engine reuse does not need this SDK root.

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

## 5. Pass the verified root to the build and tests

Keep `TENSORRT_ROOT` set to the absolute path of the verified `sdk` directory. From the tensoracle repository root, install the producer extra into the development environment and locate its PyPI CUDA root. The lock selects `nvidia-cuda-nvcc` and `nvidia-cuda-runtime` 13.2.86. For a separate build environment, install `nvidia-cuda-nvcc==13.2.*` and `nvidia-cuda-runtime==13.2.*` with that environment's package manager and locate `nvidia/cu13` in the same way.

```bash
uv sync --project python --extra producer
export CUDAToolkit_ROOT="$(uv run --project python --extra producer python -c 'from importlib.metadata import distribution; print(distribution("nvidia-cuda-runtime").locate_file("nvidia/cu13"))')"
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release \
  "-DTENSORRT_ROOT=$TENSORRT_ROOT" "-DCUDAToolkit_ROOT=$CUDAToolkit_ROOT"
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure
uv build examples/cubin_plugin --wheel \
  "--config-setting=cmake.define.TENSORRT_ROOT=$TENSORRT_ROOT" \
  "--config-setting=cmake.define.CUDAToolkit_ROOT=$CUDAToolkit_ROOT"
TENSORACLE_SDK_ROOT="$TENSORRT_ROOT" TENSORACLE_CUDA_ROOT="$CUDAToolkit_ROOT" \
  uv run --project python --extra producer pytest -q python/tests/test_example_plugin.py
```

The native helper reads `cuda.h` from this root and links the installed NVIDIA driver's `libcuda.so`; on WSL it searches `/usr/lib/wsl/lib`. It prints the selected headers and driver library during configuration. `CUDAToolkit_ROOT` can also be passed through the environment. Existing OS Toolkit roots and the default search through `nvcc` on `PATH` and `/usr/local/cuda` remain supported. Tests marked `gpu` require an NVIDIA GPU and driver. The SDK is a build input. The application still selects the TensorRT runtime and verifies its own plugin with that runtime.

## Use another TensorRT 11 release

Use the same acquisition and verification steps for another TensorRT 11 release, but first establish its release commit, exact wheel filename and platform, expected wheel and library digests, and expected digests of every required header. Check that release's exact `NV_TENSORRT_MAJOR`, `MINOR`, `PATCH`, and `BUILD` values and library SONAME. The caller must verify and record those inputs; the 11.2.1.2 values above apply only to that SDK and cannot verify another release. Build and test the caller's plugin with the selected SDK and runtime.

## Windows x86_64

Use NVIDIA's official Windows ZIP to obtain both the headers and the MSVC import library. The TensorRT runtime wheel supplies DLLs, but it does not replace the SDK's `include/` directory and `lib/nvinfer_11.lib`. See NVIDIA's [Windows ZIP installation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-zip.html) and [TensorRT 11 downloads](https://developer.nvidia.com/tensorrt/download/11x).

The following inputs are fixed for **11.4.0.106**, Windows amd64, CUDA 13.4. Use Visual Studio 2022 with MSVC, CMake, and Ninja available in an **x64 Native Tools Command Prompt for VS 2022**, then start PowerShell from that prompt. The CUDA build inputs below come from PyPI's 13.2 wheels; an OS-installed CUDA Toolkit is not needed. tensoracle's Python runtime dependencies select the CUDA 13 TensorRT runtime separately. Do not combine Windows import libraries with Linux shared libraries.

Choose a new workspace. The command creating it must fail if the directory already exists. Download the fixed ZIP, record its digest, and extract it without installing software system-wide:

```powershell
$ErrorActionPreference = 'Stop'
$SdkWork = Join-Path $env:USERPROFILE '.cache/tensoracle-sdk-11.4.0.106-windows'
New-Item -ItemType Directory -Path $SdkWork | Out-Null
Set-Location $SdkWork
$ZipName = 'TensorRT-Enterprise-11.4.0.106-Windows-amd64-cuda-13.4-Release-external.zip'
$Url = 'https://developer.download.nvidia.com/compute/machine-learning/tensorrt/11.4.0/zip/' + $ZipName
Invoke-WebRequest -Uri $Url -OutFile $ZipName
Get-FileHash -Algorithm SHA256 $ZipName | Format-List | Out-File archive-sha256.txt
Expand-Archive -LiteralPath $ZipName -DestinationPath extracted
$env:TENSORRT_ROOT = Join-Path $SdkWork 'extracted/TensorRT-11.4.0.106'
```

Verify the extracted build inputs against these SHA-256 values. These values identify the SDK members needed by the build, including the nested Python plugin header; they are not a digest of the whole ZIP. Stop on a mismatch rather than using an incomplete or mixed SDK:

```powershell
$Expected = @'
64a59d5bfdb1dd0b7a6a98992d3b617d926b760a6134c09a8bec0e62227ff5ff lib/nvinfer_11.lib
886f693c1d958651eeee58ed4d02c0f31da355c112102539d6bd4d567f56a3a4 include/NvOnnxConfig.h
81fb4be15a2432bf45c858176e347e017a8f5d9a20d82225141cd263a5114c32 include/NvInfer.h
870a47268d1d5e08dcb72b14157292a16ba541cc9d95327a860977aba11c759a include/NvInferPluginBase.h
e5558eb96fc2b08a964f73ca096213a2bab9f356a897122f2c0c56a2a6553315 include/NvInferImpl.h
7629d76e5a6bc1ee910ab90c117cdfa1f350349321e211cc9003dbad3f03b5ef include/NvInferRuntimePlugin.h
7141cc8fc7d0274ba76c085a395b2be195b68d2ef205a381ddd41f8d0b323299 include/NvInferRuntime.h
e8c04d3cd04fd3e6abdb95a3398d2bee5c171f86eb772b3b8f30f66d21006e9a include/NvInferRuntimeBase.h
8b155e73287ca63e276fc55297ae5b8723888d925afbf63a392e908faede8e02 include/NvInferPluginUtils.h
c28e2c4cbd8f1f0ed7ea6a4fe002906b92916abd3b778e71344649e522509b39 include/NvInferVersion.h
6c30ffcd9c688bd286362aae3e00c0369961b78e7b0bffd08cae739f5d780ca1 include/NvInferRuntimeCommon.h
3ab7107afc15e4317a17a57b31d7d42d677091380918bfc2dc45eb4ab38570ad include/NvInferPlugin.h
440c0ea9cdfc8940d598daff766bc6db098b83d4250f7404c0930dd013ecd0eb include/NvOnnxParser.h
99a80d98a7cdcac515d64a5f48dafe2a6d4f920c4513ec3f2b07bacf7c4ab09f include/NvInferLegacyDims.h
2ee14ba7c2cf595af1d8f3ad33bc340fbadaa03a8e1bcea1bbfbe18235d965e7 include/impl/NvInferPythonPlugin.h
'@
$Expected -split "`n" | ForEach-Object {
    $Digest, $Relative = $_.Trim() -split ' ', 2
    $File = Join-Path $env:TENSORRT_ROOT $Relative
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $File).Hash -ne $Digest) {
        throw "SDK checksum differs: $Relative"
    }
}
$Expected | Set-Content -Encoding ascii SHA256SUMS
```

Check the exact header version using MSVC's preprocessor. It must print `11`, `4`, `0`, and `106`, in that order; the import library's filename identifies only the major version:

```powershell
@'
#include "NvInferVersion.h"
NV_TENSORRT_MAJOR
NV_TENSORRT_MINOR
NV_TENSORRT_PATCH
NV_TENSORRT_BUILD
'@ | Set-Content -Encoding ascii version.cpp
cl /nologo /EP /I "$env:TENSORRT_ROOT/include" version.cpp
if ($LASTEXITCODE) { throw 'SDK version check failed' }
```

From the tensoracle repository root, configure and test the standalone C++ project, then build the example plugin wheel with the same verified SDK:

```powershell
$env:TENSORACLE_SDK_ROOT = $env:TENSORRT_ROOT
uv sync --project python
if ($LASTEXITCODE) { throw 'Development environment setup failed' }
uv pip install --python python/.venv/Scripts/python.exe `
    'nvidia-cuda-nvcc==13.2.*' 'nvidia-cuda-runtime==13.2.*'
if ($LASTEXITCODE) { throw 'PyPI CUDA installation failed' }
$env:CUDAToolkit_ROOT = (& python/.venv/Scripts/python.exe -c 'from importlib.metadata import distribution; print(distribution("nvidia-cuda-runtime").locate_file("nvidia/cu13"))')
if ($LASTEXITCODE) { throw 'PyPI CUDA root lookup failed' }
$env:TENSORACLE_CUDA_ROOT = $env:CUDAToolkit_ROOT
cmake -S cpp -B build/windows -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DTENSORRT_ROOT=$env:TENSORRT_ROOT" "-DCUDAToolkit_ROOT=$env:TENSORACLE_CUDA_ROOT"
if ($LASTEXITCODE) { throw 'CMake configuration failed' }
cmake --build build/windows --config Release
if ($LASTEXITCODE) { throw 'C++ build failed' }
ctest --test-dir build/windows -C Release --output-on-failure
if ($LASTEXITCODE) { throw 'CTest failed' }
uv build examples/cubin_plugin --wheel `
    "--config-setting=cmake.define.TENSORRT_ROOT=$env:TENSORRT_ROOT" `
    "--config-setting=cmake.define.CUDAToolkit_ROOT=$env:TENSORACLE_CUDA_ROOT"
if ($LASTEXITCODE) { throw 'Example wheel build failed' }
uv run --project python pytest python/tests/test_example_plugin.py
if ($LASTEXITCODE) { throw 'Example plugin tests failed' }
```

The example build uses `cpp/cmake/TensoracleCubinPlugin.cmake`, which finds `nvinfer_11.lib` in the SDK's `lib/` directory, `cuda.h` in the PyPI root's `include/`, and the CUDA driver import library at `lib/x64/cuda.lib` in that root. The built plugin is a DLL. Keep its TensorRT runtime available through the application's installed Python dependencies; the SDK directory is a build input and does not need to be added to the runtime `PATH`. For another TensorRT 11 release, establish new expected digests and check that release's exact header version before building.
