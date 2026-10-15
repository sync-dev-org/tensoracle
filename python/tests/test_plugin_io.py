"""Native cubin plugins declare tensor types, formats, and their output shape source."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorrt as trt
from cuda.bindings import runtime as cudart

from tensoracle import BuilderSettings, EngineBuild, InputShape, build_serialized_engine

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent if PROJECT.name == "python" else PROJECT
FIXTURES = PROJECT / "tests/cubin_io"


@pytest.fixture(scope="module")
def native_tools():
    sdk = os.environ.get("TENSORACLE_SDK_ROOT")
    cuda = os.environ.get("TENSORACLE_CUDA_ROOT")
    if not sdk or not cuda:
        pytest.skip("TensorRT SDK and CUDA toolkit were not supplied")
    return Path(sdk), Path(cuda)


def _compile(source, output, native_tools, *options):
    sdk, cuda = native_tools
    result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wno-deprecated-declarations",
            f"-I{ROOT / 'cpp/include'}",
            f"-I{sdk / 'include'}",
            f"-I{cuda / 'include'}",
            str(source),
            str(ROOT / "cpp/src/cubin_plugin.cpp"),
            f"-L{cuda / 'lib64/stubs'}",
            "-lcuda",
            *options,
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def native_contract(native_tools, tmp_path_factory):
    work = tmp_path_factory.mktemp("plugin-io-contract")
    executable = work / "check"
    _compile(FIXTURES / "check.cpp", executable, native_tools)
    return executable


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.parametrize("case", ["formats", "counts", "types", "shapes", "defaults", "invalid"])
def test_native_plugin_checks_declared_io(native_contract, case):
    """Plugins enforce each tensor's declared format, type, count, and shape validation."""
    subprocess.run([str(native_contract), case], check=True, timeout=10)


def _cuda(result):
    status, *values = result
    assert status == cudart.cudaError_t.cudaSuccess, status
    return values[0] if values else None


@pytest.fixture(scope="module")
def half_plugin(native_tools, tmp_path_factory):
    _, cuda = native_tools
    work = tmp_path_factory.mktemp("plugin-half-io")
    device = _cuda(cudart.cudaGetDevice())
    properties = _cuda(cudart.cudaGetDeviceProperties(device))
    sm = properties.major * 10 + properties.minor
    cubin = work / "half_io.cubin"
    subprocess.run(
        [
            str(cuda / "bin/nvcc"),
            "--cubin",
            f"-arch=sm_{sm}",
            str(FIXTURES / "kernels.cu"),
            "-o",
            str(cubin),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    (work / "kernel_bytes.hpp").write_text(
        "alignas(16) unsigned char const kKernelBytes[] = {"
        + ",".join(str(byte) for byte in cubin.read_bytes())
        + "};\n",
        encoding="utf-8",
    )
    plugin = work / "libhalf_io.so"
    _compile(FIXTURES / "plugin.cpp", plugin, native_tools, "-shared", "-fPIC", f"-I{work}")
    return plugin


def _plan(plugin, kind, shapes, *, dtype=trt.float16):
    def graph(network):
        creator = trt.get_plugin_registry().get_creator(
            f"TensoracleHalf{kind}", "1", "tensoracle.tests.io"
        )
        assert creator is not None
        native = creator.create_plugin("io", trt.PluginFieldCollection([]), trt.TensorRTPhase.BUILD)
        assert native is not None
        inputs = [network.add_input(name, dtype, shape) for name, shape in shapes.items()]
        layer = network.add_plugin_v3(inputs, [], native)
        if layer is None:
            raise RuntimeError("TensorRT rejected the plugin IO declaration")
        result = layer.get_output(0)
        result.name = "result"
        network.mark_output(result)

    return build_serialized_engine(
        graph,
        EngineBuild(
            {name: InputShape.fixed(shape) for name, shape in shapes.items()},
            BuilderSettings(workspace_limit=16 << 20, optimization_level=0),
        ),
        plugins=(plugin,),
    )


def _execute(plan, inputs):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    try:
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        context = engine.create_execution_context()
        assert context is not None
        output = np.empty(tuple(context.get_tensor_shape("result")), dtype=np.float16)
        buffers = {}
        stream = _cuda(cudart.cudaStreamCreate())
        try:
            for name, value in {**inputs, "result": output}.items():
                # Network I/O is LINEAR; TensorRT reformats to the plugin's HWC8 declaration.
                assert engine.get_tensor_format(name) == trt.TensorFormat.LINEAR
                buffers[name] = _cuda(cudart.cudaMalloc(value.nbytes))
                assert context.set_tensor_address(name, int(buffers[name]))
                if name in inputs:
                    _cuda(
                        cudart.cudaMemcpy(
                            buffers[name],
                            value.ctypes.data,
                            value.nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        )
                    )
            assert context.execute_async_v3(int(stream))
            _cuda(cudart.cudaStreamSynchronize(stream))
            _cuda(
                cudart.cudaMemcpy(
                    output.ctypes.data,
                    buffers["result"],
                    output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )
            return output
        finally:
            _cuda(cudart.cudaStreamSynchronize(stream))
            del context, engine
            for pointer in buffers.values():
                _cuda(cudart.cudaFree(pointer))
            _cuda(cudart.cudaStreamDestroy(stream))
    finally:
        del runtime


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.parametrize("kind", ["Unary", "Residual"])
def test_half_hwc8_plugin_matches_numpy(half_plugin, kind, tmp_path, monkeypatch):
    """One-input and two-input fp16 HWC8 plugins execute and match NumPy's output."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    shape = (1, 48, 3, 5)
    x = (np.arange(np.prod(shape)).reshape(shape) % 31 / 8).astype(np.float16)
    inputs = {"x": x}
    expected = x + x
    if kind == "Residual":
        residual = (np.flip(x, axis=1) / 2).astype(np.float16).copy()
        inputs["residual"] = residual
        expected = x + residual
    plan = _plan(half_plugin, kind, {name: value.shape for name, value in inputs.items()})
    np.testing.assert_array_equal(_execute(plan, inputs), expected)


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-004")
def test_plugin_output_shape_follows_selected_input(half_plugin, tmp_path, monkeypatch):
    """A plugin can copy input one's shape when input zero has a different shape."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    inputs = {
        "x": np.zeros((1, 48, 1, 1), dtype=np.float16),
        "residual": np.arange(720, dtype=np.float16).reshape(1, 48, 3, 5),
    }
    plan = _plan(half_plugin, "Select", {name: value.shape for name, value in inputs.items()})
    np.testing.assert_array_equal(_execute(plan, inputs), inputs["residual"])


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.parametrize("fault", ["missing-input", "extra-input", "shape", "type"])
def test_plugin_rejects_network_outside_declaration(half_plugin, fault, tmp_path):
    """A network with wrong input count, differing residual shapes, or wrong types fails."""
    environment = dict(os.environ, XDG_CACHE_HOME=str(tmp_path / "cache"))
    # Keep TensorRT's error state and any failed native graph in its own process.
    result = subprocess.run(
        [sys.executable, str(PROJECT / "tests/plugin_io_runtime.py"), str(half_plugin), fault],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "declaration rejected"
