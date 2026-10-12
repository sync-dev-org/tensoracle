"""Build small TensorRT graphs from safetensors without torch."""

import re
from importlib.metadata import requires
from pathlib import Path

import numpy as np
import pytest
import tensorrt as trt
from cuda.bindings import runtime as cudart
from safetensors.numpy import load_file, save_file

from tensoracle import BuilderSettings, EngineBuild, InputShape, build_serialized_engine


@pytest.mark.req("SP-TENSORACLE-005")
def test_package_requires_torch_only_for_checkpoint_conversion():
    """Only the conversion extra declares torch, leaving inference dependencies unchanged."""
    dependencies = requires("tensoracle") or []
    torch_dependencies = [
        item for item in dependencies if re.match(r"torch(?:[<>=;\[]|\s|$)", item.lower())
    ]
    assert len(torch_dependencies) == 1
    marker = torch_dependencies[0].partition(";")[2].strip()
    assert re.fullmatch(r"extra\s*==\s*['\"]convert['\"]", marker)


def _check_cuda(result):
    status, *values = result
    assert status == cudart.cudaError_t.cudaSuccess
    return values[0] if values else None


def _weights(path: Path) -> None:
    save_file(
        {
            "first.weight": np.array([[1.0, 0.5], [-0.25, 1.5]], dtype=np.float32),
            "first.bias": np.array([0.5, -0.25], dtype=np.float32),
            "second.weight": np.array([[0.75, -0.5], [0.25, 1.25]], dtype=np.float32),
            "second.bias": np.array([-0.125, 0.375], dtype=np.float32),
        },
        str(path),
    )


def _graph(path: Path, precision: str, input_shape: tuple[int, ...] = (1, -1, 2)):
    def define(network):
        weights = load_file(path)
        dtype = np.float16 if precision == "fp16" else np.float32
        trt_dtype = trt.float16 if precision == "fp16" else trt.float32
        value = network.add_input("features", trt_dtype, input_shape)
        for prefix in ("first", "second"):
            matrix = np.ascontiguousarray(weights[f"{prefix}.weight"][None], dtype=dtype)
            bias = np.ascontiguousarray(weights[f"{prefix}.bias"][None, None], dtype=dtype)
            constant = network.add_constant(matrix.shape, matrix).get_output(0)
            value = network.add_matrix_multiply(
                value, trt.MatrixOperation.NONE, constant, trt.MatrixOperation.NONE
            ).get_output(0)
            value = network.add_elementwise(
                value,
                network.add_constant(bias.shape, bias).get_output(0),
                trt.ElementWiseOperation.SUM,
            ).get_output(0)
        value.name = "result"
        network.mark_output(value)

    return define


def _expected(value: np.ndarray, path: Path) -> np.ndarray:
    weights = load_file(path)
    for prefix in ("first", "second"):
        value = value @ weights[f"{prefix}.weight"] + weights[f"{prefix}.bias"]
    return value


def _run(plan: bytes, value: np.ndarray) -> np.ndarray:
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context.set_input_shape("features", value.shape)
    output = np.empty(tuple(context.get_tensor_shape("result")), dtype=value.dtype)
    input_ptr = _check_cuda(cudart.cudaMalloc(value.nbytes))
    output_ptr = _check_cuda(cudart.cudaMalloc(output.nbytes))
    stream = _check_cuda(cudart.cudaStreamCreate())
    try:
        _check_cuda(
            cudart.cudaMemcpy(
                input_ptr,
                value.ctypes.data,
                value.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        assert context.set_tensor_address("features", int(input_ptr))
        assert context.set_tensor_address("result", int(output_ptr))
        assert context.execute_async_v3(int(stream))
        _check_cuda(cudart.cudaStreamSynchronize(stream))
        _check_cuda(
            cudart.cudaMemcpy(
                output.ctypes.data,
                output_ptr,
                output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
        )
        return output
    finally:
        _check_cuda(cudart.cudaStreamDestroy(stream))
        _check_cuda(cudart.cudaFree(output_ptr))
        _check_cuda(cudart.cudaFree(input_ptr))


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-007")
def test_fixed_shape_build_runs_at_its_declared_size(tmp_path):
    """A fixed input shape produces a plan that computes the declared graph."""
    path = tmp_path / "weights.safetensors"
    _weights(path)
    declaration = EngineBuild(
        precision="fp32",
        inputs={"features": InputShape.fixed((1, 3, 2))},
        settings=BuilderSettings(optimization_level=0),
    )
    plan = build_serialized_engine(_graph(path, "fp32", (1, 3, 2)), declaration)
    assert plan
    value = np.arange(6, dtype=np.float32).reshape(1, 3, 2) / 4
    np.testing.assert_allclose(_run(plan, value), _expected(value, path), rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-007")
def test_range_build_runs_at_minimum_optimum_and_maximum_and_rejects_outside(tmp_path):
    """A ranged input accepts its three declared sizes and rejects an outside size."""
    path = tmp_path / "weights.safetensors"
    _weights(path)
    declaration = EngineBuild(
        precision="fp32",
        inputs={"features": InputShape.ranged((1, 1, 2), (1, 3, 2), (1, 5, 2))},
        settings=BuilderSettings(optimization_level=0),
    )
    plan = build_serialized_engine(_graph(path, "fp32"), declaration)
    for length in (1, 3, 5):
        value = np.arange(length * 2, dtype=np.float32).reshape(1, length, 2) / 4
        np.testing.assert_allclose(_run(plan, value), _expected(value, path), rtol=1e-5, atol=1e-5)

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    assert not context.set_input_shape("features", (1, 6, 2))


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-013")
def test_fp16_build_runs_a_graph_with_fp16_tensors(tmp_path):
    """A declared fp16 plan runs the graph's fp16 tensors."""
    path = tmp_path / "weights.safetensors"
    _weights(path)
    declaration = EngineBuild(
        precision="fp16",
        inputs={"features": InputShape.ranged((1, 1, 2), (1, 2, 2), (1, 3, 2))},
        settings=BuilderSettings(optimization_level=0),
    )
    plan = build_serialized_engine(_graph(path, "fp16"), declaration)
    value = np.arange(4, dtype=np.float16).reshape(1, 2, 2) / 4
    expected = _expected(value.astype(np.float32), path)
    np.testing.assert_allclose(_run(plan, value), expected, rtol=2e-3, atol=2e-3)


@pytest.mark.req("SP-TENSORACLE-013")
def test_tf32_request_is_rejected_before_graph_is_built():
    """An explicit request to enable TF32 fails without invoking the graph."""
    declaration = EngineBuild(
        precision="fp32",
        inputs={"features": InputShape.fixed((1, 3, 2))},
        settings=BuilderSettings(tf32=True),
    )
    with pytest.raises(ValueError, match="TF32"):
        build_serialized_engine(lambda network: pytest.fail("graph was built"), declaration)


@pytest.mark.req("SP-TENSORACLE-013")
def test_tf32_is_disabled_and_other_builder_settings_are_applied(monkeypatch):
    """The builder clears TF32 while applying the declared workspace and optimization level."""
    import tensoracle._build as module

    class Config:
        builder_optimization_level = None

        def __init__(self):
            self.flags = {trt.BuilderFlag.TF32}
            self.workspace = None

        def clear_flag(self, flag):
            self.flags.discard(flag)

        def set_memory_pool_limit(self, pool, limit):
            self.workspace = (pool, limit)

    class Network:
        num_inputs = 1
        num_outputs = 0

        def get_input(self, index):
            return type("Input", (), {"name": "features", "shape": (1, 3, 2)})()

    class Builder:
        def __init__(self, logger):
            self.config = Config()
            observed.append(self)

        def create_network(self, flags):
            return Network()

        def create_builder_config(self):
            return self.config

        def build_serialized_network(self, network, config):
            return b"plan"

    observed = []
    monkeypatch.setattr(module.trt, "Builder", Builder)
    declaration = EngineBuild(
        precision="fp32",
        inputs={"features": InputShape.fixed((1, 3, 2))},
        settings=BuilderSettings(workspace_limit=4096, optimization_level=2),
    )
    assert build_serialized_engine(lambda network: None, declaration) == b"plan"
    config = observed[0].config
    assert trt.BuilderFlag.TF32 not in config.flags
    assert config.workspace == (trt.MemoryPoolType.WORKSPACE, 4096)
    assert config.builder_optimization_level == 2
