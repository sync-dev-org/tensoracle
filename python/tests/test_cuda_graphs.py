"""Capture one enqueue and replay it with caller-owned CUDA graphs."""

import json
from contextlib import ExitStack, contextmanager

import numpy as np
import pytest
import tensorrt as trt
from cuda.bindings import runtime as cudart

import tensoracle
from test_build import _check_cuda
from test_execution import _open_sum

pytest_plugins = ("test_example_plugin",)


@contextmanager
def _captured(handle, batches, *, shapes):
    """Keep one execution, its addresses, and its captured graph alive together."""
    with handle.create_execution() as execution, ExitStack() as resources:
        execution.set_input_shapes(shapes)
        output = np.empty(
            execution.get_tensor_shape("result"),
            dtype=trt.nptype(execution.get_tensor_dtype("result")),
        )
        stream = _check_cuda(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
        resources.callback(lambda: _check_cuda(cudart.cudaStreamDestroy(stream)))
        addresses = {}
        for name, value in {**batches[0], "result": output}.items():
            pointer = _check_cuda(cudart.cudaMalloc(value.nbytes))
            resources.callback(lambda pointer=pointer: _check_cuda(cudart.cudaFree(pointer)))
            addresses[name] = int(pointer)

        def copy_inputs(batch):
            for name, value in batch.items():
                _check_cuda(
                    cudart.cudaMemcpyAsync(
                        addresses[name],
                        value.ctypes.data,
                        value.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        stream,
                    )
                )

        def read_output():
            _check_cuda(
                cudart.cudaMemcpyAsync(
                    output.ctypes.data,
                    addresses["result"],
                    output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
            )
            _check_cuda(cudart.cudaStreamSynchronize(stream))
            return output.copy()

        try:
            # The first ordinary enqueue flushes deferred shape and plugin initialization.
            ordinary = []
            for batch in batches:
                copy_inputs(batch)
                execution.enqueue(addresses, stream=int(stream))
                ordinary.append(read_output())

            _check_cuda(
                cudart.cudaStreamBeginCapture(
                    stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
                )
            )
            try:
                execution.enqueue(addresses, stream=int(stream))
            finally:
                graph = _check_cuda(cudart.cudaStreamEndCapture(stream))
            resources.callback(lambda: _check_cuda(cudart.cudaGraphDestroy(graph)))
            instance = _check_cuda(cudart.cudaGraphInstantiate(graph, 0))
            resources.callback(lambda: _check_cuda(cudart.cudaGraphExecDestroy(instance)))

            def replay(index):
                copy_inputs(batches[index])
                # An empty graph or a launch writing to the wrong address must fail.
                _check_cuda(
                    cudart.cudaMemsetAsync(addresses["result"], 0xFF, output.nbytes, stream)
                )
                _check_cuda(cudart.cudaGraphLaunch(instance, stream))
                actual = read_output()
                np.testing.assert_array_equal(actual, ordinary[index])
                return actual

            yield replay
        finally:
            # Finish work before graphs, buffers, streams, and the execution are released.
            _check_cuda(cudart.cudaStreamSynchronize(stream))


def _sum_batches(rows):
    return [
        {
            "features": np.arange(rows * 2, dtype=np.float32).reshape(rows, 2) / 4 + offset,
            "extra": np.full((rows, 2), -0.25 - offset / 2, dtype=np.float32),
        }
        for offset in (0, 2)
    ]


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-028")
def test_standard_engine_graph_replay_matches_ordinary_enqueue(tmp_path, monkeypatch):
    """One captured enqueue on a nondefault stream reproduces ordinary sums for new input data."""
    handle = _open_sum(tmp_path, monkeypatch, fixed=True)
    batches = _sum_batches(3)
    with _captured(handle, batches, shapes={}) as replay:
        for index in (0, 1, 0):
            np.testing.assert_array_equal(
                replay(index), batches[index]["features"] + batches[index]["extra"]
            )


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-028")
def test_plugin_engine_graph_replay_matches_ordinary_enqueue(installed_example, tmp_path):
    """A captured cubin plugin runs repeatedly and matches ordinary enqueue with new inputs."""
    library = installed_example["library"]
    manifest = json.loads((library.parent / "bundle/bundle.json").read_text())
    digest = np.frombuffer(manifest["bundle_sha256"].encode(), dtype=np.uint8)
    bias = np.arange(16, dtype=np.float32) / 8

    def graph(network):
        creator = trt.get_plugin_registry().get_creator(
            "TensoracleExampleBias", "1", "tensoracle.example"
        )
        assert creator is not None
        fields = trt.PluginFieldCollection(
            [
                trt.PluginField("bundle_sha256", digest, trt.PluginFieldType.CHAR),
                trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32),
            ]
        )
        plugin = creator.create_plugin("bias", fields, trt.TensorRTPhase.BUILD)
        assert plugin is not None
        features = network.add_input("features", trt.float32, (3, 16))
        result = network.add_plugin_v3([features], [], plugin).get_output(0)
        result.name = "result"
        network.mark_output(result)

    plan = tensoracle.build_serialized_engine(
        graph,
        tensoracle.EngineBuild(
            {"features": tensoracle.InputShape.fixed((3, 16))},
            tensoracle.BuilderSettings(optimization_level=0),
        ),
        plugins=(library,),
    )
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    handle = tensoracle.EngineHandle(engine, runtime, tmp_path / "bias.engine")
    batches = [
        {"features": np.arange(48, dtype=np.float32).reshape(3, 16) / 4 + offset}
        for offset in (0, 2)
    ]
    with _captured(handle, batches, shapes={}) as replay:
        for index in (0, 1, 0):
            np.testing.assert_array_equal(replay(index), batches[index]["features"] + bias)


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-028")
def test_ranged_engine_keeps_a_graph_and_execution_per_shape(tmp_path, monkeypatch):
    """Two shapes keep dedicated executions and graphs that replay correctly in alternation."""
    handle = _open_sum(tmp_path, monkeypatch)
    batches = [_sum_batches(rows) for rows in (2, 5)]
    with ExitStack() as resources:
        replays = [
            resources.enter_context(
                _captured(
                    handle,
                    values,
                    shapes={name: value.shape for name, value in values[0].items()},
                )
            )
            for values in batches
        ]
        for index in (0, 1, 0):
            for replay, values in zip(replays, batches, strict=True):
                np.testing.assert_array_equal(
                    replay(index), values[index]["features"] + values[index]["extra"]
                )
