"""Caller-owned graph for the two cubin plugin examples."""

from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors.numpy import load_file

from tensoracle import BuilderSettings, EngineBuild, EngineIO, InputShape, TensorSpec


def library_path() -> Path:
    """Find this wheel's native plugin on the current host."""
    candidates = [
        path
        for path in Path(__file__).parent.glob("*tensoracle_example_plugin*")
        if path.suffix in (".so", ".dll", ".dylib")
    ]
    if len(candidates) != 1:
        raise FileNotFoundError("expected one native example plugin library")
    return candidates[0]


def weights_path() -> Path:
    return Path(__file__).parent / "weights.safetensors"


def declaration() -> EngineBuild:
    return EngineBuild(
        "fp32",
        {"features": InputShape.ranged((1, 32), (16, 32), (32, 32))},
        BuilderSettings(optimization_level=0),
    )


def expected_io() -> EngineIO:
    return EngineIO(
        inputs={"features": TensorSpec(trt.float32, (-1, 32))},
        outputs={"result": TensorSpec(trt.float32, (-1, 16))},
    )


def _fields(
    bundle_digest: str, name: str
) -> tuple[trt.PluginFieldCollection, tuple[np.ndarray, ...]]:
    weights = load_file(weights_path())
    digest = np.frombuffer(bundle_digest.encode("ascii"), dtype=np.int8)
    if name == "dot":
        matrix = np.ascontiguousarray(weights["matrix"])
        scale = np.ascontiguousarray(weights["scale"])
        owners = (digest, matrix, scale)
        fields = [
            trt.PluginField("bundle_sha256", digest, trt.PluginFieldType.CHAR),
            trt.PluginField("weights", matrix, trt.PluginFieldType.FLOAT32),
            trt.PluginField("scale", scale, trt.PluginFieldType.FLOAT32),
        ]
    else:
        bias = np.ascontiguousarray(weights["bias"])
        owners = (digest, bias)
        fields = [
            trt.PluginField("bundle_sha256", digest, trt.PluginFieldType.CHAR),
            trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32),
        ]
    return trt.PluginFieldCollection(fields), owners


def graph(network: trt.INetworkDefinition) -> None:
    """Build a dynamic dot plugin followed by a separate bias plugin."""
    from tensoracle.bundle import inspect_bundle

    digest = inspect_bundle(Path(__file__).parent / "bundle", plugin=library_path())
    registry = trt.get_plugin_registry()
    features = network.add_input("features", trt.float32, (-1, 32))
    dot_creator = registry.get_creator("TensoracleExampleDot", "1", "tensoracle.example")
    bias_creator = registry.get_creator("TensoracleExampleBias", "1", "tensoracle.example")
    if dot_creator is None or bias_creator is None:
        raise RuntimeError(
            "the example plugin creators were not registered before graph construction"
        )
    dot_fields, dot_owners = _fields(digest, "dot")
    bias_fields, bias_owners = _fields(digest, "bias")
    dot = dot_creator.create_plugin("dot", dot_fields, trt.TensorRTPhase.BUILD)
    bias = bias_creator.create_plugin("bias", bias_fields, trt.TensorRTPhase.BUILD)
    del dot_owners, bias_owners
    if dot is None or bias is None:
        raise RuntimeError("the example plugin fields were rejected")
    dotted = network.add_plugin_v3([features], [], dot).get_output(0)
    result = network.add_plugin_v3([dotted], [], bias).get_output(0)
    result.name = "result"
    network.mark_output(result)
