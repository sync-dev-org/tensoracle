"""Build serialized TensorRT engines from caller-defined network graphs."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tensorrt as trt

from ._memory import release_host_memory
from ._plugins import register_libraries


@dataclass(frozen=True)
class InputShape:
    """The minimum, preferred, and maximum dimensions of one input."""

    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.minimum
            or len(self.minimum) != len(self.optimum)
            or len(self.minimum) != len(self.maximum)
        ):
            raise ValueError("input shapes must have the same nonzero rank")
        if any(
            not isinstance(low, int)
            or not isinstance(mid, int)
            or not isinstance(high, int)
            or not 0 < low <= mid <= high
            for low, mid, high in zip(self.minimum, self.optimum, self.maximum, strict=True)
        ):
            raise ValueError(
                "input dimensions must be positive and ordered minimum <= optimum <= maximum"
            )

    @classmethod
    def fixed(cls, shape: tuple[int, ...]) -> "InputShape":
        """Declare one fixed input shape."""
        return cls(shape, shape, shape)

    @classmethod
    def ranged(
        cls, minimum: tuple[int, ...], optimum: tuple[int, ...], maximum: tuple[int, ...]
    ) -> "InputShape":
        """Declare the three shapes of one TensorRT optimization profile."""
        return cls(minimum, optimum, maximum)


@dataclass(frozen=True)
class BuilderSettings:
    """TensorRT builder options that affect the serialized engine."""

    workspace_limit: int = 1 << 30
    optimization_level: int = 3
    tf32: bool = False


@dataclass(frozen=True)
class EngineBuild:
    """One engine's precision, named input shapes, and builder settings."""

    precision: Literal["fp32", "fp16"]
    inputs: Mapping[str, InputShape]
    settings: BuilderSettings = field(default_factory=BuilderSettings)


def build_serialized_engine(
    graph: Callable[[trt.INetworkDefinition], None],
    declaration: EngineBuild,
    *,
    plugins: Sequence[Path] = (),
    logger: trt.ILogger | None = None,
) -> bytes:
    """Register ``plugins`` before calling ``graph`` and serializing its engine."""
    if declaration.precision not in ("fp32", "fp16"):
        raise ValueError(f"unsupported precision: {declaration.precision}")
    if declaration.settings.tf32:
        raise ValueError("TF32 cannot be enabled")
    if declaration.settings.workspace_limit <= 0:
        raise ValueError("workspace_limit must be positive")
    if not 0 <= declaration.settings.optimization_level <= 5:
        raise ValueError("optimization_level must be between 0 and 5")

    register_libraries(plugins)

    builder = network = config = profile = serialized = tensor = None
    inputs = {}
    try:
        builder = trt.Builder(logger or trt.Logger(trt.Logger.ERROR))
        network = builder.create_network(0)
        graph(network)
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, declaration.settings.workspace_limit
        )
        config.builder_optimization_level = declaration.settings.optimization_level

        inputs = {
            network.get_input(index).name: network.get_input(index)
            for index in range(network.num_inputs)
        }
        if inputs.keys() != declaration.inputs.keys():
            raise ValueError("declared input names must match the graph inputs")

        for name, tensor in inputs.items():
            shape = declaration.inputs[name]
            network_shape = tuple(tensor.shape)
            if len(network_shape) != len(shape.minimum):
                raise ValueError(f"declared rank does not match graph input {name}")
            for dimension, low, mid, high in zip(
                network_shape, shape.minimum, shape.optimum, shape.maximum, strict=True
            ):
                if dimension != -1 and (dimension != low or dimension != mid or dimension != high):
                    raise ValueError(f"declared dimensions do not match graph input {name}")
            if -1 in network_shape:
                if profile is None:
                    profile = builder.create_optimization_profile()
                profile.set_shape(name, shape.minimum, shape.optimum, shape.maximum)
                if not profile:
                    raise ValueError(f"TensorRT rejected the shape range for graph input {name}")
        if profile is not None:
            config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT failed to build the serialized engine")
        return bytes(serialized)
    finally:
        # Tensor references also keep the network alive, including in exception frames.
        inputs.clear()
        del tensor, profile, serialized, config, network, builder
        release_host_memory()
