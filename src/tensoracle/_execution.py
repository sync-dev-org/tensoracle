"""Execute TensorRT engines with caller-owned addresses and CUDA streams."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Self

import tensorrt as trt

from ._build import _warn_shape_tensor_outputs

if TYPE_CHECKING:
    from ._cache import EngineHandle


class EngineExecution:
    """One independent TensorRT context retained with its checked engine handle.

    Use ``handle.create_execution()`` or construct this type with an ``EngineHandle``.
    The caller owns buffers, streams, their ordering, and completion. This object
    neither copies data nor synchronizes. Finish outstanding work before changing
    shapes or streams, closing, or reusing the object from another thread.
    """

    def __init__(self, handle: EngineHandle) -> None:
        self._handle: EngineHandle | None = handle
        self._context: trt.IExecutionContext | None = None
        self._ready = False
        engine = handle.engine
        self._names = tuple(engine.get_tensor_name(i) for i in range(engine.num_io_tensors))
        self._ranges: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        initial_shapes = {}
        try:
            for name in self._names:
                if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                    raise ValueError(
                        f"tensor {name!r} has HOST location; execution requires DEVICE"
                    )
                if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                    continue
                if engine.is_shape_inference_io(name):
                    raise ValueError(f"input {name!r} is an external shape tensor; not supported")
                shape = tuple(engine.get_tensor_shape(name))
                if -1 in shape:
                    low, optimum, high = (
                        tuple(part) for part in engine.get_tensor_profile_shape(name, 0)
                    )
                    initial_shapes[name] = optimum
                    if low != high:
                        self._ranges[name] = (low, high)
                else:
                    initial_shapes[name] = shape
            self._context = engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT failed to create an execution context")
            _warn_shape_tensor_outputs(handle._shape_tensor_outputs, stacklevel=2)
            # Probe output dimensions before accepting the engine. A ranged engine
            # still requires an explicit caller setting after this capability check.
            self._set_shapes(initial_shapes, creating=True)
            self._ready = not self._ranges
        except BaseException:
            self.close()
            raise

    def set_input_shapes(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        """Set every ranged input together, validating all shapes before TensorRT.

        Fixed inputs, including single-point profiles, are initialized at creation
        and omitted here. A TensorRT failure invalidates the setting until a complete
        subsequent call succeeds; partial settings are not rolled back.
        """
        self._require_open()
        self._check_names(shapes, self._ranges.keys(), "input shape")
        for name, shape in shapes.items():
            low, high = self._ranges[name]
            if (
                not isinstance(shape, tuple)
                or len(shape) != len(low)
                or any(
                    type(value) is not int or not minimum <= value <= maximum
                    for value, minimum, maximum in zip(shape, low, high, strict=True)
                )
            ):
                raise ValueError(
                    f"input {name!r} shape {shape!r} is outside range {low!r}..{high!r}"
                )
        self._set_shapes(shapes)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Read a concrete input or output shape after all input shapes are ready."""
        context = self._require_ready()
        self._check_tensor(name)
        shape = tuple(context.get_tensor_shape(name))
        if any(dimension < 0 for dimension in shape):
            self._ready = False
            raise RuntimeError(f"tensor {name!r} shape {shape!r} is unresolved")
        return shape

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        """Read a tensor's TensorRT dtype for allocating a caller-owned buffer."""
        self._require_open()
        self._check_tensor(name)
        return self._handle.engine.get_tensor_dtype(name)

    def enqueue(self, addresses: Mapping[str, int], *, stream: int) -> None:
        """Enqueue all tensor addresses on the caller's stream without waiting.

        Addresses must be positive integers, and the stream must be a nonnegative
        integer (0 is the default stream). Bool is rejected. This method cannot
        check buffer dtype, dimensions, layout, size, lifetime, or device selection.
        """
        self._require_open()
        self._check_names(addresses, self._names, "address")
        for name, address in addresses.items():
            if type(address) is not int or address <= 0:
                raise ValueError(f"tensor {name!r} address {address!r} must be a positive integer")
        if type(stream) is not int or stream < 0:
            raise ValueError(f"stream {stream!r} must be a nonnegative integer")
        context = self._require_ready()
        for name in self._names:
            if not context.set_tensor_address(name, addresses[name]):
                raise RuntimeError(f"TensorRT rejected tensor {name!r} address {addresses[name]!r}")
        if not context.execute_async_v3(stream):
            raise RuntimeError(f"TensorRT enqueue failed on stream {stream!r}")

    def close(self) -> None:
        """Release context before handle without synchronization; repeated calls do nothing.

        The caller must confirm that all enqueued work has completed before closing.
        """
        self._ready = False
        self._context = None
        self._handle = None

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _require_open(self) -> trt.IExecutionContext:
        if self._context is None:
            raise RuntimeError("engine execution is closed")
        return self._context

    def _require_ready(self) -> trt.IExecutionContext:
        context = self._require_open()
        if not self._ready:
            raise RuntimeError("input shapes need a complete successful setting before execution")
        return context

    def _check_tensor(self, name: str) -> None:
        if name not in self._names:
            raise ValueError(f"unknown engine tensor {name!r}")

    @staticmethod
    def _check_names(values: Mapping, names, field: str) -> None:
        expected = set(names)
        for name in sorted(expected - values.keys(), key=repr):
            raise ValueError(f"tensor {name!r} {field} is missing")
        for name in sorted(values.keys() - expected, key=repr):
            raise ValueError(f"unexpected tensor {name!r} {field} {values[name]!r}")

    def _set_shapes(self, shapes: Mapping[str, tuple[int, ...]], *, creating: bool = False) -> None:
        self._ready = False
        for name, shape in shapes.items():
            try:
                accepted = self._context.set_input_shape(name, shape)
            except Exception as error:
                raise RuntimeError(
                    f"TensorRT failed to set input {name!r} shape {shape!r}"
                ) from error
            if not accepted:
                raise RuntimeError(f"TensorRT rejected input {name!r} shape {shape!r}")
        missing = self._context.infer_shapes()
        if missing:
            raise RuntimeError(f"TensorRT shape inference requires tensors {missing!r}")
        for name in self._names:
            shape = tuple(self._context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                if creating:
                    raise ValueError(
                        f"tensor {name!r} has data-dependent shape {shape!r}; not supported"
                    )
                raise RuntimeError(f"tensor {name!r} shape {shape!r} is unresolved")
        self._ready = True
