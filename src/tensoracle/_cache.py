"""Open TensorRT engines from the user cache."""

import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import tensorrt as trt

from ._build import EngineBuild, build_serialized_engine
from ._plugins import register_libraries

if TYPE_CHECKING:
    from ._execution import EngineExecution

_MAGIC = b"TENSORACLE\x01"
_WEIGHT_RECORD_VERSION = 1


@dataclass(frozen=True)
class TensorSpec:
    """Expected dtype, build-time dimensions, and optional location and format."""

    dtype: trt.DataType
    shape: tuple[int, ...]
    location: trt.TensorLocation | None = None
    format: trt.TensorFormat | None = None


@dataclass(frozen=True)
class EngineIO:
    """Expected named inputs and outputs of a TensorRT engine."""

    inputs: Mapping[str, TensorSpec]
    outputs: Mapping[str, TensorSpec]


@dataclass(frozen=True)
class EngineHandle:
    """Keep the deserialized engine and its owning runtime alive together."""

    engine: trt.ICudaEngine
    runtime: trt.Runtime
    path: Path

    def create_execution(self) -> "EngineExecution":
        """Create an independent execution context with explicit close semantics."""
        from ._execution import EngineExecution

        return EngineExecution(self)


class EngineCacheError(ValueError):
    """A stored engine cannot be used for this declaration."""


class CorruptEngineError(EngineCacheError):
    """A stored engine is incomplete or corrupt."""


class EngineConditionError(EngineCacheError):
    """A stored engine belongs to different build conditions."""


class EngineIOError(EngineCacheError):
    """The engine's input or output signature differs from the declaration."""


def open_engine(
    graph: Callable[[trt.INetworkDefinition], None],
    declaration: EngineBuild,
    *,
    name: str,
    graph_version: str,
    weights: Sequence[Path],
    io: EngineIO,
    plugins: Sequence[Path] = (),
    force_rebuild: bool = False,
    logger: trt.ILogger | None = None,
) -> EngineHandle:
    """Register ``plugins`` and build or load an engine for the supplied conditions."""
    registered_digests = register_libraries(plugins)
    conditions = _conditions(
        name, graph_version, weights, plugins, declaration, force_rebuild=force_rebuild
    )
    if tuple(conditions["plugins"]) != registered_digests:
        raise ValueError("plugin library changed while determining engine conditions")
    digest = hashlib.sha256(_json_bytes(conditions)).hexdigest()
    path = _cache_directory() / f"{digest}.engine"
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_logger = logger or trt.Logger(trt.Logger.ERROR)

    if force_rebuild:
        path.write_bytes(b"")
    else:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            plan = _read_plan(path, conditions)
            return _open_plan(plan, path, declaration, io, actual_logger)
        else:
            os.close(fd)

    temporary: Path | None = None
    try:
        plan = build_serialized_engine(graph, declaration, plugins=plugins, logger=actual_logger)
        handle = _open_plan(plan, path, declaration, io, actual_logger)
        header = {"conditions": conditions, "plan_sha256": hashlib.sha256(plan).hexdigest()}
        header_bytes = _json_bytes(header)
        fd, filename = tempfile.mkstemp(prefix=f".{digest}.", dir=path.parent)
        temporary = Path(filename)
        with os.fdopen(fd, "wb") as stream:
            stream.write(_MAGIC + len(header_bytes).to_bytes(8, "big") + header_bytes)
            stream.write(plan)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return handle
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        raise


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_directory() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache_root / "tensoracle"


def _weight_sha256(path: Path, *, force_rebuild: bool) -> str:
    absolute_path = Path(os.path.abspath(path))
    before = absolute_path.stat()
    identity = {
        "path": str(absolute_path),
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "inode": before.st_ino,
    }
    path_digest = hashlib.sha256(os.fsencode(absolute_path)).hexdigest()
    record_path = _cache_directory() / f"weight-{path_digest}.json"

    if not force_rebuild:
        try:
            record = json.loads(record_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            record = None
        if (
            isinstance(record, dict)
            and record.keys() == identity.keys() | {"version", "sha256", "checksum"}
            and record["version"] == _WEIGHT_RECORD_VERSION
            and type(record["version"]) is int
            and all(
                type(record[key]) is type(value) and record[key] == value
                for key, value in identity.items()
            )
            and isinstance(record["sha256"], str)
            and len(record["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in record["sha256"])
            and record["checksum"]
            == hashlib.sha256(
                _json_bytes({key: value for key, value in record.items() if key != "checksum"})
            ).hexdigest()
        ):
            return record["sha256"]

    digest = _file_sha256(absolute_path)
    try:
        after = absolute_path.stat()
    except OSError:
        return digest
    if (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        payload = {"version": _WEIGHT_RECORD_VERSION, **identity, "sha256": digest}
        _write_weight_record(
            record_path, {**payload, "checksum": hashlib.sha256(_json_bytes(payload)).hexdigest()}
        )
    return digest


def _write_weight_record(path: Path, record: dict) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, filename = tempfile.mkstemp(prefix=f".{path.stem}.", dir=path.parent)
        temporary = Path(filename)
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _gpu_name() -> str:
    library_name = ctypes.util.find_library("cuda") or ctypes.util.find_library("nvcuda")
    if library_name is None:
        raise RuntimeError("CUDA driver library was not found")
    cuda = ctypes.CDLL(library_name)
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    if cuda.cuInit(0) != 0:
        raise RuntimeError("CUDA driver initialization failed")
    device = ctypes.c_int()
    if (
        cuda.cuCtxGetDevice(ctypes.byref(device)) != 0
        and cuda.cuDeviceGet(ctypes.byref(device), 0) != 0
    ):
        raise RuntimeError("CUDA device selection failed")
    name = ctypes.create_string_buffer(256)
    if cuda.cuDeviceGetName(name, len(name), device.value) != 0:
        raise RuntimeError("CUDA device name lookup failed")
    return name.value.decode("utf-8")


def _conditions(
    name: str,
    graph_version: str,
    weights: Sequence[Path],
    plugins: Sequence[Path],
    declaration: EngineBuild,
    *,
    force_rebuild: bool = False,
) -> dict:
    if not name or not graph_version or not weights:
        raise ValueError("engine name, graph version, and at least one weight file are required")
    if declaration.settings.tf32:
        raise ValueError("TF32 cannot be enabled")
    return {
        "name": name,
        "graph_version": graph_version,
        "weights": [_weight_sha256(Path(path), force_rebuild=force_rebuild) for path in weights],
        "plugins": [_file_sha256(Path(path)) for path in plugins],
        "precision": declaration.precision,
        "inputs": {
            name: {
                "minimum": list(shape.minimum),
                "optimum": list(shape.optimum),
                "maximum": list(shape.maximum),
            }
            for name, shape in declaration.inputs.items()
        },
        "builder_settings": asdict(declaration.settings),
        "gpu_name": _gpu_name(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "tensorrt_version": trt.__version__,
    }


def _read_plan(path: Path, conditions: dict) -> memoryview:
    data = path.read_bytes()
    offset = len(_MAGIC)
    if not data.startswith(_MAGIC) or len(data) < offset + 8:
        raise CorruptEngineError(f"incomplete TensorRT engine file: {path}")
    header_size = int.from_bytes(data[offset : offset + 8], "big")
    header_end = offset + 8 + header_size
    if header_size == 0 or header_end >= len(data):
        raise CorruptEngineError(f"invalid TensorRT engine header: {path}")
    try:
        header = json.loads(data[offset + 8 : header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorruptEngineError(f"invalid TensorRT engine header: {path}") from error
    if not isinstance(header, dict) or not isinstance(header.get("conditions"), dict):
        raise CorruptEngineError(f"invalid TensorRT engine header: {path}")
    if header["conditions"] != conditions:
        raise EngineConditionError(f"TensorRT engine conditions differ: {path}")
    plan = memoryview(data)[header_end:]
    if header.get("plan_sha256") != hashlib.sha256(plan).hexdigest():
        raise CorruptEngineError(f"TensorRT engine checksum differs: {path}")
    return plan


def _open_plan(
    plan: bytes | memoryview,
    path: Path,
    declaration: EngineBuild,
    io: EngineIO,
    logger: trt.ILogger,
) -> EngineHandle:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise CorruptEngineError(
            f"TensorRT could not deserialize engine: {path}. Check that plugins= includes the "
            "libraries providing the plugin creators (name, version, namespace) used by this engine."
        )
    names = {engine.get_tensor_name(index) for index in range(engine.num_io_tensors)}
    expected_names = io.inputs.keys() | io.outputs.keys()
    for name in sorted(names ^ expected_names):
        _io_mismatch(path, name, "name", sorted(names), sorted(expected_names))
    for specs, mode in ((io.inputs, trt.TensorIOMode.INPUT), (io.outputs, trt.TensorIOMode.OUTPUT)):
        for name, expected in specs.items():
            for field, observed, value in (
                ("mode", engine.get_tensor_mode(name), mode),
                ("dtype", engine.get_tensor_dtype(name), expected.dtype),
                ("shape", tuple(engine.get_tensor_shape(name)), expected.shape),
            ):
                if observed != value:
                    _io_mismatch(path, name, field, observed, value)
            if expected.location is not None:
                observed = engine.get_tensor_location(name)
                if observed != expected.location:
                    _io_mismatch(path, name, "location", observed, expected.location)
            if expected.format is not None:
                observed = engine.get_tensor_format(name)
                if observed != expected.format:
                    _io_mismatch(path, name, "format", observed, expected.format)
    for name in sorted(declaration.inputs.keys() ^ io.inputs.keys()):
        _io_mismatch(
            path, name, "name (build input)", sorted(io.inputs), sorted(declaration.inputs)
        )
    for name, shape in declaration.inputs.items():
        expected_range = (shape.minimum, shape.optimum, shape.maximum)
        if -1 in io.inputs[name].shape:
            actual_range = tuple(tuple(part) for part in engine.get_tensor_profile_shape(name, 0))
        else:
            actual_range = (io.inputs[name].shape,) * 3
        if actual_range != expected_range:
            _io_mismatch(path, name, "profile range", actual_range, expected_range)
    return EngineHandle(engine, runtime, path)


def _io_mismatch(path: Path, name: str, field: str, observed: object, expected: object) -> None:
    raise EngineIOError(
        f"TensorRT engine tensor {name!r} {field} differs: "
        f"observed={observed!r}, expected={expected!r}; path={path}"
    )
