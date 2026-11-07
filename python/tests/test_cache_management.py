"""Record successful engine opens without making timestamps a condition of reuse."""

import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest
import tensorrt as trt

import tensoracle
import tensoracle._cache as cache


@pytest.fixture
def open_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_identity", lambda: {"name": "test GPU"})
    builder = SimpleNamespace(
        create_network=lambda flags: SimpleNamespace(num_inputs=0, num_outputs=0),
        create_builder_config=lambda: SimpleNamespace(
            clear_flag=lambda flag: None,
            set_memory_pool_limit=lambda pool, limit: None,
            create_timing_cache=lambda data: object(),
            set_timing_cache=lambda timing, ignore_mismatch: True,
        ),
        build_serialized_network=lambda network, config: b"TensorRT plan",
    )
    monkeypatch.setattr(trt, "Builder", lambda logger: builder)
    monkeypatch.setattr(
        trt,
        "Runtime",
        lambda logger: SimpleNamespace(
            deserialize_cuda_engine=lambda plan: SimpleNamespace(num_io_tensors=0)
        ),
    )
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"weights")
    built = []

    def graph(network):
        built.append(True)

    def open_engine(*, io=None, settings=None, force_rebuild=False):
        return tensoracle.open_engine(
            graph,
            tensoracle.EngineBuild({}, settings or tensoracle.BuilderSettings()),
            name="timestamp",
            graph_version="1",
            weights=(weight,),
            io=io or tensoracle.EngineIO({}, {}),
            force_rebuild=force_rebuild,
        )

    return open_engine, built


@pytest.mark.req("SP-TENSORACLE-052")
@pytest.mark.parametrize("workspace_limit", [None, 8192])
def test_workspace_limit_changes_reuse_saved_engine(open_cached, workspace_limit):
    """Changing only the workspace limit reuses the same file without invoking the graph."""
    open_engine, built = open_cached
    first = open_engine(settings=tensoracle.BuilderSettings(workspace_limit=4096))
    contents = first.path.read_bytes()
    reopened = open_engine(settings=tensoracle.BuilderSettings(workspace_limit=workspace_limit))
    assert reopened.path == first.path
    assert built == [True]
    assert first.path.read_bytes() == contents


@pytest.mark.req("SP-TENSORACLE-052")
@pytest.mark.req("SP-TENSORACLE-011")
@pytest.mark.parametrize("workspace_limit", [None, 4096])
def test_saved_engine_records_workspace_limit_and_updates_it_only_on_rebuild(
    open_cached, workspace_limit
):
    """Headers record the actual build's workspace outside conditions and forced rebuilds replace it."""
    open_engine, built = open_cached

    def header(path):
        with path.open("rb") as stream:
            return cache._read_header(stream, path, path.stat().st_size)

    first = open_engine(settings=tensoracle.BuilderSettings(workspace_limit=workspace_limit))
    original = header(first.path)
    assert original["workspace_limit"] == workspace_limit
    assert original["conditions"]["builder_settings"] == {"optimization_level": 3, "tf32": False}
    changed_limit = 8192 if workspace_limit is None else None
    rebuilt = open_engine(
        settings=tensoracle.BuilderSettings(workspace_limit=changed_limit), force_rebuild=True
    )
    assert rebuilt.path == first.path
    assert header(rebuilt.path)["workspace_limit"] == changed_limit
    assert header(rebuilt.path)["conditions"] == original["conditions"]
    assert built == [True, True]


@pytest.mark.req("SP-TENSORACLE-042")
@pytest.mark.req("SP-TENSORACLE-048")
@pytest.mark.req("SP-TENSORACLE-046")
def test_build_and_successful_reopen_record_current_time_without_rebuilding(open_cached):
    """Saving starts the mtime record and a successful cache hit advances it without a rebuild."""
    open_engine, built = open_cached
    started = time.time_ns()
    first = open_engine()
    assert started <= first.path.stat().st_mtime_ns <= time.time_ns()
    contents = first.path.read_bytes()
    os.utime(first.path, ns=(1_000_000_000, 1_000_000_000))
    started = time.time_ns()
    reopened = open_engine()
    assert reopened.path == first.path
    assert started <= first.path.stat().st_mtime_ns <= time.time_ns()
    assert first.path.read_bytes() == contents
    assert built == [True]
    assert sorted(path.suffix for path in first.path.parent.iterdir()) == [".engine", ".json"]


@pytest.mark.req("SP-TENSORACLE-048")
@pytest.mark.parametrize("unavailable", ["permission", "disappeared"])
def test_unavailable_timestamp_does_not_prevent_reuse(open_cached, monkeypatch, unavailable):
    """A denied timestamp update or a removed file leaves the opened engine usable."""
    open_engine, built = open_cached
    first = open_engine()
    os.utime(first.path, ns=(1_000_000_000, 1_000_000_000))
    real_utime = os.utime
    attempts = []

    def unavailable_utime(path, *args, **kwargs):
        if path == first.path:
            attempts.append(path)
            if unavailable == "disappeared":
                first.path.unlink()
                return real_utime(path, *args, **kwargs)
            raise PermissionError("read-only cache")
        return real_utime(path, *args, **kwargs)

    monkeypatch.setattr(os, "utime", unavailable_utime)
    assert open_engine().path == first.path
    assert built == [True]
    assert attempts == [first.path]
    if unavailable == "disappeared":
        assert not first.path.exists()
    else:
        assert first.path.stat().st_mtime_ns == 1_000_000_000


@pytest.mark.req("SP-TENSORACLE-042")
@pytest.mark.parametrize("failure", ["header", "plan", "conditions", "io", "deserialize"])
def test_failed_engine_open_does_not_advance_recorded_time(open_cached, monkeypatch, failure):
    """Invalid headers, plans, conditions, I/O, and deserialization preserve the previous mtime."""
    open_engine, _ = open_cached
    first = open_engine()
    data = first.path.read_bytes()
    error = tensoracle.CorruptEngineError
    io = None
    if failure == "header":
        first.path.write_bytes(b"invalid header")
    elif failure == "plan":
        first.path.write_bytes(data[:-1] + b"!")
    elif failure == "conditions":
        first.path.write_bytes(data.replace(b'"name":"timestamp"', b'"name":"different"'))
        error = tensoracle.EngineConditionError
    elif failure == "io":
        io = tensoracle.EngineIO({}, {"unexpected": tensoracle.TensorSpec(trt.float32, (1,))})
        error = tensoracle.EngineIOError
    else:
        monkeypatch.setattr(
            trt,
            "Runtime",
            lambda logger: SimpleNamespace(deserialize_cuda_engine=lambda plan: None),
        )
    os.utime(first.path, ns=(1_000_000_000, 1_000_000_000))
    with pytest.raises(error):
        open_engine(io=io)
    assert first.path.stat().st_mtime_ns == 1_000_000_000


@pytest.mark.req("SP-TENSORACLE-049")
@pytest.mark.parametrize(
    "header", [[], {}, {"conditions": []}, {"conditions": {}, "shape_tensor_outputs": [1]}]
)
def test_invalid_saved_headers_are_rejected_without_deserializing(tmp_path, header):
    """The shared header reader rejects invalid objects and shape-output records."""
    path = tmp_path / "invalid.engine"
    encoded = json.dumps(header).encode()
    path.write_bytes(b"TENSORACLE\x01" + len(encoded).to_bytes(8, "big") + encoded + b"plan")
    with pytest.raises(tensoracle.CorruptEngineError):
        cache._read_plan(path, {})


@pytest.mark.req("SP-TENSORACLE-009")
def test_shared_reader_preserves_checksum_validation(tmp_path):
    """Header-only inspection does not remove checksum validation from engine reuse."""
    path = tmp_path / "corrupt.engine"
    header = json.dumps(
        {"conditions": {}, "plan_sha256": hashlib.sha256(b"good").hexdigest()}
    ).encode()
    path.write_bytes(b"TENSORACLE\x01" + len(header).to_bytes(8, "big") + header + b"bad")
    with pytest.raises(tensoracle.CorruptEngineError, match="checksum"):
        cache._read_plan(path, {})
