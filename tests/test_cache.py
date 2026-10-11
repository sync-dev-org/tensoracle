"""Persist and validate TensorRT engines built from safetensors weights."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorrt as trt
from safetensors.numpy import save_file

import tensoracle
from test_build import _graph, _weights


def _heartbeat_during(operation):
    stopped = threading.Event()
    ready = threading.Event()
    gaps = []

    def heartbeat():
        previous = time.perf_counter()
        ready.set()
        while not stopped.wait(0.001):
            now = time.perf_counter()
            gaps.append(now - previous)
            previous = now
        gaps.append(time.perf_counter() - previous)

    thread = threading.Thread(target=heartbeat)
    thread.start()
    ready.wait()
    started = time.perf_counter()
    try:
        result = operation()
        elapsed = time.perf_counter() - started
    finally:
        stopped.set()
        thread.join()
    return result, {"elapsed_ms": elapsed * 1000, "heartbeat_gap_ms": max(gaps) * 1000}


@pytest.mark.req("REQ-TENSORACLE-007")
def test_read_large_saved_plan_keeps_other_threads_running(tmp_path):
    """Reading a 512 MiB saved plan leaves a heartbeat gap below 50 ms.

    The original implementation paused the heartbeat for 199-321 ms; the view
    paused it for 1.2-4.0 ms. The 50 ms limit is one quarter of the smallest
    original pause, rounded to whole tens of milliseconds.
    Three reads measure the maximum heartbeat gap separately; the best read must
    meet the limit so unrelated scheduling delays do not fail every trial.
    """
    import tensoracle._cache as cache

    plan = b"\0" * (512 * 1024 * 1024)
    digest = hashlib.sha256(plan).hexdigest()
    conditions = {"name": "heartbeat"}
    header = cache._json_bytes({"conditions": conditions, "plan_sha256": digest})
    path = tmp_path / "large.engine"
    with path.open("wb") as stream:
        stream.write(cache._MAGIC + len(header).to_bytes(8, "big") + header)
        stream.write(plan)
    del plan

    measurements = []
    for _ in range(3):
        (loaded, _), measured = _heartbeat_during(lambda: cache._read_plan(path, conditions))
        assert len(loaded) == 512 * 1024 * 1024
        assert hashlib.sha256(loaded).hexdigest() == digest
        measurements.append(measured)
        del loaded
    print(f"read 512 MiB: {measurements}")
    assert min(item["heartbeat_gap_ms"] for item in measurements) < 50, measurements


@pytest.mark.req("REQ-TENSORACLE-007")
def test_save_large_built_plan_keeps_other_threads_running(tmp_path, monkeypatch):
    """Saving a 512 MiB built plan leaves a heartbeat gap below 50 ms.

    The original concatenation paused the heartbeat for 197-207 ms; separate
    writes paused it for 1.2-5.1 ms. The 50 ms limit is one quarter of the smallest
    original pause, rounded to whole tens of milliseconds.
    Three saves measure the maximum heartbeat gap separately; the best save must
    meet the limit so unrelated scheduling delays do not fail every trial.
    """
    import tensoracle._cache as cache

    plan = b"\0" * (512 * 1024 * 1024)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    builder = SimpleNamespace(
        create_network=lambda flags: SimpleNamespace(num_inputs=0, num_outputs=0),
        create_builder_config=lambda: SimpleNamespace(
            clear_flag=lambda flag: None,
            set_memory_pool_limit=lambda pool, limit: None,
        ),
        build_serialized_network=lambda network, config: plan,
    )
    monkeypatch.setattr(trt, "Builder", lambda logger: builder)
    monkeypatch.setattr(
        trt,
        "Runtime",
        lambda logger: SimpleNamespace(
            deserialize_cuda_engine=lambda serialized: SimpleNamespace(num_io_tensors=0)
        ),
    )
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    declaration = tensoracle.EngineBuild("fp32", {})

    def save():
        return tensoracle.open_engine(
            lambda network: None,
            declaration,
            name="heartbeat",
            graph_version="1",
            weights=(weights,),
            io=tensoracle.EngineIO({}, {}),
            force_rebuild=True,
        )

    measurements = []
    for _ in range(3):
        handle, measured = _heartbeat_during(save)
        conditions = cache._conditions("heartbeat", "1", (weights,), (), declaration)
        loaded, _ = cache._read_plan(handle.path, conditions)
        assert len(loaded) == len(plan)
        assert hashlib.sha256(loaded).digest() == hashlib.sha256(plan).digest()
        measurements.append(measured)
        del loaded
    print(f"save 512 MiB: {measurements}")
    assert min(item["heartbeat_gap_ms"] for item in measurements) < 50, measurements


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-006")
def test_two_named_engines_from_one_weight_file_are_saved_separately(tmp_path, monkeypatch):
    """Two engine roles using one safetensors file have separate usable cache entries."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights.safetensors"
    _weights(weights)
    declaration = tensoracle.EngineBuild(
        precision="fp32",
        inputs={"features": tensoracle.InputShape.fixed((1, 3, 2))},
        settings=tensoracle.BuilderSettings(optimization_level=0),
    )
    io = tensoracle.EngineIO(
        inputs={"features": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
        outputs={"result": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
    )

    first = tensoracle.open_engine(
        _graph(weights, "fp32", (1, 3, 2)),
        declaration,
        name="encoder",
        graph_version="1",
        weights=(weights,),
        io=io,
    )
    second = tensoracle.open_engine(
        _graph(weights, "fp32", (1, 3, 2)),
        declaration,
        name="decoder",
        graph_version="1",
        weights=(weights,),
        io=io,
    )

    assert first.path != second.path
    assert first.path.is_file() and second.path.is_file()
    assert len(list((tmp_path / "cache" / "tensoracle").glob("*.engine"))) == 2
    assert first.engine.get_tensor_name(0) == "features"
    assert second.engine.get_tensor_name(0) == "features"


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-007")
def test_cached_engine_is_reused_in_another_process_without_calling_graph(tmp_path, monkeypatch):
    """A process restart reuses a verified plan while leaving the graph callback untouched."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights.safetensors"
    _weights(weights)
    declaration = tensoracle.EngineBuild(
        "fp32",
        {"features": tensoracle.InputShape.fixed((1, 3, 2))},
        tensoracle.BuilderSettings(optimization_level=0),
    )
    io = tensoracle.EngineIO(
        {"features": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
        {"result": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
    )
    first = tensoracle.open_engine(
        _graph(weights, "fp32", (1, 3, 2)),
        declaration,
        name="encoder",
        graph_version="1",
        weights=(weights,),
        io=io,
    )
    child = """
import sys
import tensorrt as trt
from pathlib import Path
from tensoracle import BuilderSettings, EngineBuild, EngineIO, InputShape, TensorSpec, open_engine
weights = Path(sys.argv[1])
def forbidden(network):
    raise AssertionError('cached graph callback was invoked')
handle = open_engine(
    forbidden,
    EngineBuild('fp32', {'features': InputShape.fixed((1, 3, 2))}, BuilderSettings(optimization_level=0)),
    name='encoder', graph_version='1', weights=(weights,),
    io=EngineIO({'features': TensorSpec(trt.float32, (1, 3, 2))},
                {'result': TensorSpec(trt.float32, (1, 3, 2))}),
)
print(handle.path)
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(weights)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(first.path)


@pytest.mark.req("REQ-TENSORACLE-008")
def test_saved_engine_deserialization_failure_guides_caller_to_check_plugins(tmp_path, monkeypatch):
    """An unreadable saved engine reports its path and asks the caller to check plugin libraries."""
    import tensoracle._cache as cache

    cache_root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weights = tmp_path / "weights.safetensors"
    _weights(weights)
    declaration = tensoracle.EngineBuild(
        "fp32", {"features": tensoracle.InputShape.fixed((1, 3, 2))}
    )
    io = tensoracle.EngineIO(
        {"features": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
        {"result": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
    )
    conditions = cache._conditions("encoder", "1", (weights,), (), declaration)
    digest = hashlib.sha256(cache._json_bytes(conditions)).hexdigest()
    path = cache_root / "tensoracle" / f"{digest}.engine"
    plan = b"saved TensorRT plan"
    header = cache._json_bytes(
        {"conditions": conditions, "plan_sha256": hashlib.sha256(plan).hexdigest()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cache._MAGIC + len(header).to_bytes(8, "big") + header + plan)
    deserialized_plans = []

    class FailingRuntime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, serialized_plan) -> None:
            deserialized_plans.append(serialized_plan)

    monkeypatch.setattr(trt, "Runtime", FailingRuntime)
    with pytest.raises(tensoracle.CorruptEngineError) as caught:
        tensoracle.open_engine(
            lambda network: pytest.fail("cached graph was invoked"),
            declaration,
            name="encoder",
            graph_version="1",
            weights=(weights,),
            io=io,
        )

    assert deserialized_plans == [plan]
    assert type(caught.value) is tensoracle.CorruptEngineError
    message = str(caught.value)
    assert str(path) in message
    assert re.search(r"(?i)(check|ensure|verify|confirm).*plugins=", message)
    assert "libraries" in message and "creators" in message


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-008")
def test_corrupt_plan_and_changed_header_fail_before_deserialization(tmp_path, monkeypatch):
    """Corrupt plan bytes and changed conditions are rejected before TensorRT sees the plan."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights.safetensors"
    _weights(weights)
    declaration = tensoracle.EngineBuild(
        "fp32",
        {"features": tensoracle.InputShape.fixed((1, 3, 2))},
        tensoracle.BuilderSettings(optimization_level=0),
    )
    io = tensoracle.EngineIO(
        {"features": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
        {"result": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
    )

    def open_cached(*, force_rebuild=False):
        return tensoracle.open_engine(
            _graph(weights, "fp32", (1, 3, 2)),
            declaration,
            name="encoder",
            graph_version="1",
            weights=(weights,),
            io=io,
            force_rebuild=force_rebuild,
        )

    first = open_cached()
    intact = first.path.read_bytes()
    corrupted = bytearray(intact)
    corrupted[-1] ^= 1
    first.path.write_bytes(corrupted)
    with monkeypatch.context() as patch:
        patch.setattr(trt, "Runtime", lambda logger: pytest.fail("TensorRT deserialize was called"))
        with pytest.raises(tensoracle.CorruptEngineError, match=re.escape(str(first.path))):
            open_cached()

    first.path.write_bytes(intact.replace(b'"name":"encoder"', b'"name":"decoder"'))
    with monkeypatch.context() as patch:
        patch.setattr(trt, "Runtime", lambda logger: pytest.fail("TensorRT deserialize was called"))
        with pytest.raises(tensoracle.EngineConditionError, match=re.escape(str(first.path))):
            open_cached()

    rebuilt = open_cached(force_rebuild=True)
    assert rebuilt.path == first.path
    assert rebuilt.engine.get_tensor_name(0) == "features"
    assert rebuilt.path.read_bytes() != corrupted
    calls = []

    def rebuilt_graph(network):
        calls.append(1)
        _graph(weights, "fp32", (1, 3, 2))(network)

    tensoracle.open_engine(
        rebuilt_graph,
        declaration,
        name="encoder",
        graph_version="1",
        weights=(weights,),
        io=io,
        force_rebuild=True,
    )
    assert calls == [1]


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-002")
@pytest.mark.req("REQ-TENSORACLE-008")
def test_force_rebuild_uses_changed_weight_contents_and_preserves_old_engine(tmp_path, monkeypatch):
    """A forced rebuild saves under the new digest without replacing the old engine."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights.safetensors"
    _weights(weights)
    declaration = tensoracle.EngineBuild(
        "fp32",
        {"features": tensoracle.InputShape.fixed((1, 3, 2))},
        tensoracle.BuilderSettings(optimization_level=0),
    )
    io = tensoracle.EngineIO(
        {"features": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
        {"result": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
    )

    def open_cached(*, force_rebuild=False):
        return tensoracle.open_engine(
            _graph(weights, "fp32", (1, 3, 2)),
            declaration,
            name="encoder",
            graph_version="1",
            weights=(weights,),
            io=io,
            force_rebuild=force_rebuild,
        )

    first = open_cached()
    old_engine = first.path.read_bytes()
    assert old_engine.startswith(b"TENSORACLE\x01")
    offset = len(b"TENSORACLE\x01")
    old_header_size = int.from_bytes(old_engine[offset : offset + 8], "big")
    old_header = json.loads(old_engine[offset + 8 : offset + 8 + old_header_size])
    assert old_header.keys() == {"conditions", "plan_sha256"}

    before = weights.stat()
    changed = bytearray(weights.read_bytes())
    changed[-4:] = np.float32(0.625).tobytes()
    weights.write_bytes(changed)
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (weights.stat().st_size, weights.stat().st_mtime_ns, weights.stat().st_ino) == (
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
    )

    new_conditions = dict(old_header["conditions"])
    new_conditions["weights"] = [hashlib.sha256(changed).hexdigest()]
    expected_digest = hashlib.sha256(
        json.dumps(new_conditions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    rebuilt = open_cached(force_rebuild=True)
    assert rebuilt.path == first.path.parent / f"{expected_digest}.engine"
    assert rebuilt.path != first.path
    assert first.path.read_bytes() == old_engine
    new_engine = rebuilt.path.read_bytes()
    assert new_engine.startswith(b"TENSORACLE\x01")
    new_header_size = int.from_bytes(new_engine[offset : offset + 8], "big")
    new_header = json.loads(new_engine[offset + 8 : offset + 8 + new_header_size])
    assert new_header.keys() == old_header.keys()
    assert new_header["conditions"] == new_conditions


@pytest.mark.req("REQ-TENSORACLE-008")
def test_build_failure_removes_placeholder_and_partial_files(tmp_path, monkeypatch):
    """A failed graph leaves neither a placeholder nor a complete cache file."""
    import tensoracle._cache as cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    monkeypatch.setattr(
        cache, "_build_serialized_engine", lambda graph, declaration, plugins, logger: graph(None)
    )
    weights = tmp_path / "weights.safetensors"
    _weights(weights)

    def fail(network):
        raise RuntimeError("graph failed")

    with pytest.raises(RuntimeError, match="graph failed"):
        tensoracle.open_engine(
            fail,
            tensoracle.EngineBuild("fp32", {"features": tensoracle.InputShape.fixed((1, 3, 2))}),
            name="encoder",
            graph_version="1",
            weights=(weights,),
            io=tensoracle.EngineIO(
                {"features": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
                {"result": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
            ),
        )
    cache_files = list((tmp_path / "cache" / "tensoracle").iterdir())
    assert len(cache_files) == 1
    assert cache_files[0].suffix == ".json"


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-007")
def test_unset_xdg_cache_home_uses_user_cache_directory(tmp_path, monkeypatch):
    """Without XDG_CACHE_HOME, the saved engine is under the user's home."""
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))
    weights = tmp_path / "weights.safetensors"
    _weights(weights)

    handle = tensoracle.open_engine(
        _graph(weights, "fp32", (1, 3, 2)),
        tensoracle.EngineBuild(
            "fp32",
            {"features": tensoracle.InputShape.fixed((1, 3, 2))},
            tensoracle.BuilderSettings(optimization_level=0),
        ),
        name="encoder",
        graph_version="1",
        weights=(weights,),
        io=tensoracle.EngineIO(
            {"features": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
            {"result": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
        ),
    )
    assert handle.path.is_file()
    assert handle.path.parent == tmp_path / "user-home" / ".cache" / "tensoracle"


@pytest.mark.req("REQ-TENSORACLE-007")
def test_each_declared_condition_selects_a_distinct_cache_entry(tmp_path, monkeypatch):
    """Each identity condition independently causes a new build and cache path."""
    import tensoracle._cache as cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weights = tmp_path / "weights.safetensors"
    _weights(weights)
    same_contents = tmp_path / "same.safetensors"
    same_contents.write_bytes(weights.read_bytes())
    changed_weights = tmp_path / "changed.safetensors"
    save_file({"different": np.array([1, 2], dtype=np.float32)}, str(changed_weights))
    compiler = shutil.which("cc")
    if compiler is None or sys.platform != "linux":
        pytest.skip("a Linux C compiler is needed to build plugin libraries")
    plugin_a = tmp_path / "plugin-a.so"
    plugin_b = tmp_path / "plugin-b.so"
    for label, path in (("a", plugin_a), ("b", plugin_b)):
        source = tmp_path / f"plugin-{label}.c"
        source.write_text(
            f"int {label}(void) {{ return 1; }}\n"
            "void *getCreators(int *count) { *count = 0; return 0; }\n",
            encoding="utf-8",
        )
        subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(path)], check=True)

    class PluginRegistry:
        def load_library(self, path):
            return object()

    monkeypatch.setattr(trt, "get_plugin_registry", PluginRegistry)

    state = {"builds": 0, "graph_calls": 0, "shape": (1, 3, 2), "range": None}

    class FakeEngine:
        num_io_tensors = 2

        def get_tensor_name(self, index):
            return ("features", "result")[index]

        def get_tensor_mode(self, name):
            return trt.TensorIOMode.INPUT if name == "features" else trt.TensorIOMode.OUTPUT

        def get_tensor_dtype(self, name):
            return trt.float32

        def get_tensor_shape(self, name):
            return state["shape"]

        def get_tensor_profile_shape(self, name, index):
            return state["range"]

    class FakeRuntime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, plan):
            return FakeEngine()

    class FakeNetwork:
        num_inputs = 1
        num_outputs = 0

        def get_input(self, index):
            return type("Input", (), {"name": "features", "shape": state["shape"]})()

    class FakeProfile:
        def set_shape(self, name, minimum, optimum, maximum):
            return None

    class FakeConfig:
        def clear_flag(self, flag):
            pass

        def set_memory_pool_limit(self, pool, limit):
            pass

        def add_optimization_profile(self, profile):
            pass

    class FakeBuilder:
        def __init__(self, logger):
            pass

        def create_network(self, flags):
            return FakeNetwork()

        def create_builder_config(self):
            return FakeConfig()

        def create_optimization_profile(self):
            return FakeProfile()

        def build_serialized_network(self, network, config):
            state["builds"] += 1
            return b"small plan"

    monkeypatch.setattr(trt, "Builder", FakeBuilder)
    monkeypatch.setattr(trt, "Runtime", FakeRuntime)

    def open_case(
        *,
        name="encoder",
        graph_version="1",
        weight_files=None,
        plugin_files=(),
        precision="fp32",
        shape=(1, 3, 2),
        input_range=None,
        settings=None,
        graph=None,
    ):
        state["shape"] = (1, -1, 2) if input_range else shape
        state["range"] = input_range
        input_shape = (
            tensoracle.InputShape.ranged(*input_range)
            if input_range
            else tensoracle.InputShape.fixed(shape)
        )
        declaration = tensoracle.EngineBuild(
            precision,
            {"features": input_shape},
            settings or tensoracle.BuilderSettings(optimization_level=0),
        )
        io = tensoracle.EngineIO(
            {"features": tensoracle.TensorSpec(trt.float32, state["shape"])},
            {"result": tensoracle.TensorSpec(trt.float32, state["shape"])},
        )

        def counted(network):
            state["graph_calls"] += 1

        return tensoracle.open_engine(
            graph or counted,
            declaration,
            name=name,
            graph_version=graph_version,
            weights=weight_files or (weights,),
            plugins=plugin_files,
            io=io,
        )

    first = open_case()
    assert state["builds"] == 1
    assert open_case(weight_files=(same_contents,)).path == first.path
    assert state["builds"] == 1
    variants = [
        {"name": "decoder"},
        {"graph_version": "2"},
        {"weight_files": (changed_weights,)},
        {"plugin_files": (plugin_a,)},
        {"plugin_files": (plugin_b,)},
        {"precision": "fp16"},
        {"shape": (1, 4, 2)},
        {"input_range": ((1, 1, 2), (1, 3, 2), (1, 5, 2))},
        {"input_range": ((1, 1, 2), (1, 3, 2), (1, 6, 2))},
        {"settings": tensoracle.BuilderSettings(workspace_limit=2048, optimization_level=0)},
        {"settings": tensoracle.BuilderSettings(optimization_level=1)},
    ]
    paths = {open_case(**variant).path for variant in variants}
    with monkeypatch.context() as patch:
        patch.setattr(cache, "_gpu_name", lambda: "GPU B")
        paths.add(open_case().path)
    with monkeypatch.context() as patch:
        patch.setattr(cache.platform, "system", lambda: "Other OS")
        paths.add(open_case().path)
    with monkeypatch.context() as patch:
        patch.setattr(cache.platform, "machine", lambda: "other-arch")
        paths.add(open_case().path)
    with monkeypatch.context() as patch:
        patch.setattr(trt, "__version__", "11.other")
        paths.add(open_case().path)
    assert first.path not in paths
    assert len(paths) == len(variants) + 4
    assert state["builds"] == len(paths) + 1
    assert (
        open_case(graph=lambda network: pytest.fail("graph changed without version bump")).path
        == first.path
    )
    assert state["builds"] == len(paths) + 1


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-002")
def test_open_rejects_mismatched_names_types_dimensions_and_input_range(tmp_path, monkeypatch):
    """A stored engine must match every declared input and output property."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights.safetensors"
    _weights(weights)
    declaration = tensoracle.EngineBuild(
        "fp32",
        {"features": tensoracle.InputShape.ranged((1, 1, 2), (1, 3, 2), (1, 5, 2))},
        tensoracle.BuilderSettings(optimization_level=0),
    )
    input_spec = tensoracle.TensorSpec(trt.float32, (1, -1, 2))
    output_spec = tensoracle.TensorSpec(trt.float32, (1, -1, 2))
    expected = tensoracle.EngineIO({"features": input_spec}, {"result": output_spec})

    def open_with(io):
        return tensoracle.open_engine(
            lambda network: pytest.fail("cached graph was invoked"),
            declaration,
            name="decoder",
            graph_version="1",
            weights=(weights,),
            io=io,
        )

    first = tensoracle.open_engine(
        _graph(weights, "fp32"),
        declaration,
        name="decoder",
        graph_version="1",
        weights=(weights,),
        io=expected,
    )
    assert open_with(expected).path == first.path
    cases = [
        tensoracle.EngineIO({"other": input_spec}, {"result": output_spec}),
        tensoracle.EngineIO({"features": input_spec}, {"other": output_spec}),
        tensoracle.EngineIO(
            {"features": tensoracle.TensorSpec(trt.float16, (1, -1, 2))},
            {"result": output_spec},
        ),
        tensoracle.EngineIO(
            {"features": tensoracle.TensorSpec(trt.float32, (1, -1))},
            {"result": output_spec},
        ),
        tensoracle.EngineIO(
            {"features": tensoracle.TensorSpec(trt.float32, (2, -1, 2))},
            {"result": output_spec},
        ),
        tensoracle.EngineIO(
            {"features": tensoracle.TensorSpec(trt.float32, (1, 3, -1))},
            {"result": output_spec},
        ),
        tensoracle.EngineIO(
            {"features": input_spec},
            {"result": tensoracle.TensorSpec(trt.float16, (1, -1, 2))},
        ),
        tensoracle.EngineIO(
            {"features": input_spec},
            {"result": tensoracle.TensorSpec(trt.float32, (1, -1, 3))},
        ),
    ]
    for io in cases:
        with pytest.raises(tensoracle.EngineIOError, match=re.escape(str(first.path))):
            open_with(io)

    class WrongProfile:
        def __getattr__(self, name):
            return getattr(first.engine, name)

        def get_tensor_profile_shape(self, name, index):
            return ((1, 1, 2), (1, 3, 2), (1, 6, 2))

    class FakeRuntime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, plan):
            return WrongProfile()

    with monkeypatch.context() as patch:
        patch.setattr(trt, "Runtime", FakeRuntime)
        with pytest.raises(tensoracle.EngineIOError, match="range"):
            open_with(expected)


@pytest.mark.gpu
@pytest.mark.req("REQ-TENSORACLE-008")
def test_second_process_rejects_exclusive_placeholder_and_force_rebuilds_it(tmp_path, monkeypatch):
    """One process builds while another sees its placeholder as a corrupt engine."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = tmp_path / "weights.safetensors"
    _weights(weights)
    declaration = tensoracle.EngineBuild(
        "fp32",
        {"features": tensoracle.InputShape.fixed((1, 3, 2))},
        tensoracle.BuilderSettings(optimization_level=0),
    )
    io = tensoracle.EngineIO(
        {"features": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
        {"result": tensoracle.TensorSpec(trt.float32, (1, 3, 2))},
    )
    child = """
import sys
import tensorrt as trt
from pathlib import Path
from tensoracle import (
    BuilderSettings, CorruptEngineError, EngineBuild, EngineIO, InputShape, TensorSpec, open_engine
)
try:
    open_engine(
        lambda network: (_ for _ in ()).throw(AssertionError('second builder started')),
        EngineBuild('fp32', {'features': InputShape.fixed((1, 3, 2))}, BuilderSettings(optimization_level=0)),
        name='encoder', graph_version='1', weights=(Path(sys.argv[1]),),
        io=EngineIO({'features': TensorSpec(trt.float32, (1, 3, 2))},
                    {'result': TensorSpec(trt.float32, (1, 3, 2))}),
    )
except CorruptEngineError:
    print('placeholder rejected')
else:
    raise AssertionError('placeholder was accepted')
"""
    graph_calls = []

    def graph(network):
        graph_calls.append(1)
        result = subprocess.run(
            [sys.executable, "-c", child, str(weights)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "placeholder rejected"
        _graph(weights, "fp32", (1, 3, 2))(network)

    first = tensoracle.open_engine(
        graph,
        declaration,
        name="encoder",
        graph_version="1",
        weights=(weights,),
        io=io,
    )
    assert graph_calls == [1]
    assert first.path.is_file()
    first.path.write_bytes(b"")
    with pytest.raises(tensoracle.CorruptEngineError):
        tensoracle.open_engine(
            lambda network: pytest.fail("placeholder caused another build"),
            declaration,
            name="encoder",
            graph_version="1",
            weights=(weights,),
            io=io,
        )
    forced = tensoracle.open_engine(
        _graph(weights, "fp32", (1, 3, 2)),
        declaration,
        name="encoder",
        graph_version="1",
        weights=(weights,),
        io=io,
        force_rebuild=True,
    )
    assert forced.path == first.path
    assert forced.engine.get_tensor_name(0) == "features"
