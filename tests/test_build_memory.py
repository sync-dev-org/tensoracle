"""Return host memory used only while building TensorRT engines."""

import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorrt as trt

from tensoracle import EngineBuild, build_serialized_engine


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-001")
@pytest.mark.parametrize("entry", ["build_serialized_engine", "open_engine"])
@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_build_returns_unused_host_memory_after_success_or_failure(tmp_path, entry, outcome):
    """A second trim frees at most 32 MiB after success or a no-output build failure.

    The unchanged implementation freed 1,840,877,568 / 1,842,180,096 bytes after success
    and 335,601,664 / 335,933,440 bytes after failure (direct / cached entry). The limit
    is one tenth of the smallest observed reduction, rounded down to whole MiB.
    """
    if sys.platform != "linux":
        pytest.skip("RSS measurement requires Linux /proc")
    library = ctypes.util.find_library("c")
    if library is None or not hasattr(ctypes.CDLL(library), "malloc_trim"):
        pytest.skip("RSS regression requires glibc malloc_trim")
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_memory_runtime.py")),
            entry,
            outcome,
            str(tmp_path),
        ],
        env={**os.environ, "XDG_CACHE_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    measured = json.loads(completed.stdout.splitlines()[-1])
    print(f"{entry} {outcome}: {measured}")
    assert measured["reduction"] <= 32 * 1024 * 1024, measured


@pytest.mark.req("SP-TENSORACLE-015")
@pytest.mark.parametrize("unavailable", ["library", "loader", "symbol"])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_unavailable_host_memory_release_preserves_the_build_result(monkeypatch, unavailable, fail):
    """Absent libc, an unloadable libc, or a missing trim leaves bytes and errors unchanged."""
    failure = RuntimeError("original builder failure")

    def serialize(network, config):
        if fail:
            raise failure
        return b"serialized plan"

    builder = SimpleNamespace(
        create_network=lambda flags: SimpleNamespace(num_inputs=0, num_outputs=0),
        create_builder_config=lambda: SimpleNamespace(
            clear_flag=lambda flag: None,
            set_memory_pool_limit=lambda pool, limit: None,
        ),
        build_serialized_network=serialize,
    )
    monkeypatch.setattr(trt, "Builder", lambda logger: builder)
    monkeypatch.setattr(
        ctypes.util, "find_library", lambda name: None if unavailable == "library" else "libc"
    )

    def load(name):
        if unavailable == "library":
            pytest.fail("a missing libc must not be loaded")
        if unavailable == "loader":
            raise OSError("libc cannot be loaded")
        return SimpleNamespace()

    monkeypatch.setattr(ctypes, "CDLL", load)
    declaration = EngineBuild("fp32", {})
    if fail:
        with pytest.raises(RuntimeError) as caught:
            build_serialized_engine(lambda network: None, declaration)
        assert caught.value is failure
    else:
        assert build_serialized_engine(lambda network: None, declaration) == b"serialized plan"
