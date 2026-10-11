"""Plugin libraries are checked before TensorRT sees their creators."""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import tensorrt as trt

from tensoracle import (
    EngineBuild,
    EngineIO,
    InputShape,
    TensorSpec,
    build_serialized_engine,
    open_engine,
)
from tensoracle._plugins import PluginLibraryError, _pe_exports, exported_data, register_libraries


def _library(tmp_path: Path, name: str, source: str) -> Path:
    compiler = shutil.which("cc")
    if compiler is None or sys.platform != "linux":
        pytest.skip("a Linux C compiler is needed for native library checks")
    path = tmp_path / f"lib{name}.so"
    source_path = tmp_path / f"{name}.c"
    source_path.write_text(source, encoding="utf-8")
    subprocess.run([compiler, "-shared", "-fPIC", str(source_path), "-o", str(path)], check=True)
    return path


@pytest.mark.req("REQ-TENSORACLE-002")
def test_library_without_get_creators_is_rejected_before_loading(tmp_path, monkeypatch):
    """A library with only a static constructor cannot destabilize the registry."""
    marker = tmp_path / "loaded"
    library = _library(
        tmp_path,
        "unsafe",
        f"#include <stdio.h>\n__attribute__((constructor)) void start(void) {{ "
        f'FILE *f = fopen("{marker}", "w"); if (f) fclose(f); }}\n',
    )
    monkeypatch.setattr(trt, "get_plugin_registry", lambda: pytest.fail("registry was used"))
    with pytest.raises(PluginLibraryError, match="getCreators"):
        register_libraries((library,))
    assert not marker.exists()


@pytest.mark.req("REQ-TENSORACLE-003")
def test_library_is_registered_once_before_graph_and_kept_alive(tmp_path, monkeypatch):
    """Repeated calls register one library once before a graph needs its creator."""
    library = _library(
        tmp_path,
        "valid",
        "void *getCreators(int *count) { *count = 0; return 0; }\n",
    )
    observed = []
    handle = object()

    class Registry:
        def load_library(self, path):
            observed.append(Path(path))
            return handle

    monkeypatch.setattr(trt, "get_plugin_registry", Registry)
    register_libraries((library, library))
    register_libraries((library,))
    assert observed == [library.resolve()]

    class Builder:
        def __init__(self, logger):
            pass

        def create_network(self, flags):
            return object()

    monkeypatch.setattr(trt, "Builder", Builder)
    declaration = EngineBuild("fp32", {"x": InputShape.fixed((1,))})

    def graph(network):
        assert observed == [library.resolve()]
        raise RuntimeError("graph reached")

    with pytest.raises(RuntimeError, match="graph reached"):
        build_serialized_engine(graph, declaration, plugins=(library,))
    assert observed == [library.resolve()]


@pytest.mark.req("REQ-TENSORACLE-003")
def test_changed_library_at_same_path_is_not_silently_reused(tmp_path, monkeypatch):
    """Changing loaded library bytes cannot label a new engine with an old creator."""
    library = _library(tmp_path, "versioned", "void *getCreators(int *n) { *n = 0; return 0; }\n")

    class Registry:
        def load_library(self, path):
            return object()

    monkeypatch.setattr(trt, "get_plugin_registry", Registry)
    register_libraries((library,))
    library.write_bytes(library.read_bytes() + b"changed")
    with pytest.raises(PluginLibraryError, match="changed"):
        register_libraries((library,))


@pytest.mark.req("REQ-TENSORACLE-008")
def test_missing_plugin_has_a_library_error_before_engine_work(tmp_path, monkeypatch):
    """A missing plugin is reported as a plugin problem, not a damaged engine."""
    monkeypatch.setattr(trt, "get_plugin_registry", lambda: pytest.fail("registry was used"))
    with pytest.raises(PluginLibraryError, match="missing"):
        register_libraries((tmp_path / "missing.so",))


@pytest.mark.req("REQ-TENSORACLE-008")
def test_open_engine_reports_missing_plugin_before_cache_or_graph(tmp_path):
    """A missing library is identified without reading a saved engine or building a graph."""
    declaration = EngineBuild("fp32", {"x": InputShape.fixed((1,))})
    io = EngineIO({"x": TensorSpec(trt.float32, (1,))}, {"y": TensorSpec(trt.float32, (1,))})
    with pytest.raises(PluginLibraryError, match="missing"):
        open_engine(
            lambda network: pytest.fail("graph was built"),
            declaration,
            name="missing-plugin",
            graph_version="1",
            weights=(tmp_path / "unused.safetensors",),
            io=io,
            plugins=(tmp_path / "missing.so",),
        )


@pytest.mark.req("REQ-TENSORACLE-011")
def test_pe_export_inspection_uses_names_and_data_without_loading(monkeypatch):
    """Windows export names and bundle data can be inspected without running a DLL."""
    dll = bytearray(1024)
    dll[:2] = b"MZ"
    struct.pack_into("<I", dll, 0x3C, 0x80)
    dll[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", dll, 0x86, 1)
    struct.pack_into("<H", dll, 0x94, 0xF0)
    struct.pack_into("<H", dll, 0x98, 0x20B)
    struct.pack_into("<I", dll, 0x98 + 112, 0x1000)
    struct.pack_into("<IIII", dll, 0x188 + 8, 0x200, 0x1000, 0x200, 0x200)
    struct.pack_into("<I", dll, 0x200 + 24, 1)
    struct.pack_into("<I", dll, 0x200 + 28, 0x1030)
    struct.pack_into("<I", dll, 0x200 + 32, 0x1040)
    struct.pack_into("<I", dll, 0x200 + 36, 0x1048)
    struct.pack_into("<I", dll, 0x230, 0x1080)
    struct.pack_into("<I", dll, 0x240, 0x1050)
    struct.pack_into("<H", dll, 0x248, 0)
    dll[0x250:0x269] = b"tensoracle_bundle_sha256\0"
    dll[0x280:0x285] = b"abcd\0"
    assert _pe_exports(bytes(dll)) == {"tensoracle_bundle_sha256"}
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "win32")
        assert exported_data(bytes(dll), "tensoracle_bundle_sha256", 5) == b"abcd\0"
