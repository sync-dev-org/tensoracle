"""Linked cubin plugins register and inspect memory in executables and shared modules."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from native_build import ROOT


def run(*command, **kwargs):
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=180, **kwargs
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def embedded_sdk(tmp_path_factory):
    work = tmp_path_factory.mktemp("embedded-sdk")
    build = work / "build"
    environment = {**os.environ, "TENSORACLE_CONAN_OFFLINE": "ON"}
    run(
        "cmake",
        "-S",
        str(ROOT / "cpp"),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DBUILD_TESTING=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        env=environment,
    )
    run("cmake", "--build", str(build), "--parallel", "8", env=environment)
    prefix = work / "prefix"
    run("cmake", "--install", str(build), "--prefix", str(prefix), env=environment)
    return work, build, prefix, environment


@pytest.fixture(
    scope="module",
    params=[("STATIC", False), ("OBJECT", False), ("STATIC", True), ("SHARED", True)],
)
def embedded_consumer(embedded_sdk, request):
    return build_consumer(embedded_sdk, *request.param)


@pytest.fixture(scope="module")
def relative_consumer(embedded_sdk):
    return build_consumer(embedded_sdk, "STATIC", True, "relative")


def build_consumer(embedded_sdk, kind, shared, name=None):
    work, sdk_build, prefix, environment = embedded_sdk
    directory = work / (name or f"{kind}-{'shared' if shared else 'executable'}")
    source, build = directory / "source", directory / "build"
    shutil.copytree(ROOT / "cpp/tests/embedded_plugin", source)
    bundle = ROOT / "examples/cubin_plugin/bundle"
    manifest = (bundle / "bundle.json").read_text()
    entries = json.loads(manifest)["entries"]
    generated = ["#include <tensoracle/cubin.h>\n#include <cstddef>\n"]
    for entry in entries:
        generated.append(f'extern "C" const unsigned char {entry["embedded_symbol"]}[];\n')
    generated.append(f"const char* embedded_manifest() {{ return {json.dumps(manifest)}; }}\n")
    generated.append("const tensoracle_bundle_file* const* embedded_files(size_t& count) {\n")
    for i, entry in enumerate(entries):
        generated.append(
            f"static const tensoracle_bundle_file file{i}{{sizeof(tensoracle_bundle_file),"
            f"{json.dumps(entry['file'])},{entry['embedded_symbol']},{entry['bytes']}}};\n"
        )
    generated.append("static const tensoracle_bundle_file* files[]{")
    generated.append(",".join(f"&file{i}" for i in range(len(entries))))
    generated.append("}; count=sizeof(files)/sizeof(files[0]); return files; }\n")
    (source / "bundle_data.cpp").write_text("".join(generated))
    run(
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DCMAKE_TOOLCHAIN_FILE={sdk_build / 'conan/conan_toolchain.cmake'}",
        f"-DEXAMPLE_ROOT={ROOT / 'examples/cubin_plugin'}",
        f"-DPLUGIN_TYPE={kind}",
        f"-DOWNER_SHARED={'ON' if shared else 'OFF'}",
        *(("-DCMAKE_EXE_LINKER_FLAGS=-no-pie",) if kind == "STATIC" and not shared else ()),
        env=environment,
    )
    run("cmake", "--build", str(build), "--parallel", "8", env=environment)
    shutil.rmtree(source)
    import tensorrt_libs

    environment = {**environment, "LD_LIBRARY_PATH": str(Path(tensorrt_libs.__file__).parent)}
    return directory, build, environment


def probe(consumer, weights=None):
    directory, build, environment = consumer
    output = run(
        str(build / "probe"),
        *((str(weights),) if weights else ()),
        cwd=directory,
        env={**environment, "XDG_CACHE_HOME": str(directory / "cache")},
    )
    return json.loads(output.splitlines()[-1])


def relative_probe(consumer, weights=None):
    directory, build, environment = consumer
    output = run(
        str(build / "relative_probe"),
        "./libmodel.so",
        str(directory),
        *((str(weights),) if weights else ()),
        cwd=build,
        env={**environment, "XDG_CACHE_HOME": str(directory / "cache")},
    )
    return json.loads(output.splitlines()[-1])


@pytest.mark.req("SP-TENSORACLE-077")
@pytest.mark.req("SP-TENSORACLE-078")
def test_relative_module_registration_survives_changing_working_directory(relative_consumer):
    """A relative dlopen followed by chdir still identifies the loaded model file."""
    _, build, _ = relative_consumer
    result = relative_probe(relative_consumer)
    container = Path((build / "container.txt").read_text())
    assert result["identity"] == hashlib.sha256(container.read_bytes()).hexdigest()


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-077")
@pytest.mark.req("SP-TENSORACLE-078")
@pytest.mark.req("SP-TENSORACLE-009")
def test_relative_module_builder_reuses_registered_identity_after_chdir(relative_consumer):
    """The builder checks the containing file after chdir and reuses only matching bytes."""
    _, build, _ = relative_consumer
    weights = ROOT / "examples/cubin_plugin/example_cubin_plugin/weights.safetensors"
    first = relative_probe(relative_consumer, weights)
    reused = relative_probe(relative_consumer, weights)
    assert (first["build_calls"], reused["build_calls"]) == (1, 0)
    assert first["path"] == reused["path"]
    container = Path((build / "container.txt").read_text())
    container.write_bytes(container.read_bytes() + b"changed relative module")
    changed = relative_probe(relative_consumer, weights)
    assert changed["identity"] == hashlib.sha256(container.read_bytes()).hexdigest()
    assert changed["identity"] != first["identity"]
    assert changed["build_calls"] == 1
    assert changed["path"] != first["path"]


@pytest.mark.req("SP-TENSORACLE-077")
@pytest.mark.req("SP-TENSORACLE-078")
def test_linked_creators_and_bundle_need_no_separate_plugin_or_bundle_files(embedded_consumer):
    """Linked creators register in process and identify their complete containing file."""
    _, build, _ = embedded_consumer
    result = probe(embedded_consumer)
    container = Path((build / "container.txt").read_text())
    assert result["identity"] == hashlib.sha256(container.read_bytes()).hexdigest()
    manifest = json.loads((ROOT / "examples/cubin_plugin/bundle/bundle.json").read_text())
    assert result["bundle"] == manifest["bundle_sha256"]


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-077")
@pytest.mark.req("SP-TENSORACLE-078")
@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.req("SP-TENSORACLE-055")
def test_embedded_plugin_executes_reuses_after_restart_and_rebuilds_for_changed_file(
    embedded_consumer,
):
    """A linked cubin plugin executes, reuses its saved engine and rebuilds after container changes."""
    _, build, _ = embedded_consumer
    weights = ROOT / "examples/cubin_plugin/example_cubin_plugin/weights.safetensors"
    first, reused = probe(embedded_consumer, weights), probe(embedded_consumer, weights)
    assert (first["build_calls"], reused["build_calls"]) == (1, 0)
    assert first["path"] == reused["path"]
    container = Path((build / "container.txt").read_text())
    container.write_bytes(container.read_bytes() + b"changed containing file")
    changed = probe(embedded_consumer, weights)
    assert changed["identity"] == hashlib.sha256(container.read_bytes()).hexdigest()
    assert changed["identity"] != first["identity"]
    assert changed["build_calls"] == 1
    assert changed["path"] != first["path"]
