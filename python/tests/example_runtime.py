"""Run the installed example wheel in a separate, producer-free Python process."""

import argparse
import ctypes
import gc
import importlib.util
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from example_cubin_plugin import declaration, expected_io, graph, library_path, weights_path
from safetensors.numpy import load_file

from tensoracle import build_serialized_engine, compare_outputs, open_engine
from tensoracle._plugins import PluginLibraryError, register_libraries


def _cuda(result):
    status, *values = result
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA runtime returned {status}")
    return values[0] if values else None


def _run(handle, rows):
    execution = handle.create_execution()
    values = load_file(weights_path())
    inputs = np.full((rows, 32), np.float32(1.0001), dtype=np.float32)
    expected = (inputs @ values["matrix"]) * values["scale"] + values["bias"]
    execution.set_input_shapes({"features": inputs.shape})
    output = np.empty(
        execution.get_tensor_shape("result"),
        dtype=trt.nptype(execution.get_tensor_dtype("result")),
    )
    source = _cuda(cudart.cudaMalloc(inputs.nbytes))
    target = _cuda(cudart.cudaMalloc(output.nbytes))
    stream = _cuda(cudart.cudaStreamCreate())
    try:
        _cuda(
            cudart.cudaMemcpy(
                source,
                inputs.ctypes.data,
                inputs.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        execution.enqueue({"features": int(source), "result": int(target)}, stream=int(stream))
        _cuda(cudart.cudaStreamSynchronize(stream))
        _cuda(
            cudart.cudaMemcpy(
                output.ctypes.data,
                target,
                output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
        )
        comparison = compare_outputs(output, expected, atol=1e-4, rtol=0)
        return {
            "rows": rows,
            "actual": float(output[0, 0]),
            "expected": float(expected[0, 0]),
            "outside_tolerance_count": comparison.outside_tolerance_count,
        }
    finally:
        # Wait before closing even if a later copy or comparison fails.
        _cuda(cudart.cudaStreamSynchronize(stream))
        execution.close()
        _cuda(cudart.cudaStreamDestroy(stream))
        _cuda(cudart.cudaFree(target))
        _cuda(cudart.cudaFree(source))


def _open(mode, extra):
    library = library_path()
    if mode == "changed":
        library = Path(extra) / library.name
        shutil.copyfile(library_path(), library)
        with library.open("ab") as stream:
            stream.write(b"changed plugin bytes")
    if mode == "reuse":

        def fail(_):
            raise AssertionError("the graph was called while reopening a saved engine")

        graph_function = fail
    else:
        graph_function = graph
    handle = open_engine(
        graph_function,
        declaration(),
        name="cubin-example",
        graph_version="1",
        weights=(weights_path(),),
        plugins=(library,),
        io=expected_io(),
    )
    return handle


def _control():
    """Run the caller's TF32 control cubin without Triton or a CUDA SDK."""
    manifest = json.loads((library_path().parent / "bundle/bundle.json").read_text())
    _cuda(cudart.cudaFree(0))  # Establish the runtime's CUDA context.
    device = _cuda(cudart.cudaGetDevice())
    properties = _cuda(cudart.cudaGetDeviceProperties(device))
    sm = f"sm{properties.major * 10 + properties.minor}"
    entry = next(item for item in manifest["entries"] if item["id"] == f"{sm}.dot.tf32-control")
    image = (library_path().parent / "bundle" / entry["file"]).read_bytes()
    driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")

    def call(name, *args):
        method = getattr(driver, name)
        method.restype = ctypes.c_int
        result = method(*args)
        if result != 0:
            raise RuntimeError(f"{name} returned CUDA error {result}")

    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    image_buffer = ctypes.create_string_buffer(image)
    call("cuModuleLoadData", ctypes.byref(module), ctypes.cast(image_buffer, ctypes.c_void_p))
    try:
        call(
            "cuModuleGetFunction",
            ctypes.byref(function),
            module,
            ctypes.c_char_p(entry["entry_symbol"].encode()),
        )
        inputs = np.full((16, 32), np.float32(1.0001), dtype=np.float32)
        matrix = load_file(weights_path())["matrix"]
        output = np.empty((16, 16), dtype=np.float32)
        source = _cuda(cudart.cudaMalloc(inputs.nbytes))
        weight = _cuda(cudart.cudaMalloc(matrix.nbytes))
        target = _cuda(cudart.cudaMalloc(output.nbytes))
        try:
            for pointer, array in ((source, inputs), (weight, matrix)):
                _cuda(
                    cudart.cudaMemcpy(
                        pointer,
                        array.ctypes.data,
                        array.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )
                )
            arguments = [
                ctypes.c_uint64(int(source)),
                ctypes.c_uint64(int(weight)),
                ctypes.c_uint64(int(target)),
                ctypes.c_int32(16),
                ctypes.c_uint64(0),
                ctypes.c_uint64(0),
            ]
            pointers = (ctypes.c_void_p * len(arguments))(
                *(ctypes.addressof(argument) for argument in arguments)
            )
            call(
                "cuLaunchKernel",
                function,
                ctypes.c_uint(1),
                ctypes.c_uint(1),
                ctypes.c_uint(1),
                ctypes.c_uint(128),
                ctypes.c_uint(1),
                ctypes.c_uint(1),
                ctypes.c_uint(entry["resources"]["launch_dynamic_shared_memory_bytes"]),
                ctypes.c_void_p(),
                pointers,
                ctypes.c_void_p(),
            )
            call("cuCtxSynchronize")
            _cuda(
                cudart.cudaMemcpy(
                    output.ctypes.data,
                    target,
                    output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )
            return {"tf32": float(output[0, 0]), "ieee_expected": float((inputs @ matrix)[0, 0])}
        finally:
            _cuda(cudart.cudaFree(target))
            _cuda(cudart.cudaFree(weight))
            _cuda(cudart.cudaFree(source))
    finally:
        call("cuModuleUnload", module)


def _separate_cuda_contexts():
    """Reopen the cached plan in two owner contexts at the same time."""
    driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")

    def call(name, *args):
        method = getattr(driver, name)
        method.restype = ctypes.c_int
        status = method(*args)
        if status:
            raise RuntimeError(f"{name} returned CUDA error {status}")

    call("cuInit", ctypes.c_uint(0))

    def run(rows):
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        call("cuDeviceGet", ctypes.byref(device), ctypes.c_int(0))
        call("cuCtxCreate_v2", ctypes.byref(context), ctypes.c_uint(0), device)
        try:
            handle = _open("reuse", None)
            result = _run(handle, rows)
            del handle
            gc.collect()
            return result
        finally:
            call("cuCtxDestroy_v2", context)

    with ThreadPoolExecutor(max_workers=2) as pool:
        return {"rows": list(pool.map(run, (3, 16)))}


def _duplicate_creator(directory):
    """A second library with the same creators must not silently replace the first."""
    import tensorrt as trt

    original = library_path()
    register_libraries((original,))
    registry = trt.get_plugin_registry()
    assert registry.get_creator("TensoracleExampleDot", "1", "tensoracle.example") is not None
    changed = Path(directory) / f"same-creators-changed-bytes{original.suffix}"
    shutil.copyfile(original, changed)
    with changed.open("ab") as stream:
        stream.write(b"different bytes with the same creators")

    def fail(_):
        raise AssertionError("a graph used the old creator for changed plugin bytes")

    try:
        build_serialized_engine(fail, declaration(), plugins=(changed,))
    except PluginLibraryError as error:
        assert registry.get_creator("TensoracleExampleDot", "1", "tensoracle.example") is not None
        return {"error_type": type(error).__name__, "graph_called": False}
    raise AssertionError("TensorRT accepted different bytes for an existing creator")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "build",
            "reuse",
            "changed",
            "concurrent",
            "separate-contexts",
            "duplicate-creator",
            "serialized",
            "control",
        ),
    )
    parser.add_argument("--extra")
    arguments = parser.parse_args()
    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("triton") is None
    if arguments.mode == "control":
        result = _control()
    elif arguments.mode == "separate-contexts":
        result = _separate_cuda_contexts()
    elif arguments.mode == "duplicate-creator":
        result = _duplicate_creator(arguments.extra)
    elif arguments.mode == "serialized":
        plan = build_serialized_engine(graph, declaration(), plugins=(library_path(),))
        result = {"plan_bytes": len(plan)}
    else:
        handle = _open(arguments.mode, arguments.extra)
        if arguments.mode == "concurrent":
            with ThreadPoolExecutor(max_workers=2) as pool:
                result_rows = list(pool.map(lambda n: _run(handle, n), (3, 16)))
        else:
            result_rows = [_run(handle, 3), _run(handle, 16)]
        native = ctypes.CDLL(str(library_path()))
        native.tensoracle_example_specialized_launches.restype = ctypes.c_uint64
        native.tensoracle_example_unspecialized_launches.restype = ctypes.c_uint64
        result = {
            "path": str(handle.path),
            "rows": result_rows,
            "specialized": native.tensoracle_example_specialized_launches(),
            "unspecialized": native.tensoracle_example_unspecialized_launches(),
        }
    assert "triton" not in sys.modules
    assert "torch" not in sys.modules
    assert "tensoracle.producer" not in sys.modules
    print(json.dumps(result))


if __name__ == "__main__":
    main()
