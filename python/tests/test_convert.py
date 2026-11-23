"""Convert tensor checkpoints without executing checkpoint code or requiring torch for inference."""

import ctypes
import os
import subprocess
import sys
from collections import OrderedDict
from types import MappingProxyType

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch", reason="checkpoint conversion requires tensoracle[convert]")


def _convert(source, destination):
    from tensoracle.convert import pth_to_safetensors

    pth_to_safetensors(source, destination)


def _assert_tensors_match(destination, expected):
    from safetensors.torch import load_file

    actual = load_file(destination)
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert actual[name].dtype == tensor.dtype
        assert actual[name].shape == tensor.shape
        assert actual[name].equal(tensor)


def _read_metadata_with_tensoracle(path):
    from tensoracle import _native

    major = _native.version()[0]
    library = ctypes.CDLL(
        f"tensoracle-{major}.dll" if os.name == "nt" else f"libtensoracle.so.{major}"
    )

    class MetadataEntry(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_size_t),
            ("key", ctypes.c_void_p),
            ("key_size", ctypes.c_size_t),
            ("value", ctypes.c_void_p),
            ("value_size", ctypes.c_size_t),
        ]

    handle = ctypes.c_void_p
    output = ctypes.POINTER(handle)
    signatures = {
        "tensoracle_weights_open": [ctypes.c_char_p, output, output],
        "tensoracle_weights_metadata_count": [handle, ctypes.POINTER(ctypes.c_size_t), output],
        "tensoracle_weights_metadata_at": [
            handle,
            ctypes.c_size_t,
            ctypes.POINTER(MetadataEntry),
            output,
        ],
        "tensoracle_weights_destroy": [handle],
        "tensoracle_error_message": [handle, ctypes.POINTER(ctypes.c_char_p), output],
        "tensoracle_error_destroy": [handle],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int32

    error = handle()
    weights = handle()

    def check(status):
        if status:
            message = ctypes.c_char_p()
            library.tensoracle_error_message(error, ctypes.byref(message), None)
            pytest.fail(f"tensoracle status {status}: {message.value!r}")

    try:
        check(
            library.tensoracle_weights_open(
                str(path).encode("utf-8"), ctypes.byref(weights), ctypes.byref(error)
            )
        )
        count = ctypes.c_size_t()
        check(
            library.tensoracle_weights_metadata_count(
                weights, ctypes.byref(count), ctypes.byref(error)
            )
        )
        metadata = {}
        for index in range(count.value):
            entry = MetadataEntry(struct_size=ctypes.sizeof(MetadataEntry))
            check(
                library.tensoracle_weights_metadata_at(
                    weights, index, ctypes.byref(entry), ctypes.byref(error)
                )
            )
            key = ctypes.string_at(entry.key, entry.key_size).decode("utf-8")
            value = ctypes.string_at(entry.value, entry.value_size).decode("utf-8")
            metadata[key] = value
        return metadata
    finally:
        library.tensoracle_weights_destroy(weights)
        library.tensoracle_error_destroy(error)


@pytest.mark.req("SP-TENSORACLE-006")
@pytest.mark.parametrize("nested", [False, True], ids=["state-dict", "nested-mapping"])
def test_function_preserves_tensor_names_dtypes_shapes_and_values(tmp_path, torch, nested):
    """Flat and nested mappings retain every tensor's name, dtype, shape, and values."""
    shared = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tensors = OrderedDict(
        weight=shared,
        alias=shared,
        transposed=shared.T,
        bias=torch.tensor([1.25, -2.5], dtype=torch.float16),
        bf16=torch.tensor([0.5, -3.0], dtype=torch.bfloat16),
        count=torch.tensor(7, dtype=torch.int64),
        mask=torch.tensor([True, False]),
        empty=torch.empty((0, 2)),
    )
    checkpoint = {"state_dict": {"layer": tensors}} if nested else tensors
    expected = {
        f"state_dict.layer.{key}" if nested else key: value for key, value in tensors.items()
    }
    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    torch.save(checkpoint, source)

    _convert(source, destination)

    _assert_tensors_match(destination, expected)


@pytest.mark.req("SP-TENSORACLE-081")
@pytest.mark.parametrize(
    "metadata",
    [
        MappingProxyType({"scale": "4", "input_color": "{...}"}),
        {"説明\0🌐": '重み\0\n"\\', "": ""},
        {},
    ],
    ids=["mapping", "unicode-nul-and-json-escapes", "empty"],
)
def test_metadata_round_trips_without_changing_tensors(tmp_path, torch, metadata):
    """Safetensors and tensoracle read the same caller metadata, including UTF-8 and NUL."""
    from safetensors import safe_open

    from tensoracle.convert import pth_to_safetensors

    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    tensors = {"weight": torch.tensor([1.25, -2.5])}
    torch.save(tensors, source)

    pth_to_safetensors(source, destination, metadata=metadata)

    with safe_open(destination, framework="pt", device="cpu") as weights:
        assert weights.metadata() == metadata
    assert _read_metadata_with_tensoracle(destination) == metadata
    _assert_tensors_match(destination, tensors)


@pytest.mark.req("SP-TENSORACLE-081")
@pytest.mark.parametrize("explicit_none", [False, True], ids=["omitted", "none"])
def test_omitted_metadata_preserves_the_previous_output_bytes(tmp_path, torch, explicit_none):
    """Omitting metadata or passing None retains the previous bytes and absent metadata."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from tensoracle.convert import pth_to_safetensors

    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    previous = tmp_path / "previous.safetensors"
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
    torch.save({"model": {"weight": tensor}}, source)
    save_file({"model.weight": tensor.contiguous()}, previous)

    kwargs = {"metadata": None} if explicit_none else {}
    pth_to_safetensors(source, destination, **kwargs)

    assert destination.read_bytes() == previous.read_bytes()
    with safe_open(destination, framework="pt", device="cpu") as weights:
        assert weights.metadata() is None
    assert _read_metadata_with_tensoracle(destination) == {}


@pytest.mark.req("SP-TENSORACLE-081")
@pytest.mark.parametrize(
    "metadata,diagnostic",
    [
        ([("scale", "4")], "mapping"),
        ({4: "scale"}, "keys.*strings"),
        ({"scale": 4}, "values.*strings"),
    ],
    ids=["not-a-mapping", "non-string-key", "non-string-value"],
)
def test_invalid_metadata_types_are_rejected_before_writing(tmp_path, torch, metadata, diagnostic):
    """Metadata requires a mapping of string keys to string values before creating output."""
    from tensoracle.convert import pth_to_safetensors

    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    torch.save({"weight": torch.ones(2)}, source)

    with pytest.raises(TypeError, match=diagnostic):
        pth_to_safetensors(source, destination, metadata=metadata)

    assert not destination.exists()


@pytest.mark.req("SP-TENSORACLE-081")
@pytest.mark.parametrize("metadata", [{"\ud800": "value"}, {"key": "\udfff"}])
@pytest.mark.parametrize("existing", [False, True], ids=["absent", "existing"])
def test_non_utf8_metadata_is_rejected_before_writing(tmp_path, torch, metadata, existing):
    """Unpaired surrogates in metadata cannot create or overwrite a safetensors file."""
    from tensoracle.convert import pth_to_safetensors

    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    torch.save({"weight": torch.ones(2)}, source)
    if existing:
        destination.write_bytes(b"previous output")

    with pytest.raises(ValueError, match="(?i)utf-8"):
        pth_to_safetensors(source, destination, metadata=metadata)

    if existing:
        assert destination.read_bytes() == b"previous output"
    else:
        assert not destination.exists()


@pytest.mark.req("SP-TENSORACLE-081")
@pytest.mark.parametrize("character", ["a", "é", "\n"], ids=["ascii", "utf8", "escaped"])
def test_metadata_cannot_exceed_the_complete_header_limit(tmp_path, torch, character):
    """UTF-8 bytes, JSON escapes, and tensor descriptors all count toward the header limit."""
    from tensoracle.convert import pth_to_safetensors

    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    torch.save({"weight": torch.ones(2)}, source)
    # This metadata alone would fit, but the tensor descriptor takes it over the limit.
    available = 100_000_000 - len(b'{"__metadata__":{"x":""}}')
    width = len(character.encode("utf-8")) if character != "\n" else 2
    metadata = {"x": character * (available // width)}

    with pytest.raises(ValueError, match="100000000"):
        pth_to_safetensors(source, destination, metadata=metadata)

    assert not destination.exists()


@pytest.mark.req("SP-TENSORACLE-081")
def test_metadata_at_the_header_limit_is_accepted(tmp_path, torch):
    """Metadata that fills the allowed header size remains readable."""
    from safetensors import safe_open

    from tensoracle.convert import pth_to_safetensors

    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    torch.save({}, source)
    metadata = {"x": "a" * (100_000_000 - len(b'{"__metadata__":{"x":""}}'))}

    pth_to_safetensors(source, destination, metadata=metadata)

    with destination.open("rb") as file:
        assert int.from_bytes(file.read(8), "little") == 100_000_000
    with safe_open(destination, framework="pt", device="cpu") as weights:
        assert weights.metadata() == metadata


@pytest.mark.req("SP-TENSORACLE-006")
def test_command_line_writes_the_same_tensors_as_the_function(tmp_path, torch):
    """The module command writes the same flattened weights as the Python function."""
    source = tmp_path / "weights.pth"
    from_function = tmp_path / "function.safetensors"
    from_command = tmp_path / "command.safetensors"
    weight = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    torch.save({"model": {"weight": weight}}, source)
    _convert(str(source), str(from_function))

    result = subprocess.run(
        [sys.executable, "-m", "tensoracle.convert", str(source), str(from_command)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    _assert_tensors_match(from_command, {"model.weight": weight})
    assert from_command.read_bytes() == from_function.read_bytes()


@pytest.mark.req("SP-TENSORACLE-006")
@pytest.mark.parametrize(
    "invalid", [3, "metadata", None, [1, 2]], ids=["number", "text", "null", "list"]
)
def test_checkpoint_metadata_is_rejected_before_writing(tmp_path, torch, invalid):
    """A non-tensor leaf rejects the checkpoint and preserves the destination file."""
    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    torch.save({"model": {"weight": torch.ones(2), "metadata": invalid}}, source)
    destination.write_bytes(b"previous output")

    with pytest.raises(TypeError):
        _convert(source, destination)

    assert destination.read_bytes() == b"previous output"


@pytest.mark.req("SP-TENSORACLE-006")
def test_flattened_name_collisions_are_rejected(tmp_path, torch):
    """Dotted keys and nested keys cannot silently overwrite the same tensor name."""
    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    torch.save({"layer.weight": torch.ones(2), "layer": {"weight": torch.zeros(2)}}, source)

    with pytest.raises(ValueError, match="layer.weight"):
        _convert(source, destination)

    assert not destination.exists()


@pytest.mark.req("SP-TENSORACLE-006")
def test_non_string_mapping_keys_are_rejected(tmp_path, torch):
    """Mapping keys must be strings rather than silently changing the weight names."""
    source = tmp_path / "weights.pth"
    destination = tmp_path / "weights.safetensors"
    torch.save({1: torch.ones(2)}, source)

    with pytest.raises(TypeError):
        _convert(source, destination)

    assert not destination.exists()


class _WritesMarkerWhenLoaded:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",)


@pytest.mark.req("SP-TENSORACLE-006")
def test_checkpoint_code_is_rejected_without_execution(tmp_path, torch):
    """A checkpoint object that would write a marker is rejected without writing it."""
    source = tmp_path / "unsafe.pth"
    destination = tmp_path / "weights.safetensors"
    marker = tmp_path / "executed"
    torch.save({"weight": torch.ones(2), "object": _WritesMarkerWhenLoaded(marker)}, source)

    with pytest.raises(ValueError):
        _convert(source, destination)

    assert not marker.exists()
    assert not destination.exists()


@pytest.mark.req("SP-TENSORACLE-005")
@pytest.mark.req("SP-TENSORACLE-006")
@pytest.mark.parametrize("entry", ["function", "command"])
def test_missing_torch_reports_the_conversion_extra(tmp_path, entry):
    """Both conversion entries explain that torch and tensoracle[convert] are required."""
    source = """
import importlib.abc
import runpy
import sys

class MissingTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError("No module named 'torch'", name='torch')

sys.meta_path.insert(0, MissingTorch())
if sys.argv[1] == 'function':
    from tensoracle.convert import pth_to_safetensors
    pth_to_safetensors(sys.argv[2], sys.argv[3])
else:
    sys.argv = ['tensoracle.convert', sys.argv[2], sys.argv[3]]
    runpy.run_module('tensoracle.convert', run_name='__main__')
"""
    destination = tmp_path / "weights.safetensors"

    result = subprocess.run(
        [sys.executable, "-c", source, entry, str(tmp_path / "weights.pth"), str(destination)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert "torch" in result.stderr
    assert "tensoracle[convert]" in result.stderr
    assert not destination.exists()


@pytest.mark.req("SP-TENSORACLE-005")
def test_utility_and_conversion_imports_leave_torch_unloaded():
    """Importing the package and comparison or conversion entries does not load torch."""
    source = """
import sys
import tensoracle
assert 'torch' not in sys.modules
from tensoracle import compare_outputs
assert 'torch' not in sys.modules
from tensoracle.convert import pth_to_safetensors
assert 'torch' not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False, timeout=30
    )

    assert result.returncode == 0, result.stderr
