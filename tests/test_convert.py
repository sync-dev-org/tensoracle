"""Convert tensor checkpoints without executing checkpoint code or requiring torch for inference."""

import subprocess
import sys
from collections import OrderedDict

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
def test_inference_and_conversion_imports_leave_torch_unloaded():
    """Importing the package and engine or conversion entries does not load torch."""
    source = """
import sys
import tensoracle
assert 'torch' not in sys.modules
from tensoracle import build_serialized_engine, open_engine
assert 'torch' not in sys.modules
from tensoracle.convert import pth_to_safetensors
assert 'torch' not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False, timeout=30
    )

    assert result.returncode == 0, result.stderr
