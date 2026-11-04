"""Array conversions preserve the C++ comparison's Python results."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tensoracle import OutputComparison, _native, compare_outputs


@pytest.mark.req("SP-TENSORACLE-053")
@pytest.mark.req("SP-TENSORACLE-014")
def test_native_comparison_returns_python_measurements():
    """The native entry point converts the shared library's measurements to Python values."""
    values = _native.compare_outputs(np.array([2.0]), np.array([1.0]), atol=0.0, rtol=0.0)
    assert values == (0, 0, 1.0, 1.0, 1.0, 1, 1.0, None, None)


@pytest.mark.req("SP-TENSORACLE-014")
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "longdouble"])
def test_float_storage_formats_and_strided_arrays_preserve_element_order(dtype):
    """Native, opposite-endian, and strided floating arrays compare in logical order."""
    reference = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    candidate = np.array([[2, 2, 4], [4, 7, 6]], dtype=dtype)
    expected = compare_outputs(candidate, reference, atol=0.1, rtol=0.0)
    opposite = candidate.astype(candidate.dtype.newbyteorder("S"))
    for left, right in [
        (opposite, reference),
        (candidate.T, reference.T),
        (candidate[:, ::-1], reference[:, ::-1]),
        (np.asfortranarray(candidate), reference),
    ]:
        actual = compare_outputs(left, right, atol=0.1, rtol=0.0)
        assert actual.max_absolute_error == expected.max_absolute_error
        assert actual.relative_l2_error == pytest.approx(expected.relative_l2_error)
        assert actual.cosine_similarity == pytest.approx(expected.cosine_similarity)
        assert actual.outside_tolerance_count == expected.outside_tolerance_count == 3


@pytest.mark.req("SP-TENSORACLE-014")
@pytest.mark.parametrize("dtype", ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "?"])
def test_integer_and_boolean_storage_formats_preserve_exact_values(dtype):
    """Every supported integer width and boolean format reports one differing element."""
    left = np.array([1, 0], dtype=dtype)
    right = np.array([1, 1], dtype=dtype).astype(np.dtype(dtype).newbyteorder("S"))
    result = compare_outputs(left[::-1], right[::-1])
    assert result.mismatch_count == 1
    assert result.mismatch_fraction == 0.5
    assert result.max_absolute_error is None


@pytest.mark.req("SP-TENSORACLE-014")
def test_scalar_arrays_and_unaligned_storage_are_converted_without_changing_shape():
    """Scalar shapes and unaligned data retain their value and shape requirements."""
    result = compare_outputs(np.array(2.0), np.array(1.0))
    assert result.max_absolute_error == 1.0
    with pytest.raises(ValueError):
        compare_outputs(np.array(2.0), np.array([1.0]))
    unaligned = np.ndarray((2,), dtype="f8", buffer=bytearray(17), offset=1)
    unaligned[:] = [2, 3]
    assert compare_outputs(unaligned, np.array([1.0, 3.0])).max_absolute_error == 1.0


@pytest.mark.req("SP-TENSORACLE-014")
@pytest.mark.parametrize("value", [[], [1], 1.0, None])
def test_nonarray_inputs_raise_type_errors(value):
    """A Python value without a NumPy array representation is rejected as a type error."""
    with pytest.raises(TypeError):
        compare_outputs(value, np.array([1.0]))
    with pytest.raises(TypeError):
        compare_outputs(np.array([1.0]), value)


@pytest.mark.req("SP-TENSORACLE-014")
@pytest.mark.parametrize("dtype", ["complex128", "object", "S1", "datetime64[ns]"])
def test_unsupported_array_kinds_raise_value_errors(dtype):
    """Complex, object, text, and datetime arrays have no numeric comparison result."""
    values = np.ones(1).astype(dtype)
    with pytest.raises(ValueError):
        compare_outputs(values, values)


@pytest.mark.req("SP-TENSORACLE-014")
@pytest.mark.parametrize("bad", [-1, np.inf, np.nan, "invalid", object(), [0]])
def test_invalid_tolerances_raise_value_errors(bad):
    """Each tolerance must convert to a finite nonnegative number."""
    values = np.array([1.0])
    with pytest.raises(ValueError):
        compare_outputs(values, values, atol=bad, rtol=0)
    with pytest.raises(ValueError):
        compare_outputs(values, values, atol=0, rtol=bad)


@pytest.mark.req("SP-TENSORACLE-014")
def test_result_retains_frozen_dataclass_value_behavior():
    """Measurement values preserve construction, equality, hashing, and immutability."""
    result = compare_outputs(np.array([True]), np.array([False]))
    expected = OutputComparison(0, 0, mismatch_count=1, mismatch_fraction=1.0)
    assert result == expected
    assert hash(result) == hash(expected)
    assert repr(result).startswith("OutputComparison(")
    with pytest.raises(FrozenInstanceError):
        result.mismatch_count = 0
