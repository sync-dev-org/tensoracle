"""Compare one CPU output array with its reference array."""

import numpy as np
import pytest

from tensoracle import compare_outputs


@pytest.mark.req("REQ-TENSORACLE-010")
def test_float_metrics_match_known_differences_and_reference_relative_tolerance():
    """One output pair reports hand-calculated distance, direction, and tolerance misses."""
    reference = np.array([1.0, -1.0], dtype=np.float32)
    candidate = np.array([2.0, 1.0], dtype=np.float32)

    result = compare_outputs(candidate, reference, atol=0.5, rtol=0.5)

    assert result.candidate_nonfinite_count == 0
    assert result.reference_nonfinite_count == 0
    assert result.max_absolute_error == pytest.approx(2.0)
    assert result.relative_l2_error == pytest.approx(np.sqrt(5 / 2))
    assert result.cosine_similarity == pytest.approx(1 / np.sqrt(10))
    assert result.outside_tolerance_count == 1
    assert result.outside_tolerance_fraction == pytest.approx(0.5)

    reference = np.array([10.0, 1.0], dtype=np.float32)
    candidate = np.array([11.5, 1.0], dtype=np.float32)
    forward = compare_outputs(candidate, reference, atol=0.0, rtol=0.14)
    reverse = compare_outputs(reference, candidate, atol=0.0, rtol=0.14)
    assert forward.outside_tolerance_count == 1
    assert forward.outside_tolerance_fraction == pytest.approx(0.5)
    assert reverse.outside_tolerance_count == 0
    assert reverse.outside_tolerance_fraction == pytest.approx(0.0)


@pytest.mark.req("REQ-TENSORACLE-010")
def test_integer_and_boolean_mismatches_preserve_large_integer_values():
    """Exact integer and boolean comparisons count each unequal element."""
    reference = np.array([2**53, 5], dtype=np.uint64)
    candidate = np.array([2**53 + 1, 5], dtype=np.uint64)
    integers = compare_outputs(candidate, reference)
    assert integers.mismatch_count == 1
    assert integers.mismatch_fraction == pytest.approx(0.5)
    assert integers.candidate_nonfinite_count == 0
    assert integers.reference_nonfinite_count == 0
    assert integers.max_absolute_error is None

    booleans = compare_outputs(np.array([True, False, True]), np.array([True, True, True]))
    assert booleans.mismatch_count == 1
    assert booleans.mismatch_fraction == pytest.approx(1 / 3)


@pytest.mark.req("REQ-TENSORACLE-010")
def test_mixed_float_precision_is_calculated_in_float64():
    """An fp16 output and fp32 reference use float64 arithmetic for every distance."""
    candidate = np.array([1.5, -2.0], dtype=np.float16)
    reference = np.array([1.25, -2.25], dtype=np.float32)
    result = compare_outputs(candidate, reference, atol=0.1, rtol=0.0)
    assert result.max_absolute_error == pytest.approx(0.25)
    assert result.relative_l2_error == pytest.approx(np.sqrt(0.125 / 6.625))
    assert result.cosine_similarity == pytest.approx(6.375 / np.sqrt(6.25 * 6.625))
    assert result.outside_tolerance_count == 2


@pytest.mark.req("REQ-TENSORACLE-010")
def test_zero_norm_only_removes_metrics_that_need_that_norm():
    """Zero reference or output norms leave the other defined float metrics available."""
    zero_reference = compare_outputs(np.array([1.0, 0.0]), np.array([0.0, 0.0]), atol=0.1, rtol=0.0)
    assert zero_reference.relative_l2_error is None
    assert zero_reference.cosine_similarity is None
    assert zero_reference.max_absolute_error == pytest.approx(1.0)
    assert zero_reference.outside_tolerance_count == 1

    zero_candidate = compare_outputs(np.array([0.0, 0.0]), np.array([1.0, 2.0]), atol=0.1, rtol=0.0)
    assert zero_candidate.relative_l2_error == pytest.approx(1.0)
    assert zero_candidate.cosine_similarity is None
    assert zero_candidate.max_absolute_error == pytest.approx(2.0)
    assert zero_candidate.outside_tolerance_count == 2


@pytest.mark.req("REQ-TENSORACLE-002")
@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        (np.array([1.0]), np.array([[1.0]])),
        (np.array([1.0]), np.array([1])),
        (np.array([True]), np.array([1])),
        (np.array([], dtype=np.float32), np.array([], dtype=np.float32)),
    ],
)
def test_incompatible_or_empty_arrays_are_rejected(candidate, reference):
    """A pair with different shapes or kinds, or no elements, has no comparison result."""
    with pytest.raises(ValueError):
        compare_outputs(candidate, reference)


@pytest.mark.req("REQ-TENSORACLE-002")
def test_nonfinite_values_on_either_side_suppress_float_metrics():
    """Each side's NaN and infinities are counted without returning misleading distances."""
    for candidate, reference, expected_counts in [
        (np.array([1.0, 2.0, 3.0]), np.array([np.nan, np.inf, 3.0]), (0, 2)),
        (np.array([np.nan, -np.inf, 3.0]), np.array([1.0, 2.0, 3.0]), (2, 0)),
    ]:
        result = compare_outputs(candidate, reference, atol=0.1, rtol=0.1)
        assert (
            result.candidate_nonfinite_count,
            result.reference_nonfinite_count,
        ) == expected_counts
        assert result.max_absolute_error is None
        assert result.relative_l2_error is None
        assert result.cosine_similarity is None
        assert result.outside_tolerance_count is None
        assert result.outside_tolerance_fraction is None


@pytest.mark.req("REQ-TENSORACLE-002")
def test_caller_can_decide_tolerance_acceptance_from_the_result():
    """A caller accepts a finite close output and sees one element outside its tolerance."""
    reference = np.array([1.0, 2.0])
    close = compare_outputs(np.array([1.05, 2.0]), reference, atol=0.1, rtol=0.0)
    outside = compare_outputs(np.array([1.15, 2.0]), reference, atol=0.1, rtol=0.0)
    assert close.candidate_nonfinite_count == close.reference_nonfinite_count == 0
    assert close.outside_tolerance_count == 0
    assert outside.outside_tolerance_count == 1


@pytest.mark.req("REQ-TENSORACLE-010")
def test_tolerance_metrics_are_absent_until_caller_supplies_both_limits():
    """A caller receives distances without a tolerance verdict until both limits are set."""
    result = compare_outputs(np.array([1.0]), np.array([1.1]))
    assert result.max_absolute_error == pytest.approx(0.1)
    assert result.outside_tolerance_count is None
    assert result.outside_tolerance_fraction is None
    with pytest.raises(ValueError):
        compare_outputs(np.array([1.0]), np.array([1.1]), atol=0.1)
