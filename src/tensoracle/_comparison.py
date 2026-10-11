"""Model-independent measurements for one pair of CPU output arrays."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OutputComparison:
    """Measurements for a candidate output against its reference output.

    Float measurements are ``None`` for integer or boolean arrays and when
    either float array contains a non-finite value. Tolerance measurements are
    also ``None`` when the caller has not supplied both tolerances. Mismatch
    measurements apply only to integer and boolean arrays.
    """

    candidate_nonfinite_count: int
    reference_nonfinite_count: int
    max_absolute_error: float | None = None
    relative_l2_error: float | None = None
    cosine_similarity: float | None = None
    outside_tolerance_count: int | None = None
    outside_tolerance_fraction: float | None = None
    mismatch_count: int | None = None
    mismatch_fraction: float | None = None


def compare_outputs(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> OutputComparison:
    """Measure one candidate output against the corresponding reference output.

    Both inputs must be nonempty NumPy arrays of equal shape and the same kind:
    floating point, integer, or boolean. Floating point inputs are converted
    to float64 before arithmetic. Tolerances must be finite, nonnegative, and
    supplied together. This function reports measurements; the caller chooses
    the limits that constitute an acceptable output.
    """
    if not isinstance(candidate, np.ndarray) or not isinstance(reference, np.ndarray):
        raise TypeError("candidate and reference must be NumPy arrays")
    if candidate.shape != reference.shape or candidate.size == 0:
        raise ValueError("candidate and reference must have the same nonempty shape")

    candidate_kind = candidate.dtype.kind
    reference_kind = reference.dtype.kind
    floats = candidate_kind == reference_kind == "f"
    integers = candidate_kind in "iu" and reference_kind in "iu"
    booleans = candidate_kind == reference_kind == "b"
    if not (floats or integers or booleans):
        raise ValueError("candidate and reference must have the same numeric kind")

    if (atol is None) != (rtol is None):
        raise ValueError("atol and rtol must be supplied together")
    if atol is not None and rtol is not None:
        try:
            atol = float(atol)
            rtol = float(rtol)
        except (TypeError, ValueError) as error:
            raise ValueError("atol and rtol must be finite nonnegative numbers") from error
        if not (np.isfinite(atol) and np.isfinite(rtol) and atol >= 0 and rtol >= 0):
            raise ValueError("atol and rtol must be finite nonnegative numbers")

    if not floats:
        # Mixed signed and unsigned widths can otherwise promote to float64,
        # which would erase differences between neighboring large integers.
        if candidate_kind != reference_kind:
            unequal = candidate.astype(object) != reference.astype(object)
        else:
            unequal = candidate != reference
        mismatch_count = int(np.count_nonzero(unequal))
        return OutputComparison(
            candidate_nonfinite_count=0,
            reference_nonfinite_count=0,
            mismatch_count=mismatch_count,
            mismatch_fraction=mismatch_count / candidate.size,
        )

    candidate_nonfinite_count = int(np.count_nonzero(~np.isfinite(candidate)))
    reference_nonfinite_count = int(np.count_nonzero(~np.isfinite(reference)))
    if candidate_nonfinite_count or reference_nonfinite_count:
        return OutputComparison(candidate_nonfinite_count, reference_nonfinite_count)

    candidate64 = candidate.astype(np.float64)
    reference64 = reference.astype(np.float64)
    difference = candidate64 - reference64
    reference_norm = float(np.hypot.reduce(reference64.ravel()))
    candidate_norm = float(np.hypot.reduce(candidate64.ravel()))
    difference_norm = float(np.hypot.reduce(difference.ravel()))

    outside_count = None
    outside_fraction = None
    if atol is not None and rtol is not None:
        outside_count = int(
            np.count_nonzero(~np.isclose(candidate64, reference64, atol=atol, rtol=rtol))
        )
        outside_fraction = outside_count / candidate.size

    cosine_similarity = None
    if candidate_norm and reference_norm:
        cosine_similarity = float(
            np.dot(candidate64.ravel() / candidate_norm, reference64.ravel() / reference_norm)
        )

    return OutputComparison(
        candidate_nonfinite_count=0,
        reference_nonfinite_count=0,
        max_absolute_error=float(np.max(np.abs(difference))),
        relative_l2_error=difference_norm / reference_norm if reference_norm else None,
        cosine_similarity=cosine_similarity,
        outside_tolerance_count=outside_count,
        outside_tolerance_fraction=outside_fraction,
    )
