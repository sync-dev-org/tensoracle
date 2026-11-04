"""Model-independent measurements for one pair of CPU output arrays."""

from dataclasses import dataclass

import numpy as np

from . import _native


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
    return OutputComparison(*_native.compare_outputs(candidate, reference, atol=atol, rtol=rtol))
