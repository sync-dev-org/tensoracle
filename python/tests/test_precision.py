"""Accept caller-defined bf16 and explicitly quantized graphs against an fp32 engine."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from cuda.bindings import runtime as cudart
from safetensors.numpy import load_file

from precision_runtime import _cuda, open_linear, run, sample_input, write_weights
from tensoracle import compare_outputs


def _require_precision(precision, sm):
    if precision == "fp8" and sm < 89:
        pytest.skip(f"FP8 requires sm89 or sm120; detected sm{sm}")


@pytest.mark.req("SP-TENSORACLE-013")
@pytest.mark.req("SP-TENSORACLE-015")
@pytest.mark.parametrize("sm", [86, 89, 120])
@pytest.mark.parametrize("precision", ["bf16", "int8", "fp8"])
def test_precision_gpu_generation_skips_only_fp8_on_sm86(precision, sm):
    """The supported GPU generations skip fp8 on sm86 and retain bf16 and int8."""
    if precision == "fp8" and sm == 86:
        with pytest.raises(pytest.skip.Exception, match="sm86"):
            _require_precision(precision, sm)
    else:
        _require_precision(precision, sm)


@pytest.mark.gpu
@pytest.mark.req("SP-TENSORACLE-013")
@pytest.mark.req("SP-TENSORACLE-014")
@pytest.mark.req("SP-TENSORACLE-020")
@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.parametrize(
    "precision,atol,relative_l2_limit",
    [("bf16", 0.01, 0.01), ("int8", 0.02, 0.02), ("fp8", 0.12, 0.08)],
)
def test_lower_precision_matches_fp32_and_reuses_saved_engines_after_restart(
    tmp_path, monkeypatch, precision, atol, relative_l2_limit
):
    """Caller-typed engines meet caller limits and run unchanged after a process restart.

    This fixture measured maximum absolute / relative L2 errors of 0.00357 /
    0.00294 (bf16), 0.00710 / 0.00572 (int8), and 0.0501 / 0.0382 (fp8).
    The limits allow at least twice those errors for different GPU tactics;
    they are this test's quality decision, not package-wide defaults.
    """
    device = _cuda(cudart.cudaGetDevice())
    properties = _cuda(cudart.cudaGetDeviceProperties(device))
    _require_precision(precision, properties.major * 10 + properties.minor)
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    weights_path = tmp_path / "linear.safetensors"
    write_weights(weights_path)
    weights = load_file(weights_path)
    reference_handle = open_linear(weights_path, "fp32")
    candidate_handle = open_linear(weights_path, precision)
    assert reference_handle.path != candidate_handle.path
    paths = {"fp32": reference_handle.path, precision: candidate_handle.path}
    assert set((cache / "tensoracle").glob("*.engine")) == set(paths.values())
    before = {
        key: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for key, path in paths.items()
    }

    reference = run(reference_handle)
    candidate = run(candidate_handle)
    np.testing.assert_allclose(
        reference, sample_input() @ weights["matrix"] + weights["bias"], atol=2e-6, rtol=1e-5
    )
    comparison = compare_outputs(candidate, reference, atol=atol, rtol=0)
    print(f"{precision}: {json.dumps(asdict(comparison), sort_keys=True)}")
    assert comparison.candidate_nonfinite_count == comparison.reference_nonfinite_count == 0
    assert comparison.outside_tolerance_count == 0, comparison
    assert comparison.relative_l2_error is not None
    assert comparison.relative_l2_error < relative_l2_limit, comparison
    # The inputs are not exactly representable at lower precision. An fp32 cache
    # entry accidentally selected for the candidate must not pass this check.
    assert comparison.max_absolute_error > 1e-5, comparison

    outputs_path = tmp_path / "reopened.npz"
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("precision_runtime.py")),
            str(weights_path),
            precision,
            str(outputs_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == {key: str(path) for key, path in paths.items()}
    with np.load(outputs_path) as reopened:
        np.testing.assert_array_equal(reopened["fp32"], reference)
        np.testing.assert_array_equal(reopened[precision], candidate)
    assert before == {
        key: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for key, path in paths.items()
    }
