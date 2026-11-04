"""Native cubin plugins declare tensor types, formats, and their output shape source."""

import os
import subprocess
from pathlib import Path

import pytest

from native_build import build_native, cuda_root

PROJECT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT / "tests/cubin_io"


@pytest.fixture(scope="module")
def native_tools():
    sdk = os.environ.get("TENSORACLE_SDK_ROOT")
    cuda = cuda_root()
    if not sdk or not cuda:
        pytest.skip("TensorRT SDK and PyPI CUDA headers were not supplied")
    return Path(sdk), Path(cuda)


@pytest.fixture(scope="module")
def native_contract(native_tools, tmp_path_factory):
    work = tmp_path_factory.mktemp("plugin-io-contract")
    return build_native(work, "check", (FIXTURES / "check.cpp",), tools=native_tools)


@pytest.mark.req("SP-TENSORACLE-004")
@pytest.mark.parametrize("case", ["formats", "counts", "types", "shapes", "defaults", "invalid"])
def test_native_plugin_checks_declared_io(native_contract, case):
    """Plugins enforce each tensor's declared format, type, count, and shape validation."""
    subprocess.run([str(native_contract), case], check=True, timeout=10)
