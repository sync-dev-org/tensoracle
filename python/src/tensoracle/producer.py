"""Compile caller-owned CUDA C++ kernels using the C++ producer and PyPI tools."""

import json
import sysconfig
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import _native


@dataclass(frozen=True)
class KernelSpec:
    """CUDA C++ source, an extern C entry, ordered arguments, and compile constants."""

    name: str
    source: str
    signature: tuple[tuple[str, str], ...]
    constexprs: dict[str, int]
    entry_symbol: str
    divisible_by: dict[str, int] = field(default_factory=dict)
    dynamic_shared_memory_bytes: int = 0


def compile_bundle(
    specs: tuple[KernelSpec, ...],
    output: Path,
    *,
    ptxas: Path | None = None,
    cuobjdump: Path | None = None,
    nvrtc: Path | None = None,
    cuda_root: Path | None = None,
) -> dict:
    """Build sm86, sm89, and sm120 cubins without a GPU or OS CUDA Toolkit.

    The C++ implementation validates ptxas >=13.2,<14 before writing output.
    Omitted tools come from the nvidia/cu13 directory in the Python environment.
    """
    root = Path(sysconfig.get_path("purelib")) / "nvidia/cu13" if cuda_root is None else cuda_root
    return json.loads(
        _native.compile_bundle(
            [asdict(spec) for spec in specs],
            str(output),
            str(root),
            None if ptxas is None else str(ptxas),
            None if cuobjdump is None else str(cuobjdump),
            None if nvrtc is None else str(nvrtc),
        )
    )
