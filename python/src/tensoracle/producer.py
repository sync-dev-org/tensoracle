"""Compile pinned Triton kernels into a portable cubin bundle.

Install tensoracle[producer] to supply Triton and the NVIDIA wheel tools.
Importing tensoracle for inference never imports this module or Triton.
"""

import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from .bundle import bundle_sha256, cubin_sm, inspect_bundle


@dataclass(frozen=True)
class KernelSpec:
    """One caller-owned Triton entry and its public launch arguments."""

    name: str
    function: Any
    signature: tuple[tuple[str, str], ...]
    constexprs: dict[str, int]
    divisible_by: dict[str, int] = field(default_factory=dict)


def _wheel_tool(package: str, relative: str) -> Path:
    tool = Path(relative).name
    try:
        path = Path(distribution(package).locate_file(relative))
    except PackageNotFoundError as exc:
        raise ImportError(
            f"{tool} requires the {package} wheel; install it with pip install 'tensoracle[producer]'"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"{tool} was not found in the {package} wheel at {path}; "
            "install it with pip install 'tensoracle[producer]'"
        )
    return path


def _ptxas_version(path: Path) -> str:
    output = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=True)
    match = re.search(r"V(\d+\.\d+\.\d+)", output.stdout + output.stderr)
    if match is None:
        raise ValueError("ptxas did not report its version")
    return match.group(1)


def _parameter_layout(cuobjdump: Path, cubin: Path, sizes: list[int]) -> int:
    elf = subprocess.run(
        [str(cuobjdump), "-elf", str(cubin)], capture_output=True, text=True, check=True
    ).stdout
    usage = subprocess.run(
        [str(cuobjdump), "--dump-resource-usage", str(cubin)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fields = re.findall(
        r"Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x([0-9a-f]+)\s+Size\s*:\s*0x([0-9a-f]+)",
        elf,
    )
    actual = sorted(
        (int(ordinal, 16), int(offset, 16), int(size, 16)) for ordinal, offset, size in fields
    )
    expected = []
    offset = 0
    for ordinal, size in enumerate([*sizes, 8, 8]):
        offset = (offset + size - 1) // size * size
        expected.append((ordinal, offset, size))
        offset += size
    if actual != expected:
        raise ValueError(f"unsupported Triton hidden parameters or public ABI: {cubin.name}")
    match = re.search(r"\bREG:(\d+)\b", usage)
    if match is None:
        raise ValueError(f"cuobjdump did not report registers for {cubin.name}")
    return int(match.group(1))


def _cpp_array(symbol: str, data: bytes) -> str:
    rows = [
        "    " + ", ".join(f"0x{value:02x}" for value in data[offset : offset + 16]) + ","
        for offset in range(0, len(data), 16)
    ]
    return (
        f'extern "C" TENSORACLE_EXPORT alignas(16) unsigned char const {symbol}[] = {{\n'
        + "\n".join(rows)
        + "\n};\n"
        + f'extern "C" TENSORACLE_EXPORT unsigned long long const {symbol}_size = sizeof({symbol});\n'
    )


def compile_bundle(
    specs: tuple[KernelSpec, ...],
    output: Path,
    *,
    ptxas: Path | None = None,
    cuobjdump: Path | None = None,
) -> dict:
    """Produce and inspect sm86, sm89, and sm120 cubins without importing torch.

    Omitted tool paths use the NVIDIA wheels installed by tensoracle[producer].
    Explicit ptxas and cuobjdump paths override those defaults independently.
    """
    if "torch" in sys.modules:
        raise ValueError("torch must not be imported in the producer")
    try:
        import triton
    except ImportError as exc:
        raise ImportError(
            "Cubin production requires Triton; install it with pip install 'tensoracle[producer]'"
        ) from exc
    from triton.backends.compiler import GPUTarget

    if triton.__version__ != "3.8.0":
        raise ValueError(f"Triton 3.8.0 is required, got {triton.__version__}")
    if ptxas is None:
        ptxas = _wheel_tool("nvidia-cuda-nvcc-cu12", "nvidia/cuda_nvcc/bin/ptxas")
    if cuobjdump is None:
        cuobjdump = _wheel_tool("nvidia-cuda-cuobjdump", "nvidia/cu13/bin/cuobjdump")
    version = _ptxas_version(ptxas)
    if version != "12.9.86":
        raise ValueError(f"ptxas 12.9.86 is required, got {version}")
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("kernel names must be unique and nonempty")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(ptxas)
    os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    entries = []
    payloads = {}
    for sm in (86, 89, 120):
        target = GPUTarget("cuda", sm, 32)
        for spec in specs:
            if not spec.name or not re.fullmatch(r"[a-zA-Z0-9_.-]+", spec.name):
                raise ValueError(f"invalid kernel name: {spec.name}")
            names = tuple(spec.function.arg_names)
            if tuple(name for name, _ in spec.signature) != tuple(
                name for name in names if name not in spec.constexprs
            ):
                raise ValueError(f"kernel signature differs from public arguments: {spec.name}")
            attrs = {
                (names.index(name),): [["tt.divisibility", divisor]]
                for name, divisor in spec.divisible_by.items()
            }
            source = triton.compiler.ASTSource(
                fn=spec.function,
                signature=dict(spec.signature),
                constexprs=spec.constexprs,
                attrs=attrs,
            )
            compiled = triton.compile(
                source, target=target, options={"num_stages": 3, "num_warps": 4}
            )
            data = bytes(compiled.asm["cubin"])
            relative = f"cubins/sm{sm}/{spec.name}.cubin"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            if cubin_sm(data) != f"sm{sm}":
                raise ValueError(f"compiled cubin has the wrong SM: {spec.name}")
            resources = {
                "compiled_shared_memory_bytes": int(compiled.metadata.shared),
                "launch_dynamic_shared_memory_bytes": int(compiled.metadata.shared),
                "registers_per_thread": _parameter_layout(
                    cuobjdump,
                    path,
                    [8 if type_.startswith("*") else 4 for _, type_ in spec.signature],
                ),
                "global_scratch_bytes": int(compiled.metadata.global_scratch_size),
                "profile_scratch_bytes": int(compiled.metadata.profile_scratch_size),
            }
            if resources["global_scratch_bytes"] or resources["profile_scratch_bytes"]:
                raise ValueError(f"nonzero scratch is unsupported: {spec.name}")
            symbol = "tensoracle_cubin_" + re.sub(r"[^a-zA-Z0-9_]", "_", f"sm{sm}_{spec.name}")
            payloads[symbol] = data
            entries.append(
                {
                    "id": f"sm{sm}.{spec.name}",
                    "file": relative,
                    "target_sm": f"sm{sm}",
                    "entry_symbol": str(compiled.metadata.name),
                    "embedded_symbol": symbol,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "signature": [{"name": name, "type": type_} for name, type_ in spec.signature],
                    "hidden_trailing_parameters": [
                        {"name": "global_scratch", "parameter_bytes": 8},
                        {"name": "profile_scratch", "parameter_bytes": 8},
                    ],
                    "resources": resources,
                    "specialization": {"divisible_by": spec.divisible_by},
                    "source_sha256": hashlib.sha256(spec.function.src.encode("utf-8")).hexdigest(),
                }
            )
    manifest = {
        "schema_version": 1,
        "producer": {"triton": triton.__version__, "ptxas": version},
        "entries": entries,
    }
    digest = bundle_sha256(manifest)
    manifest["bundle_sha256"] = digest
    (output / "bundle.json").write_text(
        __import__("json").dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    generated = [
        "// Generated by tensoracle.producer; do not edit.\n",
        '#include "tensoracle/cubin_plugin.hpp"\n#include <cstring>\n',
        "#if defined(_WIN32)\n#define TENSORACLE_EXPORT __declspec(dllexport)\n",
        '#else\n#define TENSORACLE_EXPORT __attribute__((visibility("default")))\n#endif\n',
        f'extern "C" TENSORACLE_EXPORT char const tensoracle_bundle_sha256[] = "{digest}";\n',
    ]
    for symbol, data in payloads.items():
        generated.append(_cpp_array(symbol, data))
    generated.append(
        "namespace {\nstruct BundleEntry { char const* id; tensoracle::cubin::KernelImage image; };\n"
    )
    for entry in entries:
        rules_symbol = f"{entry['embedded_symbol']}_arguments"
        generated.append(f"tensoracle::cubin::ArgumentRule const {rules_symbol}[] = {{\n")
        divisors = entry["specialization"]["divisible_by"]
        for argument in entry["signature"]:
            type_ = argument["type"]
            kind = (
                "kPointer"
                if type_.startswith("*")
                else ("kInt32" if type_ == "i32" else "kFloat32")
            )
            generated.append(
                f"    {{tensoracle::cubin::ArgumentKind::{kind}, "
                f"{divisors.get(argument['name'], 1)}}},\n"
            )
        generated.append("};\n")
    generated.append("BundleEntry const entries[] = {\n")
    for entry in entries:
        resources = entry["resources"]
        generated.append(
            f'{{"{entry["id"]}", {{'
            f"{entry['embedded_symbol']}, {entry['embedded_symbol']}_size, "
            f'"{entry["sha256"]}", "{entry["entry_symbol"]}", '
            f"{entry['target_sm'][2:]}, {resources['registers_per_thread']}, "
            f"{resources['launch_dynamic_shared_memory_bytes']}, "
            f"{resources['global_scratch_bytes']}, {resources['profile_scratch_bytes']}, "
            f"{len(entry['signature'])}, {entry['embedded_symbol']}_arguments"
            "}},\n"
        )
    generated.append(
        "};\n}\nnamespace tensoracle::cubin {\n"
        "KernelImage const* find_bundle_image(char const* id) noexcept {\n"
        "    if (id == nullptr) return nullptr;\n"
        "    for (auto const& entry : entries)\n"
        "        if (std::strcmp(entry.id, id) == 0) return &entry.image;\n"
        "    return nullptr;\n}\n}\n"
    )
    (output / "bundle.cpp").write_text("".join(generated), encoding="ascii")
    inspect_bundle(output)
    if "torch" in sys.modules:
        raise ValueError("producer imported torch")
    return manifest
