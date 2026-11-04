"""CUDA C++ kernel declarations for the caller's example bundle."""

from pathlib import Path

from tensoracle.producer import KernelSpec


def kernel_specs() -> tuple[KernelSpec, ...]:
    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    dot_signature = (
        ("x_ptr", "*fp32"),
        ("weight_ptr", "*fp32"),
        ("out_ptr", "*fp32"),
        ("rows", "i32"),
    )
    block = {"BLOCK_M": 16, "BLOCK_K": 32, "BLOCK_N": 16}
    return (
        KernelSpec(
            "dot.specialized",
            source,
            dot_signature,
            block,
            "dot_kernel",
            divisible_by={"x_ptr": 16, "weight_ptr": 16, "out_ptr": 16, "rows": 16},
        ),
        KernelSpec("dot.unspecialized", source, dot_signature, block, "dot_kernel"),
        KernelSpec(
            "dot.tf32-control", source, dot_signature, {**block, "TF32_CONTROL": 1}, "dot_kernel"
        ),
        KernelSpec(
            "scale",
            source,
            (("x_ptr", "*fp32"), ("out_ptr", "*fp32"), ("factor", "fp32"), ("count", "i32")),
            block,
            "scale_kernel",
        ),
        KernelSpec(
            "bias",
            source,
            (("x_ptr", "*fp32"), ("bias_ptr", "*fp32"), ("out_ptr", "*fp32"), ("count", "i32")),
            block,
            "bias_kernel",
        ),
    )
