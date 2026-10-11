"""Small caller-owned kernels used by the cubin plugin example."""

import triton
import triton.language as tl

from tensoracle.producer import KernelSpec


@triton.jit
def dot_ieee(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    column = tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + row[:, None] * BLOCK_K + depth[None, :], row[:, None] < rows, 0)
    weight = tl.load(weight_ptr + depth[:, None] * BLOCK_N + column[None, :])
    result = tl.dot(x, weight, input_precision="ieee")
    tl.store(out_ptr + row[:, None] * BLOCK_N + column[None, :], result, row[:, None] < rows)


@triton.jit
def dot_tf32(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    column = tl.arange(0, BLOCK_N)
    depth = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + row[:, None] * BLOCK_K + depth[None, :], row[:, None] < rows, 0)
    weight = tl.load(weight_ptr + depth[:, None] * BLOCK_N + column[None, :])
    result = tl.dot(x, weight)
    tl.store(out_ptr + row[:, None] * BLOCK_N + column[None, :], result, row[:, None] < rows)


@triton.jit
def scale(x_ptr, out_ptr, factor, count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x_ptr + offset, offset < count, 0)
    tl.store(out_ptr + offset, value * factor, offset < count)


@triton.jit
def add_bias(x_ptr, bias_ptr, out_ptr, count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x_ptr + offset, offset < count, 0)
    bias = tl.load(bias_ptr + offset % 16, offset < count, 0)
    tl.store(out_ptr + offset, value + bias, offset < count)


def kernel_specs() -> tuple[KernelSpec, ...]:
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
            dot_ieee,
            dot_signature,
            block,
            divisible_by={"x_ptr": 16, "weight_ptr": 16, "out_ptr": 16, "rows": 16},
        ),
        KernelSpec("dot.unspecialized", dot_ieee, dot_signature, block),
        KernelSpec("dot.tf32-control", dot_tf32, dot_signature, block),
        KernelSpec(
            "scale",
            scale,
            (("x_ptr", "*fp32"), ("out_ptr", "*fp32"), ("factor", "fp32"), ("count", "i32")),
            {"BLOCK": 128},
        ),
        KernelSpec(
            "bias",
            add_bias,
            (("x_ptr", "*fp32"), ("bias_ptr", "*fp32"), ("out_ptr", "*fp32"), ("count", "i32")),
            {"BLOCK": 128},
        ),
    )
