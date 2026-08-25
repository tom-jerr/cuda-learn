"""全部算子的测试：test_<op>() + @bench 装饰。

每个测试函数体只做 op 调用；正确性对比（ref）与计时由
python -m cuda_learn.bench 驱动，函数名即 benchmark 的 key。
"""

import torch
import torch.nn.functional as F

from cuda_learn import ops
from cuda_learn.bench import bench
from cuda_learn.flash_attn_baseline import flash_attn_2

DEVICE = "cuda"


# ---------- vector_add ----------


def _make_vec():
    return torch.rand(1 << 22, device=DEVICE), torch.rand(1 << 22, device=DEVICE)


@bench(make_inputs=_make_vec, ref=lambda a, b: a + b, flops=lambda a, b: a.numel())
def test_vector_add(a, b):
    return ops.vector_add(a, b)


@bench(make_inputs=_make_vec, ref=lambda a, b: a + b, flops=lambda a, b: a.numel())
def test_vector_add_grid_loop(a, b):
    return ops.vector_add_grid_loop(a, b)


@bench(make_inputs=_make_vec, ref=lambda a, b: a + b, flops=lambda a, b: a.numel())
def test_vector_add_vectorized(a, b):
    return ops.vector_add_vectorized(a, b)


# ---------- transpose ----------


def _make_mat():
    return (torch.rand(2048, 2048, device=DEVICE),)


@bench(make_inputs=_make_mat, ref=lambda x: x.t(), flops=lambda x: x.numel())
def test_transpose(x):
    return ops.transpose(x)


@bench(make_inputs=_make_mat, ref=lambda x: x.t(), flops=lambda x: x.numel())
def test_transpose_tile(x):
    return ops.transpose_tile(x)


# ---------- dot_product ----------


@bench(
    make_inputs=_make_vec,
    ref=lambda a, b: (a * b).sum(),
    flops=lambda a, b: a.numel() * 2,
    rtol=1e-3,
    atol=1e-2,
)
def test_dot_product(a, b):
    return ops.dot_product(a, b)[0]


# ---------- gemm ----------


def _make_gemm():
    return (
        torch.randn(1024, 1024, device=DEVICE),
        torch.randn(1024, 1024, device=DEVICE),
    )


def _make_gemm_mma():
    return tuple(x.to(torch.bfloat16) for x in _make_gemm())


@bench(
    make_inputs=_make_gemm,
    ref=lambda a, b: a @ b,
    flops=lambda a, b: 2 * a.shape[0] * a.shape[1] * b.shape[1],
    rtol=1e-3,
    atol=1e-3,
    baselines={"cuBLAS": ops.cublas_gemm},
)
def test_gemm(a, b):
    return ops.gemm(a, b)


@bench(
    make_inputs=_make_gemm_mma,
    ref=lambda a, b: a @ b,
    flops=lambda a, b: 2 * a.shape[0] * a.shape[1] * b.shape[1],
    rtol=1e-2,
    atol=1e-2,
    baselines={"cuBLAS BF16": ops.cublas_gemm_bf16},
)
def test_gemm_mma(a, b):
    return ops.gemm_mma(a, b)


# ---------- flash_attn（FP16 [B,H,N,64]，N 需为 64 的倍数）----------


def _make_fa():
    shape = (2, 8, 1024, 64)
    return tuple(torch.randn(shape, device=DEVICE, dtype=torch.float16)
                 for _ in range(3))


def _fa_ref(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


@bench(
    make_inputs=_make_fa,
    ref=_fa_ref,
    flops=lambda q, k, v: 4 * q.shape[0] * q.shape[1] * q.shape[2] ** 2
    * q.shape[3],
    rtol=2e-2,
    atol=2e-2,
    baselines={"flash-attn 2.8.4": flash_attn_2},
)
def test_flash_attn(q, k, v):
    return ops.flash_attn(q, k, v)


@bench(
    make_inputs=_make_fa,
    ref=_fa_ref,
    flops=lambda q, k, v: 4 * q.shape[0] * q.shape[1] * q.shape[2] ** 2
    * q.shape[3],
    rtol=2e-2,
    atol=2e-2,
    baselines={
        "original": ops.flash_attn,
        "flash-attn 2.8.4": flash_attn_2,
    },
)
def test_flash_attn_optimized(q, k, v):
    return ops.flash_attn_optimized(q, k, v)


@bench(
    make_inputs=_make_fa,
    ref=lambda q, k, v: F.scaled_dot_product_attention(
        q, k, v, is_causal=True),
    flops=lambda q, k, v: 2 * q.shape[0] * q.shape[1] * q.shape[2]
    * (q.shape[2] + 1) * q.shape[3],
    rtol=2e-2,
    atol=2e-2,
    baselines={
        "flash-attn 2.8.4": lambda q, k, v: flash_attn_2(
            q, k, v, causal=True),
    },
)
def test_flash_attn_optimized_causal(q, k, v):
    return ops.flash_attn_optimized(q, k, v, causal=True)


@bench(
    make_inputs=_make_fa,
    ref=_fa_ref,
    flops=lambda q, k, v: 4 * q.shape[0] * q.shape[1] * q.shape[2] ** 2
    * q.shape[3],
    rtol=2e-2,
    atol=2e-2,
    baselines={
        "PAD=8 8-warp": ops.flash_attn_optimized,
        "flash-attn 2.8.4": flash_attn_2,
    },
)
def test_flash_attn_swizzled(q, k, v):
    return ops.flash_attn_swizzled(q, k, v)


@bench(
    make_inputs=_make_fa,
    ref=lambda q, k, v: F.scaled_dot_product_attention(
        q, k, v, is_causal=True),
    flops=lambda q, k, v: 2 * q.shape[0] * q.shape[1] * q.shape[2]
    * (q.shape[2] + 1) * q.shape[3],
    rtol=2e-2,
    atol=2e-2,
    baselines={
        "PAD=8 8-warp": lambda q, k, v: ops.flash_attn_optimized(
            q, k, v, causal=True),
        "flash-attn 2.8.4": lambda q, k, v: flash_attn_2(
            q, k, v, causal=True),
    },
)
def test_flash_attn_swizzled_causal(q, k, v):
    return ops.flash_attn_swizzled(q, k, v, causal=True)


# ---------- fusion ----------


def _make_silu():
    return (
        torch.randn(1 << 22, device=DEVICE) * 0.5,
        torch.randn(1 << 22, device=DEVICE),
    )


@bench(
    make_inputs=_make_silu,
    ref=lambda g, u: F.silu(g) * u,
    flops=lambda g, u: g.numel() * 4,
)
def test_silu_and_mul(gate, up):
    return ops.silu_and_mul(gate, up)


def _make_rms():
    return (
        torch.randn(64, 1024, device=DEVICE),
        torch.randn(64, 1024, device=DEVICE),
        torch.rand(1024, device=DEVICE) + 0.5,
    )


def _rms_ref(x, residual, weight, eps=1e-5):
    value = x + residual
    rms = torch.sqrt((value * value).mean(dim=-1, keepdim=True) + eps)
    return value / rms * weight, value


@bench(make_inputs=_make_rms, ref=_rms_ref, flops=lambda x, r, w: x.numel() * 5)
def test_rmsnorm_and_add(x, residual, weight):
    return ops.rmsnorm_and_add(x, residual, weight, 1e-5)


# ---------- softmax ----------


def _make_softmax():
    return (torch.randn(4096, device=DEVICE),)


@bench(
    make_inputs=_make_softmax,
    ref=lambda x: torch.softmax(x, dim=0),
    flops=lambda x: x.numel() * 5,
    baselines={"cuDNN": ops.cudnn_softmax},
)
def test_softmax(x):
    return ops.softmax(x)
