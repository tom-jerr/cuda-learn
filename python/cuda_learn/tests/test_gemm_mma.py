import torch
import torch.nn.functional as F

from cuda_learn import ops
from cuda_learn.bench import bench

DEVICE = "cuda"


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


@bench(
    make_inputs=_make_gemm_mma,
    ref=lambda a, b: a @ b,
    flops=lambda a, b: 2 * a.shape[0] * a.shape[1] * b.shape[1],
    rtol=1e-2,
    atol=1e-2,
    baselines={"ordinary MMA": ops.gemm_mma},
)
def test_gemm_mma_l2(a, b):
    return ops.gemm_mma_l2(a, b, swizzle_n=8)
