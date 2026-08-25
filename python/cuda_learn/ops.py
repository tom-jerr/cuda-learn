"""统一 pytorch binding：每个 kernel 一个纯函数，torch.Tensor 进出。

约定：
- 全局函数名 = "cuda_learn." + Python 函数名；
- 输出张量由 Python 预分配、作为 out 参数传入（统一 out-param 形式）；
- 输入必须连续（tvm_ffi from_dlpack 的硬性检查），此处统一 .contiguous()。
"""

import torch

from ._ffi import call


def _contig(t):
    if not t.is_cuda:
        raise TypeError(f"cuda_learn ops require CUDA tensors, got {t.device}")
    if t.dtype != torch.float32:
        raise TypeError(f"cuda_learn ops require float32 tensors, got {t.dtype}")
    return t.contiguous() if not t.is_contiguous() else t


def _contig_bf16(t):
    if not t.is_cuda:
        raise TypeError(f"cuda_learn ops require CUDA tensors, got {t.device}")
    if t.dtype != torch.bfloat16:
        raise TypeError(f"cuda_learn ops require bfloat16 tensors, got {t.dtype}")
    return t.contiguous() if not t.is_contiguous() else t


def _contig_fp16(t):
    if not t.is_cuda:
        raise TypeError(f"cuda_learn ops require CUDA tensors, got {t.device}")
    if t.dtype != torch.float16:
        raise TypeError(f"cuda_learn ops require float16 tensors, got {t.dtype}")
    return t.contiguous() if not t.is_contiguous() else t


def vector_add(a, b):
    a, b = _contig(a), _contig(b)
    out = torch.empty_like(a)
    call("cuda_learn.vector_add", a, b, out)
    return out


def vector_add_grid_loop(a, b):
    a, b = _contig(a), _contig(b)
    out = torch.empty_like(a)
    call("cuda_learn.vector_add_grid_loop", a, b, out)
    return out


def vector_add_vectorized(a, b):
    a, b = _contig(a), _contig(b)
    out = torch.empty_like(a)
    call("cuda_learn.vector_add_vectorized", a, b, out)
    return out


def _transpose(input, name):
    input = _contig(input)
    m, n = input.shape
    out = torch.empty(n, m, device=input.device, dtype=input.dtype)
    call(name, input, out)
    return out


def transpose(input):
    return _transpose(input, "cuda_learn.transpose")


def transpose_tile(input):
    return _transpose(input, "cuda_learn.transpose_tile")


def dot_product(a, b):
    a, b = _contig(a), _contig(b)
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    call("cuda_learn.dot_product", a, b, result)
    return result


def gemm(a, b):
    a, b = _contig(a), _contig(b)
    c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    call("cuda_learn.gemm", a, b, c)
    return c


def cublas_gemm(a, b):
    """FP32 cuBLAS SGEMM benchmark baseline."""
    a, b = _contig(a), _contig(b)
    c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    call("cuda_learn.cublas_gemm", a, b, c)
    return c


def cublas_gemm_bf16(a, b):
    """BF16 cuBLAS SGEMM benchmark baseline."""
    a, b = _contig_bf16(a), _contig_bf16(b)
    c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    call("cuda_learn.cublas_gemm_bf16", a, b, c)
    return c


def gemm_mma(a, b):
    a, b = _contig_bf16(a), _contig_bf16(b)
    c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    call("cuda_learn.gemm_mma", a, b, c)
    return c


def flash_attn(q, k, v):
    """FP16 non-causal forward; Q/K/V layout is [B, H, N, 64]."""
    q, k, v = _contig_fp16(q), _contig_fp16(k), _contig_fp16(v)
    if q.ndim != 4 or q.shape[-1] != 64:
        raise ValueError(f"flash_attn expects q shaped [B,H,N,64], got {q.shape}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("flash_attn expects q, k and v to have identical shapes")
    if q.shape[2] == 0 or q.shape[2] % 64:
        raise ValueError("flash_attn sequence length must be a positive multiple of 64")
    out = torch.empty_like(q)
    call("cuda_learn.flash_attn", q, k, v, out)
    return out


def flash_attn_optimized(q, k, v, causal=False):
    """128x128 phase-pipelined FP16 attention with optional causal masking."""
    q, k, v = _contig_fp16(q), _contig_fp16(k), _contig_fp16(v)
    if q.ndim != 4 or q.shape[-1] != 64:
        raise ValueError(
            f"flash_attn_optimized expects q shaped [B,H,N,64], got {q.shape}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "flash_attn_optimized expects q, k and v to have identical shapes")
    if q.shape[2] == 0 or q.shape[2] % 64:
        raise ValueError(
            "flash_attn_optimized sequence length must be a positive multiple of 64")
    out = torch.empty_like(q)
    call("cuda_learn.flash_attn_optimized", q, k, v, out, int(bool(causal)))
    return out


def flash_attn_swizzled(q, k, v, causal=False):
    """4-warp 128x128 attention with a 48 KiB XOR-swizzled Q/K/V layout."""
    q, k, v = _contig_fp16(q), _contig_fp16(k), _contig_fp16(v)
    if q.ndim != 4 or q.shape[-1] != 64:
        raise ValueError(
            f"flash_attn_swizzled expects q shaped [B,H,N,64], got {q.shape}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "flash_attn_swizzled expects q, k and v to have identical shapes")
    if q.shape[2] == 0 or q.shape[2] % 64:
        raise ValueError(
            "flash_attn_swizzled sequence length must be a positive multiple of 64")
    out = torch.empty_like(q)
    call("cuda_learn.flash_attn_swizzled", q, k, v, out, int(bool(causal)))
    return out


def silu_and_mul(gate, up):
    gate, up = _contig(gate), _contig(up)
    out = torch.empty_like(gate)
    call("cuda_learn.silu_and_mul", gate, up, out)
    return out


def rmsnorm_and_add(x, residual, weight, epsilon=1e-5):
    """返回 (output, residual)；residual 被原地更新为 x + residual。"""
    x, residual, weight = _contig(x), _contig(residual), _contig(weight)
    out = torch.empty_like(x)
    call("cuda_learn.rmsnorm_and_add", x, residual, weight, out, float(epsilon))
    return out, residual


def softmax(input):
    input = _contig(input)
    out = torch.empty_like(input)
    call("cuda_learn.softmax", input, out)
    return out


def cudnn_softmax(input):
    """FP32 cuDNN accurate softmax benchmark baseline."""
    input = _contig(input)
    out = torch.empty_like(input)
    call("cuda_learn.cudnn_softmax", input, out)
    return out
