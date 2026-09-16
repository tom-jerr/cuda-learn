"""统一 pytorch binding：每个 kernel 一个纯函数，torch.Tensor 进出。

约定：
- 全局函数名 = "cuda_learn." + Python 函数名；
- 输出张量由 Python 预分配、作为 out 参数传入（统一 out-param 形式）；
- 输入必须连续（tvm_ffi from_dlpack 的硬性检查），此处统一 .contiguous()。
"""

import math

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


def _inplace_cache_fp16(t, name):
    """Validate an in-place FP16 cache without silently copying its storage."""
    if not t.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor, got {t.device}")
    if t.dtype != torch.float16:
        raise TypeError(f"{name} must be float16, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous because it is updated in place")
    return t


def _contig_i32(t):
    if not t.is_cuda:
        raise TypeError(f"cuda_learn ops require CUDA tensors, got {t.device}")
    if t.dtype != torch.int32:
        raise TypeError(f"cuda_learn ops require int32 tensors, got {t.dtype}")
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


def sgemv(weight, x):
    """FP32 row-major SGEMV optimized for batch-1/token decoding."""
    weight, x = _contig(weight), _contig(x)
    if weight.ndim != 2 or x.ndim != 1:
        raise ValueError("sgemv expects weight[rows, cols] and x[cols]")
    if weight.shape[1] != x.shape[0]:
        raise ValueError("sgemv weight and x reduction dimensions must match")
    out = torch.empty(weight.shape[0], device=weight.device, dtype=weight.dtype)
    call("cuda_learn.sgemv", weight, x, out)
    return out


def sgemv_async(weight, x):
    """Teaching variant that stages x through shared memory with cp.async."""
    weight, x = _contig(weight), _contig(x)
    if weight.ndim != 2 or x.ndim != 1:
        raise ValueError("sgemv_async expects weight[rows, cols] and x[cols]")
    if weight.shape[1] != x.shape[0]:
        raise ValueError(
            "sgemv_async weight and x reduction dimensions must match")
    out = torch.empty(weight.shape[0], device=weight.device, dtype=weight.dtype)
    call("cuda_learn.sgemv_async", weight, x, out)
    return out


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


def gemm_cutlass_ampere(a, b):
    """BF16 row-major GEMM using CUTLASS's SM80 multistage Tensor Core path."""
    a, b = _contig_bf16(a), _contig_bf16(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemm_cutlass_ampere expects two matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("gemm_cutlass_ampere reduction dimensions must match")
    if a.shape[1] % 8 or b.shape[1] % 8:
        raise ValueError(
            "gemm_cutlass_ampere requires K and N to be multiples of 8")
    c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    call("cuda_learn.gemm_cutlass_ampere", a, b, c)
    return c


def gemm_mma_l2(a, b, swizzle_n=8):
    """BF16 MMA GEMM with an N-cohort CTA raster for L2 locality."""
    if swizzle_n not in (2, 4, 8):
        raise ValueError(f"swizzle_n must be 2, 4, or 8, got {swizzle_n}")
    a, b = _contig_bf16(a), _contig_bf16(b)
    c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    call("cuda_learn.gemm_mma_l2", a, b, c, swizzle_n)
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


def flash_attn_multistage(q, k, v, causal=False):
    """64x64, 4-warp FP16 attention with two XOR-swizzled K/V stages."""
    q, k, v = _contig_fp16(q), _contig_fp16(k), _contig_fp16(v)
    if q.ndim != 4 or q.shape[-1] != 64:
        raise ValueError(
            f"flash_attn_multistage expects q shaped [B,H,N,64], got {q.shape}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "flash_attn_multistage expects q, k and v to have identical shapes")
    if q.shape[2] == 0 or q.shape[2] % 64:
        raise ValueError(
            "flash_attn_multistage sequence length must be a positive multiple of 64")
    out = torch.empty_like(q)
    call("cuda_learn.flash_attn_multistage", q, k, v, out,
         int(bool(causal)))
    return out


def flash_attn_cutlass3(q, k, v, causal=False):
    """CUTLASS 3.x CuTe FA2: FP16 [B,H,N,64], N divisible by 64."""
    q, k, v = _contig_fp16(q), _contig_fp16(k), _contig_fp16(v)
    if q.ndim != 4 or q.shape[-1] != 64:
        raise ValueError(
            f"flash_attn_cutlass3 expects q shaped [B,H,N,64], got {q.shape}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "flash_attn_cutlass3 expects q, k and v to have identical shapes")
    if q.shape[2] == 0 or q.shape[2] % 64:
        raise ValueError(
            "flash_attn_cutlass3 sequence length must be a positive multiple of 64")
    out = torch.empty_like(q)
    call("cuda_learn.flash_attn_cutlass3", q, k, v, out,
         int(bool(causal)))
    return out


def flash_attn_multistage_kvcache(
        q, k_cache, v_cache, cache_seqlens, k=None, v=None, causal=True):
    """Attend to a continuous KV cache and optionally append current K/V.

    Layouts use this repository's head-major convention:
      q/out: [B,Hq,Q,64], cache: [B,Hkv,capacity,64], lengths: int32 [B].
    Q must be in [1,64]. Cache-only mode supports GQA; fused append currently
    requires Hq == Hkv and k/v shaped [B,Hkv,Q,64]. ``cache_seqlens`` contains
    lengths before append and is deliberately not incremented by this call.
    """
    q = _contig_fp16(q)
    k_cache = _inplace_cache_fp16(k_cache, "k_cache")
    v_cache = _inplace_cache_fp16(v_cache, "v_cache")
    cache_seqlens = _contig_i32(cache_seqlens)
    if q.ndim != 4 or q.shape[-1] != 64 or not 1 <= q.shape[2] <= 64:
        raise ValueError(
            f"expected q [B,Hq,Q,64] with 1<=Q<=64, got {q.shape}")
    if k_cache.ndim != 4 or k_cache.shape[-1] != 64:
        raise ValueError(
            f"expected k_cache [B,Hkv,capacity,64], got {k_cache.shape}")
    if v_cache.shape != k_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")
    if k_cache.shape[0] != q.shape[0] or q.shape[1] % k_cache.shape[1]:
        raise ValueError("cache batch must match q and Hq must be divisible by Hkv")
    if cache_seqlens.shape != (q.shape[0],):
        raise ValueError("cache_seqlens must be int32 [B]")
    if (k is None) != (v is None):
        raise ValueError("k and v must either both be provided or both be None")

    out = torch.empty_like(q)
    if k is None:
        call("cuda_learn.flash_attn_multistage_kvcache", q, k_cache,
             v_cache, cache_seqlens, out, int(bool(causal)))
    else:
        k, v = _contig_fp16(k), _contig_fp16(v)
        expected = (q.shape[0], k_cache.shape[1], q.shape[2], 64)
        if k.shape != expected or v.shape != expected:
            raise ValueError(f"k/v must have shape {expected}")
        if q.shape[1] != k_cache.shape[1]:
            raise ValueError("fused append currently requires Hq == Hkv")
        call("cuda_learn.flash_attn_multistage_kvcache_append", q,
             k_cache, v_cache, cache_seqlens, k, v, out,
             int(bool(causal)))
    return out


def flash_attn3_hopper(q, k, v, causal=False):
    """Raw-PTX Hopper FA3 reproduction (BF16, forward, [B,H,N,64])."""
    q, k, v = _contig_bf16(q), _contig_bf16(k), _contig_bf16(v)
    if q.ndim != 4 or q.shape[-1] != 64:
        raise ValueError(
            f"flash_attn3_hopper expects q shaped [B,H,N,64], got {q.shape}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "flash_attn3_hopper expects q, k and v to have identical shapes")
    if q.shape[2] == 0 or q.shape[2] % 128:
        raise ValueError(
            "flash_attn3_hopper sequence length must be a positive multiple "
            "of 128")
    out = torch.empty_like(q)
    call("cuda_learn.flash_attn3_hopper", q, k, v, out, int(bool(causal)))
    return out


def _mla_decode(q, kv_cache, block_table, cache_seqlens, softmax_scale,
                name):
    """Fixed-shape DeepSeek-V3 MLA decode used by the native CUDA examples.

    q: [B, 128, 576] BF16
    kv_cache: [num_pages, 64, 576] BF16; columns [:512] are also V
    block_table: [B, max_pages] int32
    cache_seqlens: [B] int32
    """
    q = _contig_bf16(q)
    kv_cache = _contig_bf16(kv_cache)
    block_table = _contig_i32(block_table)
    cache_seqlens = _contig_i32(cache_seqlens)
    if q.ndim != 3 or q.shape[1:] != (128, 576):
        raise ValueError(f"MLA q must have shape [B,128,576], got {q.shape}")
    if kv_cache.ndim != 3 or kv_cache.shape[1:] != (64, 576):
        raise ValueError(
            "MLA kv_cache must have shape [num_pages,64,576], got "
            f"{kv_cache.shape}")
    if block_table.ndim != 2 or block_table.shape[0] != q.shape[0]:
        raise ValueError("MLA block_table must have shape [B,max_pages]")
    if cache_seqlens.shape != (q.shape[0],):
        raise ValueError("MLA cache_seqlens must have shape [B]")
    if len({q.device, kv_cache.device, block_table.device,
            cache_seqlens.device}) != 1:
        raise ValueError("all MLA tensors must be on the same CUDA device")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(576)
    softmax_scale = float(softmax_scale)
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    out = torch.empty(q.shape[0], 128, 512, device=q.device,
                      dtype=q.dtype)
    lse = torch.empty(q.shape[0], 128, device=q.device,
                      dtype=torch.float32)
    call(name, q, kv_cache, block_table, cache_seqlens, out, lse,
         softmax_scale)
    return out, lse


def mla_decode_native(q, kv_cache, block_table, cache_seqlens,
                      softmax_scale=None):
    """One-query-head-per-CTA CUDA-core MLA decoding baseline."""
    return _mla_decode(q, kv_cache, block_table, cache_seqlens,
                       softmax_scale, "cuda_learn.mla_decode_native")


def mla_decode_optimized(q, kv_cache, block_table, cache_seqlens,
                         softmax_scale=None):
    """Eight-head/CTA MLA decode with cp.async page sharing and online softmax."""
    return _mla_decode(q, kv_cache, block_table, cache_seqlens,
                       softmax_scale, "cuda_learn.mla_decode_optimized")


def rope_neox(q, k, cos_cache, sin_cache, position_ids):
    """Apply in-place split-half FP16 RoPE and return ``(q, k)``.

    Q and K use layouts ``[tokens, heads, head_dim]``. The FP16 caches use
    ``[max_position, rotary_dim // 2]`` and position_ids is a CUDA int32 vector.
    If a non-contiguous Q or K is supplied, its contiguous copy is returned and
    modified instead of the original view.
    """
    q, k = _contig_fp16(q), _contig_fp16(k)
    cos_cache = _contig_fp16(cos_cache)
    sin_cache = _contig_fp16(sin_cache)
    position_ids = _contig_i32(position_ids)

    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("rope_neox expects q and k shaped [tokens,heads,head_dim]")
    if q.shape[0] == 0 or q.shape[1] == 0 or k.shape[1] == 0 or q.shape[2] == 0:
        raise ValueError("rope_neox tensor dimensions must be positive")
    if k.shape[0] != q.shape[0] or k.shape[2] != q.shape[2]:
        raise ValueError("rope_neox q and k token/head dimensions must match")
    if cos_cache.ndim != 2 or sin_cache.shape != cos_cache.shape:
        raise ValueError(
            "rope_neox cos_cache and sin_cache must have identical 2D shapes")
    if cos_cache.shape[0] == 0 or cos_cache.shape[1] == 0:
        raise ValueError("rope_neox cache dimensions must be positive")
    if position_ids.ndim != 1 or position_ids.shape[0] != q.shape[0]:
        raise ValueError("rope_neox position_ids length must equal the token count")
    rotary_half = cos_cache.shape[1]
    if rotary_half * 2 > q.shape[2]:
        raise ValueError("rope_neox rotary_dim inferred from cache exceeds head_dim")
    if rotary_half % 2 or q.shape[2] % 2:
        raise ValueError(
            "rope_neox requires rotary_dim % 4 == 0 and head_dim % 2 == 0")
    if len({q.device, k.device, cos_cache.device, sin_cache.device,
            position_ids.device}) != 1:
        raise ValueError("rope_neox tensors must be on the same CUDA device")
    if q.data_ptr() == k.data_ptr():
        raise ValueError("rope_neox q and k must not alias")

    call("cuda_learn.rope_neox", q, k, cos_cache, sin_cache, position_ids)
    return q, k


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


def _rmsnorm(input, epsilon, name):
    input = _contig(input)
    if input.ndim != 2:
        raise ValueError(f"rmsnorm expects a 2D tensor, got shape {input.shape}")
    if input.shape[0] == 0 or input.shape[1] == 0:
        raise ValueError("rmsnorm dimensions must be positive")
    if not isinstance(epsilon, (int, float)):
        raise TypeError(f"rmsnorm epsilon must be a number, got {type(epsilon)}")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("rmsnorm epsilon must be finite and non-negative")
    out = torch.empty_like(input)
    call(name, input, out, epsilon)
    return out


def rmsnorm(input, epsilon=1e-5):
    """RMS normalization over the last dimension (without affine weight)."""
    return _rmsnorm(input, epsilon, "cuda_learn.rmsnorm")


def rmsnorm_baseline(input, epsilon=1e-5):
    """Scalar/shared-memory reduction baseline used for performance comparison."""
    return _rmsnorm(input, epsilon, "cuda_learn.rmsnorm_baseline")


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
