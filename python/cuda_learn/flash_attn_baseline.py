"""Lazy benchmark-only binding for the vendored flash-attention 2 kernel."""

import math
import os
from pathlib import Path

import torch

_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    fa_root = root / "third_party" / "flash-attention" / "csrc"
    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(
        str(x) for x in torch.cuda.get_device_capability())
    try:
        _MODULE = load(
            name="cuda_learn_flash_attn_2_d64",
            sources=[
                str(root / "benchmarks" / "flash_attn_2_d64_binding.cpp"),
                str(fa_root / "flash_attn" / "src"
                    / "flash_fwd_hdim64_fp16_sm80.cu"),
                str(fa_root / "flash_attn" / "src"
                    / "flash_fwd_hdim64_fp16_causal_sm80.cu"),
                str(fa_root / "flash_attn" / "src"
                    / "flash_fwd_split_hdim64_fp16_sm80.cu"),
            ],
            extra_include_paths=[
                str(fa_root / "flash_attn"),
                str(fa_root / "flash_attn" / "src"),
                str(fa_root / "cutlass" / "include"),
            ],
            extra_cflags=[
                "-O3", "-std=c++17", "-DFLASHATTENTION_DISABLE_DROPOUT",
                "-DFLASHATTENTION_DISABLE_ALIBI",
                "-DFLASHATTENTION_DISABLE_SOFTCAP",
                "-DFLASHATTENTION_DISABLE_LOCAL",
                "-DFLASHATTENTION_DISABLE_UNEVEN_K",
            ],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--use_fast_math",
                "--expt-relaxed-constexpr", "--expt-extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-DFLASHATTENTION_DISABLE_DROPOUT",
                "-DFLASHATTENTION_DISABLE_ALIBI",
                "-DFLASHATTENTION_DISABLE_SOFTCAP",
                "-DFLASHATTENTION_DISABLE_LOCAL",
                "-DFLASHATTENTION_DISABLE_UNEVEN_K",
            ],
            verbose=False,
        )
    finally:
        if old_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch_list
    return _MODULE


def flash_attn_2(q, k, v, causal=False):
    """Run vendored flash-attention 2.8.3 on physical [B,H,N,64] tensors."""
    out = torch.empty_like(q)
    softmax_lse = torch.empty(q.shape[:3], device=q.device, dtype=torch.float32)
    _load_module().forward(q, k, v, out, softmax_lse, bool(causal))
    return out


def _num_splits_heuristic(batch_heads, num_sms, num_n_blocks):
    if batch_heads >= 0.8 * num_sms:
        return 1
    max_splits = min(128, num_sms, num_n_blocks)
    candidates = []
    for splits in range(1, max_splits + 1):
        blocks = (num_n_blocks + splits - 1) // splits
        previous = ((num_n_blocks + splits - 2) // (splits - 1)
                    if splits > 1 else None)
        if splits > 1 and blocks == previous:
            candidates.append(0.0)
            continue
        waves = batch_heads * splits / num_sms
        candidates.append(waves / math.ceil(waves))
    threshold = 0.85 * max(candidates)
    return next(i + 1 for i, efficiency in enumerate(candidates)
                if efficiency >= threshold)


def flash_attn_2_kvcache(q, k_cache, v_cache, cache_seqlens,
                         num_splits=0):
    """Official FA2 v2.8.3 continuous-cache decode baseline (MHA, D=64)."""
    batch, heads, q_len, dim = q.shape
    if dim != 64 or q_len != 1:
        raise ValueError("benchmark-only FA2 KV-cache baseline expects [B,H,1,64]")
    if num_splits == 0:
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_n_blocks = math.ceil(k_cache.shape[2] / 256)
        num_splits = _num_splits_heuristic(
            batch * heads * math.ceil(q_len / 64), num_sms * 2,
            num_n_blocks)
    out = torch.empty_like(q)
    softmax_lse = torch.empty(
        batch, heads, q_len, device=q.device, dtype=torch.float32)
    softmax_lse_accum = torch.empty(
        num_splits, batch, heads, q_len, device=q.device,
        dtype=torch.float32)
    out_accum = torch.empty(
        num_splits, batch, heads, q_len, 64, device=q.device,
        dtype=torch.float32)
    _load_module().forward_kvcache(
        q, k_cache, v_cache, cache_seqlens, out, softmax_lse,
        softmax_lse_accum, out_accum, num_splits)
    return out
