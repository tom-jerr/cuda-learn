"""Lazy benchmark-only binding for the vendored flash-attention 2 kernel."""

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
    """Run vendored flash-attention 2.8.4 on physical [B,H,N,64] tensors."""
    out = torch.empty_like(q)
    softmax_lse = torch.empty(q.shape[:3], device=q.device, dtype=torch.float32)
    _load_module().forward(q, k, v, out, softmax_lse, bool(causal))
    return out
