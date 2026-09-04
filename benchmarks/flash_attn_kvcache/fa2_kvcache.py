"""Minimal loader and cache construction for the vendored FA2 KV-cache kernel."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

_MODULE = None


def load_extension():
    """Build only FA2's FP16/D64 regular and split-KV specializations."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    root = here.parents[1]
    fa_root = root / "third_party" / "flash-attention" / "csrc"
    required = fa_root / "flash_attn" / "src" / "flash_fwd_split_hdim64_fp16_sm80.cu"
    if not required.exists():
        raise FileNotFoundError(
            "vendored flash-attention is missing; run "
            "`git submodule update --init third_party/flash-attention`"
        )

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(
        str(x) for x in torch.cuda.get_device_capability()
    )
    try:
        _MODULE = load(
            name="cuda_learn_fa2_kvcache_d64",
            sources=[
                str(here / "fa2_kvcache_binding.cpp"),
                str(fa_root / "flash_attn" / "src" / "flash_fwd_hdim64_fp16_sm80.cu"),
                str(fa_root / "flash_attn" / "src" / "flash_fwd_split_hdim64_fp16_sm80.cu"),
            ],
            extra_include_paths=[
                str(fa_root / "flash_attn"),
                str(fa_root / "flash_attn" / "src"),
                str(fa_root / "cutlass" / "include"),
            ],
            extra_cflags=[
                "-O3", "-std=c++17", "-DFLASHATTENTION_DISABLE_DROPOUT",
                "-DFLASHATTENTION_DISABLE_ALIBI", "-DFLASHATTENTION_DISABLE_SOFTCAP",
                "-DFLASHATTENTION_DISABLE_LOCAL", "-DFLASHATTENTION_DISABLE_UNEVEN_K",
            ],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "--use_fast_math",
                "--expt-relaxed-constexpr", "--expt-extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-DFLASHATTENTION_DISABLE_DROPOUT",
                "-DFLASHATTENTION_DISABLE_ALIBI", "-DFLASHATTENTION_DISABLE_SOFTCAP",
                "-DFLASHATTENTION_DISABLE_LOCAL", "-DFLASHATTENTION_DISABLE_UNEVEN_K",
            ],
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
    return _MODULE


@dataclass
class Case:
    name: str
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    cache_seqlens: torch.Tensor
    block_table: torch.Tensor | None
    mode: int

    def run(self) -> torch.Tensor:
        load_extension().forward(
            self.q, self.k, self.v, self.out, self.lse,
            self.cache_seqlens, self.block_table, self.mode,
        )
        return self.out


def make_cases(
    *,
    batch: int,
    q_len: int,
    kv_len: int,
    q_heads: int,
    kv_heads: int,
    page_size: int,
    fragmentation: int,
    seed: int,
) -> dict[str, Case]:
    """Create four caches containing exactly the same logical K/V sequence."""
    if kv_len % page_size:
        raise ValueError("kv_len must be divisible by page_size")
    if page_size % 256:
        raise ValueError("FA2 requires page_size to be divisible by 256")
    if q_heads % kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    if fragmentation < 1:
        raise ValueError("fragmentation must be >= 1")

    device = "cuda"
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(seed)
    logical_k = torch.randn(
        batch, kv_len, kv_heads, 64, device=device, dtype=dtype, generator=generator
    )
    logical_v = torch.randn(
        batch, kv_len, kv_heads, 64, device=device, dtype=dtype, generator=generator
    )
    q = torch.randn(
        batch, q_len, q_heads, 64, device=device, dtype=dtype, generator=generator
    )
    seqlens = torch.full((batch,), kv_len, device=device, dtype=torch.int32)

    pages_per_seq = kv_len // page_size
    used_pages = batch * pages_per_seq
    physical_pages = used_pages * fragmentation
    page_shape = (physical_pages, page_size, kv_heads, 64)

    # Both page pools have the same allocated size. The contiguous table selects
    # adjacent pages; the random table selects and orders pages across the pool.
    seq_k = torch.empty(page_shape, device=device, dtype=dtype)
    seq_v = torch.empty_like(seq_k)
    seq_table = torch.arange(used_pages, device=device, dtype=torch.int32).view(
        batch, pages_per_seq
    )
    logical_k_pages = logical_k.view(used_pages, page_size, kv_heads, 64)
    logical_v_pages = logical_v.view_as(logical_k_pages)
    seq_k[:used_pages].copy_(logical_k_pages)
    seq_v[:used_pages].copy_(logical_v_pages)

    rand_k = torch.empty(page_shape, device=device, dtype=dtype)
    rand_v = torch.empty_like(rand_k)
    rand_ids = torch.randperm(
        physical_pages, device=device, dtype=torch.int32, generator=generator
    )[:used_pages]
    rand_table = rand_ids.view(batch, pages_per_seq)
    rand_k[rand_ids.long()] = logical_k_pages
    rand_v[rand_ids.long()] = logical_v_pages

    def case(name: str, k: torch.Tensor, v: torch.Tensor,
             table: torch.Tensor | None, mode: int) -> Case:
        return Case(
            name=name,
            q=q,
            k=k,
            v=v,
            out=torch.empty_like(q),
            lse=torch.empty(batch, q_heads, q_len, device=device, dtype=torch.float32),
            cache_seqlens=seqlens,
            block_table=table,
            mode=mode,
        )

    return {
        "dense": case("dense", logical_k, logical_v, None, 0),
        "dense_split": case("dense_split", logical_k, logical_v, None, 1),
        "paged_contiguous": case("paged_contiguous", seq_k, seq_v, seq_table, 2),
        "paged_random": case("paged_random", rand_k, rand_v, rand_table, 2),
    }
