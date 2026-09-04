#!/usr/bin/env python3
"""Run one deterministic FA2 KV-cache case for Nsight Compute/Systems."""

from __future__ import annotations

import argparse

import torch

from fa2_kvcache import make_cases, load_extension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[
        "dense", "dense_split", "paged_contiguous", "paged_random"
    ], required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--fragmentation", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=1)
    args = parser.parse_args()

    load_extension()
    case = make_cases(
        batch=args.batch, q_len=args.q_len, kv_len=args.kv_len,
        q_heads=args.q_heads, kv_heads=args.kv_heads,
        page_size=args.page_size, fragmentation=args.fragmentation, seed=0,
    )[args.case]
    for _ in range(args.warmup):
        case.run()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"fa2_kvcache_{args.case}")
    for _ in range(args.profile_iters):
        case.run()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
