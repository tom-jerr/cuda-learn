#!/usr/bin/env python3
"""CUPTI-backed kernel timing fallback when NCU counters are unavailable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fa2_kvcache import make_cases, load_extension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--fragmentation", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    load_extension()
    cases = make_cases(
        batch=args.batch, q_len=args.q_len, kv_len=args.kv_len,
        q_heads=args.q_heads, kv_heads=args.kv_heads,
        page_size=args.page_size, fragmentation=args.fragmentation, seed=0,
    )
    result = {}
    for name in ("dense_split", "paged_contiguous", "paged_random"):
        case = cases[name]
        for _ in range(args.warmup):
            case.run()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                case.run()
            torch.cuda.synchronize()
        kernels = [e for e in prof.key_averages() if "flash_fwd_splitkv_kernel" in e.key]
        if len(kernels) != 1:
            raise RuntimeError(f"expected one FA2 kernel row for {name}, got {len(kernels)}")
        event = kernels[0]
        # Kineto reports CUDA time in microseconds.
        result[name] = {
            "calls": event.count,
            "cuda_total_us": event.device_time_total,
            "cuda_mean_us": event.device_time_total / event.count,
        }
        print(f"{name:18s} {result[name]['cuda_mean_us']:.3f} us/kernel")

    result["ratios"] = {
        "random_over_contiguous": (
            result["paged_random"]["cuda_mean_us"] /
            result["paged_contiguous"]["cuda_mean_us"]
        ),
        "paged_over_dense_split": (
            result["paged_contiguous"]["cuda_mean_us"] /
            result["dense_split"]["cuda_mean_us"]
        ),
    }
    print(json.dumps(result["ratios"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
