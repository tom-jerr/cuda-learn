#!/usr/bin/env python3
"""Paired large-batch benchmark for contiguous vs random FA2 page tables."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from fa2_kvcache import load_extension, make_cases


def paired_time(cases, warmup: int, iters: int):
    names = ("paged_contiguous", "paged_random")
    for iteration in range(warmup):
        order = names if iteration % 2 == 0 else names[::-1]
        for name in order:
            cases[name].run()
    torch.cuda.synchronize()

    events = {
        name: [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iters)
        ]
        for name in names
    }
    for iteration in range(iters):
        order = names if iteration % 2 == 0 else names[::-1]
        for name in order:
            start, end = events[name][iteration]
            start.record()
            cases[name].run()
            end.record()
    torch.cuda.synchronize()
    return {
        name: [start.elapsed_time(end) * 1000.0 for start, end in events[name]]
        for name in names
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--fragmentation", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    load_extension()
    rows = []
    print(f"GPU: {torch.cuda.get_device_name()}")
    for batch in args.batches:
        for kv_len in args.kv_lens:
            cases = make_cases(
                batch=batch, q_len=1, kv_len=kv_len,
                q_heads=args.q_heads, kv_heads=args.kv_heads,
                page_size=args.page_size, fragmentation=args.fragmentation, seed=0,
            )
            timings = paired_time(cases, args.warmup, args.iters)
            contiguous = statistics.median(timings["paged_contiguous"])
            random = statistics.median(timings["paged_random"])
            useful_bytes = batch * kv_len * args.kv_heads * 64 * 2 * 2  # K + V, FP16
            row = {
                "batch": batch,
                "kv_len": kv_len,
                "paged_contiguous_median_us": contiguous,
                "paged_random_median_us": random,
                "random_over_contiguous": random / contiguous,
                "contiguous_useful_gbps": useful_bytes / (contiguous * 1e-6) / 1e9,
                "random_useful_gbps": useful_bytes / (random * 1e-6) / 1e9,
            }
            rows.append(row)
            print(
                f"BS={batch:2d} KV={kv_len:5d}: contiguous={contiguous:9.3f} us, "
                f"random={random:9.3f} us, ratio={random / contiguous:.4f}x, "
                f"BW={row['contiguous_useful_gbps']:.1f}/{row['random_useful_gbps']:.1f} GB/s"
            )
            del cases
            torch.cuda.empty_cache()

    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "config": {
            "q_len": 1, "q_heads": args.q_heads, "kv_heads": args.kv_heads,
            "head_dim": 64, "dtype": "float16", "page_size": args.page_size,
            "fragmentation": args.fragmentation, "warmup": args.warmup,
            "iters": args.iters, "timing": "paired, alternating launch order",
        },
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
