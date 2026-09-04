#!/usr/bin/env python3
"""Benchmark sequential versus randomly paged FA2 KV-cache reads."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from fa2_kvcache import make_cases, load_extension

CASE_ORDER = ("dense", "dense_split", "paged_contiguous", "paged_random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--kv-lens", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--fragmentation", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def time_case(case, warmup: int, iters: int) -> list[float]:
    """Measure the steady-state cache layout after an independent warmup."""
    for _ in range(warmup):
        case.run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        case.run()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    load_extension()
    rows = []
    print(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | FA2 vendored 2.8.3")
    print(
        f"B={args.batch}, Sq={args.q_len}, Hq/Hkv={args.q_heads}/{args.kv_heads}, "
        f"D=64, FP16, page={args.page_size}, fragmentation={args.fragmentation}x"
    )

    for kv_len in args.kv_lens:
        cases = make_cases(
            batch=args.batch, q_len=args.q_len, kv_len=kv_len,
            q_heads=args.q_heads, kv_heads=args.kv_heads,
            page_size=args.page_size, fragmentation=args.fragmentation,
            seed=args.seed,
        )
        # Correctness is equality between layouts, avoiding a different math backend.
        for case in cases.values():
            case.run()
        torch.cuda.synchronize()
        reference = cases["dense"].out.float()
        for name in CASE_ORDER[1:]:
            error = (cases[name].out.float() - reference).abs().max().item()
            if error > 5e-3:
                raise RuntimeError(f"{name} max error {error} exceeds 5e-3")

        timings = {
            name: time_case(cases[name], args.warmup, args.iters)
            for name in CASE_ORDER
        }
        medians = {name: statistics.median(values) for name, values in timings.items()}
        print(f"\nKV length {kv_len}")
        for name in CASE_ORDER:
            values = timings[name]
            row = {
                "kv_len": kv_len,
                "case": name,
                "median_us": medians[name],
                "mean_us": statistics.mean(values),
                "min_us": min(values),
                "random_over_case": medians["paged_random"] / medians[name],
            }
            rows.append(row)
            print(
                f"  {name:18s} median {row['median_us']:9.3f} us  "
                f"mean {row['mean_us']:9.3f} us  min {row['min_us']:9.3f} us"
            )
        print(
            "  locality penalty: random / paged_contiguous = "
            f"{medians['paged_random'] / medians['paged_contiguous']:.3f}x; "
            "paged indirection: paged_contiguous / dense_split = "
            f"{medians['paged_contiguous'] / medians['dense_split']:.3f}x; "
            "API path total: random / dense = "
            f"{medians['paged_random'] / medians['dense']:.3f}x"
        )
        del cases
        torch.cuda.empty_cache()

    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "flash_attn": "vendored 2.8.3 official FP16/D64 specializations",
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".json":
            args.output.write_text(json.dumps(metadata, indent=2) + "\n")
        else:
            with args.output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
