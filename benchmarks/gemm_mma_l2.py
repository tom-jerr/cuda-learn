"""Compare the ordinary and L2-cohort rasters of the BF16 MMA GEMM.

Run from the repository root after building libcuda_learn.so:

  source scripts/env.sh
  PYTHONPATH=python python benchmarks/gemm_mma_l2.py
"""

import argparse
import statistics

import torch

from cuda_learn import ops


def time_variants(variants, warmup, iters):
    """Interleave variants so boost/thermal drift is shared by every kernel."""
    for _ in range(warmup):
        for _, fn in variants:
            fn()
    torch.cuda.synchronize()

    pending = []
    count = len(variants)
    for iteration in range(iters):
        # Rotate which implementation goes first in each round.
        order = variants[iteration % count:] + variants[:iteration % count]
        for name, fn in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            pending.append((name, start, end))
    torch.cuda.synchronize()

    samples = {name: [] for name, _ in variants}
    for name, start, end in pending:
        samples[name].append(start.elapsed_time(end))
    return {name: (statistics.median(values), min(values))
            for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["4096x4096x4096",
                                                        "8192x8192x8192"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("shape is MxNxK; values are median/min milliseconds")

    for shape in args.shapes:
        m, n, k = map(int, shape.lower().split("x"))
        if m % 64 or n % 64 or k % 32:
            raise ValueError(f"unaligned shape: {shape}")
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)

        ref = ops.gemm_mma(a, b)
        l2 = ops.gemm_mma_l2(a, b, 8)
        torch.testing.assert_close(l2, ref, rtol=0, atol=0)

        variants = [("ordinary", lambda: ops.gemm_mma(a, b))]
        variants += [(f"swizzle-{s}",
                      lambda s=s: ops.gemm_mma_l2(a, b, s))
                     for s in (2, 4, 8)]

        timings = time_variants(variants, args.warmup, args.iters)
        results = [(name, *timings[name]) for name, _ in variants]

        baseline = results[0][1]
        print(f"\n{m}x{n}x{k}")
        for name, median, minimum in results:
            tflops = 2.0 * m * n * k / (median * 1e-3) / 1e12
            speedup = baseline / median
            print(f"  {name:10s} {median:8.3f}/{minimum:8.3f} ms"
                  f"  {tflops:7.2f} TFLOPS  {speedup:6.3f}x")


if __name__ == "__main__":
    main()
