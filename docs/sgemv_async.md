# Coalesced + `cp.async` SGEMV

## Workload

The benchmark models one FP32 linear projection in batch-1/token-by-token LLM
decoding:

```text
y[11008] = W[11008, 4096] * x[4096]
```

This is the `4096 -> 11008` gated-MLP projection shape used by the original
Llama family.  The row-major weight is 172 MiB, so it does not fit in cache and
each invocation must stream essentially the whole matrix from DRAM.

## Implementation

`src/sgemv_async.cu` contains two kernels:

- `sgemv_coalesced_kernel`: one warp owns one output row; every lane loads one
  aligned `float4`, so a warp reads contiguous weight segments.
- `sgemv_async_kernel`: eight warps still own eight rows, while the whole block
  cooperatively stages 1024-float tiles of `x` into two shared-memory buffers.
  The next tile is issued with `cp.async` before the current tile's weight loads
  and FMAs.

The async kernel uses 256 threads, 38 registers/thread, 8192 bytes of static
shared memory, and reaches 100% theoretical occupancy on the tested GPU.  The
sm_89 SASS contains `LDGSTS.E.BYPASS.128.ZFILL` (the machine instruction emitted
for `cp.async`), `LDS.128` for staged `x`, and `LDG.E.128` for weights.

## Build and reproduce

```bash
cmake -S . -B build
cmake --build build --target sgemv_async_bench -j
./build/benchmarks/sgemv_async_bench --warmup 50 --iters 500
```

To collect Nsight Compute counters after enabling access to NVIDIA performance
counters:

```bash
ncu --kernel-name-base demangled \
  --kernel-name 'regex:.*sgemv_async_kernel.*' \
  --launch-count 1 --set full \
  -o build/sgemv_async_ncu \
  ./build/benchmarks/sgemv_async_bench \
  --kernel async --warmup 0 --iters 1 --skip-check
```

## Result and bottleneck

Measured on an RTX 4060 Laptop GPU (sm_89, 128-bit GDDR6 at 8001 MHz, 256.0
GB/s theoretical DRAM bandwidth).  The table reports the median of five runs;
each run used 50 warmups and 500 timed iterations.

| implementation | latency | effective bandwidth | throughput |
| --- | ---: | ---: | ---: |
| coalesced + `cp.async` | 0.9063 ms | 199.1 GB/s | 0.100 TFLOP/s |
| coalesced only | 0.9037 ms | 199.6 GB/s | 0.100 TFLOP/s |
| cuBLAS SGEMV | 0.8871 ms | 203.4 GB/s | 0.102 TFLOP/s |

The operation performs about 90.2 MFLOP while reading about 180.4 MB, giving a
logical arithmetic intensity of only 0.5 FLOP/byte.  Its bandwidth roof is
therefore roughly 128 GFLOP/s on this GPU.  The async kernel reaches about 78%
of peak DRAM bandwidth and about 100 GFLOP/s, while already having full
theoretical occupancy.  It is DRAM-bandwidth bound on the one-use weight
matrix, rather than compute- or occupancy-bound.

`cp.async` does not improve this workload: it is 0.3% slower than the plain
coalesced kernel at the median (within normal laptop clock/thermal variation).
Only the 16 KiB input vector is staged; it is tiny and cache-resident, whereas
the 172 MiB weight matrix has no reuse.  The async path consequently adds
shared-memory reads and barriers without removing the dominant weight traffic.
Copying weights through shared memory would likewise add traffic because every
weight is consumed only once.

Nsight Compute was invoked on this machine, but hardware-counter collection was
blocked with `ERR_NVGPUCTRPERM`.  Nsight Systems captured CUDA API calls but the
current WDDM/WSL setup did not expose GPU activity records.  The conclusion
above therefore uses CUDA-event timings, device bandwidth attributes, compiler
resource data, and SASS inspection; the command above is ready for a counter
run once host performance-counter permissions are enabled.
