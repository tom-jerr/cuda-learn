#pragma once

#include <cuda_runtime.h>

struct SgemvKernelInfo {
  int threads_per_block;
  int registers_per_thread;
  int static_smem_bytes;
  int active_blocks_per_sm;
};

// Row-major SGEMV: y[rows] = weight[rows, cols] * x[cols].
// The optimized path uses float4 loads when cols is a multiple of four;
// arbitrary sizes fall back to a coalesced scalar kernel.
void launch_sgemv_coalesced(const float *weight, const float *x, float *y,
                            int rows, int cols, cudaStream_t stream = nullptr);

// Ampere+ implementation.  Each warp owns one output row, while all warps in
// a block cooperatively stage x with a two-stage cp.async pipeline.
void launch_sgemv_async(const float *weight, const float *x, float *y, int rows,
                        int cols, cudaStream_t stream = nullptr);

// Shared-memory-free path: one warp computes several rows and reuses each
// vector load of x from registers. rows_per_warp must be 2, 4, 8, or 16.
void launch_sgemv_register_reuse(const float *weight, const float *x, float *y,
                                 int rows, int cols, int rows_per_warp,
                                 cudaStream_t stream = nullptr);

SgemvKernelInfo get_sgemv_async_kernel_info();
SgemvKernelInfo get_sgemv_register_kernel_info();
