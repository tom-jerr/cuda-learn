#include "include/sgemv_async.cuh"

#include <cuda_runtime.h>

#ifndef CUDA_LEARN_SGEMV_NO_FFI
#include "include/ffi_common.h"
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
constexpr int kVectorWidth = 4;
constexpr int kTileCols = 1024;

static_assert(kTileCols == kThreads * kVectorWidth);

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ void cp_async_16(void *dst, const void *src,
                                            int valid_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned dst_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :
               : "r"(dst_addr), "l"(src), "r"(valid_bytes));
#else
  float *dst_float = static_cast<float *>(dst);
  const float *src_float = static_cast<const float *>(src);
#pragma unroll
  for (int i = 0; i < kVectorWidth; ++i) {
    dst_float[i] = i * static_cast<int>(sizeof(float)) < valid_bytes
                       ? src_float[i]
                       : 0.0f;
  }
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : :);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" : :);
#endif
}

__device__ __forceinline__ void load_x_tile_async(float *dst, const float *x,
                                                   int tile_begin, int cols) {
  const int local_col = threadIdx.x * kVectorWidth;
  const int global_col = tile_begin + local_col;
  const int remaining = cols - global_col;
  const int valid_floats = remaining <= 0 ? 0 : (remaining < 4 ? remaining : 4);
  // Keep the source address inside the allocation when cp.async is asked to
  // zero-fill an entirely out-of-range vector.
  const float *safe_src = valid_floats == 0 ? x : x + global_col;
  cp_async_16(dst + local_col, safe_src,
              valid_floats * static_cast<int>(sizeof(float)));
}

__global__ void sgemv_coalesced_kernel(const float *__restrict__ weight,
                                       const float *__restrict__ x,
                                       float *__restrict__ y, int rows,
                                       int cols) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  const float *row_ptr = weight + static_cast<size_t>(row) * cols;
  float sum = 0.0f;
  int col = lane * kVectorWidth;
  for (; col + 3 < cols; col += kWarpSize * kVectorWidth) {
    const float4 w = *reinterpret_cast<const float4 *>(row_ptr + col);
    const float4 xv = *reinterpret_cast<const float4 *>(x + col);
    sum = fmaf(w.x, xv.x, sum);
    sum = fmaf(w.y, xv.y, sum);
    sum = fmaf(w.z, xv.z, sum);
    sum = fmaf(w.w, xv.w, sum);
  }
  for (; col < cols; ++col) {
    sum = fmaf(row_ptr[col], x[col], sum);
  }

  sum = warp_sum(sum);
  if (lane == 0) {
    y[row] = sum;
  }
}

__global__ void sgemv_scalar_kernel(const float *__restrict__ weight,
                                    const float *__restrict__ x,
                                    float *__restrict__ y, int rows, int cols) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }
  const float *row_ptr = weight + static_cast<size_t>(row) * cols;
  float sum = 0.0f;
  for (int col = lane; col < cols; col += kWarpSize) {
    sum = fmaf(row_ptr[col], x[col], sum);
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    y[row] = sum;
  }
}

template <int RowsPerWarp, int WarpsPerBlock>
__global__ void sgemv_register_reuse_kernel(
    const float *__restrict__ weight, const float *__restrict__ x,
    float *__restrict__ y, int rows, int cols) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int first_row =
      (blockIdx.x * WarpsPerBlock + warp) * RowsPerWarp;
  if (first_row >= rows) {
    return;
  }

  float sums[RowsPerWarp] = {};
  for (int col = lane * kVectorWidth; col < cols;
       col += kWarpSize * kVectorWidth) {
    const float4 xv = *reinterpret_cast<const float4 *>(x + col);
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
      const int row = first_row + r;
      if (row < rows) {
        const float *row_ptr = weight + static_cast<size_t>(row) * cols;
        const float4 w = *reinterpret_cast<const float4 *>(row_ptr + col);
        sums[r] = fmaf(w.x, xv.x, sums[r]);
        sums[r] = fmaf(w.y, xv.y, sums[r]);
        sums[r] = fmaf(w.z, xv.z, sums[r]);
        sums[r] = fmaf(w.w, xv.w, sums[r]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    sums[r] = warp_sum(sums[r]);
    if (lane == 0 && first_row + r < rows) {
      y[first_row + r] = sums[r];
    }
  }
}

__global__ void sgemv_async_kernel(const float *__restrict__ weight,
                                   const float *__restrict__ x,
                                   float *__restrict__ y, int rows, int cols) {
  __shared__ __align__(16) float x_tiles[2][kTileCols];

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  const float *row_ptr =
      row < rows ? weight + static_cast<size_t>(row) * cols : weight;
  const int tile_count = (cols + kTileCols - 1) / kTileCols;
  float sum = 0.0f;

  load_x_tile_async(x_tiles[0], x, 0, cols);
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  for (int tile = 0; tile < tile_count; ++tile) {
    const int current = tile & 1;
    const int next = current ^ 1;
    const int tile_begin = tile * kTileCols;
    const int valid_cols =
        cols - tile_begin < kTileCols ? cols - tile_begin : kTileCols;

    // Start the next x transfer before consuming the current tile.  The
    // matrix loads and FMAs below provide the overlap window.
    if (tile + 1 < tile_count) {
      load_x_tile_async(x_tiles[next], x, tile_begin + kTileCols, cols);
      cp_async_commit();
    }

    if (row < rows) {
      for (int local_col = lane * kVectorWidth; local_col < valid_cols;
           local_col += kWarpSize * kVectorWidth) {
        if (local_col + 3 < valid_cols) {
          const float4 w = *reinterpret_cast<const float4 *>(
              row_ptr + tile_begin + local_col);
          const float4 xv = *reinterpret_cast<const float4 *>(
              x_tiles[current] + local_col);
          sum = fmaf(w.x, xv.x, sum);
          sum = fmaf(w.y, xv.y, sum);
          sum = fmaf(w.z, xv.z, sum);
          sum = fmaf(w.w, xv.w, sum);
        } else {
          for (int i = local_col; i < valid_cols; ++i) {
            sum = fmaf(row_ptr[tile_begin + i], x_tiles[current][i], sum);
          }
        }
      }
    }

    if (tile + 1 < tile_count) {
      cp_async_wait_all();
      // Every thread has completed both its current-tile reads and its
      // next-tile copy before either shared-memory buffer can be reused.
      __syncthreads();
    }
  }

  if (row < rows) {
    sum = warp_sum(sum);
    if (lane == 0) {
      y[row] = sum;
    }
  }
}

} // namespace

void launch_sgemv_coalesced(const float *weight, const float *x, float *y,
                            int rows, int cols, cudaStream_t stream) {
  const dim3 block(kThreads);
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if ((cols & (kVectorWidth - 1)) != 0) {
    sgemv_scalar_kernel<<<grid, block, 0, stream>>>(weight, x, y, rows, cols);
    return;
  }
  sgemv_coalesced_kernel<<<grid, block, 0, stream>>>(weight, x, y, rows, cols);
}

void launch_sgemv_async(const float *weight, const float *x, float *y, int rows,
                        int cols, cudaStream_t stream) {
  const dim3 block(kThreads);
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if ((cols & (kVectorWidth - 1)) != 0) {
    sgemv_scalar_kernel<<<grid, block, 0, stream>>>(weight, x, y, rows, cols);
    return;
  }
  sgemv_async_kernel<<<grid, block, 0, stream>>>(weight, x, y, rows, cols);
}

void launch_sgemv_register_reuse(const float *weight, const float *x, float *y,
                                 int rows, int cols, int rows_per_warp,
                                 cudaStream_t stream) {
  const dim3 block(kThreads);
  if ((cols & (kVectorWidth - 1)) != 0) {
    const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    sgemv_scalar_kernel<<<grid, block, 0, stream>>>(weight, x, y, rows, cols);
    return;
  }

  switch (rows_per_warp) {
  case 2: {
    const dim3 grid((rows + kWarpsPerBlock * 2 - 1) /
                    (kWarpsPerBlock * 2));
    sgemv_register_reuse_kernel<2, kWarpsPerBlock>
        <<<grid, block, 0, stream>>>(weight, x, y, rows, cols);
    break;
  }
  case 4: {
    const dim3 grid((rows + kWarpsPerBlock * 4 - 1) /
                    (kWarpsPerBlock * 4));
    sgemv_register_reuse_kernel<4, kWarpsPerBlock>
        <<<grid, block, 0, stream>>>(weight, x, y, rows, cols);
    break;
  }
  case 8: {
    constexpr int warps = 4;
    const dim3 tuned_block(warps * kWarpSize);
    const dim3 grid((rows + warps * 8 - 1) / (warps * 8));
    sgemv_register_reuse_kernel<8, warps>
        <<<grid, tuned_block, 0, stream>>>(weight, x, y, rows, cols);
    break;
  }
  case 16: {
    constexpr int warps = 4;
    const dim3 tuned_block(warps * kWarpSize);
    const dim3 grid((rows + warps * 16 - 1) / (warps * 16));
    sgemv_register_reuse_kernel<16, warps>
        <<<grid, tuned_block, 0, stream>>>(weight, x, y, rows, cols);
    break;
  }
  default:
    // Keep this low-level launcher non-throwing; use the conservative path for
    // unsupported tuning values.
    launch_sgemv_coalesced(weight, x, y, rows, cols, stream);
    break;
  }
}

SgemvKernelInfo get_sgemv_async_kernel_info() {
  cudaFuncAttributes attributes{};
  cudaFuncGetAttributes(&attributes, sgemv_async_kernel);
  int active_blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, sgemv_async_kernel, kThreads, 0);
  return {kThreads, attributes.numRegs,
          static_cast<int>(attributes.sharedSizeBytes), active_blocks};
}

SgemvKernelInfo get_sgemv_register_kernel_info() {
  constexpr int warps = 4;
  constexpr int threads = warps * kWarpSize;
  cudaFuncAttributes attributes{};
  cudaFuncGetAttributes(&attributes, sgemv_register_reuse_kernel<8, warps>);
  int active_blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, sgemv_register_reuse_kernel<8, warps>, threads, 0);
  return {threads, attributes.numRegs,
          static_cast<int>(attributes.sharedSizeBytes), active_blocks};
}

#ifndef CUDA_LEARN_SGEMV_NO_FFI
namespace {

using tvm::ffi::TensorView;

void sgemv_async_ffi(TensorView weight, TensorView x, TensorView y) {
  check_tensor(weight, "weight", 2);
  check_tensor(x, "x", 1);
  check_tensor(y, "y", 1);
  const int rows = static_cast<int>(dim(weight, 0));
  const int cols = static_cast<int>(dim(weight, 1));
  if (dim(x, 0) != cols || dim(y, 0) != rows) {
    TVM_FFI_THROW(RuntimeError)
        << "sgemv_async: expected weight[rows, cols] @ x[cols] -> y[rows]";
  }
  launch_sgemv_async(static_cast<const float *>(weight.data_ptr()),
                     static_cast<const float *>(x.data_ptr()),
                     static_cast<float *>(y.data_ptr()), rows, cols,
                     get_stream(weight));
  CUDA_LEARN_CHECK(cudaGetLastError());
}

void sgemv_register_reuse_ffi(TensorView weight, TensorView x, TensorView y) {
  check_tensor(weight, "weight", 2);
  check_tensor(x, "x", 1);
  check_tensor(y, "y", 1);
  const int rows = static_cast<int>(dim(weight, 0));
  const int cols = static_cast<int>(dim(weight, 1));
  if (dim(x, 0) != cols || dim(y, 0) != rows) {
    TVM_FFI_THROW(RuntimeError)
        << "sgemv: expected weight[rows, cols] @ x[cols] -> y[rows]";
  }
  launch_sgemv_register_reuse(
      static_cast<const float *>(weight.data_ptr()),
      static_cast<const float *>(x.data_ptr()),
      static_cast<float *>(y.data_ptr()), rows, cols, 8, get_stream(weight));
  CUDA_LEARN_CHECK(cudaGetLastError());
}

} // namespace

CUDA_LEARN_REGISTER("cuda_learn.sgemv_async", sgemv_async_ffi);
CUDA_LEARN_REGISTER("cuda_learn.sgemv", sgemv_register_reuse_ffi);
#endif
