#include "ffi_common.h"

#include <cmath>
#include <limits>

using tvm::ffi::TensorView;

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// One block handles one row. The power-of-two shared-memory reduction is
// deliberately simple: profiling on sm_89 showed that, for wide rows, replacing
// it with warp shuffles or float4/register caching made the memory-bound kernel
// slower. The optimized entry point instead avoids idle warps on narrow rows.
__global__ void rmsnorm_kernel(const float *__restrict__ input,
                               float *__restrict__ output, int hidden_size,
                               float epsilon) {
  extern __shared__ float sum[];
  const int row_offset = blockIdx.x * hidden_size;
  const int tid = threadIdx.x;
  float square_sum = 0.0f;

  for (int col = tid; col < hidden_size; col += blockDim.x) {
    const float value = input[row_offset + col];
    square_sum = fmaf(value, value, square_sum);
  }
  sum[tid] = square_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      sum[tid] += sum[tid + offset];
    }
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(sum[0] / static_cast<float>(hidden_size) + epsilon);
  for (int col = tid; col < hidden_size; col += blockDim.x) {
    output[row_offset + col] = input[row_offset + col] * inverse_rms;
  }
}

// Four independent rows per block for D <= 128. Each row fits in one warp and
// each lane retains at most one float4, so normalization needs only one input
// read and no block-wide barrier.
__global__ void rmsnorm_small_kernel(const float *__restrict__ input,
                                     float *__restrict__ output, int rows,
                                     int hidden_size, float epsilon) {
  constexpr int kWarpsPerBlock = 4;
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x / 32;
  const int row = blockIdx.x * kWarpsPerBlock + warp_id;
  if (row >= rows)
    return;

  // per warp per row, 4(float4) * 32(warp_size)
  const int vector_size = hidden_size / 4;
  const int vector_index = row * vector_size + lane;
  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  float4 *output4 = reinterpret_cast<float4 *>(output);
  float4 value = {0.0f, 0.0f, 0.0f, 0.0f};
  if (lane < vector_size) {
    value = input4[vector_index];
  }
  float square_sum = value.x * value.x + value.y * value.y + value.z * value.z +
                     value.w * value.w;
  square_sum = warp_reduce_sum(square_sum);
  float inverse_rms = 0.0f;
  if (lane == 0) {
    inverse_rms =
        rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
  }
  inverse_rms = __shfl_sync(0xffffffff, inverse_rms, 0);
  if (lane < vector_size) {
    output4[vector_index] = {value.x * inverse_rms, value.y * inverse_rms,
                             value.z * inverse_rms, value.w * inverse_rms};
  }
}

int launch_threads(int hidden_size) {
  if (hidden_size <= 128)
    return 32;
  if (hidden_size <= 256)
    return 64;
  if (hidden_size <= 512)
    return 128;
  return 256;
}

void check_rmsnorm_args(const TensorView &input, const TensorView &output,
                        double epsilon, const char *op_name) {
  check_tensor(input, "input", 2);
  check_tensor(output, "output", 2);
  if (dim(input, 0) != dim(output, 0) || dim(input, 1) != dim(output, 1)) {
    TVM_FFI_THROW(RuntimeError) << op_name << ": shape mismatch";
  }
  if (input.device().device_id != output.device().device_id) {
    TVM_FFI_THROW(RuntimeError) << op_name << ": device mismatch";
  }
  if (dim(input, 0) <= 0 || dim(input, 1) <= 0) {
    TVM_FFI_THROW(RuntimeError) << op_name << ": dimensions must be positive";
  }
  if (dim(input, 0) > std::numeric_limits<int>::max() ||
      dim(input, 1) > std::numeric_limits<int>::max()) {
    TVM_FFI_THROW(RuntimeError) << op_name << ": dimensions exceed int32 range";
  }
  if (dim(input, 0) > std::numeric_limits<int>::max() / dim(input, 1)) {
    TVM_FFI_THROW(RuntimeError)
        << op_name << ": tensor indexing exceeds int32 range";
  }
  if (!std::isfinite(epsilon) || epsilon < 0.0 ||
      epsilon > std::numeric_limits<float>::max()) {
    TVM_FFI_THROW(RuntimeError)
        << op_name << ": epsilon must be finite, non-negative, and fit float32";
  }
}

void launch_rmsnorm(const TensorView &input, const TensorView &output,
                    double epsilon, int threads) {
  const int rows = static_cast<int>(dim(input, 0));
  const int hidden_size = static_cast<int>(dim(input, 1));
  rmsnorm_kernel<<<rows, threads, threads * sizeof(float), get_stream(input)>>>(
      static_cast<const float *>(input.data_ptr()),
      static_cast<float *>(output.data_ptr()), hidden_size,
      static_cast<float>(epsilon));
  CUDA_LEARN_CHECK(cudaGetLastError());
}

} // namespace

void rmsnorm(TensorView input, TensorView output, double epsilon) {
  check_rmsnorm_args(input, output, epsilon, "rmsnorm");
  const int rows = static_cast<int>(dim(input, 0));
  const int hidden_size = static_cast<int>(dim(input, 1));
  if (hidden_size <= 128 && hidden_size % 4 == 0) {
    constexpr int threads = 128;
    constexpr int warps_per_block = threads / 32;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;
    rmsnorm_small_kernel<<<blocks, threads, 0, get_stream(input)>>>(
        static_cast<const float *>(input.data_ptr()),
        static_cast<float *>(output.data_ptr()), rows, hidden_size,
        static_cast<float>(epsilon));
    CUDA_LEARN_CHECK(cudaGetLastError());
  } else {
    launch_rmsnorm(input, output, epsilon, launch_threads(hidden_size));
  }
}

// Fixed 256-thread version matching the original mapping, retained for A/B
// benchmark comparisons.
void rmsnorm_baseline(TensorView input, TensorView output, double epsilon) {
  check_rmsnorm_args(input, output, epsilon, "rmsnorm_baseline");
  launch_rmsnorm(input, output, epsilon, 256);
}

CUDA_LEARN_REGISTER("cuda_learn.rmsnorm", rmsnorm);
CUDA_LEARN_REGISTER("cuda_learn.rmsnorm_baseline", rmsnorm_baseline);
