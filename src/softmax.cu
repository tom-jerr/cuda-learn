#include "include/ffi_common.h"

using tvm::ffi::TensorView;

#include <cfloat>

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ float block_reduce_sum(float *shared, float val) {
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warp_reduce_sum(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0)
      shared[0] = val; // 第二阶段结果写回 shared[0]
  }
  __syncthreads();
  float result = shared[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float block_reduce_max(float *shared, float val) {
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warp_reduce_max(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    val = warp_reduce_max(val);
    if (threadIdx.x == 0)
      shared[0] = val; // 第二阶段结果写回 shared[0]
  }
  __syncthreads();
  float result = shared[0];
  __syncthreads();
  return result;
}

__global__ void softmax_kernel(const float *input, float *output, int N) {
  extern __shared__ float shared[];
  float max_val = -FLT_MAX;
  float sum_val = 0.0f;

  // Step 1: Compute the maximum value in the input array
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    max_val = fmaxf(max_val, input[i]);
  }
  max_val = block_reduce_max(shared, max_val);

  // Step 2: Compute the sum of exponentials
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    sum_val += expf(input[i] - max_val);
  }
  sum_val = block_reduce_sum(shared, sum_val);

  // Step 3: Write the softmax output
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    output[i] = expf(input[i] - max_val) / sum_val;
  }
}

void softmax(TensorView input, TensorView output) {
  check_tensor(input, "input", 1);
  check_tensor(output, "output", 1);
  int N = static_cast<int>(dim(input, 0));
  if (dim(output, 0) != N) {
    TVM_FFI_THROW(RuntimeError) << "softmax: length mismatch";
  }
  // 单 block 版本：blockDim.x = 256，动态 smem 作 block reduce 暂存。
  int threads = 256;
  softmax_kernel<<<1, threads, threads * sizeof(float), get_stream(input)>>>(
      static_cast<const float *>(input.data_ptr()),
      static_cast<float *>(output.data_ptr()), N);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.softmax", softmax);
