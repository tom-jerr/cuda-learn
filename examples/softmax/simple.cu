#include <cfloat>
#include <cuda_runtime.h>

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
    if (threadIdx.x == 0) shared[0] = val;  // 第二阶段结果写回 shared[0]
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
    if (threadIdx.x == 0) shared[0] = val;  // 第二阶段结果写回 shared[0]
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

int main() {
  const int N = 1024;
  float *d_input, *d_output;
  cudaMalloc(&d_input, N * sizeof(float));
  cudaMalloc(&d_output, N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    float val = static_cast<float>(rand()) / RAND_MAX;
    cudaMemcpy(d_input + i, &val, sizeof(float), cudaMemcpyHostToDevice);
  }

  // Initialize input data (omitted for brevity)

  softmax_kernel<<<1, 256, 256 * sizeof(float)>>>(d_input, d_output, N);

  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}