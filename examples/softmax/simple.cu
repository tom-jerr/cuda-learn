#include <cfloat>
#include <cuda_runtime.h>
#include <iostream>

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

__global__ void softmax_2d_kernel(const float *input, float *output, int rows,
                                  int cols) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  float max_val = -FLT_MAX;
  float sum_val = 0.0f;

  // Step 1: Compute the maximum value in the row
  for (int col = tid; col < cols; col += blockDim.x) {
    max_val = fmaxf(max_val, input[row * cols + col]);
  }
  max_val = block_reduce_max(shared, max_val);

  // Step 2: Compute the sum of exponentials
  for (int col = tid; col < cols; col += blockDim.x) {
    sum_val += expf(input[row * cols + col] - max_val);
  }
  sum_val = block_reduce_sum(shared, sum_val);

  // Step 3: Write the softmax output
  for (int col = tid; col < cols; col += blockDim.x) {
    output[row * cols + col] =
        expf(input[row * cols + col] - max_val) / sum_val;
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

  const int rows = 4;
  const int cols = 8;
  float h_input[rows * cols] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  float h_output[rows * cols];

  float *d2_input, *d2_output;
  cudaMalloc(&d2_input, rows * cols * sizeof(float));
  cudaMalloc(&d2_output, rows * cols * sizeof(float));
  cudaMemcpy(d2_input, h_input, rows * cols * sizeof(float),
             cudaMemcpyHostToDevice);
  softmax_2d_kernel<<<rows, 256, 256 * sizeof(float)>>>(d2_input, d2_output,
                                                        rows, cols);
  cudaMemcpy(h_output, d2_output, rows * cols * sizeof(float),
             cudaMemcpyDeviceToHost);

  float *reference_output = new float[rows * cols];
  // Compute reference output (omitted for brevity)
  for (int i = 0; i < rows; ++i) {
    float max_val = -FLT_MAX;
    for (int j = 0; j < cols; ++j) {
      max_val = fmaxf(max_val, h_input[i * cols + j]);
    }
    float sum_val = 0.0f;
    for (int j = 0; j < cols; ++j) {
      sum_val += expf(h_input[i * cols + j] - max_val);
    }
    for (int j = 0; j < cols; ++j) {
      reference_output[i * cols + j] =
          expf(h_input[i * cols + j] - max_val) / sum_val;
    }
  }

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (fabs(h_output[i * cols + j] - reference_output[i * cols + j]) >
          1e-5) {
        std::cout << "Mismatch at row " << i << ", col " << j << ": GPU "
                  << h_output[i * cols + j] << ", CPU "
                  << reference_output[i * cols + j] << std::endl;
      }
    }
  }

  cudaFree(d2_input);
  cudaFree(d2_output);
  return 0;
}