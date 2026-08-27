#include <cuda_runtime.h>
#include <iostream>

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
  static __shared__ float shared[32]; // 32 warps per block
  const int lane = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;

  value = warp_reduce_sum(value);

  if (lane == 0)
    shared[warp_id] = value;
  __syncthreads();

  value = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
  if (warp_id == 0) {
    shared[0] = warp_reduce_sum(value);
  }
  __syncthreads();
  float res = shared[0];
  // __syncthreads();
  return res;
}

__global__ void reduce_2dim_kernel(const float *input, float *output, int rows,
                                   int cols) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    sum += input[row * cols + col];
  }
  sum = block_reduce_sum(sum);
  if (tid == 0) {
    output[row] = sum;
  }
}

int main() {
  const int rows = 4;
  const int cols = 8;
  float h_input[rows * cols] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  float h_output[rows];

  float *d_input, *d_output;
  cudaMalloc(&d_input, rows * cols * sizeof(float));
  cudaMalloc(&d_output, rows * sizeof(float));
  cudaMemcpy(d_input, h_input, rows * cols * sizeof(float),
             cudaMemcpyHostToDevice);
  reduce_2dim_kernel<<<rows, 256>>>(d_input, d_output, rows, cols);
  cudaMemcpy(h_output, d_output, rows * sizeof(float), cudaMemcpyDeviceToHost);

  float *reference_output = new float[rows];

  for (int i = 0; i < rows; ++i) {
    reference_output[i] = 0;
    for (int j = 0; j < cols; ++j) {
      reference_output[i] += h_input[i * cols + j];
    }
  }

  for (int i = 0; i < rows; ++i) {
    std::cout << "Row " << i << " sum: " << h_output[i] << std::endl;
    if (h_output[i] != reference_output[i]) {
      std::cerr << "Error: expected " << reference_output[i] << ", got "
                << h_output[i] << std::endl;
    }
  }
}