#include <cstdio>
#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void dot_product(const float *a, const float *b, float *result,
                            int n) {
  __shared__ float shared[32]; // assuming blockDim.x <= 1024
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int lane = tid % 32;
  int warp_id = tid / 32;

  float sum = 0.0f;

  // grid-stride loop to handle large arrays
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    sum += a[i] * b[i];
  }

  sum = warp_reduce_sum(sum);

  if (lane == 0) {
    shared[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    sum = shared[lane];
    sum = warp_reduce_sum(sum);
    if (tid == 0) {
      atomicAdd(result, sum);
    }
  }
}

int main() {
  constexpr int n = 1024;
  constexpr int bytes = n * sizeof(float);

  float *a, *b, *result;
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&result, sizeof(float));

  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
  }
  *result = 0.0f;

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  dot_product<<<grid_size, block_size>>>(a, b, result, n);
  cudaDeviceSynchronize();

  float expected_result =
      (n - 1) * n * (2 * n - 1) / 6.0f; // sum of squares formula
  if (*result != expected_result) {
    printf("Mismatch: expected %f, got %f\n", expected_result, *result);
    return 1;
  }

  printf("Dot product: %f\n", *result);

  cudaFree(a);
  cudaFree(b);
  cudaFree(result);

  return 0;
}