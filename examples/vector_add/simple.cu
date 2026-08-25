#include <cstdio>
#include <cuda_runtime.h>
// per thread calc one element
__global__ void vector_add(const float *a, const float *b, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

// grid-loop
__global__ void vector_add_grid_loop(const float *a, const float *b, float *out,
                                     int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    out[idx] = a[idx] + b[idx];
  }
}

// vectorized version: each thread calc 4 elements
__global__ void vector_add_vectorized(const float *a, const float *b,
                                      float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vector_size = 4;
  int start_idx = idx * vector_size;
  if (start_idx + 3 < n) {
    float4 a_vec = reinterpret_cast<const float4 *>(a)[idx];
    float4 b_vec = reinterpret_cast<const float4 *>(b)[idx];
    float4 out_vec;
    out_vec.x = a_vec.x + b_vec.x;
    out_vec.y = a_vec.y + b_vec.y;
    out_vec.z = a_vec.z + b_vec.z;
    out_vec.w = a_vec.w + b_vec.w;
    reinterpret_cast<float4 *>(out)[idx] = out_vec;
  } else {
    for (int i = start_idx; i < n; ++i) {
      out[i] = a[i] + b[i];
    }
  }
}

int main() {
  constexpr int n = 1024;
  constexpr int bytes = n * sizeof(float);

  float *a, *b, *out;
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&out, bytes);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
  }

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  vector_add_grid_loop<<<grid_size, block_size>>>(a, b, out, n);
  cudaDeviceSynchronize();
  for (int i = 0; i < n; ++i) {
    if (out[i] != a[i] + b[i]) {
      printf("Error at index %d: %f + %f != %f\n", i, a[i], b[i], out[i]);
      return 1;
    }
  }
  vector_add_vectorized<<<grid_size, block_size>>>(a, b, out, n);
  cudaDeviceSynchronize();
  for (int i = 0; i < n; ++i) {
    if (out[i] != a[i] + b[i]) {
      printf("Error at index %d: %f + %f != %f\n", i, a[i], b[i], out[i]);
      return 1;
    }
  }
  return 0;
}