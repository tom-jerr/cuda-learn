#include <cuda_runtime.h>
#include <iostream>

// Histogram
// grid(N/128), block(128)
// a: Nx1, y: count histogram
__global__ void histogram_native_kernel(int *a, int *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    atomicAdd(&(y[a[idx]]), 1);
}

void histogram_cpu(int *a, int *y, int N) {
  for (int i = 0; i < N; i++) {
    y[a[i]]++;
  }
}

int main() {
  const int N = 1024;
  int h_a[N];
  for (int i = 0; i < N; i++) {
    h_a[i] = i % 256;
  }
  int h_y[256] = {0};

  int *d_a, *d_y;
  cudaMalloc(&d_a, N * sizeof(int));
  cudaMalloc(&d_y, 256 * sizeof(int));
  cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, 256 * sizeof(int), cudaMemcpyHostToDevice);

  histogram_native_kernel<<<(N + 127) / 128, 128>>>(d_a, d_y, N);
  cudaMemcpy(h_y, d_y, 256 * sizeof(int), cudaMemcpyDeviceToHost);

  int *reference_output = new int[256];
  for (int i = 0; i < 256; i++) {
    reference_output[i] = 0;
  }
  histogram_cpu(h_a, reference_output, N);
  for (int i = 0; i < 256; i++) {
    if (h_y[i] != reference_output[i]) {
      std::cout << "Mismatch at index " << i << ": GPU = " << h_y[i]
                << ", CPU = " << reference_output[i] << std::endl;
      return -1;
    }
  }
  std::cout << "Histogram computation is correct!" << std::endl;
}