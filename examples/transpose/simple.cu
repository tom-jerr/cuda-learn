#include <cstdio>
#include <cuda_runtime.h>

// per thread transpose one element in matrix
__global__ void matrix_transpose(const float *input, float *output, int M,
                                 int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < N && row < M) {
    output[col * M + row] = input[row * N + col];
  }
}

// tile transpose: each thread transpose one tile and use smem to reduce global
// memory accesses
__global__ void matrix_transpose_tile(const float *input, float *output, int M,
                                      int N) {
  __shared__ float tile[16][16 + 1]; // +1 to avoid bank conflicts
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < N && row < M) {
    tile[threadIdx.y][threadIdx.x] = input[row * N + col];
  }
  __syncthreads();
  int new_col = blockIdx.y * blockDim.y + threadIdx.x;
  int new_row = blockIdx.x * blockDim.x + threadIdx.y;
  if (new_col < M && new_row < N) {
    output[new_row * M + new_col] = tile[threadIdx.x][threadIdx.y];
  }
}

int main() {
  constexpr int M = 128;
  constexpr int N = 128;

  float *input, *output;
  cudaMallocManaged(&input, M * N * sizeof(float));
  cudaMallocManaged(&output, M * N * sizeof(float));

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      input[i * N + j] = static_cast<float>(i * N + j);
    }
  }

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  matrix_transpose_tile<<<grid, block>>>(input, output, M, N);
  cudaDeviceSynchronize();

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (output[j * M + i] != input[i * N + j]) {
        printf("Mismatch at (%d, %d): expected %f, got %f\n", i, j,
               input[i * N + j], output[j * M + i]);
        return 1;
      }
    }
  }
}