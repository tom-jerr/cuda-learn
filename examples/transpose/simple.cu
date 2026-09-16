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

// Matrix Transpose: output[j, i] = input[i, j]
//
// input: MxN，按行连续存储
// output: NxM，按行连续存储，使用独立于 input 的缓冲区
//
// block(32, 32)：1024 个线程，本 kernel 按此配置启动
// grid(ceil(N / 32), ceil(M / 32))
//
// 每个 block：
//   负责 input 中一个 16x16 的 tile；
//   将其转置后写入 output 中对应的 16x16 区域。
//   输入 tile 的位置为 (blockIdx.y, blockIdx.x)，
//   输出 tile 的位置为 (blockIdx.x, blockIdx.y)。
//
// 每个线程：
//   读取一个 input 元素，存入 shared memory；
//   同步后，取出另一个线程加载的元素，写入 output。
//   边界 tile 中的线程通过条件判断跳过越界访问。
__global__ void matrix_transpose_tile(const float *input, float *output, int M,
                                      int N) {
  __shared__ float tile[32][32 + 1]; // +1 to avoid bank conflicts
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

  dim3 block(32, 32);
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