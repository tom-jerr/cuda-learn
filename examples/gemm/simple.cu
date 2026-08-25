#include <cstdio>
#include <cuda_runtime.h>

// b_m, b_n, b_k, t_m, t_n tiled gemm
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;

constexpr int TM = 4;
constexpr int TN = 4;

constexpr int BLOCK_X = BN / TN;
constexpr int BLOCK_Y = BM / TM;
constexpr int THREADS_PER_BLOCK = BLOCK_X * BLOCK_Y;

__global__ void tiled_gemm(const float *A, const float *B, float *C, int M,
                           int N, int K) {
  // C[M,N] = A[M,K] * B[K,N]，矩阵均为 row-major。
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  // current block [block_row, block_col] 对应 C 的左上角。
  int block_row = blockIdx.y * BM;
  int block_col = blockIdx.x * BN;
  // current tile 内线程的二维索引。
  int thread_row = threadIdx.y * TM;
  int thread_col = threadIdx.x * TN;
  // thread id
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  float acc[TM][TN] = {};

  // K tile
  for (int k = 0; k < K; k += BK) {
    // g2s load A tile
    for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
      int a_shrad_row = idx / BK;
      int a_shrad_col = idx % BK;
      int a_global_row = block_row + a_shrad_row;
      int a_global_col = k + a_shrad_col;
      if (a_global_row < M && a_global_col < K) {
        As[a_shrad_row][a_shrad_col] = A[a_global_row * K + a_global_col];
      } else {
        As[a_shrad_row][a_shrad_col] = 0.0f;
      }
    }
    // g2s load B tile
    for (int idx = tid; idx < BK * BN; idx += THREADS_PER_BLOCK) {
      int b_shrad_row = idx / BN;
      int b_shrad_col = idx % BN;
      int b_global_row = k + b_shrad_row;
      int b_global_col = block_col + b_shrad_col;
      if (b_global_row < K && b_global_col < N) {
        Bs[b_shrad_row][b_shrad_col] = B[b_global_row * N + b_global_col];
      } else {
        Bs[b_shrad_row][b_shrad_col] = 0.0f;
      }
    }
    __syncthreads();

    // s2r + compute
    for (int bk = 0; bk < BK; ++bk) {
      float a_reg[TM];
      float b_reg[TN];
      for (int tm = 0; tm < TM; ++tm) {
        a_reg[tm] = As[thread_row + tm][bk];
      }
      for (int tn = 0; tn < TN; ++tn) {
        b_reg[tn] = Bs[bk][thread_col + tn];
      }
      for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
          acc[tm][tn] += a_reg[tm] * b_reg[tn];
        }
      }
    }
    __syncthreads();
  }
  // write back to global memory
  for (int tm = 0; tm < TM; ++tm) {
    for (int tn = 0; tn < TN; ++tn) {
      int c_global_row = block_row + thread_row + tm;
      int c_global_col = block_col + thread_col + tn;
      if (c_global_row < M && c_global_col < N) {
        C[c_global_row * N + c_global_col] = acc[tm][tn];
      }
    }
  }
}

int main() {
  float *A, *B, *C;
  int M = 128, N = 128, K = 128;
  cudaMallocManaged(&A, M * K * sizeof(float));
  cudaMallocManaged(&B, K * N * sizeof(float));
  cudaMallocManaged(&C, M * N * sizeof(float));
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      A[i * K + j] = static_cast<float>(i * K + j);
      B[j * N + i] = static_cast<float>(j * N + i);
    }
  }
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  tiled_gemm<<<grid, block>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float expected = 0.0f;
      for (int k = 0; k < K; ++k) {
        expected += A[i * K + k] * B[k * N + j];
      }
      if (abs(expected - C[i * N + j]) > 1e-5) {
        printf("Error at (%d, %d): expected %f, got %f\n", i, j, expected,
               C[i * N + j]);
      }
    }
  }
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return 0;
}