#include "ffi_common.h"

using tvm::ffi::TensorView;

// b_m, b_n, b_k, t_m, t_n tiled gemm
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;

constexpr int TM = 4;
constexpr int TN = 4;

constexpr int BLOCK_X = BN / TN;
constexpr int BLOCK_Y = BM / TM;
constexpr int THREADS_PER_BLOCK = BLOCK_X * BLOCK_Y;

__global__ void tiled_gemm_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
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

void gemm(TensorView a, TensorView b, TensorView c) {
  check_tensor(a, "a", 2);
  check_tensor(b, "b", 2);
  check_tensor(c, "c", 2);
  int M = static_cast<int>(dim(a, 0));
  int K = static_cast<int>(dim(a, 1));
  int N = static_cast<int>(dim(b, 1));
  if (dim(b, 0) != K || dim(c, 0) != M || dim(c, 1) != N) {
    TVM_FFI_THROW(RuntimeError)
        << "gemm: shapes must be a[M,K] @ b[K,N] -> c[M,N]";
  }
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  tiled_gemm_kernel<<<grid, block, 0, get_stream(a)>>>(
      static_cast<const float *>(a.data_ptr()),
      static_cast<const float *>(b.data_ptr()),
      static_cast<float *>(c.data_ptr()), M, N, K);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.gemm", gemm);
