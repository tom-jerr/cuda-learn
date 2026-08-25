#include "ampere_primitives.cuh"
#include "ffi_common.h"

#include <cuda_bf16.h>

using tvm::ffi::TensorView;

// HBM --cp.async--> shared memory --ldmatrix--> registers --mma.sync--> regs

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int PAD = 8; // 8 half = 16 B; changes the bank mapping of each row.
constexpr int THREADS = 128; // four warps, arranged as a 2x2 warp grid.

__global__ void gemm_mma_kernel(const __nv_bfloat16 *__restrict__ a,
                                const __nv_bfloat16 *__restrict__ b,
                                __nv_bfloat16 *__restrict__ c, int m, int n,
                                int k) {
  __shared__ __align__(16) __nv_bfloat16 a_smem[BM][BK + PAD];
  __shared__ __align__(16) __nv_bfloat16 b_smem[BK][BN + PAD];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp >> 1; // 0,0,1,1: each warp owns 32 output rows.
  const int warp_n = warp & 1;  // 0,1,0,1: each warp owns 32 output cols.
  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  /**
   * @brief
    2   : warp tile 在 M 方向有 2 个 mma tile
    4   : warp tile 在 N 方向有 4 个 mma tile
    4   : 每个 mma tile，每个 lane 有 4 个 accumulator registers
   */

  float accum[2][4][4] = {};

  for (int bk = 0; bk < k; bk += BK) {
    // HBM->SMEM: 每个 warp 负责加载 32x32 的 tile
    for (int linear = tid * 8; linear < BM * BK; linear += THREADS * 8) {
      const int row = linear / BK;
      const int col = linear % BK;
      ampere::cp_async_16(ampere::shared_u32(&a_smem[row][col]),
                          &a[(block_m + row) * k + bk + col]);
    }
    for (int linear = tid * 8; linear < BK * BN; linear += THREADS * 8) {
      const int row = linear / BN;
      const int col = linear % BN;
      ampere::cp_async_16(ampere::shared_u32(&b_smem[row][col]),
                          &b[(bk + row) * n + block_n + col]);
    }
    ampere::cp_async_commit();
    ampere::cp_async_wait_all();
    __syncthreads(); // cp.async completion is not a CTA visibility barrier.

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t a_frag[2][4];
      uint32_t b_frag[4][2];
/**
 * @brief
  lane 0–7 → M0 的行 0–7（行基址 +0，列 +0）
  lane 8–15 → M1 的行 0–7（行基址 +8，列 +0）
  lane 16–23 → M2 的行 0–7（行基址 +0，列 +8）
  lane 24–31 → M3 的行 0–7（行基址 +8，列 +8）
 */
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int row = warp_m * 32 + mi * 16 + (lane & 15);
        const int col = kk + (lane >> 4) * 8;
        ampere::ldmatrix_x4(a_frag[mi], ampere::shared_u32(&a_smem[row][col]));
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int row = kk + (lane & 15);
        const int col = warp_n * 32 + ni * 8 + (lane >> 4);
        ampere::ldmatrix_x2_trans(b_frag[ni],
                                  ampere::shared_u32(&b_smem[row][col]));
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          ampere::mma_m16n8k16_bf16_f32(accum[mi][ni], a_frag[mi],
                                         b_frag[ni]);
        }
      }
    }
    __syncthreads();
  }
// reg->hbm
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const int row0 = block_m + warp_m * 32 + mi * 16 + lane / 4;
      const int row1 = row0 + 8;
      const int col = block_n + warp_n * 32 + nj * 8 + (lane % 4) * 2;
      c[row0 * n + col] = __float2bfloat16_rn(accum[mi][nj][0]);
      c[row0 * n + col + 1] = __float2bfloat16_rn(accum[mi][nj][1]);
      c[row1 * n + col] = __float2bfloat16_rn(accum[mi][nj][2]);
      c[row1 * n + col + 1] = __float2bfloat16_rn(accum[mi][nj][3]);
    }
  }
}

void gemm_mma(TensorView a, TensorView b, TensorView c) {
  auto check_bf16 = [](const TensorView &tensor, const char *name) {
    if (tensor.ndim() != 2 || tensor.dtype().code != kDLBfloat ||
        tensor.dtype().bits != 16 || tensor.device().device_type != kDLCUDA) {
      TVM_FFI_THROW(RuntimeError)
          << name << ": expected a 2D CUDA bfloat16 tensor";
    }
  };
  check_bf16(a, "a");
  check_bf16(b, "b");
  check_bf16(c, "c");
  int M = static_cast<int>(dim(a, 0));
  int K = static_cast<int>(dim(a, 1));
  int N = static_cast<int>(dim(b, 1));

  if (dim(b, 0) != K || dim(c, 0) != M || dim(c, 1) != N ||
      M % BM != 0 || N % BN != 0 || K % BK != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "gemm_mma: shapes must be aligned a[M,K] @ b[K,N] -> c[M,N] "
           "with M/N multiples of 64 and K a multiple of 32";
  }

  dim3 block(THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  gemm_mma_kernel<<<grid, block, 0, get_stream(a)>>>(
      reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), M, N, K);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.gemm_mma", gemm_mma);
