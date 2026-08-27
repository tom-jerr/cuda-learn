#include "ampere_primitives.cuh"
#include "ffi_common.h"

#include <cuda_bf16.h>

using tvm::ffi::TensorView;

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int PAD = 8;
constexpr int THREADS = 128;

// Pack consecutive CTAs into SWIZZLE_N-wide strips of output tiles.  Compared
// with the ordinary grid (which visits every N tile for one M tile before
// moving to the next M tile), this bounds both reuse distances:
//
//   ordinary: (m0,n0), ... (m0,n_last), (m1,n0), ...
//   swizzled: (m0,n0), ... (m0,n7),    (m1,n0), ...
//
// A[m,:] is reused by the CTAs in one strip, while B[:,n0:n7] is revisited
// after only SWIZZLE_N CTAs.  The CUDA scheduler is free to run CTAs in a
// different order, so this improves the probability of L2 reuse rather than
// imposing an SM scheduling order.
template <int SWIZZLE_N>
__global__ void gemm_mma_l2_kernel(const __nv_bfloat16 *__restrict__ a,
                                   const __nv_bfloat16 *__restrict__ b,
                                   __nv_bfloat16 *__restrict__ c, int m, int n,
                                   int k) {
  __shared__ __align__(16) __nv_bfloat16 a_smem[BM][BK + PAD];
  __shared__ __align__(16) __nv_bfloat16 b_smem[BK][BN + PAD];

  // grid.x enumerates M tiles and the N position inside a cohort; grid.y
  // enumerates cohorts.  CUDA linearizes x before y.
  const int tile_m = blockIdx.x / SWIZZLE_N;
  const int tile_n = blockIdx.y * SWIZZLE_N + blockIdx.x % SWIZZLE_N;
  if (tile_n * BN >= n) {
    return; // Padding CTA in the final, incomplete N cohort.
  }

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;
  const int block_m = tile_m * BM;
  const int block_n = tile_n * BN;

  float accum[2][4][4] = {};

  for (int bk = 0; bk < k; bk += BK) {
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
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t a_frag[2][4];
      uint32_t b_frag[4][2];

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
          ampere::mma_m16n8k16_bf16_f32(accum[mi][ni], a_frag[mi], b_frag[ni]);
        }
      }
    }
    __syncthreads();
  }

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

void check_gemm_mma_l2_tensor(const TensorView &tensor, const char *name) {
  if (tensor.ndim() != 2 || tensor.dtype().code != kDLBfloat ||
      tensor.dtype().bits != 16 || tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected a 2D CUDA bfloat16 tensor";
  }
}

template <int SWIZZLE_N>
void launch_gemm_mma_l2(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                        __nv_bfloat16 *c, int m, int n, int k,
                        cudaStream_t stream) {
  const int tiles_m = m / BM;
  const int tiles_n = n / BN;
  const dim3 block(THREADS);
  const dim3 grid(tiles_m * SWIZZLE_N, (tiles_n + SWIZZLE_N - 1) / SWIZZLE_N);
  gemm_mma_l2_kernel<SWIZZLE_N><<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

} // namespace

void gemm_mma_l2(TensorView a, TensorView b, TensorView c, int swizzle_n) {
  check_gemm_mma_l2_tensor(a, "a");
  check_gemm_mma_l2_tensor(b, "b");
  check_gemm_mma_l2_tensor(c, "c");

  const int M = static_cast<int>(dim(a, 0));
  const int K = static_cast<int>(dim(a, 1));
  const int N = static_cast<int>(dim(b, 1));
  if (dim(b, 0) != K || dim(c, 0) != M || dim(c, 1) != N || M % BM != 0 ||
      N % BN != 0 || K % BK != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "gemm_mma_l2: shapes must be aligned a[M,K] @ b[K,N] -> c[M,N] "
           "with M/N multiples of 64 and K a multiple of 32";
  }

  const auto *a_ptr = reinterpret_cast<const __nv_bfloat16 *>(a.data_ptr());
  const auto *b_ptr = reinterpret_cast<const __nv_bfloat16 *>(b.data_ptr());
  auto *c_ptr = reinterpret_cast<__nv_bfloat16 *>(c.data_ptr());
  const cudaStream_t stream = get_stream(a);

  switch (swizzle_n) {
  case 2:
    launch_gemm_mma_l2<2>(a_ptr, b_ptr, c_ptr, M, N, K, stream);
    break;
  case 4:
    launch_gemm_mma_l2<4>(a_ptr, b_ptr, c_ptr, M, N, K, stream);
    break;
  case 8:
    launch_gemm_mma_l2<8>(a_ptr, b_ptr, c_ptr, M, N, K, stream);
    break;
  default:
    TVM_FFI_THROW(RuntimeError)
        << "gemm_mma_l2: swizzle_n must be one of {2, 4, 8}, got " << swizzle_n;
  }
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.gemm_mma_l2", gemm_mma_l2);
