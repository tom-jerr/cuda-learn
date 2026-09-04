// Standalone mapping example for one 64x64 FlashAttention tile on Ampere/Ada.
//
// Build:
//   make -C examples flash_attn/ldmatrix_x4_mapping
// Run:
//   ./examples/flash_attn/ldmatrix_x4_mapping
//
// Fixed configuration:
//   D=64, BR=64, BC=64, THREADS=128 (4 warps)
//
// This file makes the two implicit CuTe mappings explicit:
//
//   1. Copy mapping: (thread, vector-repeat, vector-value) -> tensor (row,col)
//   2. MMA mapping:  (warp, lane, register-value, repeat) -> tensor (m,n,k)
//
// Global -> shared copy
// ---------------------
// Each cp.async moves 8 half values (16 bytes).  For rho=0..3,
//
//   linear = tid * 8 + rho * THREADS * 8
//          = tid * 8 + rho * 1024
//   row    = linear / D = tid / 8 + 16 * rho
//   col    = linear % D = 8 * (tid % 8)
//
// and vector element v=0..7 maps to (row,col+v).  Therefore
//
//   128 threads * 4 vectors/thread * 8 half/vector = 64 * 64 half.
//
// Shared memory uses physical offset row * (D+PAD) + col = row * 72 + col.
// Since 72 half = 144 bytes and col is a multiple of 8 half, every vector and
// every ldmatrix row address below remains 16-byte aligned.
//
// Conceptual TiledMMA
// -------------------
// The atom is m16n8k16 and all 4 warps are placed along M:
//
//   warp layout = (W_M,W_N,W_K) = (4,1,1)
//   QK repeats   = (R_M,R_N,R_K) = (1,8,4)
//   PV repeats   = (R_M,R_N,R_K) = (1,8,4)
//
// Let lane=4*g+t, where g=lane/4 and t=lane%4.  Warp w owns rows
// 16*w..16*w+15.  The exact per-lane coordinates are documented beside each
// ldmatrix/MMA loop below.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int D = 64;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int THREADS = 128;
constexpr int PAD = 8;
constexpr int STRIDE = D + PAD;
constexpr int TILE_ELEMS = BR * D;
constexpr int SMEM_ELEMS = BR * STRIDE;

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    const cudaError_t status = (expr);                                         \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__device__ __forceinline__ uint32_t shared_u32(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ void cp_async_16(uint32_t dst_smem,
                                             const void *src_gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(dst_smem), "l"(src_gmem));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&dst)[4],
                                             uint32_t src_smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
               "{%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
               : "r"(src_smem));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&dst)[4],
                                                   uint32_t src_smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
               "{%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
               : "r"(src_smem));
}

// A has 8 half values/lane in four packed registers.  A single B fragment has
// 4 half values/lane, so pass the selected pair from an x4 load directly.
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                              const uint32_t (&a)[4],
                                              uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0),
                 "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float x, float y) {
  union Pack {
    half2 h2;
    uint32_t u32;
  } result;
  result.h2 = __floats2half2_rn(x, y);
  return result.u32;
}

__device__ __forceinline__ float subgroup4_max(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1, 4));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2, 4));
  return x;
}

__device__ __forceinline__ float subgroup4_sum(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1, 4);
  x += __shfl_xor_sync(0xffffffffu, x, 2, 4);
  return x;
}

__global__ __launch_bounds__(THREADS)
void flash_attn_64_x4_kernel(const half *__restrict__ q,
                             const half *__restrict__ k,
                             const half *__restrict__ v,
                             half *__restrict__ out) {
  __shared__ __align__(16) half q_smem[SMEM_ELEMS];
  __shared__ __align__(16) half k_smem[SMEM_ELEMS];
  __shared__ __align__(16) half v_smem[SMEM_ELEMS];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int row0 = warp * 16 + g;
  const int row1 = row0 + 8;

  // Explicit G2S copy mapping:
  //   rho = 0..3
  //   row = tid/8 + 16*rho
  //   col = 8*(tid%8)
  // Q, K and V have the same logical row-major mapping in this example.
#pragma unroll
  for (int linear = tid * 8; linear < TILE_ELEMS;
       linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    cp_async_16(shared_u32(&q_smem[row * STRIDE + col]), &q[linear]);
    cp_async_16(shared_u32(&k_smem[row * STRIDE + col]), &k[linear]);
    cp_async_16(shared_u32(&v_smem[row * STRIDE + col]), &v[linear]);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  // -----------------------------------------------------------------------
  // Q @ K^T
  // -----------------------------------------------------------------------
  // One warp owns 16x64 scores.  C mapping after all kd iterations:
  //
  //   s_frag[nj][0] = S[row0, 8*nj + 2*t]
  //   s_frag[nj][1] = S[row0, 8*nj + 2*t + 1]
  //   s_frag[nj][2] = S[row1, 8*nj + 2*t]
  //   s_frag[nj][3] = S[row1, 8*nj + 2*t + 1]
  float s_frag[8][4] = {};

#pragma unroll
  for (int kd = 0; kd < D; kd += 16) {
    // Q A fragment.  The four x4 source matrices are:
    //
    //   source lanes  0.. 7: rows warp*16+0..7,  cols kd+0..7
    //   source lanes  8..15: rows warp*16+8..15, cols kd+0..7
    //   source lanes 16..23: rows warp*16+0..7,  cols kd+8..15
    //   source lanes 24..31: rows warp*16+8..15, cols kd+8..15
    //
    // After ldmatrix distribution, this lane receives:
    //   q_frag[0] = Q[row0, kd+2*t : kd+2*t+2]
    //   q_frag[1] = Q[row1, kd+2*t : kd+2*t+2]
    //   q_frag[2] = Q[row0, kd+8+2*t : kd+8+2*t+2]
    //   q_frag[3] = Q[row1, kd+8+2*t : kd+8+2*t+2]
    uint32_t q_frag[4];
    const int q_addr_row = warp * 16 + (lane & 15);
    const int q_addr_col = kd + (lane >> 4) * 8;
    ldmatrix_x4(q_frag,
                shared_u32(&q_smem[q_addr_row * STRIDE + q_addr_col]));

    // Group adjacent K N-fragments.  For pair=0, x4 loads K rows 0..15;
    // for pair=1, rows 16..31; and so on.  matrix=lane/8 selects:
    //
    //   matrix 0: K[n_base+0..7,  kd+0..7]
    //   matrix 1: K[n_base+0..7,  kd+8..15]
    //   matrix 2: K[n_base+8..15, kd+0..7]
    //   matrix 3: K[n_base+8..15, kd+8..15]
    //
    // Consequently {k4[0],k4[1]} is B for score fragment 2*pair,
    // and {k4[2],k4[3]} is B for score fragment 2*pair+1.
#pragma unroll
    for (int pair = 0; pair < BC / 16; ++pair) {
      uint32_t k4[4];
      const int matrix = lane >> 3;
      const int k_addr_row = pair * 16 + (matrix >> 1) * 8 + (lane & 7);
      const int k_addr_col = kd + (matrix & 1) * 8;
      ldmatrix_x4(k4,
                  shared_u32(&k_smem[k_addr_row * STRIDE + k_addr_col]));

      mma_m16n8k16(s_frag[2 * pair], q_frag, k4[0], k4[1]);
      mma_m16n8k16(s_frag[2 * pair + 1], q_frag, k4[2], k4[3]);
    }
  }

  // One tile is the whole attention matrix here, so ordinary row softmax is
  // sufficient.  Four adjacent lanes have the same row0/row1 and collectively
  // own all 64 columns: 4 lanes * (8 fragments * 2 values) = 64.
  float local_max0 = -INFINITY;
  float local_max1 = -INFINITY;
#pragma unroll
  for (int nj = 0; nj < BC / 8; ++nj) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      s_frag[nj][item] *= 0.125f; // 1/sqrt(64)
    }
    local_max0 = fmaxf(local_max0,
                       fmaxf(s_frag[nj][0], s_frag[nj][1]));
    local_max1 = fmaxf(local_max1,
                       fmaxf(s_frag[nj][2], s_frag[nj][3]));
  }
  const float row_max0 = subgroup4_max(local_max0);
  const float row_max1 = subgroup4_max(local_max1);

  float local_sum0 = 0.0f;
  float local_sum1 = 0.0f;
#pragma unroll
  for (int nj = 0; nj < BC / 8; ++nj) {
    s_frag[nj][0] = __expf(s_frag[nj][0] - row_max0);
    s_frag[nj][1] = __expf(s_frag[nj][1] - row_max0);
    s_frag[nj][2] = __expf(s_frag[nj][2] - row_max1);
    s_frag[nj][3] = __expf(s_frag[nj][3] - row_max1);
    local_sum0 += s_frag[nj][0] + s_frag[nj][1];
    local_sum1 += s_frag[nj][2] + s_frag[nj][3];
  }
  const float row_sum0 = subgroup4_sum(local_sum0);
  const float row_sum1 = subgroup4_sum(local_sum1);

  // -----------------------------------------------------------------------
  // P @ V
  // -----------------------------------------------------------------------
  float o_acc[D / 8][4] = {};

#pragma unroll
  for (int pk = 0; pk < BC; pk += 16) {
    const int pn = pk / 8;
    // Two adjacent 16x8 C fragments become one 16x16 A fragment without any
    // cross-lane movement.  Only FP32->FP16 conversion and packing are real.
    uint32_t p_frag[4] = {
        pack_half2(s_frag[pn][0], s_frag[pn][1]),
        pack_half2(s_frag[pn][2], s_frag[pn][3]),
        pack_half2(s_frag[pn + 1][0], s_frag[pn + 1][1]),
        pack_half2(s_frag[pn + 1][2], s_frag[pn + 1][3]),
    };

    // Group adjacent output-column fragments for V.  With matrix=lane/8:
    //
    //   matrix 0: V[pk+0..7,  d_base+0..7]
    //   matrix 1: V[pk+8..15, d_base+0..7]
    //   matrix 2: V[pk+0..7,  d_base+8..15]
    //   matrix 3: V[pk+8..15, d_base+8..15]
    //
    // .trans converts row-major shared V into the column-major B fragment.
    // {v4[0],v4[1]} feeds output fragment 2*pair and {v4[2],v4[3]}
    // feeds output fragment 2*pair+1.
#pragma unroll
    for (int pair = 0; pair < D / 16; ++pair) {
      uint32_t v4[4];
      const int matrix = lane >> 3;
      const int v_addr_row = pk + (matrix & 1) * 8 + (lane & 7);
      const int v_addr_col = pair * 16 + (matrix >> 1) * 8;
      ldmatrix_x4_trans(
          v4, shared_u32(&v_smem[v_addr_row * STRIDE + v_addr_col]));

      mma_m16n8k16(o_acc[2 * pair], p_frag, v4[0], v4[1]);
      mma_m16n8k16(o_acc[2 * pair + 1], p_frag, v4[2], v4[3]);
    }
  }

  // MMA C mapping for output:
  //   o_acc[nj][0:2] -> O[row0, 8*nj+2*t : +2]
  //   o_acc[nj][2:4] -> O[row1, 8*nj+2*t : +2]
  // Scatter to the now-dead q_smem, then gather contiguous uint4 vectors.
  const float inv_sum0 = 1.0f / row_sum0;
  const float inv_sum1 = 1.0f / row_sum1;
#pragma unroll
  for (int nj = 0; nj < D / 8; ++nj) {
    const int col = nj * 8 + 2 * t;
    *reinterpret_cast<uint32_t *>(&q_smem[row0 * STRIDE + col]) =
        pack_half2(o_acc[nj][0] * inv_sum0, o_acc[nj][1] * inv_sum0);
    *reinterpret_cast<uint32_t *>(&q_smem[row1 * STRIDE + col]) =
        pack_half2(o_acc[nj][2] * inv_sum1, o_acc[nj][3] * inv_sum1);
  }
  __syncthreads();

#pragma unroll
  for (int linear = tid * 8; linear < TILE_ELEMS;
       linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    const uint4 packed =
        *reinterpret_cast<const uint4 *>(&q_smem[row * STRIDE + col]);
    *reinterpret_cast<uint4 *>(&out[linear]) = packed;
  }
}

void print_mapping_example() {
  std::printf("configuration: D=BR=BC=64, THREADS=128, PAD=8\n");
  std::printf("G2S: row=tid/8+16*rho, col=8*(tid%%8), rho=0..3\n");
  for (int tid : {0, 5, 127}) {
    std::printf("  tid=%3d:", tid);
    for (int rho = 0; rho < 4; ++rho) {
      const int row = tid / 8 + 16 * rho;
      const int col = 8 * (tid % 8);
      std::printf(" rho%d->(%d,%d:%d)", rho, row, col, col + 7);
    }
    std::printf("\n");
  }

  constexpr int warp = 2;
  constexpr int lane = 5;
  constexpr int g = lane / 4;
  constexpr int t = lane % 4;
  constexpr int kd = 16;
  constexpr int n_pair = 1;
  constexpr int pk = 32;
  constexpr int d_pair = 2;
  const int row0 = 16 * warp + g;
  const int row1 = row0 + 8;

  std::printf("\nworked lane: warp=%d lane=%d -> g=%d t=%d rows=(%d,%d)\n",
              warp, lane, g, t, row0, row1);
  std::printf("  Q kd=%d: r0=(%d,%d:%d), r1=(%d,%d:%d), "
              "r2=(%d,%d:%d), r3=(%d,%d:%d)\n",
              kd, row0, kd + 2 * t, kd + 2 * t + 1, row1,
              kd + 2 * t, kd + 2 * t + 1, row0, kd + 8 + 2 * t,
              kd + 8 + 2 * t + 1, row1, kd + 8 + 2 * t,
              kd + 8 + 2 * t + 1);
  std::printf("  K pair=%d: MMA0 key=%d, MMA1 key=%d for this lane's g\n",
              n_pair, n_pair * 16 + g, n_pair * 16 + 8 + g);
  std::printf("  V pk=%d pair=%d: B registers supply output columns %d and %d "
              "for this lane's g\n",
              pk, d_pair, d_pair * 16 + g, d_pair * 16 + 8 + g);
}

std::vector<half> reference_attention(const std::vector<half> &q,
                                      const std::vector<half> &k,
                                      const std::vector<half> &v) {
  std::vector<float> p(BR * BC);
  std::vector<half> p_half(BR * BC);
  std::vector<half> out(BR * D);

  for (int m = 0; m < BR; ++m) {
    float row_max = -INFINITY;
    for (int n = 0; n < BC; ++n) {
      float score = 0.0f;
      for (int d = 0; d < D; ++d) {
        score += __half2float(q[m * D + d]) *
                 __half2float(k[n * D + d]);
      }
      p[m * BC + n] = score * 0.125f;
      row_max = std::max(row_max, p[m * BC + n]);
    }
    float row_sum = 0.0f;
    for (int n = 0; n < BC; ++n) {
      p[m * BC + n] = std::exp(p[m * BC + n] - row_max);
      row_sum += p[m * BC + n];
    }
    for (int n = 0; n < BC; ++n) {
      // The kernel converts unnormalized P to FP16 before PV, then divides O by
      // the FP32 row sum.  Mirror that behavior in the reference.
      p_half[m * BC + n] = __float2half_rn(p[m * BC + n]);
    }
    for (int d = 0; d < D; ++d) {
      float value = 0.0f;
      for (int n = 0; n < BC; ++n) {
        value += __half2float(p_half[m * BC + n]) *
                 __half2float(v[n * D + d]);
      }
      out[m * D + d] = __float2half_rn(value / row_sum);
    }
  }
  return out;
}

} // namespace

int main() {
  print_mapping_example();

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 8) {
    std::fprintf(stderr, "This example requires compute capability 8.0+.\n");
    return EXIT_FAILURE;
  }

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<half> hq(TILE_ELEMS), hk(TILE_ELEMS), hv(TILE_ELEMS);
  for (int i = 0; i < TILE_ELEMS; ++i) {
    hq[i] = __float2half_rn(dist(rng));
    hk[i] = __float2half_rn(dist(rng));
    hv[i] = __float2half_rn(dist(rng));
  }
  const std::vector<half> reference = reference_attention(hq, hk, hv);

  half *dq = nullptr;
  half *dk = nullptr;
  half *dv = nullptr;
  half *dout = nullptr;
  const size_t bytes = TILE_ELEMS * sizeof(half);
  CUDA_CHECK(cudaMalloc(&dq, bytes));
  CUDA_CHECK(cudaMalloc(&dk, bytes));
  CUDA_CHECK(cudaMalloc(&dv, bytes));
  CUDA_CHECK(cudaMalloc(&dout, bytes));
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), bytes, cudaMemcpyHostToDevice));

  flash_attn_64_x4_kernel<<<1, THREADS>>>(dq, dk, dv, dout);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<half> actual(TILE_ELEMS);
  CUDA_CHECK(
      cudaMemcpy(actual.data(), dout, bytes, cudaMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  int max_index = 0;
  for (int i = 0; i < TILE_ELEMS; ++i) {
    const float error =
        std::fabs(__half2float(actual[i]) - __half2float(reference[i]));
    if (error > max_abs_error) {
      max_abs_error = error;
      max_index = i;
    }
  }

  std::printf("\nvalidation: max_abs_error=%g at O[%d,%d] -> %s\n",
              max_abs_error, max_index / D, max_index % D,
              max_abs_error <= 2.0e-2f ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(dk));
  CUDA_CHECK(cudaFree(dv));
  CUDA_CHECK(cudaFree(dout));
  return max_abs_error <= 2.0e-2f ? EXIT_SUCCESS : EXIT_FAILURE;
}
