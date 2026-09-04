// Templated multi-stage + XOR-swizzle FlashAttention mapping example.
//
// Build and run:
//   make -C examples flash_attn/multistage_swizzle
//   ./examples/flash_attn/multistage_swizzle
//
// This advances ldmatrix_x4_mapping.cu in two independent dimensions:
//
//   template<int Stages, bool UseSwizzle>
//
//   Stages=1: K/V use one stage; wait for each tile before consuming it.
//   Stages=2: while consuming tile t, prefetch tile t+1 into the other stage.
//
//   UseSwizzle=false:
//       physical(row,col) = 64*row + col
//
//   UseSwizzle=true (equivalent to the D=64 form of CuTe Swizzle<3,3,3>):
//       physical(row,col) = 64*row + (col ^ ((row & 7) << 3))
//
// The XOR only changes column bits [5:3], i.e. the index of an 8-half / 16-byte
// segment.  Column bits [2:0] remain unchanged, so each cp.async vector remains
// contiguous and 16-byte aligned.
//
// There is no runtime "unswizzle instruction".  Producers and consumers keep
// logical (row,col) coordinates and apply the same logical->physical function
// when addressing shared memory.  At the segment level:
//
//       physical_segment = logical_segment ^ (row & 7)
//       logical_segment  = physical_segment ^ (row & 7)
//
// because XOR is self-inverse.  Each stage owns a separate 4096-half region;
// swizzling is applied only after selecting the stage base.

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
constexpr int SEQLEN = 256;
constexpr int TILE_ELEMS = BR * D;
constexpr int KV_TILES = SEQLEN / BC;

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    const cudaError_t status = (expr);                                         \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

template <bool UseSwizzle>
__host__ __device__ constexpr int tile_offset(int row, int col) {
  if constexpr (UseSwizzle) {
    return row * D + (col ^ ((row & 7) << 3));
  } else {
    return row * D + col;
  }
}

template <int Stages> struct SharedStorage {
  half q[TILE_ELEMS];
  half k[Stages][TILE_ELEMS];
  half v[Stages][TILE_ELEMS];
};

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
  asm volatile("cp.async.wait_group 0;\n" ::);
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

// The logical copy mapping is unchanged by swizzle:
//
//   row = tid/8 + 16*rho
//   col = 8*(tid%8), rho=0..3
//
// Only the shared destination address differs.  Because col is a multiple of
// 8, tile_offset always returns the beginning of a contiguous 16-byte segment.
template <bool UseSwizzle>
__device__ __forceinline__ void copy_tile_async(const half *src, half *dst) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int linear = tid * 8; linear < TILE_ELEMS;
       linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    cp_async_16(shared_u32(&dst[tile_offset<UseSwizzle>(row, col)]),
                &src[linear]);
  }
}

template <int Stages, bool UseSwizzle>
__global__ __launch_bounds__(THREADS)
void flash_attn_multistage_kernel(const half *__restrict__ q,
                                  const half *__restrict__ k,
                                  const half *__restrict__ v,
                                  half *__restrict__ out) {
  static_assert(Stages == 1 || Stages == 2,
                "This teaching pipeline implements one or two stages");
  __shared__ __align__(16) SharedStorage<Stages> storage;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int q_tile = blockIdx.x * BR;
  const int row0 = warp * 16 + g;
  const int row1 = row0 + 8;

  float o_acc[D / 8][4] = {};
  float row_m0 = -INFINITY;
  float row_m1 = -INFINITY;
  float row_l0 = 0.0f;
  float row_l1 = 0.0f;

  // Group 0 contains Q and KV tile 0.  It must be completely visible before
  // the first MMA.  With Stages=2, each later tile is issued into the other
  // stage immediately before computing the current tile, so its copy overlaps
  // the current QK/softmax/PV work.
  copy_tile_async<UseSwizzle>(&q[q_tile * D], storage.q);
  copy_tile_async<UseSwizzle>(&k[0], storage.k[0]);
  copy_tile_async<UseSwizzle>(&v[0], storage.v[0]);
  cp_async_commit();

#pragma unroll 1
  for (int tile = 0; tile < KV_TILES; ++tile) {
    cp_async_wait_all();
    __syncthreads();

    const int read_stage = tile % Stages;
    half *k_stage = storage.k[read_stage];
    half *v_stage = storage.v[read_stage];

    if constexpr (Stages == 2) {
      const int next_tile = tile + 1;
      if (next_tile < KV_TILES) {
        const int write_stage = next_tile % Stages;
        copy_tile_async<UseSwizzle>(&k[next_tile * BC * D],
                                    storage.k[write_stage]);
        copy_tile_async<UseSwizzle>(&v[next_tile * BC * D],
                                    storage.v[write_stage]);
        cp_async_commit();
      }
    }

    float s_frag[BC / 8][4] = {};

#pragma unroll
    for (int kd = 0; kd < D; kd += 16) {
      uint32_t q_frag[4];
      const int q_addr_row = warp * 16 + (lane & 15);
      const int q_addr_col = kd + (lane >> 4) * 8;
      ldmatrix_x4(
          q_frag,
          shared_u32(
              &storage.q[tile_offset<UseSwizzle>(q_addr_row, q_addr_col)]));

      // One x4 load contains two adjacent K B operands:
      //   k4[0:2] -> keys pair*16+0..7
      //   k4[2:4] -> keys pair*16+8..15
#pragma unroll
      for (int pair = 0; pair < BC / 16; ++pair) {
        uint32_t k4[4];
        const int matrix = lane >> 3;
        const int row = pair * 16 + (matrix >> 1) * 8 + (lane & 7);
        const int col = kd + (matrix & 1) * 8;
        ldmatrix_x4(
            k4,
            shared_u32(&k_stage[tile_offset<UseSwizzle>(row, col)]));
        mma_m16n8k16(s_frag[2 * pair], q_frag, k4[0], k4[1]);
        mma_m16n8k16(s_frag[2 * pair + 1], q_frag, k4[2], k4[3]);
      }
    }

    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
#pragma unroll
    for (int nj = 0; nj < BC / 8; ++nj) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        s_frag[nj][item] *= 0.125f;
      }
      local_max0 =
          fmaxf(local_max0, fmaxf(s_frag[nj][0], s_frag[nj][1]));
      local_max1 =
          fmaxf(local_max1, fmaxf(s_frag[nj][2], s_frag[nj][3]));
    }
    const float tile_max0 = subgroup4_max(local_max0);
    const float tile_max1 = subgroup4_max(local_max1);
    const float new_m0 = fmaxf(row_m0, tile_max0);
    const float new_m1 = fmaxf(row_m1, tile_max1);
    const float alpha0 = row_l0 == 0.0f ? 0.0f : __expf(row_m0 - new_m0);
    const float alpha1 = row_l1 == 0.0f ? 0.0f : __expf(row_m1 - new_m1);

#pragma unroll
    for (int nj = 0; nj < D / 8; ++nj) {
      o_acc[nj][0] *= alpha0;
      o_acc[nj][1] *= alpha0;
      o_acc[nj][2] *= alpha1;
      o_acc[nj][3] *= alpha1;
    }

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
#pragma unroll
    for (int nj = 0; nj < BC / 8; ++nj) {
      s_frag[nj][0] = __expf(s_frag[nj][0] - new_m0);
      s_frag[nj][1] = __expf(s_frag[nj][1] - new_m0);
      s_frag[nj][2] = __expf(s_frag[nj][2] - new_m1);
      s_frag[nj][3] = __expf(s_frag[nj][3] - new_m1);
      local_sum0 += s_frag[nj][0] + s_frag[nj][1];
      local_sum1 += s_frag[nj][2] + s_frag[nj][3];
    }
    row_l0 = alpha0 * row_l0 + subgroup4_sum(local_sum0);
    row_l1 = alpha1 * row_l1 + subgroup4_sum(local_sum1);
    row_m0 = new_m0;
    row_m1 = new_m1;

#pragma unroll
    for (int pk = 0; pk < BC; pk += 16) {
      const int pn = pk / 8;
      uint32_t p_frag[4] = {
          pack_half2(s_frag[pn][0], s_frag[pn][1]),
          pack_half2(s_frag[pn][2], s_frag[pn][3]),
          pack_half2(s_frag[pn + 1][0], s_frag[pn + 1][1]),
          pack_half2(s_frag[pn + 1][2], s_frag[pn + 1][3]),
      };

      // One x4.trans contains two adjacent V B operands:
      //   v4[0:2] -> output columns pair*16+0..7
      //   v4[2:4] -> output columns pair*16+8..15
#pragma unroll
      for (int pair = 0; pair < D / 16; ++pair) {
        uint32_t v4[4];
        const int matrix = lane >> 3;
        const int row = pk + (matrix & 1) * 8 + (lane & 7);
        const int col = pair * 16 + (matrix >> 1) * 8;
        ldmatrix_x4_trans(
            v4,
            shared_u32(&v_stage[tile_offset<UseSwizzle>(row, col)]));
        mma_m16n8k16(o_acc[2 * pair], p_frag, v4[0], v4[1]);
        mma_m16n8k16(o_acc[2 * pair + 1], p_frag, v4[2], v4[3]);
      }
    }

    // No warp may still read the current stage when a later iteration reuses
    // it.  __syncthreads does not wait for the other stage's cp.async; the next
    // iteration's cp_async_wait_all does that immediately before consumption.
    __syncthreads();
    if constexpr (Stages == 1) {
      const int next_tile = tile + 1;
      if (next_tile < KV_TILES) {
        copy_tile_async<UseSwizzle>(&k[next_tile * BC * D], storage.k[0]);
        copy_tile_async<UseSwizzle>(&v[next_tile * BC * D], storage.v[0]);
        cp_async_commit();
      }
    }
  }

  // Register C layout -> swizzled shared layout.  The logical coordinates are
  // unchanged; only tile_offset chooses a permuted physical 16-byte segment.
  const float inv_l0 = 1.0f / row_l0;
  const float inv_l1 = 1.0f / row_l1;
#pragma unroll
  for (int nj = 0; nj < D / 8; ++nj) {
    const int col = nj * 8 + 2 * t;
    *reinterpret_cast<uint32_t *>(
        &storage.q[tile_offset<UseSwizzle>(row0, col)]) =
        pack_half2(o_acc[nj][0] * inv_l0, o_acc[nj][1] * inv_l0);
    *reinterpret_cast<uint32_t *>(
        &storage.q[tile_offset<UseSwizzle>(row1, col)]) =
        pack_half2(o_acc[nj][2] * inv_l1, o_acc[nj][3] * inv_l1);
  }
  __syncthreads();

  // "Remove" swizzle by reading the same logical (row,col) through the same
  // mapping.  Lowest 3 column bits are untouched, so uint4 remains contiguous.
#pragma unroll
  for (int linear = tid * 8; linear < TILE_ELEMS;
       linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    const uint4 packed = *reinterpret_cast<const uint4 *>(
        &storage.q[tile_offset<UseSwizzle>(row, col)]);
    *reinterpret_cast<uint4 *>(&out[(q_tile + row) * D + col]) = packed;
  }
}

std::vector<half> reference_attention(const std::vector<half> &q,
                                      const std::vector<half> &k,
                                      const std::vector<half> &v) {
  std::vector<half> out(SEQLEN * D);
  std::vector<float> scores(BC);
  std::vector<half> p_half(BC);

  for (int m = 0; m < SEQLEN; ++m) {
    float row_m = -INFINITY;
    float row_l = 0.0f;
    float out_acc[D] = {};

    for (int tile = 0; tile < KV_TILES; ++tile) {
      float tile_max = -INFINITY;
      for (int n = 0; n < BC; ++n) {
        float score = 0.0f;
        const int key = tile * BC + n;
        for (int d = 0; d < D; ++d) {
          score += __half2float(q[m * D + d]) *
                   __half2float(k[key * D + d]);
        }
        scores[n] = score * 0.125f;
        tile_max = std::max(tile_max, scores[n]);
      }
      const float new_m = std::max(row_m, tile_max);
      const float alpha = row_l == 0.0f ? 0.0f : std::exp(row_m - new_m);
      for (int d = 0; d < D; ++d) {
        out_acc[d] *= alpha;
      }
      float tile_sum = 0.0f;
      for (int n = 0; n < BC; ++n) {
        const float p = std::exp(scores[n] - new_m);
        tile_sum += p;
        p_half[n] = __float2half_rn(p);
      }
      for (int d = 0; d < D; ++d) {
        for (int n = 0; n < BC; ++n) {
          const int key = tile * BC + n;
          out_acc[d] +=
              __half2float(p_half[n]) * __half2float(v[key * D + d]);
        }
      }
      row_l = alpha * row_l + tile_sum;
      row_m = new_m;
    }
    for (int d = 0; d < D; ++d) {
      out[m * D + d] = __float2half_rn(out_acc[d] / row_l);
    }
  }
  return out;
}

float max_error(const std::vector<half> &a, const std::vector<half> &b) {
  float result = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    result = std::max(
        result, std::fabs(__half2float(a[i]) - __half2float(b[i])));
  }
  return result;
}

template <int Stages, bool UseSwizzle>
std::vector<half> run_variant(const half *dq, const half *dk, const half *dv,
                              half *dout) {
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(
      &attr, flash_attn_multistage_kernel<Stages, UseSwizzle>));
  std::printf("variant Stages=%d Swizzle=%s: registers=%d, static_smem=%zu B\n",
              Stages, UseSwizzle ? "true" : "false", attr.numRegs,
              attr.sharedSizeBytes);

  flash_attn_multistage_kernel<Stages, UseSwizzle>
      <<<SEQLEN / BR, THREADS>>>(dq, dk, dv, dout);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<half> result(SEQLEN * D);
  CUDA_CHECK(cudaMemcpy(result.data(), dout, result.size() * sizeof(half),
                        cudaMemcpyDeviceToHost));
  return result;
}

void print_swizzle_table() {
  std::printf("Swizzle<3,3,3> segment mapping (segment=col/8):\n");
  for (int row = 0; row < 8; ++row) {
    std::printf("  row %d:", row);
    for (int segment = 0; segment < 8; ++segment) {
      std::printf(" %d->%d", segment, segment ^ row);
    }
    std::printf("\n");
  }
  std::printf("Applying the same row XOR again recovers the logical segment.\n\n");
}

} // namespace

int main() {
  print_swizzle_table();

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 8) {
    std::fprintf(stderr, "This example requires compute capability 8.0+.\n");
    return EXIT_FAILURE;
  }

  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<half> hq(SEQLEN * D), hk(SEQLEN * D), hv(SEQLEN * D);
  for (int i = 0; i < SEQLEN * D; ++i) {
    hq[i] = __float2half_rn(dist(rng));
    hk[i] = __float2half_rn(dist(rng));
    hv[i] = __float2half_rn(dist(rng));
  }
  const std::vector<half> reference = reference_attention(hq, hk, hv);

  half *dq = nullptr;
  half *dk = nullptr;
  half *dv = nullptr;
  half *dout = nullptr;
  const size_t bytes = SEQLEN * D * sizeof(half);
  CUDA_CHECK(cudaMalloc(&dq, bytes));
  CUDA_CHECK(cudaMalloc(&dk, bytes));
  CUDA_CHECK(cudaMalloc(&dv, bytes));
  CUDA_CHECK(cudaMalloc(&dout, bytes));
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), bytes, cudaMemcpyHostToDevice));

  const std::vector<half> one_stage =
      run_variant<1, false>(dq, dk, dv, dout);
  const std::vector<half> two_stage =
      run_variant<2, false>(dq, dk, dv, dout);
  const std::vector<half> swizzled =
      run_variant<2, true>(dq, dk, dv, dout);

  const float error_one = max_error(one_stage, reference);
  const float error_two = max_error(two_stage, reference);
  const float error_swizzle = max_error(swizzled, reference);
  const float layout_difference = max_error(two_stage, swizzled);
  std::printf("\nreference max error:\n");
  std::printf("  1-stage, no swizzle: %g\n", error_one);
  std::printf("  2-stage, no swizzle: %g\n", error_two);
  std::printf("  2-stage, swizzled:   %g\n", error_swizzle);
  std::printf("  no-swizzle vs swizzle output difference: %g\n",
              layout_difference);

  const bool pass = error_one <= 2.0e-2f && error_two <= 2.0e-2f &&
                    error_swizzle <= 2.0e-2f && layout_difference == 0.0f;
  std::printf("validation: %s\n", pass ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(dq));
  CUDA_CHECK(cudaFree(dk));
  CUDA_CHECK(cudaFree(dv));
  CUDA_CHECK(cudaFree(dout));
  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
