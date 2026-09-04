#include "ampere_primitives.cuh"
#include "ffi_common.h"

#include <cmath>
#include <cstdint>

using tvm::ffi::TensorView;

namespace {

constexpr int D = 64;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int THREADS = 128;
constexpr int TILE_ELEMS = 64 * 64;

// D=64 specialization of CuTe Swizzle<3,3,3>.  The XOR permutes 16-byte
// segments within a row while leaving the low three column bits unchanged.
// Producers and consumers both address logical (row,col) through this map;
// no explicit "unswizzle" instruction is needed.
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

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&dst)[4],
                                                  uint32_t src_smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
               "{%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
               : "r"(src_smem));
}

__device__ __forceinline__ void
mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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

// 128 threads copy one 64x64 tile as 512 aligned 16-byte transactions:
//   row = tid/8 + 16*rho, col = 8*(tid%8), rho=0..3.
template <bool UseSwizzle>
__device__ __forceinline__ void copy_tile_async(const half *src, half *dst) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int linear = tid * 8; linear < TILE_ELEMS; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    ampere::cp_async_16(
        ampere::shared_u32(&dst[tile_offset<UseSwizzle>(row, col)]),
        &src[linear]);
  }
}

// A deliberately direct production binding of the teaching example:
//   - four warps own 64 query rows (one m16 fragment per warp),
//   - K/V have two 64x64 shared-memory stages,
//   - one x4 load supplies two adjacent K or V B fragments,
//   - QK accumulator ownership is reinterpreted as the P operand for PV.
template <int Stages, bool UseSwizzle, bool Causal>
__global__ __launch_bounds__(THREADS, 2) void flash_attn_multistage_kernel(
    const half *__restrict__ q, const half *__restrict__ k,
    const half *__restrict__ v, half *__restrict__ out, int heads, int seqlen) {
  static_assert(Stages == 1 || Stages == 2);
  __shared__ __align__(16) SharedStorage<Stages> storage;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int q_block = blockIdx.x;
  const int q_tile = q_block * BR;
  const int row0 = warp * 16 + g;
  const int row1 = row0 + 8;
  const size_t bh_base =
      (static_cast<size_t>(blockIdx.z) * heads + blockIdx.y) * seqlen * D;

  float o_acc[D / 8][4] = {};
  float row_m0 = -INFINITY;
  float row_m1 = -INFINITY;
  float row_l0 = 0.0f;
  float row_l1 = 0.0f;

  const int kv_tiles = Causal ? q_block + 1 : seqlen / BC;

  // Group zero primes invariant Q and the first K/V tile.  Each subsequent
  // group is issued into the other stage before current-tile math starts.
  copy_tile_async<UseSwizzle>(&q[bh_base + size_t(q_tile) * D], storage.q);
  copy_tile_async<UseSwizzle>(&k[bh_base], storage.k[0]);
  copy_tile_async<UseSwizzle>(&v[bh_base], storage.v[0]);
  ampere::cp_async_commit();

#pragma unroll 1
  for (int tile = 0; tile < kv_tiles; ++tile) {
    ampere::cp_async_wait_all();
    __syncthreads();

    const int read_stage = tile % Stages;
    half *k_stage = storage.k[read_stage];
    half *v_stage = storage.v[read_stage];

    if constexpr (Stages == 2) {
      const int next_tile = tile + 1;
      if (next_tile < kv_tiles) {
        const int write_stage = next_tile % Stages;
        const size_t next_offset = bh_base + size_t(next_tile) * BC * D;
        copy_tile_async<UseSwizzle>(&k[next_offset], storage.k[write_stage]);
        copy_tile_async<UseSwizzle>(&v[next_offset], storage.v[write_stage]);
        ampere::cp_async_commit();
      }
    }

    float s_frag[BC / 8][4] = {};

    // Q @ K^T.  Each ldmatrix.x4 of K replaces two neighboring x2 loads;
    // registers [0:1] feed score fragment 2*pair and [2:3] feed 2*pair+1.
#pragma unroll
    for (int kd = 0; kd < D; kd += 16) {
      uint32_t q_frag[4];
      const int q_row = warp * 16 + (lane & 15);
      const int q_col = kd + (lane >> 4) * 8;
      ampere::ldmatrix_x4(
          q_frag, ampere::shared_u32(
                      &storage.q[tile_offset<UseSwizzle>(q_row, q_col)]));

#pragma unroll
      for (int pair = 0; pair < BC / 16; ++pair) {
        uint32_t k4[4];
        const int matrix = lane >> 3;
        const int row = pair * 16 + (matrix >> 1) * 8 + (lane & 7);
        const int col = kd + (matrix & 1) * 8;
        ampere::ldmatrix_x4(
            k4,
            ampere::shared_u32(&k_stage[tile_offset<UseSwizzle>(row, col)]));
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
      if constexpr (Causal) {
        if (tile == q_block) {
          const int col = nj * 8 + 2 * t;
          if (col > row0)
            s_frag[nj][0] = -INFINITY;
          if (col + 1 > row0)
            s_frag[nj][1] = -INFINITY;
          if (col > row1)
            s_frag[nj][2] = -INFINITY;
          if (col + 1 > row1)
            s_frag[nj][3] = -INFINITY;
        }
      }
      local_max0 = fmaxf(local_max0, fmaxf(s_frag[nj][0], s_frag[nj][1]));
      local_max1 = fmaxf(local_max1, fmaxf(s_frag[nj][2], s_frag[nj][3]));
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

    // P @ V.  Adjacent QK C fragments already have the exact lane ownership
    // required by one PV A fragment; only FP32->FP16 packing is performed.
#pragma unroll
    for (int pk = 0; pk < BC; pk += 16) {
      const int pn = pk / 8;
      uint32_t p_frag[4] = {
          pack_half2(s_frag[pn][0], s_frag[pn][1]),
          pack_half2(s_frag[pn][2], s_frag[pn][3]),
          pack_half2(s_frag[pn + 1][0], s_frag[pn + 1][1]),
          pack_half2(s_frag[pn + 1][2], s_frag[pn + 1][3]),
      };

#pragma unroll
      for (int pair = 0; pair < D / 16; ++pair) {
        uint32_t v4[4];
        const int matrix = lane >> 3;
        const int row = pk + (matrix & 1) * 8 + (lane & 7);
        const int col = pair * 16 + (matrix >> 1) * 8;
        ldmatrix_x4_trans(v4, ampere::shared_u32(
                                  &v_stage[tile_offset<UseSwizzle>(row, col)]));
        mma_m16n8k16(o_acc[2 * pair], p_frag, v4[0], v4[1]);
        mma_m16n8k16(o_acc[2 * pair + 1], p_frag, v4[2], v4[3]);
      }
    }

    // Proves all warps finished reading this stage before it is reused two
    // iterations later.  The next wait_all publishes the other stage.
    __syncthreads();
    if constexpr (Stages == 1) {
      const int next_tile = tile + 1;
      if (next_tile < kv_tiles) {
        const size_t next_offset = bh_base + size_t(next_tile) * BC * D;
        copy_tile_async<UseSwizzle>(&k[next_offset], storage.k[0]);
        copy_tile_async<UseSwizzle>(&v[next_offset], storage.v[0]);
        ampere::cp_async_commit();
      }
    }
  }

  // Accumulator C layout -> swizzled shared memory -> contiguous global.  The
  // second read applies the same XOR map, which is its own inverse.
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

#pragma unroll
  for (int linear = tid * 8; linear < TILE_ELEMS; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    const uint4 packed = *reinterpret_cast<const uint4 *>(
        &storage.q[tile_offset<UseSwizzle>(row, col)]);
    *reinterpret_cast<uint4 *>(&out[bh_base + size_t(q_tile + row) * D + col]) =
        packed;
  }
}

void check_fp16_tensor(const TensorView &tensor, const char *name) {
  if (tensor.data_ptr() == nullptr || tensor.ndim() != 4 ||
      tensor.dtype().code != kDLFloat || tensor.dtype().bits != 16 ||
      tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected a non-null float16 CUDA [B,H,N,64] tensor";
  }
}

void check_same_shape(const TensorView &tensor, const TensorView &q,
                      const char *name) {
  for (int axis = 0; axis < 4; ++axis) {
    if (dim(tensor, axis) != dim(q, axis)) {
      TVM_FFI_THROW(RuntimeError) << name << ": expected the same shape as q";
    }
  }
  if (tensor.device().device_id != q.device().device_id) {
    TVM_FFI_THROW(RuntimeError) << name << ": expected the same device as q";
  }
}

} // namespace

void flash_attn_multistage(TensorView q, TensorView k, TensorView v,
                           TensorView out, int64_t causal64) {
  check_fp16_tensor(q, "q");
  check_fp16_tensor(k, "k");
  check_fp16_tensor(v, "v");
  check_fp16_tensor(out, "out");
  check_same_shape(k, q, "k");
  check_same_shape(v, q, "v");
  check_same_shape(out, q, "out");

  const int64_t batch64 = dim(q, 0);
  const int64_t heads64 = dim(q, 1);
  const int64_t seqlen64 = dim(q, 2);
  if (dim(q, 3) != D || batch64 <= 0 || heads64 <= 0 || seqlen64 <= 0 ||
      seqlen64 % 64 != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn_multistage: expected [B,H,N,64] with positive B/H "
           "and N divisible by 64";
  }
  if (batch64 > 65535 || heads64 > 65535 || seqlen64 > 2147483647LL) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn_multistage: shape exceeds CUDA grid limits";
  }

  const int batch = static_cast<int>(batch64);
  const int heads = static_cast<int>(heads64);
  const int seqlen = static_cast<int>(seqlen64);
  const dim3 grid(seqlen / BR, heads, batch);
  cudaStream_t stream = get_stream(q);
  if (causal64 != 0) {
    flash_attn_multistage_kernel<2, true, true><<<grid, THREADS, 0, stream>>>(
        static_cast<const half *>(q.data_ptr()),
        static_cast<const half *>(k.data_ptr()),
        static_cast<const half *>(v.data_ptr()),
        static_cast<half *>(out.data_ptr()), heads, seqlen);
  } else {
    flash_attn_multistage_kernel<2, true, false><<<grid, THREADS, 0, stream>>>(
        static_cast<const half *>(q.data_ptr()),
        static_cast<const half *>(k.data_ptr()),
        static_cast<const half *>(v.data_ptr()),
        static_cast<half *>(out.data_ptr()), heads, seqlen);
  }
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.flash_attn_multistage", flash_attn_multistage);
