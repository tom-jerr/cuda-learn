#include "ampere_primitives.cuh"
#include "ffi_common.h"

#include <cmath>

using tvm::ffi::TensorView;

namespace {

constexpr int D = 64;
constexpr int BR = 128;
constexpr int BC = 128;
constexpr int THREADS = 128;
constexpr int WARPS = 4;
static_assert(THREADS == WARPS * 32);
constexpr int Q_ELEMS = BR * D;
constexpr int KV_ELEMS = BC * D;
constexpr size_t SWIZZLED_SMEM_BYTES =
    (Q_ELEMS + 2 * KV_ELEMS) * sizeof(half); // 48 KiB

// Logical [row, col] -> physical shared-memory element. The low three row
// bits XOR the 16-byte vector-column bits. Values inside an 8-half vector stay
// contiguous, so cp.async and the vectorized epilogue remain 16-byte aligned.
// This is the D=64 specialization of CUTE's Swizzle<3,3,3> layout.
__device__ __forceinline__ int swizzled_index(int row, int col) {
  return row * D + (col ^ ((row & 7) << 3));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  union PackedHalf2 {
    half2 h;
    uint32_t u;
  } packed;
  packed.h = __floats2half2_rn(lo, hi);
  return packed.u;
}

__device__ __forceinline__ float subgroup4_max(float value) {
#pragma unroll
  for (int delta = 1; delta < 4; delta <<= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, delta, 4));
  }
  return value;
}

__device__ __forceinline__ float subgroup4_sum(float value) {
#pragma unroll
  for (int delta = 1; delta < 4; delta <<= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, delta, 4);
  }
  return value;
}

// Four warps cover a 128-row Q tile. Each warp owns two independent 16-row
// MMA tiles (mi=0/1), separated by 64 logical rows, and keeps both score and
// output fragments in registers. Q/K/V use a compact XOR-swizzled 48 KiB
// shared layout so two CTAs can reside on an sm_89 SM.
__global__ __launch_bounds__(THREADS, 2)
void flash_attention2_fwd_d64_swizzled_4warp(
    const half *__restrict__ q, const half *__restrict__ k,
    const half *__restrict__ v, half *__restrict__ o, int heads, int seqlen,
    bool causal) {
  extern __shared__ __align__(16) unsigned char storage[];
  half *q_smem = reinterpret_cast<half *>(storage);
  half *k_smem = q_smem + Q_ELEMS;
  half *v_smem = k_smem + KV_ELEMS;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int q_tile = blockIdx.x * BR;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const size_t bh_base =
      (static_cast<size_t>(batch) * heads + head) * seqlen * D;
  const int q_valid_rows = min(BR, seqlen - q_tile);

  float o_acc[2][8][4] = {};
  float row_m0[2] = {-INFINITY, -INFINITY};
  float row_m1[2] = {-INFINITY, -INFINITY};
  float row_l0[2] = {0.0f, 0.0f};
  float row_l1[2] = {0.0f, 0.0f};

  // Prime invariant Q and K[0]. With 128 threads, every lane issues twice as
  // many 16-byte copies as the padded 8-warp kernel.
#pragma unroll
  for (int linear = tid * 8; linear < BR * D; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    const int safe_row = q_tile + min(row, q_valid_rows - 1);
    ampere::cp_async_16_zfill(
        ampere::shared_u32(&q_smem[swizzled_index(row, col)]),
        &q[bh_base + static_cast<size_t>(safe_row) * D + col],
        row < q_valid_rows);
  }

  constexpr float SOFTMAX_SCALE = 0.125f;
  const int kv_tiles = causal ? blockIdx.x + 1 : (seqlen + BC - 1) / BC;

#pragma unroll
  for (int linear = tid * 8; linear < BC * D; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    const int safe_row = min(row, seqlen - 1);
    const size_t offset = bh_base + static_cast<size_t>(safe_row) * D + col;
    ampere::cp_async_16_zfill(
        ampere::shared_u32(&k_smem[swizzled_index(row, col)]), &k[offset],
        row < seqlen);
  }
  ampere::cp_async_commit();

  for (int tile_idx = 0; tile_idx < kv_tiles; ++tile_idx) {
    ampere::cp_async_wait_all();
    __syncthreads();

    const int kv_tile = tile_idx * BC;

    // V[t] overlaps the QK phase.
#pragma unroll
    for (int linear = tid * 8; linear < BC * D;
         linear += THREADS * 8) {
      const int row = linear / D;
      const int col = linear % D;
      const int safe_row = min(kv_tile + row, seqlen - 1);
      const size_t offset = bh_base + static_cast<size_t>(safe_row) * D + col;
      ampere::cp_async_16_zfill(
          ampere::shared_u32(&v_smem[swizzled_index(row, col)]), &v[offset],
          kv_tile + row < seqlen);
    }
    ampere::cp_async_commit();

    float s_frag[2][16][4] = {};
#pragma unroll
    for (int kd = 0; kd < D; kd += 16) {
      uint32_t q_frag[2][4];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int q_row = warp * 16 + mi * 64 + (lane & 15);
        const int q_col = kd + (lane >> 4) * 8;
        ampere::ldmatrix_x4(
            q_frag[mi],
            ampere::shared_u32(&q_smem[swizzled_index(q_row, q_col)]));
      }

#pragma unroll
      for (int nj = 0; nj < 16; ++nj) {
        uint32_t k_frag[2];
        const int k_row = nj * 8 + (lane & 7);
        const int k_col = kd + ((lane >> 3) & 1) * 8;
        ampere::ldmatrix_x2(
            k_frag,
            ampere::shared_u32(&k_smem[swizzled_index(k_row, k_col)]));
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          ampere::mma_m16n8k16_f32(s_frag[mi][nj], q_frag[mi], k_frag);
        }
      }
    }

    ampere::cp_async_wait_all();
    __syncthreads();

    // K[t+1] overlaps both 16-row softmax/PV groups.
    if (tile_idx + 1 < kv_tiles) {
      const int next_kv_tile = kv_tile + BC;
#pragma unroll
      for (int linear = tid * 8; linear < BC * D;
           linear += THREADS * 8) {
        const int row = linear / D;
        const int col = linear % D;
        const int safe_row = min(next_kv_tile + row, seqlen - 1);
        const size_t offset =
            bh_base + static_cast<size_t>(safe_row) * D + col;
        ampere::cp_async_16_zfill(
            ampere::shared_u32(&k_smem[swizzled_index(row, col)]), &k[offset],
            next_kv_tile + row < seqlen);
      }
      ampere::cp_async_commit();
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      const int row0 = warp * 16 + mi * 64 + lane / 4;
      const int row1 = row0 + 8;
      float local_max0 = -INFINITY;
      float local_max1 = -INFINITY;
#pragma unroll
      for (int nj = 0; nj < 16; ++nj) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          s_frag[mi][nj][item] *= SOFTMAX_SCALE;
        }
        if ((causal && kv_tile == q_tile) || kv_tile + BC > seqlen) {
          const int col = nj * 8 + (lane % 4) * 2;
          if (kv_tile + col >= seqlen ||
              (causal && kv_tile + col > q_tile + row0)) {
            s_frag[mi][nj][0] = -INFINITY;
          }
          if (kv_tile + col + 1 >= seqlen ||
              (causal && kv_tile + col + 1 > q_tile + row0)) {
            s_frag[mi][nj][1] = -INFINITY;
          }
          if (kv_tile + col >= seqlen ||
              (causal && kv_tile + col > q_tile + row1)) {
            s_frag[mi][nj][2] = -INFINITY;
          }
          if (kv_tile + col + 1 >= seqlen ||
              (causal && kv_tile + col + 1 > q_tile + row1)) {
            s_frag[mi][nj][3] = -INFINITY;
          }
        }
        local_max0 =
            fmaxf(local_max0,
                  fmaxf(s_frag[mi][nj][0], s_frag[mi][nj][1]));
        local_max1 =
            fmaxf(local_max1,
                  fmaxf(s_frag[mi][nj][2], s_frag[mi][nj][3]));
      }

      const float tile_max0 = subgroup4_max(local_max0);
      const float tile_max1 = subgroup4_max(local_max1);
      const float new_m0 = fmaxf(row_m0[mi], tile_max0);
      const float new_m1 = fmaxf(row_m1[mi], tile_max1);
      const float alpha0 =
          row_l0[mi] == 0.0f ? 0.0f : __expf(row_m0[mi] - new_m0);
      const float alpha1 =
          row_l1[mi] == 0.0f ? 0.0f : __expf(row_m1[mi] - new_m1);

#pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        o_acc[mi][nj][0] *= alpha0;
        o_acc[mi][nj][1] *= alpha0;
        o_acc[mi][nj][2] *= alpha1;
        o_acc[mi][nj][3] *= alpha1;
      }

      float local_sum0 = 0.0f;
      float local_sum1 = 0.0f;
#pragma unroll
      for (int nj = 0; nj < 16; ++nj) {
        const float p00 = __expf(s_frag[mi][nj][0] - new_m0);
        const float p01 = __expf(s_frag[mi][nj][1] - new_m0);
        const float p10 = __expf(s_frag[mi][nj][2] - new_m1);
        const float p11 = __expf(s_frag[mi][nj][3] - new_m1);
        local_sum0 += p00 + p01;
        local_sum1 += p10 + p11;
        s_frag[mi][nj][0] = p00;
        s_frag[mi][nj][1] = p01;
        s_frag[mi][nj][2] = p10;
        s_frag[mi][nj][3] = p11;
      }
      row_l0[mi] = alpha0 * row_l0[mi] + subgroup4_sum(local_sum0);
      row_l1[mi] = alpha1 * row_l1[mi] + subgroup4_sum(local_sum1);
      row_m0[mi] = new_m0;
      row_m1[mi] = new_m1;
    }

    // P stays in registers. Process the two 16-row groups successively so the
    // compiler can retire one score half before materializing the next P MMA
    // operand. This trades some repeated V ldmatrix work for lower peak
    // register pressure, which is critical at the 255-register 2-CTA limit.
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int pk = 0; pk < BC; pk += 16) {
        const int pn = pk / 8;
        uint32_t p_frag[4] = {
            pack_half2(s_frag[mi][pn][0], s_frag[mi][pn][1]),
            pack_half2(s_frag[mi][pn][2], s_frag[mi][pn][3]),
            pack_half2(s_frag[mi][pn + 1][0], s_frag[mi][pn + 1][1]),
            pack_half2(s_frag[mi][pn + 1][2], s_frag[mi][pn + 1][3]),
        };

#pragma unroll
        for (int nj = 0; nj < 8; ++nj) {
          uint32_t v_frag[2];
          const int v_row = pk + (lane & 15);
          const int v_col = nj * 8;
          ampere::ldmatrix_x2_trans(
              v_frag,
              ampere::shared_u32(&v_smem[swizzled_index(v_row, v_col)]));
          ampere::mma_m16n8k16_f32(o_acc[mi][nj], p_frag, v_frag);
        }
      }
    }
  }

  // Reuse Q shared memory for FA2-style vectorized output layout conversion.
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    const int row0 = warp * 16 + mi * 64 + lane / 4;
    const int row1 = row0 + 8;
    const float inv_l0 = 1.0f / row_l0[mi];
    const float inv_l1 = 1.0f / row_l1[mi];
#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      const int col = nj * 8 + (lane % 4) * 2;
      *reinterpret_cast<uint32_t *>(
          &q_smem[swizzled_index(row0, col)]) =
          pack_half2(o_acc[mi][nj][0] * inv_l0,
                     o_acc[mi][nj][1] * inv_l0);
      *reinterpret_cast<uint32_t *>(
          &q_smem[swizzled_index(row1, col)]) =
          pack_half2(o_acc[mi][nj][2] * inv_l1,
                     o_acc[mi][nj][3] * inv_l1);
    }
  }
  __syncthreads();

#pragma unroll
  for (int linear = tid * 8; linear < BR * D;
       linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    if (row < q_valid_rows) {
      const uint4 output = *reinterpret_cast<const uint4 *>(
          &q_smem[swizzled_index(row, col)]);
      *reinterpret_cast<uint4 *>(
          &o[bh_base + static_cast<size_t>(q_tile + row) * D + col]) = output;
    }
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

void flash_attn_swizzled(TensorView q, TensorView k, TensorView v,
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
        << "flash_attn_swizzled: expected [B,H,N,64] with positive B/H "
           "and N divisible by 64";
  }
  if (batch64 > 65535 || heads64 > 65535 || seqlen64 > 2147483647LL) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn_swizzled: shape exceeds CUDA grid limits";
  }

  const int batch = static_cast<int>(batch64);
  const int heads = static_cast<int>(heads64);
  const int seqlen = static_cast<int>(seqlen64);
  const bool causal = causal64 != 0;

  static const bool smem_opted_in = [] {
    CUDA_LEARN_CHECK(cudaFuncSetAttribute(
        flash_attention2_fwd_d64_swizzled_4warp,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SWIZZLED_SMEM_BYTES)));
    return true;
  }();
  (void)smem_opted_in;

  const dim3 grid((seqlen + BR - 1) / BR, heads, batch);
  flash_attention2_fwd_d64_swizzled_4warp
      <<<grid, THREADS, SWIZZLED_SMEM_BYTES, get_stream(q)>>>(
          static_cast<const half *>(q.data_ptr()),
          static_cast<const half *>(k.data_ptr()),
          static_cast<const half *>(v.data_ptr()),
          static_cast<half *>(out.data_ptr()), heads, seqlen, causal);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.flash_attn_swizzled", flash_attn_swizzled);
