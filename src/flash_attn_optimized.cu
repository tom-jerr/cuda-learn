#include "ampere_primitives.cuh"
#include "ffi_common.h"

#include <cmath>

using tvm::ffi::TensorView;

namespace {

constexpr int D = 64;
constexpr int BR = 128;
constexpr int BC = 128;
constexpr int PAD = 8;
constexpr int THREADS = 256;

constexpr int Q_ELEMS = BR * (D + PAD);
constexpr int KV_ELEMS = BC * (D + PAD);

// Q, one K tile, and one V tile are the only shared-memory allocations. K and
// V deliberately use separate single-stage
// buffers: while Q @ K[t]^T runs, V[t] is copied; while softmax/P @ V[t] runs,
// K[t+1] is copied. This is the phase-interleaved pipeline used by FA2 and
// avoids the occupancy cost of duplicating complete K/V tiles.
constexpr size_t OPT_SMEM_BYTES = (Q_ELEMS + 2 * KV_ELEMS) * sizeof(half);

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

// FP16 forward attention for [B,H,N,64], with optional causal masking. Eight
// warps cover one 128-row query tile; each warp owns 16 rows. Within a warp,
// each four-lane subgroup owns two rows:
//   lanes [4g, 4g+3] cooperate on rows g and g+8.
// This exactly matches mma.m16n8k16's fp32 accumulator mapping.
__global__ void flash_attention2_fwd_d64_registers(const half *__restrict__ q,
                                                   const half *__restrict__ k,
                                                   const half *__restrict__ v,
                                                   half *__restrict__ o,
                                                   int heads, int seqlen,
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
  const int row0 = warp * 16 + lane / 4;
  const int row1 = row0 + 8;
  const size_t bh_base =
      (static_cast<size_t>(batch) * heads + head) * seqlen * D;

  // The accumulator layout per lane is:
  //   [0:1] -> row0, two adjacent output columns
  //   [2:3] -> row1, two adjacent output columns.
  // Eight N fragments cover D=64. It survives across every KV tile.
  float o_acc[8][4] = {};
  float row_m0 = -INFINITY;
  float row_m1 = -INFINITY;
  float row_l0 = 0.0f;
  float row_l1 = 0.0f;

  const int q_valid_rows = min(BR, seqlen - q_tile);

  // Q is invariant across the whole KV streaming loop. Prime Q and K[0] in
  // the same async group; a partial final 128-row tile is zero-filled.
#pragma unroll
  for (int linear = tid * 8; linear < BR * D; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    const int safe_row = q_tile + min(row, q_valid_rows - 1);
    ampere::cp_async_16_zfill(
        ampere::shared_u32(&q_smem[row * (D + PAD) + col]),
        &q[bh_base + static_cast<size_t>(safe_row) * D + col],
        row < q_valid_rows);
  }

  constexpr float SOFTMAX_SCALE = 0.125f;
  // Causal attention never touches a KV tile strictly to the right of this Q
  // tile. Because BR == BC, the final visited tile is exactly the diagonal.
  const int kv_tiles = causal ? blockIdx.x + 1 : (seqlen + BC - 1) / BC;

  // Prime K[0]. K has one shared-memory stage; it is only overwritten after
  // every warp has completed Q @ K[t]^T.
#pragma unroll
  for (int linear = tid * 8; linear < BC * D; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    const int safe_row = min(row, seqlen - 1);
    const size_t offset = bh_base + static_cast<size_t>(safe_row) * D + col;
    ampere::cp_async_16_zfill(
        ampere::shared_u32(&k_smem[row * (D + PAD) + col]), &k[offset],
        row < seqlen);
  }
  ampere::cp_async_commit();

  for (int tile_idx = 0; tile_idx < kv_tiles; ++tile_idx) {
    // K[t] was issued by the prologue or during the previous softmax/PV phase.
    // Besides publishing K[t], this barrier proves all warps finished reading
    // V[t-1] before the single V buffer is reused below.
    ampere::cp_async_wait_all();
    __syncthreads();

    const int kv_tile = tile_idx * BC;

    // V[t] is independent of Q @ K[t]^T, so copy it while Tensor Cores consume
    // the current K tile.
#pragma unroll
    for (int linear = tid * 8; linear < BC * D; linear += THREADS * 8) {
      const int row = linear / D;
      const int col = linear % D;
      const int safe_row = min(kv_tile + row, seqlen - 1);
      const size_t offset = bh_base + static_cast<size_t>(safe_row) * D + col;
      ampere::cp_async_16_zfill(
          ampere::shared_u32(&v_smem[row * (D + PAD) + col]), &v[offset],
          kv_tile + row < seqlen);
    }
    ampere::cp_async_commit();

    // --------------------------- Q @ K^T ---------------------------
    // All 16x128 scores owned by this warp remain in registers.
    float s_frag[16][4] = {};
#pragma unroll
    for (int kd = 0; kd < D; kd += 16) {
      uint32_t q_frag[4];
      const int q_row = warp * 16 + (lane & 15);
      const int q_col = kd + (lane >> 4) * 8;
      ampere::ldmatrix_x4(
          q_frag, ampere::shared_u32(&q_smem[q_row * (D + PAD) + q_col]));

#pragma unroll
      for (int nj = 0; nj < 16; ++nj) {
        uint32_t k_frag[2];
        const int k_row = nj * 8 + (lane & 7);
        const int k_col = kd + ((lane >> 3) & 1) * 8;
        ampere::ldmatrix_x2(
            k_frag, ampere::shared_u32(&k_smem[k_row * (D + PAD) + k_col]));
        ampere::mma_m16n8k16_f32(s_frag[nj], q_frag, k_frag);
      }
    }

    // V[t] must be visible before PV. This barrier also proves every warp has
    // finished reading K[t], so K[t+1] may reuse the sole K buffer.
    ampere::cp_async_wait_all();
    __syncthreads();

    if (tile_idx + 1 < kv_tiles) {
      const int next_kv_tile = kv_tile + BC;
#pragma unroll
      for (int linear = tid * 8; linear < BC * D; linear += THREADS * 8) {
        const int row = linear / D;
        const int col = linear % D;
        const int safe_row = min(next_kv_tile + row, seqlen - 1);
        const size_t offset = bh_base + static_cast<size_t>(safe_row) * D + col;
        ampere::cp_async_16_zfill(
            ampere::shared_u32(&k_smem[row * (D + PAD) + col]), &k[offset],
            next_kv_tile + row < seqlen);
      }
      ampere::cp_async_commit();
    }

    // --------------------- warp-cooperative softmax ---------------------
    // A lane has 32 scores for row0 and 32 for row1. Four adjacent lanes
    // together cover all 128 columns of each row.
    // 16 * 128
    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
#pragma unroll
    for (int nj = 0; nj < 16; ++nj) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        // 每个 lane 持有 4 个 accumulator
        s_frag[nj][item] *= SOFTMAX_SCALE;
      }
      // Whole future tiles were skipped above. Only the diagonal tile needs
      // element-level masking. Fragment items 0:1 belong to row0 and 2:3 to
      // row1; each lane owns two adjacent key columns.
      if ((causal && kv_tile == q_tile) || kv_tile + BC > seqlen) {
        const int col = nj * 8 + (lane % 4) * 2;
        if (kv_tile + col >= seqlen ||
            (causal && kv_tile + col > q_tile + row0)) {
          s_frag[nj][0] = -INFINITY;
        }
        if (kv_tile + col + 1 >= seqlen ||
            (causal && kv_tile + col + 1 > q_tile + row0)) {
          s_frag[nj][1] = -INFINITY;
        }
        if (kv_tile + col >= seqlen ||
            (causal && kv_tile + col > q_tile + row1)) {
          s_frag[nj][2] = -INFINITY;
        }
        if (kv_tile + col + 1 >= seqlen ||
            (causal && kv_tile + col + 1 > q_tile + row1)) {
          s_frag[nj][3] = -INFINITY;
        }
      }
      /**
        s_frag[nj][0] = S[lane4/4    ][nj * 8 + lane4 * 2    ];
        s_frag[nj][1] = S[lane4/4    ][nj * 8 + lane4 * 2 + 1];

        s_frag[nj][2] = S[lane4/4 + 8][nj * 8 + lane4 * 2    ];
        s_frag[nj][3] = S[lane4/4 + 8][nj * 8 + lane4 * 2 + 1];
       */
      local_max0 = fmaxf(local_max0, fmaxf(s_frag[nj][0], s_frag[nj][1]));
      local_max1 = fmaxf(local_max1, fmaxf(s_frag[nj][2], s_frag[nj][3]));
    }
    const float tile_max0 = subgroup4_max(local_max0);
    const float tile_max1 = subgroup4_max(local_max1);
    const float new_m0 = fmaxf(row_m0, tile_max0);
    const float new_m1 = fmaxf(row_m1, tile_max1);
    const float alpha0 = row_l0 == 0.0f ? 0.0f : __expf(row_m0 - new_m0);
    const float alpha1 = row_l1 == 0.0f ? 0.0f : __expf(row_m1 - new_m1);

    // Rescale the persistent output numerator in registers before adding P@V.
#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      o_acc[nj][0] *= alpha0;
      o_acc[nj][1] *= alpha0;
      o_acc[nj][2] *= alpha1;
      o_acc[nj][3] *= alpha1;
    }

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
#pragma unroll
    for (int nj = 0; nj < 16; ++nj) {
      const float p00 = __expf(s_frag[nj][0] - new_m0);
      const float p01 = __expf(s_frag[nj][1] - new_m0);
      const float p10 = __expf(s_frag[nj][2] - new_m1);
      const float p11 = __expf(s_frag[nj][3] - new_m1);
      local_sum0 += p00 + p01;
      local_sum1 += p10 + p11;
      // Keep P in the score accumulator registers. Two adjacent 8-column
      // score fragments form the four packed A registers consumed by the
      // following m16n8k16 PV MMA.
      s_frag[nj][0] = p00;
      s_frag[nj][1] = p01;
      s_frag[nj][2] = p10;
      s_frag[nj][3] = p11;
    }
    row_l0 = alpha0 * row_l0 + subgroup4_sum(local_sum0);
    row_l1 = alpha1 * row_l1 + subgroup4_sum(local_sum1);
    row_m0 = new_m0;
    row_m1 = new_m1;

    // ----------------------------- P @ V -----------------------------
    /**
      QK MMA accumulator
              ↓ softmax，保持 lane ownership
      两个 16×8 s_frag
              ↓ FP32/BF16 转 FP16，并 pack half2
      一个 16×16 P operand
              ↓
      P × V MMA
      p_frag[0]: 上半部分行，P 的前 8 列片段
      p_frag[1]: 下半部分行，P 的前 8 列片段
      p_frag[2]: 上半部分行，P 的后 8 列片段
      p_frag[3]: 下半部分行，P 的后 8 列片段
     */
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
      for (int nj = 0; nj < 8; ++nj) {
        uint32_t v_frag[2];
        const int v_row = pk + (lane & 15);
        const int v_col = nj * 8;
        ampere::ldmatrix_x2_trans(
            v_frag, ampere::shared_u32(&v_smem[v_row * (D + PAD) + v_col]));
        ampere::mma_m16n8k16_f32(o_acc[nj], p_frag, v_frag);
      }
    }
  }

  // Normalize into the now-dead Q shared-memory tile. The accumulator layout
  // is scattered across lanes, so direct stores compile to 32 scalar U16
  // stores per thread. Shared memory performs the layout conversion cheaply;
  // after the CTA barrier, all threads stream contiguous 16-byte vectors to
  // HBM, matching FA2's vectorized output epilogue.
  const float inv_l0 = 1.0f / row_l0;
  const float inv_l1 = 1.0f / row_l1;
#pragma unroll
  for (int nj = 0; nj < 8; ++nj) {
    const int col = nj * 8 + (lane % 4) * 2;
    *reinterpret_cast<uint32_t *>(&q_smem[row0 * (D + PAD) + col]) =
        pack_half2(o_acc[nj][0] * inv_l0, o_acc[nj][1] * inv_l0);
    *reinterpret_cast<uint32_t *>(&q_smem[row1 * (D + PAD) + col]) =
        pack_half2(o_acc[nj][2] * inv_l1, o_acc[nj][3] * inv_l1);
  }
  __syncthreads();

#pragma unroll
  for (int linear = tid * 8; linear < BR * D; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    if (row < q_valid_rows) {
      const uint4 output =
          *reinterpret_cast<const uint4 *>(&q_smem[row * (D + PAD) + col]);
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

void flash_attn_optimized(TensorView q, TensorView k, TensorView v,
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
        << "flash_attn_optimized: expected [B,H,N,64] with positive B/H "
           "and N divisible by 64";
  }
  if (batch64 > 65535 || heads64 > 65535 || seqlen64 > 2147483647LL) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn_optimized: shape exceeds CUDA grid limits";
  }

  const int batch = static_cast<int>(batch64);
  const int heads = static_cast<int>(heads64);
  const int seqlen = static_cast<int>(seqlen64);
  const bool causal = causal64 != 0;

  static const bool smem_opted_in = [] {
    CUDA_LEARN_CHECK(
        cudaFuncSetAttribute(flash_attention2_fwd_d64_registers,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(OPT_SMEM_BYTES)));
    return true;
  }();
  (void)smem_opted_in;

  const dim3 grid((seqlen + BR - 1) / BR, heads, batch);
  flash_attention2_fwd_d64_registers<<<grid, THREADS, OPT_SMEM_BYTES,
                                       get_stream(q)>>>(
      static_cast<const half *>(q.data_ptr()),
      static_cast<const half *>(k.data_ptr()),
      static_cast<const half *>(v.data_ptr()),
      static_cast<half *>(out.data_ptr()), heads, seqlen, causal);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.flash_attn_optimized", flash_attn_optimized);
