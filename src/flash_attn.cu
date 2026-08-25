#include "ampere_primitives.cuh"
#include "ffi_common.h"
using tvm::ffi::TensorView;

// A deliberately fixed educational kernel. One CTA owns 64 query rows and
// streams over K/V in 64-row tiles. Four warps split Q: warp w owns rows
// [16*w, 16*w+16). This is the central FA2 work-partitioning idea.
constexpr int D = 64;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int PAD = 8;
constexpr int THREADS = 128;

constexpr int Q_ELEMS = BR * (D + PAD);
constexpr int KV_ELEMS = BC * (D + PAD);
constexpr int SCORE_ELEMS = BR * BC;
constexpr int PROB_ELEMS = BR * (BC + PAD);
constexpr int OUT_ELEMS = BR * D;
constexpr size_t SMEM_BYTES =
    (Q_ELEMS + KV_ELEMS + PROB_ELEMS) * sizeof(half) +
    (SCORE_ELEMS + OUT_ELEMS + 2 * BR) * sizeof(float);

/**
 * @brief row-major fp16, N must be divisible by 64.
 * @param  q                [B,H,N,64]
 * @param  k                [B,H,N,64]
 * @param  v                [B,H,N,64]
 * @param  o                [B,H,N,64]
 * @param  heads            number of attention heads
 * @param  seqlen           sequence length, a positive multiple of 64
 * @return __global__
 */
__global__ void flash_attention2_fwd_d64(const half *__restrict__ q,
                                         const half *__restrict__ k,
                                         const half *__restrict__ v,
                                         half *__restrict__ o, int heads,
                                         int seqlen) {
  extern __shared__ __align__(16) unsigned char storage[];
  half *q_smem = reinterpret_cast<half *>(storage);
  half *kv_smem = q_smem + Q_ELEMS;
  float *score_smem = reinterpret_cast<float *>(kv_smem + KV_ELEMS);
  half *prob_smem = reinterpret_cast<half *>(score_smem + SCORE_ELEMS);
  float *out_smem = reinterpret_cast<float *>(prob_smem + PROB_ELEMS);
  float *row_m = out_smem + OUT_ELEMS;
  float *row_l = row_m + BR;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int q_tile = blockIdx.x * BR;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const size_t bh_base = static_cast<size_t>(batch) * heads * seqlen * D +
                         static_cast<size_t>(head) * seqlen * D;

  // hbm(q) -> smem(q)
  // 4096 half values / (128 threads * 8 half per copy) = 4 copies per thread.
#pragma unroll
  for (int linear = tid * 8; linear < BR * D; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    ampere::cp_async_16(ampere::shared_u32(&q_smem[row * (D + PAD) + col]),
                        &q[bh_base + (q_tile + row) * D + col]);
  }
  ampere::cp_async_commit();

  // init online-softmax state
#pragma unroll
  for (int linear = tid; linear < OUT_ELEMS; linear += THREADS) {
    out_smem[linear] = 0.0f;
  }
  if (tid < BR) {
    row_m[tid] = -1e20f;
    row_l[tid] = 0.0f;
  }
  ampere::cp_async_wait_all();
  __syncthreads(); // cp.async completion is not a CTA visibility barrier.
  const float softmax_scale = rsqrtf(static_cast<float>(D));
  for (int kv_tile = 0; kv_tile < seqlen; kv_tile += BC) {
    // 1. Q @ K^T
    // hbm(k) -> smem(k)
#pragma unroll
    for (int linear = tid * 8; linear < BC * D; linear += THREADS * 8) {
      const int row = linear / D;
      const int col = linear % D;
      ampere::cp_async_16(ampere::shared_u32(&kv_smem[row * (D + PAD) + col]),
                          &k[bh_base + (kv_tile + row) * D + col]);
    }
    ampere::cp_async_commit();
    ampere::cp_async_wait_all();
    __syncthreads(); // cp.async completion is not a CTA visibility barrier.

    // 8 次迭代，每次每个 lane 得到 4 个 accumulator registers
    float s_frag[8][4] = {};
    // smem->reg: 每次加载q 16*16, k 16 * 8
#pragma unroll
    for (int kd = 0; kd < D; kd += 16) {
      uint32_t q_frag[4];
      const int q_row = warp * 16 + (lane & 15);
      const int q_col = kd + (lane >> 4) * 8;
      ampere::ldmatrix_x4(
          q_frag, ampere::shared_u32(&q_smem[q_row * (D + PAD) + q_col]));

#pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        uint32_t k_frag[2];
        const int k_row = nj * 8 + (lane & 7);
        const int k_col = kd + ((lane >> 3) & 1) * 8;
        ampere::ldmatrix_x2(
            k_frag, ampere::shared_u32(&kv_smem[k_row * (D + PAD) + k_col]));
        ampere::mma_m16n8k16_f32(s_frag[nj], q_frag, k_frag);
      }
    }

    // reg->smem
#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      const int row0 = warp * 16 + lane / 4;
      const int row1 = row0 + 8;
      const int col = nj * 8 + (lane % 4) * 2;
      score_smem[row0 * BC + col] = s_frag[nj][0];
      score_smem[row0 * BC + col + 1] = s_frag[nj][1];
      score_smem[row1 * BC + col] = s_frag[nj][2];
      score_smem[row1 * BC + col + 1] = s_frag[nj][3];
    }
    __syncthreads();

    // 2. online softmax
    if (tid < BR) {
      const int row = tid;
      float tile_max = -1e20f;
#pragma unroll
      for (int col = 0; col < BC; ++col) {
        tile_max = fmaxf(tile_max, score_smem[row * BC + col] * softmax_scale);
      }
      const float old_m = row_m[row];
      const float new_m = fmaxf(old_m, tile_max);
      const float alpha = row_l[row] == 0.0f ? 0.0f : __expf(old_m - new_m);
      float tile_sum = 0.0f;

#pragma unroll
      for (int col = 0; col < BC; ++col) {
        const float p =
            __expf(score_smem[row * BC + col] * softmax_scale - new_m);
        prob_smem[row * (BC + PAD) + col] = __float2half_rn(p);
        tile_sum += p;
      }

#pragma unroll
      for (int col = 0; col < D; ++col) {
        out_smem[row * D + col] *= alpha;
      }
      row_m[row] = new_m;
      row_l[row] = row_l[row] * alpha + tile_sum;
    }
    __syncthreads();

    // 3. P @ V
#pragma unroll
    for (int linear = tid * 8; linear < BC * D; linear += THREADS * 8) {
      const int row = linear / D;
      const int col = linear % D;
      ampere::cp_async_16(ampere::shared_u32(&kv_smem[row * (D + PAD) + col]),
                          &v[bh_base + (kv_tile + row) * D + col]);
    }
    ampere::cp_async_commit();
    ampere::cp_async_wait_all();
    __syncthreads(); // cp.async completion is not a CTA visibility barrier.

    float o_frag[8][4] = {};
    // // Each warp computes T_w[16,64] = P_w[16,64] @ V_tile[64,64].
#pragma unroll
    for (int pk = 0; pk < BC; pk += 16) {
      uint32_t p_frag[4];
      const int p_row = warp * 16 + (lane & 15);
      const int p_col = pk + (lane >> 4) * 8;
      ampere::ldmatrix_x4(
          p_frag, ampere::shared_u32(&prob_smem[p_row * (BC + PAD) + p_col]));

#pragma unroll
      for (int nj = 0; nj < 8; ++nj) {
        uint32_t v_frag[2];
        // V is row-major [K,D], while mma's B fragment is column-major.
        // Transpose the 16x8 shared-memory slice during ldmatrix.
        const int v_row = pk + (lane & 15);
        const int v_col = nj * 8;
        ampere::ldmatrix_x2_trans(
            v_frag, ampere::shared_u32(&kv_smem[v_row * (D + PAD) + v_col]));
        ampere::mma_m16n8k16_f32(o_frag[nj], p_frag, v_frag);
      }
    }

#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
      const int row0 = warp * 16 + lane / 4;
      const int row1 = row0 + 8;
      const int col = nj * 8 + (lane % 4) * 2;
      out_smem[row0 * D + col] += o_frag[nj][0];
      out_smem[row0 * D + col + 1] += o_frag[nj][1];
      out_smem[row1 * D + col] += o_frag[nj][2];
      out_smem[row1 * D + col + 1] += o_frag[nj][3];
    }
    __syncthreads();
  }
#pragma unroll
  for (int linear = tid; linear < BR * D; linear += THREADS) {
    const int row = linear / D;
    const int col = linear % D;
    o[bh_base + static_cast<size_t>(q_tile + row) * D + col] =
        __float2half_rn(out_smem[row * D + col] / row_l[row]);
  }
}

namespace {

void check_fp16_tensor(const TensorView &tensor, const char *name) {
  if (tensor.data_ptr() == nullptr) {
    TVM_FFI_THROW(RuntimeError) << name << ": null tensor";
  }
  if (tensor.ndim() != 4) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected [B,H,N,64], got " << tensor.ndim() << " dims";
  }
  if (tensor.dtype().code != kDLFloat || tensor.dtype().bits != 16) {
    TVM_FFI_THROW(RuntimeError) << name << ": expected float16";
  }
  if (tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError) << name << ": expected a CUDA tensor";
  }
}

void check_same_shape(const TensorView &tensor, const TensorView &q,
                      const char *name) {
  for (int axis = 0; axis < 4; ++axis) {
    if (dim(tensor, axis) != dim(q, axis)) {
      TVM_FFI_THROW(RuntimeError)
          << name << ": expected the same shape as q [B,H,N,64]";
    }
  }
  if (tensor.device().device_id != q.device().device_id) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected the same CUDA device as q";
  }
}

} // namespace

void flash_attn(TensorView q, TensorView k, TensorView v, TensorView out) {
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
  if (dim(q, 3) != D) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn: head dimension must be " << D << ", got " << dim(q, 3);
  }
  if (batch64 <= 0 || heads64 <= 0 || seqlen64 <= 0 || seqlen64 % BR != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn: B and H must be positive, and N must be a positive "
           "multiple of "
        << BR;
  }
  if (batch64 > 65535 || heads64 > 65535 || seqlen64 > 2147483647LL) {
    TVM_FFI_THROW(RuntimeError) << "flash_attn: shape exceeds CUDA grid limits";
  }

  // This kernel needs more than CUDA's legacy 48 KiB dynamic-SMEM limit.
  // Static initialization makes the opt-in cost a first-call cost, outside the
  // benchmark timing window, while every launch still uses torch's current stream.
  static const bool smem_opted_in = [] {
    CUDA_LEARN_CHECK(cudaFuncSetAttribute(
        flash_attention2_fwd_d64,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SMEM_BYTES)));
    return true;
  }();
  (void)smem_opted_in;

  const int batch = static_cast<int>(batch64);
  const int heads = static_cast<int>(heads64);
  const int seqlen = static_cast<int>(seqlen64);
  const dim3 grid(seqlen / BR, heads, batch);
  flash_attention2_fwd_d64<<<grid, THREADS, SMEM_BYTES, get_stream(q)>>>(
      static_cast<const half *>(q.data_ptr()),
      static_cast<const half *>(k.data_ptr()),
      static_cast<const half *>(v.data_ptr()),
      static_cast<half *>(out.data_ptr()), heads, seqlen);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.flash_attn", flash_attn);
