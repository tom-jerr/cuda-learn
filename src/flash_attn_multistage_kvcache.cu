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

__host__ __device__ constexpr int tile_offset(int row, int col) {
  return row * D + (col ^ ((row & 7) << 3));
}

struct SharedStorage {
  half q[TILE_ELEMS];
  half k[2][TILE_ELEMS];
  half v[2][TILE_ELEMS];
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

__device__ __forceinline__ void copy_q_async(const half *src, half *dst,
                                             int q_len) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int linear = tid * 8; linear < TILE_ELEMS; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    const int safe_row = min(row, q_len - 1);
    ampere::cp_async_16_zfill(ampere::shared_u32(&dst[tile_offset(row, col)]),
                              &src[safe_row * D + col], row < q_len);
  }
}

__device__ __forceinline__ void copy_cache_tile_async(const half *src,
                                                      half *dst, int tile_start,
                                                      int actual_len,
                                                      int capacity) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int linear = tid * 8; linear < TILE_ELEMS; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    const int cache_row = tile_start + row;
    const int safe_row = min(cache_row, capacity - 1);
    ampere::cp_async_16_zfill(ampere::shared_u32(&dst[tile_offset(row, col)]),
                              &src[safe_row * D + col], cache_row < actual_len);
  }
}

// Continuous-cache teaching subset of FA2 KV-cache attention.
//
// q/out:       [B, Hq, Q, 64], 1 <= Q <= 64
// k/v cache:   [B, Hkv, capacity, 64]
// cache_lens:  [B], length before optional append
// knew/vnew:   [B, Hkv, Q, 64] when AppendKV=true
//
// Cache-only supports GQA. Fused append is restricted to Hq==Hkv by the host
// wrapper so each CTA owns a unique cache slice and no inter-CTA race exists.
template <bool AppendKV, bool Causal>
__global__ __launch_bounds__(THREADS, 2) void flash_attn_kvcache_kernel(
    const half *__restrict__ q, half *__restrict__ k_cache,
    half *__restrict__ v_cache, const int32_t *__restrict__ cache_lens,
    const half *__restrict__ knew, const half *__restrict__ vnew,
    half *__restrict__ out, int heads_q, int heads_kv, int q_len,
    int capacity) {
  __shared__ __align__(16) SharedStorage storage;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int q_head = blockIdx.x;
  const int batch = blockIdx.y;
  const int kv_head = q_head / (heads_q / heads_kv);
  const int row0 = warp * 16 + g;
  const int row1 = row0 + 8;

  const size_t q_base =
      (static_cast<size_t>(batch) * heads_q + q_head) * q_len * D;
  const size_t cache_base =
      (static_cast<size_t>(batch) * heads_kv + kv_head) * capacity * D;
  const size_t new_base =
      (static_cast<size_t>(batch) * heads_kv + kv_head) * q_len * D;

  int cache_len = cache_lens[batch];
  cache_len = max(0, min(cache_len, capacity));
  const int append_len =
      AppendKV ? min(q_len, max(0, capacity - cache_len)) : 0;
  const int actual_len = cache_len + append_len;

  // The official kernel performs the same conceptual operation inside its
  // Append_KV specialization. Vectorized global stores update the cache before
  // this CTA issues cp.async reads from it. cache_lens itself is not mutated.
  if constexpr (AppendKV) {
    for (int linear = tid * 8; linear < q_len * D; linear += THREADS * 8) {
      const int row = linear / D;
      const int col = linear % D;
      if (row < append_len) {
        const uint4 packed_k = *reinterpret_cast<const uint4 *>(
            &knew[new_base + static_cast<size_t>(row) * D + col]);
        const uint4 packed_v = *reinterpret_cast<const uint4 *>(
            &vnew[new_base + static_cast<size_t>(row) * D + col]);
        *reinterpret_cast<uint4 *>(
            &k_cache[cache_base + static_cast<size_t>(cache_len + row) * D +
                     col]) = packed_k;
        *reinterpret_cast<uint4 *>(
            &v_cache[cache_base + static_cast<size_t>(cache_len + row) * D +
                     col]) = packed_v;
      }
    }
    // __syncthreads also orders these global writes for threads in this CTA.
    __syncthreads();
  }

  float o_acc[D / 8][4] = {};
  float row_m0 = -INFINITY;
  float row_m1 = -INFINITY;
  float row_l0 = 0.0f;
  float row_l1 = 0.0f;

  copy_q_async(&q[q_base], storage.q, q_len);
  copy_cache_tile_async(&k_cache[cache_base], storage.k[0], 0, actual_len,
                        capacity);
  copy_cache_tile_async(&v_cache[cache_base], storage.v[0], 0, actual_len,
                        capacity);
  ampere::cp_async_commit();

  const int kv_tiles = (actual_len + BC - 1) / BC;
#pragma unroll 1
  for (int tile = 0; tile < kv_tiles; ++tile) {
    ampere::cp_async_wait_all();
    __syncthreads();

    half *k_stage = storage.k[tile & 1];
    half *v_stage = storage.v[tile & 1];
    if (tile + 1 < kv_tiles) {
      const int write_stage = (tile + 1) & 1;
      const int next_start = (tile + 1) * BC;
      copy_cache_tile_async(&k_cache[cache_base], storage.k[write_stage],
                            next_start, actual_len, capacity);
      copy_cache_tile_async(&v_cache[cache_base], storage.v[write_stage],
                            next_start, actual_len, capacity);
      ampere::cp_async_commit();
    }

    float s_frag[BC / 8][4] = {};
#pragma unroll
    for (int kd = 0; kd < D; kd += 16) {
      uint32_t q_frag[4];
      const int q_row = warp * 16 + (lane & 15);
      const int q_col = kd + (lane >> 4) * 8;
      ampere::ldmatrix_x4(
          q_frag, ampere::shared_u32(&storage.q[tile_offset(q_row, q_col)]));

#pragma unroll
      for (int pair = 0; pair < BC / 16; ++pair) {
        uint32_t k4[4];
        const int matrix = lane >> 3;
        const int row = pair * 16 + (matrix >> 1) * 8 + (lane & 7);
        const int col = kd + (matrix & 1) * 8;
        ampere::ldmatrix_x4(
            k4, ampere::shared_u32(&k_stage[tile_offset(row, col)]));
        mma_m16n8k16(s_frag[2 * pair], q_frag, k4[0], k4[1]);
        mma_m16n8k16(s_frag[2 * pair + 1], q_frag, k4[2], k4[3]);
      }
    }

    const int key_tile = tile * BC;
    const int causal_limit0 = actual_len - q_len + row0;
    const int causal_limit1 = actual_len - q_len + row1;
    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
#pragma unroll
    for (int nj = 0; nj < BC / 8; ++nj) {
#pragma unroll
      for (int item = 0; item < 4; ++item)
        s_frag[nj][item] *= 0.125f;
      const int col = key_tile + nj * 8 + 2 * t;
      if (row0 >= q_len || col >= actual_len || (Causal && col > causal_limit0))
        s_frag[nj][0] = -INFINITY;
      if (row0 >= q_len || col + 1 >= actual_len ||
          (Causal && col + 1 > causal_limit0))
        s_frag[nj][1] = -INFINITY;
      if (row1 >= q_len || col >= actual_len || (Causal && col > causal_limit1))
        s_frag[nj][2] = -INFINITY;
      if (row1 >= q_len || col + 1 >= actual_len ||
          (Causal && col + 1 > causal_limit1))
        s_frag[nj][3] = -INFINITY;
      local_max0 = fmaxf(local_max0, fmaxf(s_frag[nj][0], s_frag[nj][1]));
      local_max1 = fmaxf(local_max1, fmaxf(s_frag[nj][2], s_frag[nj][3]));
    }

    const float tile_max0 = subgroup4_max(local_max0);
    const float tile_max1 = subgroup4_max(local_max1);
    const float new_m0 = fmaxf(row_m0, tile_max0);
    const float new_m1 = fmaxf(row_m1, tile_max1);
    const float alpha0 =
        row_l0 == 0.0f || new_m0 == -INFINITY ? 0.0f : __expf(row_m0 - new_m0);
    const float alpha1 =
        row_l1 == 0.0f || new_m1 == -INFINITY ? 0.0f : __expf(row_m1 - new_m1);

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
      s_frag[nj][0] =
          new_m0 == -INFINITY ? 0.0f : __expf(s_frag[nj][0] - new_m0);
      s_frag[nj][1] =
          new_m0 == -INFINITY ? 0.0f : __expf(s_frag[nj][1] - new_m0);
      s_frag[nj][2] =
          new_m1 == -INFINITY ? 0.0f : __expf(s_frag[nj][2] - new_m1);
      s_frag[nj][3] =
          new_m1 == -INFINITY ? 0.0f : __expf(s_frag[nj][3] - new_m1);
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
#pragma unroll
      for (int pair = 0; pair < D / 16; ++pair) {
        uint32_t v4[4];
        const int matrix = lane >> 3;
        const int row = pk + (matrix & 1) * 8 + (lane & 7);
        const int col = pair * 16 + (matrix >> 1) * 8;
        ldmatrix_x4_trans(v4,
                          ampere::shared_u32(&v_stage[tile_offset(row, col)]));
        mma_m16n8k16(o_acc[2 * pair], p_frag, v4[0], v4[1]);
        mma_m16n8k16(o_acc[2 * pair + 1], p_frag, v4[2], v4[3]);
      }
    }
    __syncthreads();
  }

  // If actual_len == 0, the prologue copies are still pending even though the
  // KV loop did not execute. Drain them before reusing Q shared memory.
  if (kv_tiles == 0) {
    ampere::cp_async_wait_all();
    __syncthreads();
  }

  const float inv_l0 = row_l0 == 0.0f ? 0.0f : 1.0f / row_l0;
  const float inv_l1 = row_l1 == 0.0f ? 0.0f : 1.0f / row_l1;
#pragma unroll
  for (int nj = 0; nj < D / 8; ++nj) {
    const int col = nj * 8 + 2 * t;
    *reinterpret_cast<uint32_t *>(&storage.q[tile_offset(row0, col)]) =
        pack_half2(o_acc[nj][0] * inv_l0, o_acc[nj][1] * inv_l0);
    *reinterpret_cast<uint32_t *>(&storage.q[tile_offset(row1, col)]) =
        pack_half2(o_acc[nj][2] * inv_l1, o_acc[nj][3] * inv_l1);
  }
  __syncthreads();

#pragma unroll
  for (int linear = tid * 8; linear < TILE_ELEMS; linear += THREADS * 8) {
    const int row = linear / D;
    const int col = linear % D;
    if (row < q_len) {
      const uint4 packed =
          *reinterpret_cast<const uint4 *>(&storage.q[tile_offset(row, col)]);
      *reinterpret_cast<uint4 *>(
          &out[q_base + static_cast<size_t>(row) * D + col]) = packed;
    }
  }
}

void check_fp16_4d(const TensorView &tensor, const char *name) {
  if (tensor.data_ptr() == nullptr || tensor.ndim() != 4 ||
      tensor.dtype().code != kDLFloat || tensor.dtype().bits != 16 ||
      tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected a non-null float16 CUDA 4D tensor";
  }
}

void check_i32_1d(const TensorView &tensor, const char *name) {
  if (tensor.data_ptr() == nullptr || tensor.ndim() != 1 ||
      tensor.dtype().code != kDLInt || tensor.dtype().bits != 32 ||
      tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected a non-null int32 CUDA 1D tensor";
  }
}

void check_device(const TensorView &tensor, const TensorView &q,
                  const char *name) {
  if (tensor.device().device_id != q.device().device_id) {
    TVM_FFI_THROW(RuntimeError) << name << ": expected the same device as q";
  }
}

struct KVShape {
  int batch;
  int heads_q;
  int heads_kv;
  int q_len;
  int capacity;
};

KVShape check_common(TensorView q, TensorView k_cache, TensorView v_cache,
                     TensorView cache_lens, TensorView out) {
  check_fp16_4d(q, "q");
  check_fp16_4d(k_cache, "k_cache");
  check_fp16_4d(v_cache, "v_cache");
  check_fp16_4d(out, "out");
  check_i32_1d(cache_lens, "cache_lens");
  check_device(k_cache, q, "k_cache");
  check_device(v_cache, q, "v_cache");
  check_device(cache_lens, q, "cache_lens");
  check_device(out, q, "out");

  const int64_t batch = dim(q, 0);
  const int64_t heads_q = dim(q, 1);
  const int64_t q_len = dim(q, 2);
  const int64_t heads_kv = dim(k_cache, 1);
  const int64_t capacity = dim(k_cache, 2);
  if (batch <= 0 || heads_q <= 0 || heads_kv <= 0 || q_len <= 0 || q_len > BR ||
      capacity <= 0 || dim(q, 3) != D || dim(k_cache, 0) != batch ||
      dim(k_cache, 3) != D || dim(v_cache, 0) != batch ||
      dim(v_cache, 1) != heads_kv || dim(v_cache, 2) != capacity ||
      dim(v_cache, 3) != D || dim(cache_lens, 0) != batch ||
      dim(out, 0) != batch || dim(out, 1) != heads_q || dim(out, 2) != q_len ||
      dim(out, 3) != D || heads_q % heads_kv != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn_multistage_kvcache: expected q/out [B,Hq,Q,64], "
           "cache [B,Hkv,capacity,64], lens [B], 1<=Q<=64 and Hq%Hkv==0";
  }
  if (batch > 65535 || heads_q > 65535 || capacity > 2147483647LL) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn_multistage_kvcache: shape exceeds CUDA limits";
  }
  return {static_cast<int>(batch), static_cast<int>(heads_q),
          static_cast<int>(heads_kv), static_cast<int>(q_len),
          static_cast<int>(capacity)};
}

template <bool AppendKV>
void launch(TensorView q, TensorView k_cache, TensorView v_cache,
            TensorView cache_lens, const half *knew, const half *vnew,
            TensorView out, int64_t causal64, const KVShape &shape) {
  const dim3 grid(shape.heads_q, shape.batch);
  cudaStream_t stream = get_stream(q);
  if (causal64 != 0) {
    flash_attn_kvcache_kernel<AppendKV, true><<<grid, THREADS, 0, stream>>>(
        static_cast<const half *>(q.data_ptr()),
        static_cast<half *>(k_cache.data_ptr()),
        static_cast<half *>(v_cache.data_ptr()),
        static_cast<const int32_t *>(cache_lens.data_ptr()), knew, vnew,
        static_cast<half *>(out.data_ptr()), shape.heads_q, shape.heads_kv,
        shape.q_len, shape.capacity);
  } else {
    flash_attn_kvcache_kernel<AppendKV, false><<<grid, THREADS, 0, stream>>>(
        static_cast<const half *>(q.data_ptr()),
        static_cast<half *>(k_cache.data_ptr()),
        static_cast<half *>(v_cache.data_ptr()),
        static_cast<const int32_t *>(cache_lens.data_ptr()), knew, vnew,
        static_cast<half *>(out.data_ptr()), shape.heads_q, shape.heads_kv,
        shape.q_len, shape.capacity);
  }
  CUDA_LEARN_CHECK(cudaGetLastError());
}

} // namespace

void flash_attn_multistage_kvcache(TensorView q, TensorView k_cache,
                                   TensorView v_cache, TensorView cache_lens,
                                   TensorView out, int64_t causal64) {
  const KVShape shape = check_common(q, k_cache, v_cache, cache_lens, out);
  launch<false>(q, k_cache, v_cache, cache_lens, nullptr, nullptr, out,
                causal64, shape);
}

void flash_attn_multistage_kvcache_append(TensorView q, TensorView k_cache,
                                          TensorView v_cache,
                                          TensorView cache_lens,
                                          TensorView knew, TensorView vnew,
                                          TensorView out, int64_t causal64) {
  const KVShape shape = check_common(q, k_cache, v_cache, cache_lens, out);
  check_fp16_4d(knew, "knew");
  check_fp16_4d(vnew, "vnew");
  check_device(knew, q, "knew");
  check_device(vnew, q, "vnew");
  if (shape.heads_q != shape.heads_kv || dim(knew, 0) != shape.batch ||
      dim(knew, 1) != shape.heads_kv || dim(knew, 2) != shape.q_len ||
      dim(knew, 3) != D || dim(vnew, 0) != shape.batch ||
      dim(vnew, 1) != shape.heads_kv || dim(vnew, 2) != shape.q_len ||
      dim(vnew, 3) != D) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn_multistage_kvcache_append: expected knew/vnew "
           "[B,Hkv,Q,64] and fused append currently requires Hq==Hkv";
  }
  launch<true>(q, k_cache, v_cache, cache_lens,
               static_cast<const half *>(knew.data_ptr()),
               static_cast<const half *>(vnew.data_ptr()), out, causal64,
               shape);
}

CUDA_LEARN_REGISTER("cuda_learn.flash_attn_multistage_kvcache",
                    flash_attn_multistage_kvcache);
CUDA_LEARN_REGISTER("cuda_learn.flash_attn_multistage_kvcache_append",
                    flash_attn_multistage_kvcache_append);
