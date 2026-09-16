#include "ampere_primitives.cuh"
#include "ffi_common.h"

#include <cuda_bf16.h>

#include <cmath>

using tvm::ffi::TensorView;

namespace {

// The DeepSeek-V3 dense MLA decode shape.  In MLA mode K and V share one
// latent-cache row: all 576 elements participate in QK, while the first 512
// elements are reused as V.
constexpr int kHeads = 128;
constexpr int kHeadDimQK = 576;
constexpr int kHeadDimV = 512;
constexpr int kPageSize = 64;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPageElements = kPageSize * kHeadDimQK;
constexpr size_t kPageBytes = kPageElements * sizeof(__nv_bfloat16);

__device__ __forceinline__ float warp_sum_all(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_max_all(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value,
                  __shfl_down_sync(0xffffffffu, value, offset));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

// Readability baseline: one 256-thread CTA computes one query head.  Every
// token performs a block-wide QK reduction, immediately followed by the
// online-softmax/PV update.  It is deliberately simple and does not attempt
// to share an MQA cache row between query heads.
__global__ void mla_decode_native_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ kv_cache,
    const int *__restrict__ block_table,
    const int *__restrict__ cache_seqlens,
    __nv_bfloat16 *__restrict__ out, float *__restrict__ lse,
    int max_pages, float softmax_scale) {
  __shared__ __align__(16) __nv_bfloat16 q_shared[kHeadDimQK];
  __shared__ float reduction[kThreads];
  // m, l, alpha, beta for the online softmax recurrence.
  __shared__ float state[4];

  const int tid = threadIdx.x;
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int q_base = (batch * kHeads + head) * kHeadDimQK;

  for (int d = tid; d < kHeadDimQK; d += kThreads) {
    q_shared[d] = q[q_base + d];
  }
  if (tid == 0) {
    state[0] = -INFINITY;
    state[1] = 0.0f;
  }
  __syncthreads();

  // Each thread owns at most two of the 512 output dimensions.
  float out_acc0 = 0.0f;
  float out_acc1 = 0.0f;
  const int seqlen = cache_seqlens[batch];

  for (int token = 0; token < seqlen; ++token) {
    const int logical_page = token / kPageSize;
    const int page_offset = token % kPageSize;
    const int physical_page = block_table[batch * max_pages + logical_page];
    const __nv_bfloat16 *kv =
        kv_cache + (physical_page * kPageSize + page_offset) * kHeadDimQK;

    float partial = 0.0f;
#pragma unroll
    for (int d = tid; d < kHeadDimQK; d += kThreads) {
      partial = fmaf(__bfloat162float(q_shared[d]),
                     __bfloat162float(kv[d]), partial);
    }
    reduction[tid] = partial;
    __syncthreads();

#pragma unroll
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float score = reduction[0] * softmax_scale;
      const float old_m = state[0];
      const float old_l = state[1];
      const float new_m = fmaxf(old_m, score);
      const float alpha = old_l == 0.0f ? 0.0f : __expf(old_m - new_m);
      const float beta = __expf(score - new_m);
      state[0] = new_m;
      state[1] = alpha * old_l + beta;
      state[2] = alpha;
      state[3] = beta;
    }
    __syncthreads();

    if (tid < kHeadDimV) {
      out_acc0 = state[2] * out_acc0 +
                 state[3] * __bfloat162float(kv[tid]);
    }
    if (tid + kThreads < kHeadDimV) {
      out_acc1 = state[2] * out_acc1 +
                 state[3] * __bfloat162float(kv[tid + kThreads]);
    }
    // The next iteration may replace state and reduction.  This barrier also
    // keeps the shared-state lifetime obvious in the teaching baseline.
    __syncthreads();
  }

  const float inv_l = 1.0f / state[1];
  const int out_base = (batch * kHeads + head) * kHeadDimV;
  if (tid < kHeadDimV) {
    out[out_base + tid] = __float2bfloat16_rn(out_acc0 * inv_l);
  }
  if (tid + kThreads < kHeadDimV) {
    out[out_base + tid + kThreads] =
        __float2bfloat16_rn(out_acc1 * inv_l);
  }
  if (tid == 0) {
    lse[batch * kHeads + head] = state[0] + logf(state[1]);
  }
}

// MQA-aware version: one 8-warp CTA computes eight query heads.  A complete
// 64x576 paged-cache tile is copied once with cp.async and reused by all eight
// warps.  A warp owns one head, keeps Q and the 512-D output in registers, and
// performs one online-softmax update per 64-token page (rather than per token).
__global__ void mla_decode_optimized_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ kv_cache,
    const int *__restrict__ block_table,
    const int *__restrict__ cache_seqlens,
    __nv_bfloat16 *__restrict__ out, float *__restrict__ lse,
    int max_pages, float softmax_scale) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto *page = reinterpret_cast<__nv_bfloat16 *>(shared_raw);

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int batch = blockIdx.y;
  const int head = blockIdx.x * kWarps + warp;
  const int q_base = (batch * kHeads + head) * kHeadDimQK;

  // 576 / 32 = 18 Q values per lane; 512 / 32 = 16 output values.
  float q_reg[18];
  float out_acc[16] = {};
#pragma unroll
  for (int i = 0; i < 18; ++i) {
    q_reg[i] = __bfloat162float(q[q_base + lane + i * 32]);
  }

  float row_m = -INFINITY;
  float row_l = 0.0f;
  const int seqlen = cache_seqlens[batch];
  const int num_pages = (seqlen + kPageSize - 1) / kPageSize;

  for (int logical_page = 0; logical_page < num_pages; ++logical_page) {
    const int physical_page =
        block_table[batch * max_pages + logical_page];
    const __nv_bfloat16 *global_page =
        kv_cache + physical_page * kPageElements;

    // Both page bases and every 8-bfloat16 chunk are naturally 16-B aligned.
    // 36,864 elements / (256 threads * 8 elements) = 18 copies per thread.
#pragma unroll
    for (int linear = tid * 8; linear < kPageElements;
         linear += kThreads * 8) {
      ampere::cp_async_16(ampere::shared_u32(page + linear),
                          global_page + linear);
    }
    ampere::cp_async_commit();
    ampere::cp_async_wait_all();
    __syncthreads();

    const int valid_tokens = min(kPageSize, seqlen - logical_page * kPageSize);
    float scores[2] = {-INFINITY, -INFINITY};

    // The warp collaborates on one 576-D dot product at a time.  The resulting
    // 64 scores are distributed as two register values per lane.
#pragma unroll
    for (int token = 0; token < kPageSize; ++token) {
      float partial = 0.0f;
#pragma unroll
      for (int i = 0; i < 18; ++i) {
        const int d = lane + i * 32;
        partial = fmaf(q_reg[i],
                       __bfloat162float(page[token * kHeadDimQK + d]),
                       partial);
      }
      const float score = warp_sum_all(partial) * softmax_scale;
      if (lane == (token & 31)) {
        scores[token >> 5] = token < valid_tokens ? score : -INFINITY;
      }
    }

    const float tile_m = warp_max_all(fmaxf(scores[0], scores[1]));
    const float new_m = fmaxf(row_m, tile_m);
    const float alpha = row_l == 0.0f ? 0.0f : __expf(row_m - new_m);
    const float weight0 = __expf(scores[0] - new_m);
    const float weight1 = __expf(scores[1] - new_m);
    const float tile_l = warp_sum_all(weight0 + weight1);

    // Broadcast each score weight from its owner lane and form P@V.  V is the
    // first 512 columns of the same latent-cache row used as K above.
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int d = lane + j * 32;
      float pv = 0.0f;
#pragma unroll
      for (int token = 0; token < kPageSize; ++token) {
        const float local_weight = token < 32 ? weight0 : weight1;
        const float weight =
            __shfl_sync(0xffffffffu, local_weight, token & 31);
        pv = fmaf(weight,
                  __bfloat162float(page[token * kHeadDimQK + d]), pv);
      }
      out_acc[j] = alpha * out_acc[j] + pv;
    }

    row_m = new_m;
    row_l = alpha * row_l + tile_l;
    // No warp may let the next cp.async overwrite a page still used by PV.
    __syncthreads();
  }

  const float inv_l = 1.0f / row_l;
  const int out_base = (batch * kHeads + head) * kHeadDimV;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    out[out_base + lane + j * 32] =
        __float2bfloat16_rn(out_acc[j] * inv_l);
  }
  if (lane == 0) {
    lse[batch * kHeads + head] = row_m + logf(row_l);
  }
}

void check_bf16(const TensorView &tensor, const char *name, int ndim) {
  if (tensor.ndim() != ndim || tensor.dtype().code != kDLBfloat ||
      tensor.dtype().bits != 16 ||
      tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected a " << ndim << "D CUDA bfloat16 tensor";
  }
}

void check_i32(const TensorView &tensor, const char *name, int ndim) {
  if (tensor.ndim() != ndim || tensor.dtype().code != kDLInt ||
      tensor.dtype().bits != 32 ||
      tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected a " << ndim << "D CUDA int32 tensor";
  }
}

void check_shapes(const TensorView &q, const TensorView &kv_cache,
                  const TensorView &block_table,
                  const TensorView &cache_seqlens, const TensorView &out,
                  const TensorView &lse) {
  const int64_t batch = dim(q, 0);
  if (batch <= 0 || dim(q, 1) != kHeads || dim(q, 2) != kHeadDimQK) {
    TVM_FFI_THROW(RuntimeError)
        << "q must have shape [B, 128, 576] with B > 0";
  }
  if (dim(kv_cache, 0) <= 0 || dim(kv_cache, 1) != kPageSize ||
      dim(kv_cache, 2) != kHeadDimQK) {
    TVM_FFI_THROW(RuntimeError)
        << "kv_cache must have shape [num_pages, 64, 576]";
  }
  if (dim(block_table, 0) != batch || dim(block_table, 1) <= 0) {
    TVM_FFI_THROW(RuntimeError)
        << "block_table must have shape [B, max_pages]";
  }
  if (dim(cache_seqlens, 0) != batch) {
    TVM_FFI_THROW(RuntimeError) << "cache_seqlens must have shape [B]";
  }
  if (dim(out, 0) != batch || dim(out, 1) != kHeads ||
      dim(out, 2) != kHeadDimV) {
    TVM_FFI_THROW(RuntimeError) << "out must have shape [B, 128, 512]";
  }
  if (dim(lse, 0) != batch || dim(lse, 1) != kHeads) {
    TVM_FFI_THROW(RuntimeError) << "lse must have shape [B, 128]";
  }
  const int device = q.device().device_id;
  if (kv_cache.device().device_id != device ||
      block_table.device().device_id != device ||
      cache_seqlens.device().device_id != device ||
      out.device().device_id != device || lse.device().device_id != device) {
    TVM_FFI_THROW(RuntimeError) << "all MLA tensors must be on one CUDA device";
  }
}

template <bool Optimized>
void mla_decode(TensorView q, TensorView kv_cache, TensorView block_table,
                TensorView cache_seqlens, TensorView out, TensorView lse,
                double softmax_scale) {
  check_bf16(q, "q", 3);
  check_bf16(kv_cache, "kv_cache", 3);
  check_i32(block_table, "block_table", 2);
  check_i32(cache_seqlens, "cache_seqlens", 1);
  check_bf16(out, "out", 3);
  check_tensor(lse, "lse", 2);
  check_shapes(q, kv_cache, block_table, cache_seqlens, out, lse);
  if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0) {
    TVM_FFI_THROW(RuntimeError) << "softmax_scale must be finite and positive";
  }

  const int batch = static_cast<int>(dim(q, 0));
  const int max_pages = static_cast<int>(dim(block_table, 1));
  const auto *q_ptr =
      reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr());
  const auto *kv_ptr =
      reinterpret_cast<const __nv_bfloat16 *>(kv_cache.data_ptr());
  const auto *table_ptr = reinterpret_cast<const int *>(block_table.data_ptr());
  const auto *length_ptr =
      reinterpret_cast<const int *>(cache_seqlens.data_ptr());
  auto *out_ptr = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
  auto *lse_ptr = reinterpret_cast<float *>(lse.data_ptr());
  const cudaStream_t stream = get_stream(q);

  if constexpr (Optimized) {
    CUDA_LEARN_CHECK(cudaFuncSetAttribute(
        mla_decode_optimized_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kPageBytes)));
    mla_decode_optimized_kernel<<<dim3(kHeads / kWarps, batch), kThreads,
                                  kPageBytes, stream>>>(
        q_ptr, kv_ptr, table_ptr, length_ptr, out_ptr, lse_ptr, max_pages,
        static_cast<float>(softmax_scale));
  } else {
    mla_decode_native_kernel<<<dim3(kHeads, batch), kThreads, 0, stream>>>(
        q_ptr, kv_ptr, table_ptr, length_ptr, out_ptr, lse_ptr, max_pages,
        static_cast<float>(softmax_scale));
  }
  CUDA_LEARN_CHECK(cudaGetLastError());
}

void mla_decode_native(TensorView q, TensorView kv_cache,
                       TensorView block_table, TensorView cache_seqlens,
                       TensorView out, TensorView lse,
                       double softmax_scale) {
  mla_decode<false>(q, kv_cache, block_table, cache_seqlens, out, lse,
                    softmax_scale);
}

void mla_decode_optimized(TensorView q, TensorView kv_cache,
                          TensorView block_table, TensorView cache_seqlens,
                          TensorView out, TensorView lse,
                          double softmax_scale) {
  mla_decode<true>(q, kv_cache, block_table, cache_seqlens, out, lse,
                   softmax_scale);
}

} // namespace

CUDA_LEARN_REGISTER("cuda_learn.mla_decode_native", mla_decode_native);
CUDA_LEARN_REGISTER("cuda_learn.mla_decode_optimized", mla_decode_optimized);
