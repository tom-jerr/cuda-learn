#include "ffi_common.h"

#include <cstdint>
#include <cuda_bf16.h>

#if defined(CUDA_LEARN_HAS_SM90A)
#include "hopper_primitives.cuh"
#include <cuda.h>
#endif

using tvm::ffi::TensorView;

namespace {

constexpr int FA3_D = 64;
#if defined(CUDA_LEARN_HAS_SM90A)
constexpr int FA3_Q_ROWS = 64;
constexpr int FA3_KV_ROWS = 64;
constexpr int FA3_CONSUMERS = 2;
constexpr int FA3_STAGES = 2;
constexpr int FA3_THREADS = 384;
constexpr int FA3_TILE_ELEMS = FA3_Q_ROWS * FA3_D;
constexpr int FA3_TILE_BYTES = FA3_TILE_ELEMS * sizeof(__nv_bfloat16);
constexpr size_t FA3_SMEM_BYTES =
    (FA3_CONSUMERS + 2 * FA3_STAGES) * FA3_TILE_BYTES;
#endif

void check_fa3_bf16(const TensorView &tensor, const char *name) {
  if (tensor.data_ptr() == nullptr || tensor.ndim() != 4 ||
      tensor.dtype().code != kDLBfloat || tensor.dtype().bits != 16 ||
      tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected a non-null bfloat16 CUDA [B,H,N,64] tensor";
  }
}

void check_fa3_same_shape(const TensorView &tensor, const TensorView &q,
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

#if defined(CUDA_LEARN_HAS_SM90A)

struct HopperFa3Params {
  CUtensorMap q_map;
  CUtensorMap k_map;
  CUtensorMap v_map;
  CUtensorMap o_map;
  int seqlen;
  int causal;
};

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  union Bits16 {
    __nv_bfloat16 value;
    uint16_t bits;
  } a, b;
  a.value = __float2bfloat16_rn(lo);
  b.value = __float2bfloat16_rn(hi);
  return static_cast<uint32_t>(a.bits) | (static_cast<uint32_t>(b.bits) << 16);
}

__device__ __forceinline__ float subgroup4_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1, 4));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2, 4));
  return value;
}

__device__ __forceinline__ float subgroup4_sum(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, 1, 4);
  value += __shfl_xor_sync(0xffffffffu, value, 2, 4);
  return value;
}

struct OnlineSoftmax {
  float m0 = -INFINITY;
  float m1 = -INFINITY;
  float l0 = 0.0f;
  float l1 = 0.0f;
};

struct RowScale {
  float row0;
  float row1;
};

// WGMMA's FP32 m64n64 accumulator gives each lane two rows.  In every 16
// columns, registers {0,1,4,5} belong to row0 and {2,3,6,7} to row1.
__device__ __forceinline__ RowScale softmax_inplace(hopper::Accum64x64 &scores,
                                                    OnlineSoftmax &state,
                                                    int q_start, int kv_start,
                                                    bool causal) {
  constexpr float SCALE_LOG2E = 0.125f * 1.4426950408889634f;
  const int wg_lane = threadIdx.x & 127;
  const int warp = wg_lane >> 5;
  const int lane = wg_lane & 31;
  const int lane4 = lane & 3;
  const int row0 = q_start + warp * 16 + lane / 4;
  const int row1 = row0 + 8;

  float local_max0 = -INFINITY;
  float local_max1 = -INFINITY;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int in_tile = i & 7;
    const bool bottom = (in_tile & 2) != 0;
    const int col = kv_start + (i >> 3) * 16 + ((in_tile & 4) ? 8 : 0) +
                    lane4 * 2 + (in_tile & 1);
    float value = scores.x[i] * SCALE_LOG2E;
    if (causal && col > (bottom ? row1 : row0)) {
      value = -INFINITY;
    }
    scores.x[i] = value;
    if (bottom) {
      local_max1 = fmaxf(local_max1, value);
    } else {
      local_max0 = fmaxf(local_max0, value);
    }
  }

  const float tile_max0 = subgroup4_max(local_max0);
  const float tile_max1 = subgroup4_max(local_max1);
  const float new_m0 = fmaxf(state.m0, tile_max0);
  const float new_m1 = fmaxf(state.m1, tile_max1);
  const float alpha0 = state.l0 == 0.0f ? 0.0f : exp2f(state.m0 - new_m0);
  const float alpha1 = state.l1 == 0.0f ? 0.0f : exp2f(state.m1 - new_m1);

  float local_sum0 = 0.0f;
  float local_sum1 = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const bool bottom = ((i & 7) & 2) != 0;
    const float probability = exp2f(scores.x[i] - (bottom ? new_m1 : new_m0));
    scores.x[i] = probability;
    if (bottom) {
      local_sum1 += probability;
    } else {
      local_sum0 += probability;
    }
  }

  state.l0 = alpha0 * state.l0 + subgroup4_sum(local_sum0);
  state.l1 = alpha1 * state.l1 + subgroup4_sum(local_sum1);
  state.m0 = new_m0;
  state.m1 = new_m1;
  return {alpha0, alpha1};
}

__device__ __forceinline__ void scale_rows(hopper::Accum64x64 &tile,
                                           RowScale scale) {
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    tile.x[i] *= (((i & 7) & 2) != 0) ? scale.row1 : scale.row0;
  }
}

__device__ __forceinline__ void
issue_qk(hopper::Accum64x64 &scores, const void *q_smem, const void *k_smem) {
  hopper::wgmma_fence(scores);
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    hopper::wgmma_ss_qk(scores, hopper::k_major_descriptor(q_smem, chunk),
                        hopper::k_major_descriptor(k_smem, chunk), chunk != 0);
  }
  hopper::wgmma_commit_group();
}

__device__ __forceinline__ void issue_pv(hopper::Accum64x64 &output,
                                         const hopper::Accum64x64 &probability,
                                         const void *v_smem) {
  hopper::wgmma_fence(output);
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    const int base = chunk * 8;
    const uint32_t p[4] = {
        pack_bf16(probability.x[base + 0], probability.x[base + 1]),
        pack_bf16(probability.x[base + 2], probability.x[base + 3]),
        pack_bf16(probability.x[base + 4], probability.x[base + 5]),
        pack_bf16(probability.x[base + 6], probability.x[base + 7]),
    };
    hopper::wgmma_rs_pv(output, p, hopper::mn_major_descriptor(v_smem, chunk));
  }
  hopper::wgmma_commit_group();
}

__device__ __forceinline__ void pingpong_before(int consumer, int sequence,
                                                hopper::MBarrier *turn) {
  if (consumer == 0) {
    if (sequence != 0) {
      hopper::mbarrier_wait(&turn[0], (sequence - 1) & 1);
    }
  } else {
    hopper::mbarrier_wait(&turn[1], sequence & 1);
  }
}

__device__ __forceinline__ void pingpong_after(int consumer,
                                               hopper::MBarrier *turn) {
  if ((threadIdx.x & 127) == 0) {
    hopper::mbarrier_arrive(&turn[1 - consumer]);
  }
}

template <bool NextIsScore1>
__device__ __forceinline__ void
pipeline_transition(int tile, int consumer, int q_start, int &issue_sequence,
                    __nv_bfloat16 *q_smem, __nv_bfloat16 **k_smem,
                    __nv_bfloat16 **v_smem, hopper::MBarrier *kv_ready,
                    hopper::MBarrier *stage_empty, hopper::MBarrier *turn,
                    hopper::Accum64x64 &score0, hopper::Accum64x64 &score1,
                    hopper::Accum64x64 &output, OnlineSoftmax &online,
                    bool causal) {
  const int next_stage = tile & (FA3_STAGES - 1);
  const int cur_stage = (tile - 1) & (FA3_STAGES - 1);
  hopper::mbarrier_wait(&kv_ready[next_stage], (tile / FA3_STAGES) & 1);

  if constexpr (NextIsScore1) {
    pingpong_before(consumer, issue_sequence, turn);
    issue_qk(score1, q_smem, k_smem[next_stage]);
    pingpong_after(consumer, turn);
    ++issue_sequence;

    pingpong_before(consumer, issue_sequence, turn);
    issue_pv(output, score0, v_smem[cur_stage]);
    pingpong_after(consumer, turn);
    ++issue_sequence;

    hopper::wgmma_wait_group<1>();
    const RowScale alpha =
        softmax_inplace(score1, online, q_start, tile * FA3_KV_ROWS, causal);
    hopper::wgmma_wait_group<0>();
    if ((threadIdx.x & 127) == 0) {
      hopper::mbarrier_arrive(&stage_empty[cur_stage]);
    }
    scale_rows(output, alpha);
  } else {
    pingpong_before(consumer, issue_sequence, turn);
    issue_qk(score0, q_smem, k_smem[next_stage]);
    pingpong_after(consumer, turn);
    ++issue_sequence;

    pingpong_before(consumer, issue_sequence, turn);
    issue_pv(output, score1, v_smem[cur_stage]);
    pingpong_after(consumer, turn);
    ++issue_sequence;

    hopper::wgmma_wait_group<1>();
    const RowScale alpha =
        softmax_inplace(score0, online, q_start, tile * FA3_KV_ROWS, causal);
    hopper::wgmma_wait_group<0>();
    if ((threadIdx.x & 127) == 0) {
      hopper::mbarrier_arrive(&stage_empty[cur_stage]);
    }
    scale_rows(output, alpha);
  }
}

__device__ __forceinline__ __nv_bfloat16 *swizzled_ptr(__nv_bfloat16 *base,
                                                       int row, int col) {
  uintptr_t address = reinterpret_cast<uintptr_t>(base + row * FA3_D + col);
  address ^= ((address & 0x3ffULL) >> 7) << 4;
  return reinterpret_cast<__nv_bfloat16 *>(address);
}

__device__ __forceinline__ void
store_output_shared(__nv_bfloat16 *dst, hopper::Accum64x64 &output,
                    const OnlineSoftmax &state) {
  const int wg_lane = threadIdx.x & 127;
  const int warp = wg_lane >> 5;
  const int lane = wg_lane & 31;
  const int lane4 = lane & 3;
  const int row0 = warp * 16 + lane / 4;
  const int row1 = row0 + 8;
  const float inv_l0 = 1.0f / state.l0;
  const float inv_l1 = 1.0f / state.l1;

#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
    const int base = tile * 8;
    const int col0 = tile * 16 + lane4 * 2;
    const int col1 = col0 + 8;
    *reinterpret_cast<uint32_t *>(swizzled_ptr(dst, row0, col0)) =
        pack_bf16(output.x[base + 0] * inv_l0, output.x[base + 1] * inv_l0);
    *reinterpret_cast<uint32_t *>(swizzled_ptr(dst, row1, col0)) =
        pack_bf16(output.x[base + 2] * inv_l1, output.x[base + 3] * inv_l1);
    *reinterpret_cast<uint32_t *>(swizzled_ptr(dst, row0, col1)) =
        pack_bf16(output.x[base + 4] * inv_l0, output.x[base + 5] * inv_l0);
    *reinterpret_cast<uint32_t *>(swizzled_ptr(dst, row1, col1)) =
        pack_bf16(output.x[base + 6] * inv_l1, output.x[base + 7] * inv_l1);
  }
}

__device__ __forceinline__ void warpgroup_barrier(int id) {
  asm volatile("bar.sync %0, 128;" : : "r"(id) : "memory");
}

__global__ __launch_bounds__(FA3_THREADS, 1) void flash_attn3_hopper_kernel(
    const __grid_constant__ HopperFa3Params params) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  auto *tiles = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *q_smem[FA3_CONSUMERS] = {tiles, tiles + FA3_TILE_ELEMS};
  __nv_bfloat16 *k_smem[FA3_STAGES] = {tiles + 2 * FA3_TILE_ELEMS,
                                       tiles + 3 * FA3_TILE_ELEMS};
  __nv_bfloat16 *v_smem[FA3_STAGES] = {tiles + 4 * FA3_TILE_ELEMS,
                                       tiles + 5 * FA3_TILE_ELEMS};

  __shared__ hopper::MBarrier q_ready;
  __shared__ hopper::MBarrier kv_ready[FA3_STAGES];
  __shared__ hopper::MBarrier stage_empty[FA3_STAGES];
  __shared__ hopper::MBarrier turn[FA3_CONSUMERS];

  if (threadIdx.x == 0) {
    hopper::mbarrier_init(&q_ready, 1);
#pragma unroll
    for (int stage = 0; stage < FA3_STAGES; ++stage) {
      hopper::mbarrier_init(&kv_ready[stage], 1);
      hopper::mbarrier_init(&stage_empty[stage], FA3_CONSUMERS);
      hopper::mbarrier_init(&turn[stage], 1);
    }
  }
  __syncthreads();

  const int warpgroup = threadIdx.x >> 7;
  const int q_cta_start = blockIdx.x * FA3_CONSUMERS * FA3_Q_ROWS;
  const int kv_tiles = params.causal
                           ? blockIdx.x * FA3_CONSUMERS + FA3_CONSUMERS
                           : params.seqlen / FA3_KV_ROWS;

  if (warpgroup == 0) {
    hopper::setmaxnreg_dec<32>();
    if (threadIdx.x == 0) {
      hopper::mbarrier_expect_tx(&q_ready, FA3_CONSUMERS * FA3_TILE_BYTES);
      hopper::tma_load_5d(q_smem[0], &params.q_map, 0, q_cta_start, 0,
                          blockIdx.y, blockIdx.z, &q_ready);
      hopper::tma_load_5d(q_smem[1], &params.q_map, 0, q_cta_start + FA3_Q_ROWS,
                          0, blockIdx.y, blockIdx.z, &q_ready);

      for (int tile = 0; tile < kv_tiles; ++tile) {
        const int stage = tile & (FA3_STAGES - 1);
        if (tile >= FA3_STAGES) {
          hopper::mbarrier_wait(&stage_empty[stage],
                                ((tile - FA3_STAGES) / FA3_STAGES) & 1);
        }
        hopper::mbarrier_expect_tx(&kv_ready[stage], 2 * FA3_TILE_BYTES);
        const int kv_start = tile * FA3_KV_ROWS;
        hopper::tma_load_5d(k_smem[stage], &params.k_map, 0, kv_start, 0,
                            blockIdx.y, blockIdx.z, &kv_ready[stage]);
        hopper::tma_load_5d(v_smem[stage], &params.v_map, 0, kv_start, 0,
                            blockIdx.y, blockIdx.z, &kv_ready[stage]);
      }
    }
    return;
  }

  hopper::setmaxnreg_inc<160>();
  const int consumer = warpgroup - 1;
  const int q_start = q_cta_start + consumer * FA3_Q_ROWS;
  OnlineSoftmax online;
  hopper::Accum64x64 output = {};
  // Keep these as two named objects.  A pointer-swapped array makes nvcc
  // address-take the fragments and creates a large per-thread stack frame.
  hopper::Accum64x64 score0 = {};
  hopper::Accum64x64 score1 = {};
  int issue_sequence = 0;

  hopper::mbarrier_wait(&q_ready, 0);
  hopper::mbarrier_wait(&kv_ready[0], 0);
  pingpong_before(consumer, issue_sequence, turn);
  issue_qk(score0, q_smem[consumer], k_smem[0]);
  pingpong_after(consumer, turn);
  ++issue_sequence;
  hopper::wgmma_wait_group<0>();
  RowScale alpha =
      softmax_inplace(score0, online, q_start, 0, params.causal != 0);
  scale_rows(output, alpha);

  int tile = 1;
  for (; tile + 1 < kv_tiles; tile += 2) {
    // Algorithm-2 style two-level pipeline: QK(next) is the older WGMMA
    // group; PV(cur) is the younger group. wait_group<1> exposes the next
    // scores while PV remains in flight, so softmax runs concurrently.
    pipeline_transition<true>(tile, consumer, q_start, issue_sequence,
                              q_smem[consumer], k_smem, v_smem, kv_ready,
                              stage_empty, turn, score0, score1, output, online,
                              params.causal != 0);
    pipeline_transition<false>(tile + 1, consumer, q_start, issue_sequence,
                               q_smem[consumer], k_smem, v_smem, kv_ready,
                               stage_empty, turn, score0, score1, output,
                               online, params.causal != 0);
  }
  if (tile < kv_tiles) {
    pipeline_transition<true>(tile, consumer, q_start, issue_sequence,
                              q_smem[consumer], k_smem, v_smem, kv_ready,
                              stage_empty, turn, score0, score1, output, online,
                              params.causal != 0);
  }

  const int last_stage = (kv_tiles - 1) & (FA3_STAGES - 1);
  pingpong_before(consumer, issue_sequence, turn);
  if ((kv_tiles - 1) & 1) {
    issue_pv(output, score1, v_smem[last_stage]);
  } else {
    issue_pv(output, score0, v_smem[last_stage]);
  }
  pingpong_after(consumer, turn);
  hopper::wgmma_wait_group<0>();
  if ((threadIdx.x & 127) == 0) {
    hopper::mbarrier_arrive(&stage_empty[last_stage]);
  }

  // Reuse the now-dead Q tile as a WGMMA-fragment -> TMA layout conversion.
  store_output_shared(q_smem[consumer], output, online);
  const int named_barrier = consumer + 1;
  warpgroup_barrier(named_barrier);
  if ((threadIdx.x & 127) == 0) {
    hopper::tma_store_5d(&params.o_map, 0, q_start, 0, blockIdx.y, blockIdx.z,
                         q_smem[consumer]);
    hopper::tma_store_wait();
  }
  warpgroup_barrier(named_barrier);
}

void encode_fa3_tensor_map(CUtensorMap *map, void *data, int batch, int heads,
                           int seqlen) {
  const uint64_t global_dims[5] = {64, static_cast<uint64_t>(seqlen), 1,
                                   static_cast<uint64_t>(heads),
                                   static_cast<uint64_t>(batch)};
  const uint64_t global_strides[4] = {
      128, 128, static_cast<uint64_t>(seqlen) * 128,
      static_cast<uint64_t>(heads) * seqlen * 128};
  const uint32_t box_dims[5] = {64, 64, 1, 1, 1};
  const uint32_t element_strides[5] = {1, 1, 1, 1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, data, global_dims,
      global_strides, box_dims, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    const char *message = "unknown CUDA driver error";
    cuGetErrorString(result, &message);
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn3_hopper: cuTensorMapEncodeTiled failed: " << message;
  }
}

#endif // CUDA_LEARN_HAS_SM90A

} // namespace

void flash_attn3_hopper(TensorView q, TensorView k, TensorView v,
                        TensorView out, int64_t causal64) {
  check_fa3_bf16(q, "q");
  check_fa3_bf16(k, "k");
  check_fa3_bf16(v, "v");
  check_fa3_bf16(out, "out");
  check_fa3_same_shape(k, q, "k");
  check_fa3_same_shape(v, q, "v");
  check_fa3_same_shape(out, q, "out");

  const int64_t batch64 = dim(q, 0);
  const int64_t heads64 = dim(q, 1);
  const int64_t seqlen64 = dim(q, 2);
  if (dim(q, 3) != FA3_D || batch64 <= 0 || heads64 <= 0 || seqlen64 <= 0 ||
      seqlen64 % 128 != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn3_hopper: expected [B,H,N,64] with positive B/H and "
           "N divisible by 128";
  }
  if (batch64 > 65535 || heads64 > 65535 || seqlen64 > 2147483647LL) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn3_hopper: shape exceeds CUDA grid limits";
  }

#if !defined(CUDA_LEARN_HAS_SM90A)
  TVM_FFI_THROW(RuntimeError)
      << "flash_attn3_hopper was not built for Hopper; configure a separate "
         "build with -DCUDA_LEARN_ENABLE_HOPPER_FA3=ON";
#else
  cudaDeviceProp props{};
  CUDA_LEARN_CHECK(cudaGetDeviceProperties(&props, q.device().device_id));
  if (props.major != 9) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn3_hopper requires Hopper compute capability 9.0, got "
        << props.major << "." << props.minor;
  }

  const int batch = static_cast<int>(batch64);
  const int heads = static_cast<int>(heads64);
  const int seqlen = static_cast<int>(seqlen64);
  HopperFa3Params params{};
  params.seqlen = seqlen;
  params.causal = causal64 != 0;
  encode_fa3_tensor_map(&params.q_map, q.data_ptr(), batch, heads, seqlen);
  encode_fa3_tensor_map(&params.k_map, k.data_ptr(), batch, heads, seqlen);
  encode_fa3_tensor_map(&params.v_map, v.data_ptr(), batch, heads, seqlen);
  encode_fa3_tensor_map(&params.o_map, out.data_ptr(), batch, heads, seqlen);

  static const bool smem_opted_in = [] {
    CUDA_LEARN_CHECK(cudaFuncSetAttribute(
        flash_attn3_hopper_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(FA3_SMEM_BYTES)));
    return true;
  }();
  (void)smem_opted_in;

  const dim3 grid(seqlen / (FA3_CONSUMERS * FA3_Q_ROWS), heads, batch);
  flash_attn3_hopper_kernel<<<grid, FA3_THREADS, FA3_SMEM_BYTES,
                              get_stream(q)>>>(params);
  CUDA_LEARN_CHECK(cudaGetLastError());
#endif
}

CUDA_LEARN_REGISTER("cuda_learn.flash_attn3_hopper", flash_attn3_hopper);
