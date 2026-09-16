// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The fp4-to-fp8 LUT sequence below is derived from the Apache-2.0
// FlashInfer/TensorRT-LLM mixed-input utilities. The surrounding standalone
// kernel and layout checks were written for this cuda_learn example.
//
// A compact SM90a teaching kernel for the first MoE projection:
//
//   Y[e, token, channel] = X_fp8[e, token, k] * W_mxfp4[e, channel, k]^T
//
// MXFP4 weights are expanded in registers to E4M3 and consumed directly by
// RS WGMMA.  FP8 activations stay in shared memory.  This is intentionally a
// readable layout experiment rather than a replacement for a production MoE
// implementation (which would use TMA and an interleaved weight layout).

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/float8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace mxfp4_moe {

using namespace cute;
using FP8 = cutlass::float_e4m3_t;

constexpr int kTileTokens = 128;  // WGMMA N
constexpr int kTileChannels = 64; // WGMMA M
constexpr int kTileK = 32;        // WGMMA K and MXFP4 scale group
constexpr int kThreads = 128;     // one warpgroup
constexpr int kStages = 2;

using MmaAtom = SM90_64x128x32_F32E4M3E4M3_RS_TN<>;
using TiledMma = decltype(make_tiled_mma(MmaAtom{}));
// A K=32 stage is exactly one 32-byte swizzle line, so SW32 is the largest
// canonical GMMA layout that evenly divides this tile (SW64/SW128 require
// K extents of at least 64/128 FP8 elements).
using SmemAtom = GMMA::Layout_K_SW32_Atom<FP8>;
using SmemLayout =
    decltype(tile_to_shape(SmemAtom{}, Shape<Int<kTileTokens>, Int<kTileK>>{}));

static_assert(cosize(SmemLayout{}) == kTileTokens * kTileK);
static_assert(sizeof(FP8) == 1);

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = (call);                                              \
    if (status_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status_));                               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// PTX prmt wrapper.  The two 32-bit inputs form an eight-byte lookup table;
// each selector nibble chooses one byte.
__device__ __forceinline__ uint32_t prmt(uint32_t hi, uint32_t lo,
                                         uint32_t selector) {
  uint32_t result;
  asm volatile("prmt.b32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(lo), "r"(hi), "r"(selector));
  return result;
}

// Convert two packed fp4x8 registers to two packed fp8x8 registers while
// folding in the preprocessed E8M0 exponent offsets.  Low four output bytes of
// both registers use lo_exp_offset; high four use hi_exp_offset.
//
// This follows the Apache-2.0 FlashInfer/TensorRT-LLM Humming conversion idea,
// reduced here to the non-preprocessed-sign path used by this example.
__device__ __forceinline__ void
fp4x8_pair_to_fp8x8_scaled(uint32_t fp4x8_0, uint32_t fp4x8_1,
                           uint64_t &fp8x8_0, uint64_t &fp8x8_1,
                           uint32_t lo_exp_offset, uint32_t hi_exp_offset) {
  uint32_t out0[2];
  uint32_t out1[2];
  uint32_t const selector0 = fp4x8_0 & 0x77777777U;
  uint32_t const selector1 = fp4x8_1 & 0x77777777U;

  // E4M3 has three mantissa bits, hence one exponent step is encoded as 0x08.
  // Offset is not applied to zero (low byte multiplier is zero).
  constexpr uint32_t kCodes0To3 = 0x0c080000U;
  constexpr uint32_t kCodes4To7 = 0x1c181410U;
  uint32_t const lo_lut03 = lo_exp_offset * 0x08080800U + kCodes0To3;
  uint32_t const lo_lut47 = lo_exp_offset * 0x08080808U + kCodes4To7;
  uint32_t const hi_lut03 = hi_exp_offset * 0x08080800U + kCodes0To3;
  uint32_t const hi_lut47 = hi_exp_offset * 0x08080808U + kCodes4To7;

  uint32_t const hb_sign0 = fp4x8_0 & 0x80808080U;
  uint32_t const hb_sign1 = fp4x8_1 & 0x80808080U;
  uint32_t const lb_sign0 = (fp4x8_0 & 0x08080808U) << 4U;
  uint32_t const lb_sign1 = (fp4x8_1 & 0x08080808U) << 4U;
  uint32_t const lo_sign0 = prmt(hb_sign0, lb_sign0, 0x5140U);
  uint32_t const lo_sign1 = prmt(hb_sign1, lb_sign1, 0x5140U);
  uint32_t const hi_sign0 = prmt(hb_sign0, lb_sign0, 0x7362U);
  uint32_t const hi_sign1 = prmt(hb_sign1, lb_sign1, 0x7362U);

  out0[0] = lo_sign0 | prmt(lo_lut47, lo_lut03, selector0);
  out1[0] = lo_sign1 | prmt(lo_lut47, lo_lut03, selector1);
  out0[1] = hi_sign0 | prmt(hi_lut47, hi_lut03, selector0 >> 16U);
  out1[1] = hi_sign1 | prmt(hi_lut47, hi_lut03, selector1 >> 16U);
  fp8x8_0 = uint64_t(out0[0]) | (uint64_t(out0[1]) << 32U);
  fp8x8_1 = uint64_t(out1[0]) | (uint64_t(out1[1]) << 32U);
}

template <class CoordTensor, class Fragment>
__device__ __forceinline__ void
load_and_expand_weight(uint8_t const *packed_weight, uint8_t const *exp_offset,
                       int expert, int channels, int k_size, int channel_tile,
                       int k_tile, CoordTensor const &a_coord,
                       Fragment &a_fragment) {
  // ALayout_64x32 gives exactly 16 logical E4M3 values per warpgroup lane.
  static_assert(decltype(size(a_coord))::value == 16);
  static_assert(decltype(size(a_fragment))::value == 16);

  uint32_t packed[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    auto coord = a_coord(i);
    int row = channel_tile + int(get<0>(coord));
    int k = k_tile + int(get<1>(coord));
    uint8_t byte =
        packed_weight[(static_cast<int64_t>(expert) * channels + row) *
                          (k_size / 2) +
                      k / 2];
    uint32_t code = (byte >> (4 * (k & 1))) & 0xfU;
    packed[i / 8] |= code << (4 * (i % 8));
  }

  // Fragment entries 0..3 and 8..11 belong to one A row; entries 4..7 and
  // 12..15 belong to the other.  Scale is constant for this K=32 tile.
  int lo_row = channel_tile + int(get<0>(a_coord(0)));
  int hi_row = channel_tile + int(get<0>(a_coord(4)));
  int scale_group = k_tile / kTileK;
  int groups_per_row = k_size / kTileK;
  uint32_t lo_offset =
      exp_offset[(static_cast<int64_t>(expert) * channels + lo_row) *
                     groups_per_row +
                 scale_group];
  uint32_t hi_offset =
      exp_offset[(static_cast<int64_t>(expert) * channels + hi_row) *
                     groups_per_row +
                 scale_group];

  auto *raw = reinterpret_cast<uint64_t *>(a_fragment.data());
  fp4x8_pair_to_fp8x8_scaled(packed[0], packed[1], raw[0], raw[1], lo_offset,
                             hi_offset);
}

template <class SmemTensor>
__device__ __forceinline__ void
load_activation_stage(uint8_t const *activation, int const *tokens_per_expert,
                      int expert, int token_tile, int k_tile,
                      int token_capacity, int k_size, SmemTensor &smem) {
  int valid_tokens = tokens_per_expert[expert];
  for (int linear = threadIdx.x; linear < kTileTokens * kTileK;
       linear += blockDim.x) {
    int token_in_tile = linear / kTileK;
    int k_in_tile = linear % kTileK;
    int token = token_tile + token_in_tile;
    uint8_t bits = 0;
    if (token < valid_tokens && token < token_capacity) {
      bits =
          activation[(static_cast<int64_t>(expert) * token_capacity + token) *
                         k_size +
                     k_tile + k_in_tile];
    }
    reinterpret_cast<uint8_t &>(smem(token_in_tile, k_in_tile)) = bits;
  }
}

__global__ __launch_bounds__(kThreads) void mxfp4_fp8_rs_gemm_kernel(
    float *output, uint8_t const *activation, uint8_t const *packed_weight,
    uint8_t const *exp_offset, float const *token_scale,
    float const *expert_residual, int const *tokens_per_expert,
    int token_capacity, int channels, int k_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  FP8 *shared_fp8 = reinterpret_cast<FP8 *>(shared_bytes);
  Tensor s_x0 = make_tensor(make_smem_ptr(shared_fp8), SmemLayout{});
  Tensor s_x1 = make_tensor(make_smem_ptr(shared_fp8 + cosize(SmemLayout{})),
                            SmemLayout{});

  int expert = blockIdx.z;
  int channel_tile = blockIdx.x * kTileChannels;
  int token_tile = blockIdx.y * kTileTokens;

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  auto a_identity =
      make_identity_tensor(Shape<Int<kTileChannels>, Int<kTileK>>{});
  auto a_coord = thr_mma.partition_A(a_identity);
  // partition_A(identity) maps values to logical coordinates.  Its strides
  // are coordinate-valued, so it must not be reused as an owning RF layout.
  // partition_shape_A gives the compact owning register shape instead.
  auto a_reg0 = make_tensor<FP8>(
      partition_shape_A(tiled_mma, Shape<Int<kTileChannels>, Int<kTileK>>{}));
  auto a_reg1 = make_tensor<FP8>(
      partition_shape_A(tiled_mma, Shape<Int<kTileChannels>, Int<kTileK>>{}));

  auto b_smem0 = thr_mma.partition_B(s_x0);
  auto b_smem1 = thr_mma.partition_B(s_x1);
  auto b_desc0 = thr_mma.make_fragment_B(b_smem0);
  auto b_desc1 = thr_mma.make_fragment_B(b_smem1);

  auto c_identity =
      make_identity_tensor(Shape<Int<kTileChannels>, Int<kTileTokens>>{});
  auto c_coord = thr_mma.partition_C(c_identity);
  auto accum = make_tensor<float>(partition_shape_C(
      tiled_mma, Shape<Int<kTileChannels>, Int<kTileTokens>>{}));
  clear(accum);

  int k_tiles = k_size / kTileK;
  load_activation_stage(activation, tokens_per_expert, expert, token_tile, 0,
                        token_capacity, k_size, s_x0);
  cutlass::arch::fence_view_async_shared();
  __syncthreads();

  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  warpgroup_fence_operand(accum);

  // Two RF A slots and two SMEM B stages are essential: conversion/loading
  // for tile k+1 proceeds while WGMMA(k) is pending, without overwriting an
  // operand that the asynchronous instruction can still read.
  for (int kt = 0; kt < k_tiles; ++kt) {
    int k_tile = kt * kTileK;
    if ((kt & 1) == 0) {
      load_and_expand_weight(packed_weight, exp_offset, expert, channels,
                             k_size, channel_tile, k_tile, a_coord, a_reg0);
      warpgroup_arrive();
      cute::gemm(tiled_mma, a_reg0, b_desc0, accum);
    } else {
      load_and_expand_weight(packed_weight, exp_offset, expert, channels,
                             k_size, channel_tile, k_tile, a_coord, a_reg1);
      warpgroup_arrive();
      cute::gemm(tiled_mma, a_reg1, b_desc1, accum);
    }
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    warpgroup_commit_batch();

    // Keep at most the newest group pending.  Consequently stage/slot k-1 is
    // complete before it is selected again for k+1.
    warpgroup_wait<1>();
    if (kt + 1 < k_tiles) {
      if ((kt & 1) == 0) {
        load_activation_stage(activation, tokens_per_expert, expert, token_tile,
                              k_tile + kTileK, token_capacity, k_size, s_x1);
      } else {
        load_activation_stage(activation, tokens_per_expert, expert, token_tile,
                              k_tile + kTileK, token_capacity, k_size, s_x0);
      }
      cutlass::arch::fence_view_async_shared();
      __syncthreads();
    }
  }

  warpgroup_wait<0>();
  warpgroup_fence_operand(accum);

  int valid_tokens = tokens_per_expert[expert];
#pragma unroll
  for (int i = 0; i < size(accum); ++i) {
    auto coord = c_coord(i);
    int channel = channel_tile + int(get<0>(coord));
    int token = token_tile + int(get<1>(coord));
    if (channel < channels && token < valid_tokens && token < token_capacity) {
      float scale =
          token_scale[static_cast<int64_t>(expert) * token_capacity + token] *
          expert_residual[expert];
      output[(static_cast<int64_t>(expert) * token_capacity + token) *
                 channels +
             channel] = accum(i) * scale;
    }
  }
#else
  (void)output;
  (void)activation;
  (void)packed_weight;
  (void)exp_offset;
  (void)token_scale;
  (void)expert_residual;
  (void)tokens_per_expert;
  (void)token_capacity;
  (void)channels;
  (void)k_size;
#endif
}

struct ShapeArgs {
  int experts = 8;
  int token_capacity = 128;
  int channels = 2 * 14336; // gate + up projection rows
  int k = 4096;             // common hidden size
  bool layout_only = false;
  bool smoke = false;
};

void print_layout() {
  TiledMma mma;
  auto a_id = make_identity_tensor(Shape<Int<kTileChannels>, Int<kTileK>>{});
  auto c_id =
      make_identity_tensor(Shape<Int<kTileChannels>, Int<kTileTokens>>{});

  std::printf("MMA: ");
  print(mma);
  std::printf("\nactivation SMEM: ");
  print(SmemLayout{});
  std::printf("\nA register layout/lane: ");
  print(mma.get_slice(0).partition_A(a_id).layout());
  std::printf("\nC accumulator layout/lane: ");
  print(mma.get_slice(0).partition_C(c_id).layout());
  std::printf("\n\nA coordinates (representative lanes):\n");
  for (int tid : {0, 1, 4, 32}) {
    auto a = mma.get_slice(tid).partition_A(a_id);
    std::printf(" lane %d:", tid);
    for (int i = 0; i < size(a); ++i) {
      auto p = a(i);
      std::printf(" (%d,%d)", int(get<0>(p)), int(get<1>(p)));
    }
    std::printf("\n");
  }
  std::printf("C coordinates (first eight values of representative lanes):\n");
  for (int tid : {0, 1, 4, 32}) {
    auto c = mma.get_slice(tid).partition_C(c_id);
    std::printf(" lane %d:", tid);
    for (int i = 0; i < 8; ++i) {
      auto p = c(i);
      std::printf(" (%d,%d)", int(get<0>(p)), int(get<1>(p)));
    }
    std::printf("\n");
  }

  // Prove the assumptions used by the pair converter and global epilogue.
  std::vector<int> a_owners(kTileChannels * kTileK, 0);
  std::vector<int> c_owners(kTileChannels * kTileTokens, 0);
  for (int tid = 0; tid < kThreads; ++tid) {
    auto a = mma.get_slice(tid).partition_A(a_id);
    auto c = mma.get_slice(tid).partition_C(c_id);
    if (size(a) != 16 || size(c) != 64) {
      std::fprintf(stderr, "unexpected fragment size at lane %d\n", tid);
      std::exit(EXIT_FAILURE);
    }
    int lo_row = int(get<0>(a(0)));
    int hi_row = int(get<0>(a(4)));
    for (int i = 0; i < size(a); ++i) {
      auto p = a(i);
      int row = int(get<0>(p));
      int k = int(get<1>(p));
      bool lo_chunk = (i % 8) < 4;
      if (row != (lo_chunk ? lo_row : hi_row)) {
        std::fprintf(stderr, "scale/chunk mismatch lane=%d value=%d\n", tid, i);
        std::exit(EXIT_FAILURE);
      }
      ++a_owners[row * kTileK + k];
    }
    for (int i = 0; i < size(c); ++i) {
      auto p = c(i);
      ++c_owners[int(get<0>(p)) * kTileTokens + int(get<1>(p))];
    }
  }
  for (int n : a_owners) {
    if (n != 1) {
      std::fprintf(stderr, "A ownership is not bijective\n");
      std::exit(EXIT_FAILURE);
    }
  }
  for (int n : c_owners) {
    if (n != 1) {
      std::fprintf(stderr, "C ownership is not bijective\n");
      std::exit(EXIT_FAILURE);
    }
  }
  std::puts("PASS layout: A=64x32 and C=64x128 each have exactly one owner; "
            "the two scale rows match fp8x4 chunks.");
}

float fp4_value(uint8_t code) {
  static constexpr float kMagnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                          2.0f, 3.0f, 4.0f, 6.0f};
  float value = kMagnitude[code & 7];
  return (code & 8) ? -value : value;
}

float fp8_value(uint8_t bits) { return static_cast<float>(FP8::bitcast(bits)); }

ShapeArgs parse_args(int argc, char **argv) {
  ShapeArgs args;
  for (int i = 1; i < argc; ++i) {
    std::string option = argv[i];
    auto value = [&](char const *name) {
      if (++i == argc) {
        std::fprintf(stderr, "%s requires a value\n", name);
        std::exit(EXIT_FAILURE);
      }
      return std::atoi(argv[i]);
    };
    if (option == "--layout") {
      args.layout_only = true;
    } else if (option == "--smoke") {
      args.smoke = true;
      args.experts = 1;
      args.token_capacity = 128;
      args.channels = 128;
      args.k = 4096;
    } else if (option == "--experts") {
      args.experts = value("--experts");
    } else if (option == "--tokens") {
      args.token_capacity = value("--tokens");
    } else if (option == "--channels") {
      args.channels = value("--channels");
    } else if (option == "--k") {
      args.k = value("--k");
    } else {
      std::fprintf(stderr, "unknown option: %s\n", option.c_str());
      std::exit(EXIT_FAILURE);
    }
  }
  if (args.token_capacity % kTileTokens || args.channels % kTileChannels ||
      args.k % kTileK) {
    std::fprintf(stderr, "tokens/channels/k must be multiples of 128/64/32\n");
    std::exit(EXIT_FAILURE);
  }
  return args;
}

void run(ShapeArgs const &args) {
  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major != 9) {
    std::printf(
        "SKIP: RS WGMMA requires Hopper SM90a; current GPU is %s (SM%d%d).\n",
        prop.name, prop.major, prop.minor);
    return;
  }

  int64_t x_count =
      static_cast<int64_t>(args.experts) * args.token_capacity * args.k;
  int64_t w_bytes =
      static_cast<int64_t>(args.experts) * args.channels * (args.k / 2);
  int64_t s_count =
      static_cast<int64_t>(args.experts) * args.channels * (args.k / kTileK);
  int64_t y_count =
      static_cast<int64_t>(args.experts) * args.token_capacity * args.channels;

  std::printf("shape: experts=%d, token_capacity=%d, N=%d, K=%d\n",
              args.experts, args.token_capacity, args.channels, args.k);
  std::printf("logical GEMM/expert: Y^T[%d,%d] = W[%d,%d] X^T[%d,%d]\n",
              args.channels, args.token_capacity, args.channels, args.k, args.k,
              args.token_capacity);

  std::vector<uint8_t> h_x(x_count);
  std::vector<uint8_t> h_w(w_bytes);
  std::vector<uint8_t> h_s(s_count);
  std::vector<float> h_token_scale(static_cast<int64_t>(args.experts) *
                                   args.token_capacity);
  std::vector<float> h_residual(args.experts);
  std::vector<int> h_tokens(args.experts);

  std::mt19937 rng(20260909);
  std::uniform_int_distribution<int> fp4_dist(0, 15);
  // Modest finite E4M3 codes, including signs; avoid NaN encodings.
  std::uniform_int_distribution<int> fp8_mag(0x18, 0x30);
  for (uint8_t &x : h_x) {
    x = static_cast<uint8_t>(fp8_mag(rng) | ((rng() & 1) ? 0x80 : 0));
  }
  for (int64_t i = 0; i < w_bytes; ++i) {
    h_w[i] = static_cast<uint8_t>(fp4_dist(rng) | (fp4_dist(rng) << 4));
  }
  // Preprocessed offsets t=e'-b+1 must lie in [1,12].  Limiting this example
  // to [4,7] keeps reference values small while still exercising scaling.
  for (int64_t i = 0; i < s_count; ++i) {
    h_s[i] = static_cast<uint8_t>(4 + (rng() & 3));
  }
  for (int e = 0; e < args.experts; ++e) {
    h_tokens[e] = args.token_capacity - (e % 3) * 16;
    h_residual[e] = std::ldexp(1.0f, (e % 3) - 1);
    for (int m = 0; m < args.token_capacity; ++m) {
      h_token_scale[e * args.token_capacity + m] =
          0.75f + 0.125f * float(m % 3);
    }
  }

  uint8_t *d_x = nullptr, *d_w = nullptr, *d_s = nullptr;
  float *d_y = nullptr, *d_token_scale = nullptr, *d_residual = nullptr;
  int *d_tokens = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, x_count));
  CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
  CUDA_CHECK(cudaMalloc(&d_s, s_count));
  CUDA_CHECK(cudaMalloc(&d_y, y_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_scale, h_token_scale.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_residual, h_residual.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tokens, h_tokens.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), w_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_s, h_s.data(), s_count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_token_scale, h_token_scale.data(),
                        h_token_scale.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(),
                        h_residual.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(),
                        h_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_y, 0, y_count * sizeof(float)));

  dim3 grid(args.channels / kTileChannels, args.token_capacity / kTileTokens,
            args.experts);
  constexpr int shared_bytes = kStages * cosize(SmemLayout{}) * sizeof(FP8);
  mxfp4_fp8_rs_gemm_kernel<<<grid, kThreads, shared_bytes>>>(
      d_y, d_x, d_w, d_s, d_token_scale, d_residual, d_tokens,
      args.token_capacity, args.channels, args.k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Validate a deterministic sample.  This remains quick for the default
  // 8 x 128 x 28672 x 4096 MoE geometry.
  std::vector<float> h_y(y_count);
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  int checks = args.smoke ? 64 : 24;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  bool valid = true;
  for (int sample = 0; sample < checks; ++sample) {
    int e = sample % args.experts;
    int token = (sample * 17) % h_tokens[e];
    int channel = (sample * 997) % args.channels;
    double reference = 0.0;
    for (int k = 0; k < args.k; ++k) {
      uint8_t byte = h_w[(static_cast<int64_t>(e) * args.channels + channel) *
                             (args.k / 2) +
                         k / 2];
      uint8_t code = (byte >> (4 * (k & 1))) & 0xf;
      uint8_t offset = h_s[(static_cast<int64_t>(e) * args.channels + channel) *
                               (args.k / kTileK) +
                           k / kTileK];
      float expanded = std::ldexp(fp4_value(code), int(offset) - 6);
      float x = fp8_value(
          h_x[(static_cast<int64_t>(e) * args.token_capacity + token) * args.k +
              k]);
      reference += double(x) * double(expanded);
    }
    reference *= h_token_scale[e * args.token_capacity + token] * h_residual[e];
    float actual = h_y[(static_cast<int64_t>(e) * args.token_capacity + token) *
                           args.channels +
                       channel];
    float abs_error = std::abs(actual - float(reference));
    float rel_error = abs_error / std::max(1.0f, std::abs(float(reference)));
    max_abs = std::max(max_abs, abs_error);
    max_rel = std::max(max_rel, rel_error);
    float tolerance = 0.02f + 1.0e-3f * std::abs(float(reference));
    valid = valid && abs_error <= tolerance;
  }
  std::printf("%s GPU sample check: %d outputs, max_abs=%g, max_rel=%g\n",
              valid ? "PASS" : "FAIL", checks, max_abs, max_rel);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_s));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_token_scale));
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_tokens));
  if (!valid) {
    std::exit(EXIT_FAILURE);
  }
}

} // namespace mxfp4_moe

int main(int argc, char **argv) {
  auto args = mxfp4_moe::parse_args(argc, argv);
  mxfp4_moe::print_layout();
  if (!args.layout_only) {
    mxfp4_moe::run(args);
  }
  return 0;
}
