#include <cute/tensor.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace cute_fa2_4096 {

using namespace cute;

using Element = half_t;
using Accum = float;

// Fixed teaching specialization:
//   Q, K, V, O: [4096, 64]
//   S = Q K^T:  [4096, 4096]
using SeqLen = Int<4096>;
using HeadDim = Int<64>;
using BlockM = Int<64>;
using BlockN = Int<64>;
using Stages = Int<2>;

static_assert(int(SeqLen{}) % int(BlockM{}) == 0);
static_assert(int(SeqLen{}) % int(BlockN{}) == 0);

// Four warps are placed only along M.  Keeping warp-N equal to one is the key
// condition that lets every warp reinterpret its complete 16x64 QK result as
// the 16x64 A operand of P@V without cross-warp exchange.
using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMma = decltype(make_tiled_mma(
    MmaAtom{}, Layout<Shape<_4, _1, _1>>{}, Tile<_64, _64, _16>{}));

// The physical row-major offset is x = row*64 + col.  Swizzle<3,3,3>
// computes x' = x ^ (((x >> 6) & 7) << 3), or equivalently
// x' = row*64 + (col ^ ((row & 7) << 3)) for this 8x64 atom.
using SmemRowAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_8, HeadDim>, Stride<HeadDim, _1>>{}));
using SmemColAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<HeadDim, _8>, Stride<_1, HeadDim>>{}));

using SmemLayoutQ = decltype(
    tile_to_shape(SmemRowAtom{}, make_shape(BlockM{}, HeadDim{})));
using SmemLayoutK = decltype(tile_to_shape(
    SmemRowAtom{}, make_shape(BlockN{}, HeadDim{}, Stages{})));
using SmemLayoutV = decltype(tile_to_shape(
    SmemColAtom{}, make_shape(HeadDim{}, BlockN{}, Stages{})));
using SmemLayoutO = SmemLayoutQ;

// One cp.async transfers eight FP16 values (16 bytes).  The row copy maps
// tid -> (row=tid/4, vector=tid%4); the column copy swaps the logical modes so
// V remains physically row-major while being exposed to MMA as (D, N).
using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>;
using G2SRow = decltype(make_tiled_copy(
    G2SAtom{}, Layout<Shape<_32, _4>, Stride<_4, _1>>{},
    Layout<Shape<_1, _8>>{}));
using G2SCol = decltype(make_tiled_copy(
    G2SAtom{}, Layout<Shape<_4, _32>, Stride<_1, _4>>{},
    Layout<Shape<_8, _1>>{}));

using S2RAtomN = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using S2RAtomT = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

using S2GAtom = Copy_Atom<UniversalCopy<uint128_t>, Element>;
using S2GRow = decltype(make_tiled_copy(
    S2GAtom{}, Layout<Shape<_32, _4>, Stride<_4, _1>>{},
    Layout<Shape<_1, _8>>{}));

constexpr int kThreads = 128;
constexpr int kKvTiles = int(SeqLen{}) / int(BlockN{});
constexpr int kSmemElements = cosize(SmemLayoutQ{}) +
                              cosize(SmemLayoutK{}) +
                              cosize(SmemLayoutV{});
constexpr int kSmemBytes = kSmemElements * sizeof(Element);

static_assert(decltype(size(TiledMma{}))::value == kThreads);
static_assert(kSmemBytes == 40 * 1024);

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

template <class Copy, class Src, class Dst>
__device__ __forceinline__ void copy_kv_tile(Copy const &copy_op,
                                             Src const &src,
                                             Dst const &dst,
                                             int tile, int stage) {
  copy(copy_op, src(_, _, _, tile), dst(_, _, _, stage));
}

__global__ __launch_bounds__(kThreads, 2)
void fa2_4096_kernel(const Element *__restrict__ q,
                     const Element *__restrict__ k,
                     const Element *__restrict__ v,
                     Element *__restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto *q_smem = reinterpret_cast<Element *>(smem_raw);
  auto *k_smem = q_smem + cosize(SmemLayoutQ{});
  auto *v_smem = k_smem + cosize(SmemLayoutK{});

  Tensor mQ = make_tensor(make_gmem_ptr(q),
                          make_shape(SeqLen{}, HeadDim{}),
                          make_stride(HeadDim{}, _1{}));
  Tensor mK = make_tensor(make_gmem_ptr(k),
                          make_shape(SeqLen{}, HeadDim{}),
                          make_stride(HeadDim{}, _1{}));
  Tensor mVt = make_tensor(make_gmem_ptr(v),
                           make_shape(HeadDim{}, SeqLen{}),
                           make_stride(_1{}, HeadDim{}));
  Tensor mO = make_tensor(make_gmem_ptr(out),
                          make_shape(SeqLen{}, HeadDim{}),
                          make_stride(HeadDim{}, _1{}));

  Tensor gQ = local_tile(mQ, make_tile(BlockM{}, HeadDim{}),
                         make_coord(blockIdx.x, 0));
  Tensor gK = local_tile(mK, make_tile(BlockN{}, HeadDim{}),
                         make_coord(_, 0));
  Tensor gVt = local_tile(mVt, make_tile(HeadDim{}, BlockN{}),
                          make_coord(0, _));
  Tensor gO = local_tile(mO, make_tile(BlockM{}, HeadDim{}),
                         make_coord(blockIdx.x, 0));

  Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(k_smem), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(v_smem), SmemLayoutV{});

  G2SRow g2s_row;
  auto g2s_row_thr = g2s_row.get_slice(threadIdx.x);
  Tensor tQgQ = g2s_row_thr.partition_S(gQ);
  Tensor tQsQ = g2s_row_thr.partition_D(sQ);
  Tensor tKgK = g2s_row_thr.partition_S(gK);
  Tensor tKsK = g2s_row_thr.partition_D(sK);

  G2SCol g2s_col;
  auto g2s_col_thr = g2s_col.get_slice(threadIdx.x);
  Tensor tVgV = g2s_col_thr.partition_S(gVt);
  Tensor tVsV = g2s_col_thr.partition_D(sV);

  // Prologue: Q is invariant; K0/V0 occupy stage zero.
  copy(g2s_row, tQgQ, tQsQ);
  copy_kv_tile(g2s_row, tKgK, tKsK, 0, 0);
  copy_kv_tile(g2s_col, tVgV, tVsV, 0, 0);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  Tensor trQ = thr_mma.partition_fragment_A(gQ);
  auto q_s2r = make_tiled_copy_A(S2RAtomN{}, tiled_mma);
  auto q_s2r_thr = q_s2r.get_slice(threadIdx.x);
  Tensor tQsQ_mma = q_s2r_thr.partition_S(sQ);
  Tensor tQrQ = q_s2r_thr.retile_D(trQ);
  copy(q_s2r, tQsQ_mma, tQrQ);

  auto score_layout = make_layout(make_shape(BlockM{}, BlockN{}),
                                  make_stride(BlockN{}, _1{}));
  auto score_tensor = make_tensor(
      make_gmem_ptr(static_cast<Accum *>(nullptr)), score_layout);
  Tensor trS = thr_mma.partition_fragment_C(score_tensor);

  Tensor trO = thr_mma.partition_fragment_C(gO);
  clear(trO);

  // C-fragment coordinates -> score-matrix offset -> A-fragment coordinates.
  // The mapping is compile-time; only the resulting register view exists at
  // run time.
  auto compact_score = make_tensor(static_cast<Element *>(nullptr),
                                   make_layout(make_shape(BlockM{}, BlockN{})));
  auto layout_as_c = thr_mma.partition_C(compact_score).layout();
  auto layout_as_a = thr_mma.partition_A(compact_score).layout();
  auto a_to_c = left_inverse(layout_as_c).compose(layout_as_a);

  auto k_s2r = make_tiled_copy_B(S2RAtomN{}, tiled_mma);
  auto k_s2r_thr = k_s2r.get_slice(threadIdx.x);
  Tensor tKsK_mma = k_s2r_thr.partition_S(sK);

  auto v_s2r = make_tiled_copy_B(S2RAtomT{}, tiled_mma);
  auto v_s2r_thr = v_s2r.get_slice(threadIdx.x);
  Tensor tVsV_mma = v_s2r_thr.partition_S(sV);

  float running_max[2] = {-INFINITY, -INFINITY};
  float running_sum[2] = {0.0f, 0.0f};
  constexpr float kScale = 1.0f / 8.0f;

  int read_stage = 0;
#pragma unroll 1
  for (int tile = 0; tile < kKvTiles; ++tile) {
    const int next_tile = tile + 1;
    const int write_stage = read_stage ^ 1;

    // Submit K[t+1]/V[t+1] before computing tile t.  The destination stage is
    // disjoint from the stage consumed below.
    if (next_tile < kKvTiles) {
      copy_kv_tile(g2s_row, tKgK, tKsK, next_tile, write_stage);
      copy_kv_tile(g2s_col, tVgV, tVsV, next_tile, write_stage);
      cp_async_fence();
    }

    Tensor trK = thr_mma.partition_fragment_B(gK(_, _, 0));
    Tensor tKrK = k_s2r_thr.retile_D(trK);
    copy(k_s2r, tKsK_mma(_, _, _, read_stage), tKrK);

    clear(trS);
    gemm(tiled_mma, trQ, trK, trS);

    // Each lane owns two values for each of two rows in every 8-column MMA_N
    // fragment.  A 4-lane subgroup owns the complete 64-column score row.
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
      for (int row_item = 0; row_item < 2; ++row_item) {
        tile_max[row_item] =
            fmaxf(tile_max[row_item],
                  fmaxf(trS(make_coord(0, row_item), 0, nj),
                        trS(make_coord(1, row_item), 0, nj)));
      }
    }
    tile_max[0] = subgroup4_max(tile_max[0]) * kScale;
    tile_max[1] = subgroup4_max(tile_max[1]) * kScale;

    const float new_max0 = fmaxf(running_max[0], tile_max[0]);
    const float new_max1 = fmaxf(running_max[1], tile_max[1]);
    const float alpha0 = running_sum[0] == 0.0f
                             ? 0.0f
                             : __expf(running_max[0] - new_max0);
    const float alpha1 = running_sum[1] == 0.0f
                             ? 0.0f
                             : __expf(running_max[1] - new_max1);

    float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
      for (int row_item = 0; row_item < 2; ++row_item) {
#pragma unroll
        for (int col_item = 0; col_item < 2; ++col_item) {
          const float shifted =
              kScale * trS(make_coord(col_item, row_item), 0, nj) -
              (row_item == 0 ? new_max0 : new_max1);
          const float probability = __expf(shifted);
          trS(make_coord(col_item, row_item), 0, nj) = probability;
          tile_sum[row_item] += probability;
        }
      }
    }
    tile_sum[0] = subgroup4_sum(tile_sum[0]);
    tile_sum[1] = subgroup4_sum(tile_sum[1]);
    running_sum[0] = alpha0 * running_sum[0] + tile_sum[0];
    running_sum[1] = alpha1 * running_sum[1] + tile_sum[1];
    running_max[0] = new_max0;
    running_max[1] = new_max1;

    // Rescale the old numerator before accumulating the current P@V tile.
#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
      for (int col_item = 0; col_item < 2; ++col_item) {
        trO(make_coord(col_item, 0), 0, nj) *= alpha0;
        trO(make_coord(col_item, 1), 0, nj) *= alpha1;
      }
    }

    Tensor trP_as_c = make_tensor_like<Element>(trS);
#pragma unroll
    for (int i = 0; i < size(trS); ++i) {
      trP_as_c(i) = Element(trS(i));
    }
    auto trP_as_a = trP_as_c.compose(a_to_c);

    Tensor trV = thr_mma.partition_fragment_B(gVt(_, _, 0));
    Tensor tVrV = v_s2r_thr.retile_D(trV);
    copy(v_s2r, tVsV_mma(_, _, _, read_stage), tVrV);
    gemm(tiled_mma, trP_as_a, trV, trO);

    if (next_tile < kKvTiles) {
      cp_async_wait<0>();
      __syncthreads();
      read_stage = write_stage;
    }
  }

  const float inv_sum0 = 1.0f / running_sum[0];
  const float inv_sum1 = 1.0f / running_sum[1];
  Tensor trO_half = make_tensor_like<Element>(trO);
#pragma unroll
  for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
    for (int col_item = 0; col_item < 2; ++col_item) {
      trO_half(make_coord(col_item, 0), 0, nj) =
          Element(trO(make_coord(col_item, 0), 0, nj) * inv_sum0);
      trO_half(make_coord(col_item, 1), 0, nj) =
          Element(trO(make_coord(col_item, 1), 0, nj) * inv_sum1);
    }
  }

  // K/V are dead.  Reuse the beginning of dynamic shared memory as a
  // swizzled output exchange: scattered MMA C stores followed by coalesced
  // 16-byte global stores.
  Tensor sO = make_tensor(make_smem_ptr(q_smem), SmemLayoutO{});
  Tensor tOsO = thr_mma.partition_C(sO);
  copy(trO_half, tOsO);
  __syncthreads();

  S2GRow s2g;
  auto s2g_thr = s2g.get_slice(threadIdx.x);
  Tensor tOsO_vec = s2g_thr.partition_S(sO);
  Tensor tOgO_vec = s2g_thr.partition_D(gO);
  copy(s2g, tOsO_vec, tOgO_vec);
}

void check_cuda(cudaError_t status, const char *expression, const char *file,
                int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                 expression, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(expr)                                                       \
  ::cute_fa2_4096::check_cuda((expr), #expr, __FILE__, __LINE__)

void print_layouts() {
  TiledMma tiled_mma;
  std::printf("TiledMMA:\n");
  print(tiled_mma);
  std::printf("\nG2S row copy:\n");
  print(G2SRow{});
  std::printf("\nG2S column copy:\n");
  print(G2SCol{});
  std::printf("\nSmem Q: ");
  print(SmemLayoutQ{});
  std::printf("\nSmem K: ");
  print(SmemLayoutK{});
  std::printf("\nSmem V: ");
  print(SmemLayoutV{});
  auto s2r_n = make_tiled_copy_B(S2RAtomN{}, tiled_mma);
  auto s2r_t = make_tiled_copy_B(S2RAtomT{}, tiled_mma);
  std::printf("\nS2R K (ldmatrix.x4):\n");
  print(s2r_n);
  std::printf("\nS2R V (ldmatrix.x4.trans):\n");
  print(s2r_t);

  auto compact = make_tensor(static_cast<Element *>(nullptr),
                             make_layout(make_shape(BlockM{}, BlockN{})));
  auto thr_mma = tiled_mma.get_slice(0);
  auto c_partition = thr_mma.partition_C(compact);
  auto a_partition = thr_mma.partition_A(compact);
  auto tr_c = make_tensor<Accum>(shape(c_partition));
  auto tr_p = make_tensor_like<Element>(tr_c);
  auto a_to_c = left_inverse(c_partition.layout())
                    .compose(a_partition.layout());
  auto tr_p_as_a = tr_p.compose(a_to_c);
  std::printf("\nScore/P as C: ");
  print(tr_p.layout());
  std::printf("\nP as A:       ");
  print(tr_p_as_a.layout());
  std::printf("\nshared memory: %d bytes\n", kSmemBytes);
}

} // namespace cute_fa2_4096

int main(int argc, char **argv) {
  using namespace cute_fa2_4096;

  const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 20;
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 8) {
    std::fprintf(stderr, "This example requires compute capability 8.0+\n");
    return EXIT_FAILURE;
  }

  constexpr int S = int(SeqLen{});
  constexpr int D = int(HeadDim{});
  const size_t elements = size_t(S) * D;
  std::vector<Element> h_q(elements);
  std::vector<Element> h_k(elements);
  std::vector<Element> h_v(elements);
  std::vector<Element> h_o(elements);

  for (size_t i = 0; i < elements; ++i) {
    h_q[i] = Element(float(int((i * 17 + 3) % 29) - 14) / 64.0f);
    h_k[i] = Element(float(int((i * 13 + 5) % 31) - 15) / 64.0f);
    h_v[i] = Element(float(int((i * 7 + 11) % 23) - 11) / 16.0f);
  }

  Element *d_q = nullptr;
  Element *d_k = nullptr;
  Element *d_v = nullptr;
  Element *d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, elements * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_k, elements * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_v, elements * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_o, elements * sizeof(Element)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), elements * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), elements * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), elements * sizeof(Element),
                        cudaMemcpyHostToDevice));

  const dim3 grid(S / int(BlockM{}));
  for (int i = 0; i < 3; ++i) {
    fa2_4096_kernel<<<grid, kThreads, kSmemBytes>>>(d_q, d_k, d_v, d_o);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    fa2_4096_kernel<<<grid, kThreads, kSmemBytes>>>(d_q, d_k, d_v, d_o);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  elapsed_ms /= iterations;

  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, elements * sizeof(Element),
                        cudaMemcpyDeviceToHost));

  // A full CPU O(S^2D) reference is unnecessarily expensive.  Check the first
  // row owned by every warp in every CTA (256 rows total), while also scanning
  // the complete output for NaN/Inf.
  std::vector<int> rows;
  rows.reserve((S / int(BlockM{})) * 4);
  for (int block = 0; block < S / int(BlockM{}); ++block) {
    for (int warp = 0; warp < 4; ++warp) {
      rows.push_back(block * int(BlockM{}) + warp * 16);
    }
  }
  std::vector<float> scores(int(BlockN{}));
  std::vector<Element> probabilities(int(BlockN{}));
  float max_error = 0.0f;
  for (int row : rows) {
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    std::array<float, D> numerator{};
    for (int tile = 0; tile < kKvTiles; ++tile) {
      float tile_max = -INFINITY;
      for (int item = 0; item < int(BlockN{}); ++item) {
        const int col = tile * int(BlockN{}) + item;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
          dot += float(h_q[size_t(row) * D + d]) *
                 float(h_k[size_t(col) * D + d]);
        }
        scores[item] = dot * 0.125f;
        tile_max = std::max(tile_max, scores[item]);
      }

      const float new_max = std::max(running_max, tile_max);
      const float alpha = running_sum == 0.0f
                              ? 0.0f
                              : std::exp(running_max - new_max);
      float tile_sum = 0.0f;
      for (int item = 0; item < int(BlockN{}); ++item) {
        const float probability = std::exp(scores[item] - new_max);
        tile_sum += probability;
        probabilities[item] = Element(probability);
      }
      running_sum = alpha * running_sum + tile_sum;
      running_max = new_max;

      for (int d = 0; d < D; ++d) {
        numerator[d] *= alpha;
        for (int item = 0; item < int(BlockN{}); ++item) {
          const int col = tile * int(BlockN{}) + item;
          numerator[d] += float(probabilities[item]) *
                          float(h_v[size_t(col) * D + d]);
        }
      }
    }

    for (int d = 0; d < D; ++d) {
      const Element reference = Element(numerator[d] / running_sum);
      max_error =
          std::max(max_error,
                   std::abs(float(reference) -
                            float(h_o[size_t(row) * D + d])));
    }
  }

  bool all_finite = true;
  int non_finite_count = 0;
  int first_non_finite = -1;
  for (int i = 0; i < int(h_o.size()); ++i) {
    if (!std::isfinite(float(h_o[i]))) {
      all_finite = false;
      ++non_finite_count;
      if (first_non_finite < 0) {
        first_non_finite = i;
      }
    }
  }

  const double flops = 4.0 * double(S) * S * D;
  const double tflops = flops / (elapsed_ms * 1.0e9);
  int active_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, fa2_4096_kernel, kThreads, kSmemBytes));
  print_layouts();
  std::printf("shape: Q/K/V=[%d,%d], score=[%d,%d]\n", S, D, S, S);
  std::printf("latency: %.4f ms, mathematical throughput: %.2f TFLOP/s\n",
              elapsed_ms, tflops);
  std::printf("occupancy limit: %d CTA/SM (%d resident warps/SM)\n",
              active_blocks_per_sm, active_blocks_per_sm * kThreads / 32);
  std::printf("sampled-row max abs error: %.8f, all finite: %s\n", max_error,
              all_finite ? "yes" : "no");
  if (!all_finite) {
    std::printf("non-finite outputs: %d, first=(row=%d,col=%d)\n",
                non_finite_count, first_non_finite / D,
                first_non_finite % D);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  const bool passed = all_finite && max_error < 4.0e-3f;
  std::printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
