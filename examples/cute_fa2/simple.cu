#include <cute/tensor.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cute_fa2 {

using namespace cute;

using Element = half_t;
using Accum = float;

using BlockM = Int<16>;
using BlockN = Int<64>;
using HeadDim = Int<64>;

// One warp owns the complete 16x64 score/output tile.  Extending the N
// footprint from the atom's natural 8 columns to 64 gives eight MMA_N modes.
using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMma = decltype(make_tiled_mma(
    MmaAtom{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _64, _16>{}));

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

template <bool UseLayoutAlgebra>
__global__ __launch_bounds__(32)
void attention_kernel(const Element *__restrict__ q,
                      const Element *__restrict__ k,
                      const Element *__restrict__ v,
                      Element *__restrict__ out) {
  // Q and K are row-major.  For P@V, mma(..._TN) wants its B tensor in
  // (output_column, reduction_row) coordinates, so V is exposed as a
  // transposed logical view with stride (1, HeadDim).
  Tensor gQ = make_tensor(make_gmem_ptr(q),
                          make_shape(BlockM{}, HeadDim{}),
                          make_stride(HeadDim{}, _1{}));
  Tensor gK = make_tensor(make_gmem_ptr(k),
                          make_shape(BlockN{}, HeadDim{}),
                          make_stride(HeadDim{}, _1{}));
  Tensor gVt = make_tensor(make_gmem_ptr(v),
                           make_shape(HeadDim{}, BlockN{}),
                           make_stride(_1{}, HeadDim{}));
  Tensor gO = make_tensor(make_gmem_ptr(out),
                          make_shape(BlockM{}, HeadDim{}),
                          make_stride(HeadDim{}, _1{}));

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  Tensor tQgQ = thr_mma.partition_A(gQ);
  Tensor tKgK = thr_mma.partition_B(gK);
  Tensor trQ = thr_mma.partition_fragment_A(gQ);
  Tensor trK = thr_mma.partition_fragment_B(gK);
  copy(tQgQ, trQ);
  copy(tKgK, trK);

  Tensor tOgO = thr_mma.partition_C(gO);
  Tensor trS = thr_mma.make_fragment_C(tOgO);
  clear(trS);

  // trQ: (V8, MMA_M=1, MMA_K=4)
  // trK: (V4, MMA_N=8, MMA_K=4)
  // trS: (V4, MMA_M=1, MMA_N=8)
  gemm(tiled_mma, trQ, trK, trS);

  // The C atom gives each lane two adjacent scores from each of two rows.
  // Four adjacent lanes collectively own all 64 columns of those rows.
  float row_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
    for (int row_item = 0; row_item < 2; ++row_item) {
      row_max[row_item] =
          fmaxf(row_max[row_item],
                fmaxf(trS(make_coord(0, row_item), 0, nj),
                      trS(make_coord(1, row_item), 0, nj)));
    }
  }
  row_max[0] = subgroup4_max(row_max[0]);
  row_max[1] = subgroup4_max(row_max[1]);

  constexpr float softmax_scale = 1.0f / 8.0f; // 1/sqrt(HeadDim)
  float row_sum[2] = {0.0f, 0.0f};
#pragma unroll
  for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
    for (int row_item = 0; row_item < 2; ++row_item) {
#pragma unroll
      for (int col_item = 0; col_item < 2; ++col_item) {
        // Apply the scale before the stable-softmax subtraction.
        const float probability =
            __expf(softmax_scale *
                       trS(make_coord(col_item, row_item), 0, nj) -
                   softmax_scale * row_max[row_item]);
        row_sum[row_item] += probability;
        trS(make_coord(col_item, row_item), 0, nj) = probability;
      }
    }
  }
  row_sum[0] = subgroup4_sum(row_sum[0]);
  row_sum[1] = subgroup4_sum(row_sum[1]);

  // P must be rounded to FP16 for the second Tensor Core operation.  It is
  // still physically stored in the C-fragment register order at this point.
  Tensor trP_as_C = make_tensor_like<Element>(trS);
#pragma unroll
  for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
    for (int row_item = 0; row_item < 2; ++row_item) {
#pragma unroll
      for (int col_item = 0; col_item < 2; ++col_item) {
        const float probability =
            trS(make_coord(col_item, row_item), 0, nj);
        trP_as_C(make_coord(col_item, row_item), 0, nj) =
            Element(probability / row_sum[row_item]);
      }
    }
  }

  Tensor tVgV = thr_mma.partition_B(gVt);
  Tensor trV = thr_mma.partition_fragment_B(gVt);
  copy(tVgV, trV);

  Tensor trO = thr_mma.make_fragment_C(tOgO);
  clear(trO);

  if constexpr (UseLayoutAlgebra) {
    // Both partitions use the same compact logical 16x64 coordinate space.
    // C_layout : C-fragment coordinate -> matrix offset
    // A_layout : A-fragment coordinate -> the same matrix offset
    auto compact_tensor = make_tensor(
        static_cast<Element *>(nullptr),
        make_layout(make_shape(BlockM{}, BlockN{})));
    auto a_partition_layout = thr_mma.partition_A(compact_tensor).layout();
    auto c_partition_layout = thr_mma.partition_C(compact_tensor).layout();

    // A-coordinate -> matrix offset -> C-coordinate.  Composing this mapping
    // with trP_as_C then yields A-coordinate -> physical P register offset.
    auto c_layout_inv = left_inverse(c_partition_layout);
    auto a_to_c = c_layout_inv.compose(a_partition_layout);
    auto trP_as_A = trP_as_C.compose(a_to_c);

    CUTE_STATIC_ASSERT_V(size<0>(trP_as_A) == size<0>(trQ));
    CUTE_STATIC_ASSERT_V(size<1>(trP_as_A) == size<1>(trQ));
    CUTE_STATIC_ASSERT_V(size<2>(trP_as_A) == size<2>(trQ));
    gemm(tiled_mma, trP_as_A, trV, trO);
  } else {
    // Split MMA_N=8 into (2,4), then join the inner 2 with the C atom's V4:
    //   (V4, M1, N8) -> ((V4,2), M1, N4) == (V8, M1, K4).
    auto c_layout_div = logical_divide(
        trP_as_C.layout(), Shape<Underscore, Underscore, _2>{});
    auto a_layout = make_layout(
        make_layout(get<0>(c_layout_div), get<2, 0>(c_layout_div)),
        get<1>(c_layout_div), get<2, 1>(c_layout_div));
    auto trP_as_A = make_tensor(trP_as_C.data(), a_layout);

    CUTE_STATIC_ASSERT_V(size<0>(trP_as_A) == size<0>(trQ));
    CUTE_STATIC_ASSERT_V(size<1>(trP_as_A) == size<1>(trQ));
    CUTE_STATIC_ASSERT_V(size<2>(trP_as_A) == size<2>(trQ));
    gemm(tiled_mma, trP_as_A, trV, trO);
  }

  copy(trO, tOgO);
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
  ::cute_fa2::check_cuda((expr), #expr, __FILE__, __LINE__)

void print_layouts() {
  using namespace cute;
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(0);
  auto compact_tensor = make_tensor(
      static_cast<Element *>(nullptr),
      make_layout(make_shape(BlockM{}, BlockN{})));
  auto c_partition = thr_mma.partition_C(compact_tensor);
  auto a_partition = thr_mma.partition_A(compact_tensor);
  auto trC = make_tensor<Accum>(shape(c_partition));
  auto trP = make_tensor_like<Element>(trC);

  auto c_layout_div = logical_divide(
      trP.layout(), Shape<Underscore, Underscore, _2>{});
  auto tri_layout = make_layout(
      make_layout(get<0>(c_layout_div), get<2, 0>(c_layout_div)),
      get<1>(c_layout_div), get<2, 1>(c_layout_div));

  auto a_to_c = left_inverse(c_partition.layout())
                    .compose(a_partition.layout());
  auto reed_tensor = trP.compose(a_to_c);

  std::printf("TiledMMA:\n");
  print(tiled_mma);
  std::printf("\nC fragment layout: ");
  print(trP.layout());
  std::printf("\nTri Dao C-as-A layout: ");
  print(tri_layout);
  std::printf("\nLayout-algebra C-as-A layout: ");
  print(reed_tensor.layout());
  std::printf("\n");
}

} // namespace cute_fa2

int main() {
  using namespace cute_fa2;

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 8) {
    std::fprintf(stderr, "This example requires compute capability 8.0+\n");
    return EXIT_FAILURE;
  }

  constexpr int M = int(BlockM{});
  constexpr int N = int(BlockN{});
  constexpr int D = int(HeadDim{});
  std::vector<Element> h_q(M * D);
  std::vector<Element> h_k(N * D);
  std::vector<Element> h_v(N * D);
  std::vector<Element> h_tri(M * D);
  std::vector<Element> h_algebra(M * D);
  std::vector<Element> h_reference(M * D);

  for (int i = 0; i < M * D; ++i) {
    h_q[i] = Element(float((i * 17 + 3) % 29 - 14) / 32.0f);
  }
  for (int i = 0; i < N * D; ++i) {
    h_k[i] = Element(float((i * 13 + 5) % 31 - 15) / 32.0f);
    h_v[i] = Element(float((i * 7 + 11) % 23 - 11) / 16.0f);
  }

  // Reference mirrors the kernel: FP32 QK accumulation, FP32 softmax,
  // probabilities rounded to FP16, FP32 PV accumulation, FP16 output.
  std::vector<float> scores(N);
  std::vector<Element> probabilities(N);
  for (int row = 0; row < M; ++row) {
    float max_score = -INFINITY;
    for (int col = 0; col < N; ++col) {
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        dot += float(h_q[row * D + d]) * float(h_k[col * D + d]);
      }
      scores[col] = dot * 0.125f;
      max_score = std::max(max_score, scores[col]);
    }
    float sum = 0.0f;
    for (int col = 0; col < N; ++col) {
      scores[col] = std::exp(scores[col] - max_score);
      sum += scores[col];
    }
    for (int col = 0; col < N; ++col) {
      probabilities[col] = Element(scores[col] / sum);
    }
    for (int d = 0; d < D; ++d) {
      float output = 0.0f;
      for (int col = 0; col < N; ++col) {
        output += float(probabilities[col]) * float(h_v[col * D + d]);
      }
      h_reference[row * D + d] = Element(output);
    }
  }

  Element *d_q = nullptr;
  Element *d_k = nullptr;
  Element *d_v = nullptr;
  Element *d_tri = nullptr;
  Element *d_algebra = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_tri, h_tri.size() * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_algebra, h_algebra.size() * sizeof(Element)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(Element),
                        cudaMemcpyHostToDevice));

  attention_kernel<false><<<1, 32>>>(d_q, d_k, d_v, d_tri);
  attention_kernel<true><<<1, 32>>>(d_q, d_k, d_v, d_algebra);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_tri.data(), d_tri, h_tri.size() * sizeof(Element),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_algebra.data(), d_algebra,
                        h_algebra.size() * sizeof(Element),
                        cudaMemcpyDeviceToHost));

  float tri_error = 0.0f;
  float algebra_error = 0.0f;
  float methods_error = 0.0f;
  int method_mismatches = 0;
  for (int i = 0; i < M * D; ++i) {
    tri_error = std::max(
        tri_error, std::abs(float(h_tri[i]) - float(h_reference[i])));
    algebra_error = std::max(
        algebra_error,
        std::abs(float(h_algebra[i]) - float(h_reference[i])));
    methods_error = std::max(
        methods_error, std::abs(float(h_tri[i]) - float(h_algebra[i])));
    method_mismatches += float(h_tri[i]) != float(h_algebra[i]);
  }

  print_layouts();
  std::printf("Tri Dao max abs error:       %.8f\n", tri_error);
  std::printf("Layout algebra max abs error: %.8f\n", algebra_error);
  std::printf("Between methods:              %.8f (%d mismatches)\n",
              methods_error, method_mismatches);

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_tri));
  CUDA_CHECK(cudaFree(d_algebra));

  const bool passed = tri_error < 2.0e-3f && algebra_error < 2.0e-3f &&
                      method_mismatches == 0;
  std::printf("%s\n", passed ? "PASS" : "FAIL");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
