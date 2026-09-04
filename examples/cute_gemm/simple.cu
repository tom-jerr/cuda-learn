#include <cute/tensor.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// A compact, standalone version of the attached SM80 CuTe HGEMM.  It computes
//   D[M,N] = A[M,K] * B[N,K]^T
// where all three allocations are row-major.  Requiring aligned shapes keeps
// predication separate from the CuTe hierarchy demonstrated here.

namespace cute_example {

using namespace cute;
using Element = half_t;

using BM = Int<128>;
using BN = Int<128>;
using BK = Int<32>;
using Stages = Int<2>;

// Logical (row, k, stage) coordinates are XOR-swizzled before becoming shared
// memory offsets.  The 8x32 atom is then repeated to the CTA tile shape.
using SmemLayoutAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    make_layout(make_shape(Int<8>{}, BK{}),
                make_stride(BK{}, Int<1>{}))));
using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtom{}, make_shape(BM{}, BK{}, Stages{})));
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtom{}, make_shape(BN{}, BK{}, Stages{})));

// One Atom is one warp-level m16n8k16 Tensor Core instruction.  The atom
// layout places 2x2 warps over MxN (128 threads), and the 32x32x16 permutation
// describes the value tile consumed by those atoms.
using MmaAtom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
using TiledMma = decltype(make_tiled_mma(
    MmaAtom{},
    make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})),
    Tile<Int<32>, Int<32>, Int<16>>{}));

// 128 threads, each cp.async moving eight FP16 values (16 bytes).  The thread
// and value layouts describe ownership; partition_S/D apply that ownership to
// actual tensors.
using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>;
using G2SCopy = decltype(make_tiled_copy(
    G2SCopyAtom{},
    make_layout(make_shape(Int<32>{}, Int<4>{}),
                make_stride(Int<4>{}, Int<1>{})),
    make_layout(make_shape(Int<1>{}, Int<8>{}))));

using S2RCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

template <class MMA, class CopyG2S, class LayoutA, class LayoutB>
__global__ void cute_hgemm_kernel(const Element* a, const Element* b,
                                  Element* d, int m, int n, int k) {
  extern __shared__ Element smem[];
  Element* a_smem = smem;
  Element* b_smem = smem + cosize(LayoutA{});

  // Tensor = Engine(pointer/address space) + Layout(shape -> offset).
  Tensor mA = make_tensor(make_gmem_ptr(a), make_shape(m, k),
                          make_stride(k, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(b), make_shape(n, k),
                          make_stride(k, Int<1>{}));
  Tensor mD = make_tensor(make_gmem_ptr(d), make_shape(m, n),
                          make_stride(n, Int<1>{}));

  // Each CTA owns one output tile; '_' preserves K as a tile-index mode.
  Tensor gA = local_tile(mA, make_tile(BM{}, BK{}),
                         make_coord(blockIdx.y, _));       // (BM,BK,k_tile)
  Tensor gB = local_tile(mB, make_tile(BN{}, BK{}),
                         make_coord(blockIdx.x, _));       // (BN,BK,k_tile)
  Tensor gD = local_tile(mD, make_tile(BM{}, BN{}),
                         make_coord(blockIdx.y, blockIdx.x));
  Tensor sA = make_tensor(make_smem_ptr(a_smem), LayoutA{}); // (BM,BK,stage)
  Tensor sB = make_tensor(make_smem_ptr(b_smem), LayoutB{}); // (BN,BK,stage)

  MMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tDgD = thr_mma.partition_C(gD);
  Tensor tDrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  Tensor tDrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  Tensor tDrD = thr_mma.make_fragment_C(tDgD);
  clear(tDrD);

  CopyG2S g2s_copy;
  auto g2s_thr = g2s_copy.get_slice(threadIdx.x);
  Tensor tAgA = g2s_thr.partition_S(gA);
  Tensor tAsA = g2s_thr.partition_D(sA);
  Tensor tBgB = g2s_thr.partition_S(gB);
  Tensor tBsB = g2s_thr.partition_D(sB);

  // Deriving these copies from TiledMMA guarantees that ldmatrix output lands
  // in exactly the register layout expected by the MMA atom.
  auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtom{}, tiled_mma);
  auto s2r_thr_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tAsA_mma = s2r_thr_a.partition_S(sA);
  Tensor tArA = s2r_thr_a.retile_D(tDrA);

  auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtom{}, tiled_mma);
  auto s2r_thr_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tBsB_mma = s2r_thr_b.partition_S(sB);
  Tensor tBrB = s2r_thr_b.retile_D(tDrB);

  constexpr int stage_count = decltype(size<3>(tAsA))::value;
  const int k_tile_count = size<3>(tAgA);
  int next_gmem_tile = 0;
  int read_stage = 0;
  int write_stage = 0;

  // Fill all but one pipeline slots.  With two stages this submits tile zero.
#pragma unroll
  for (int stage = 0; stage < stage_count - 1; ++stage) {
    copy(g2s_copy, tAgA(_, _, _, next_gmem_tile),
         tAsA(_, _, _, write_stage));
    copy(g2s_copy, tBgB(_, _, _, next_gmem_tile),
         tBsB(_, _, _, write_stage));
    cp_async_fence();
    ++next_gmem_tile;
    write_stage = (write_stage + 1) % stage_count;
  }

  cp_async_wait<stage_count - 2>();
  __syncthreads();

  // Register prefetch for the first k16 fragment.
  copy(s2r_copy_a, tAsA_mma(_, _, 0, read_stage), tArA(_, _, 0));
  copy(s2r_copy_b, tBsB_mma(_, _, 0, read_stage), tBrB(_, _, 0));

  const int mma_k_blocks = size<2>(tDrA); // BK / MMA_K = 2 here.
  for (int tile = 0; tile < k_tile_count; ++tile) {
#pragma unroll
    for (int k_block = 0; k_block < mma_k_blocks; ++k_block) {
      const int next_k_block = (k_block + 1) % mma_k_blocks;

      if (k_block == mma_k_blocks - 1) {
        cp_async_wait<stage_count - 2>();
        __syncthreads();
        read_stage = (read_stage + 1) % stage_count;
      }

      // Preload the fragment consumed by the following MMA operation.
      copy(s2r_copy_a, tAsA_mma(_, _, next_k_block, read_stage),
           tArA(_, _, next_k_block));
      copy(s2r_copy_b, tBsB_mma(_, _, next_k_block, read_stage),
           tBrB(_, _, next_k_block));

      if (k_block == 0) {
        if (next_gmem_tile < k_tile_count) {
          copy(g2s_copy, tAgA(_, _, _, next_gmem_tile),
               tAsA(_, _, _, write_stage));
          copy(g2s_copy, tBgB(_, _, _, next_gmem_tile),
               tBsB(_, _, _, write_stage));
          ++next_gmem_tile;
          write_stage = (write_stage + 1) % stage_count;
        }
        cp_async_fence();
      }

      gemm(tiled_mma, tDrA(_, _, k_block), tDrB(_, _, k_block), tDrD);
    }
  }

  // partition_C gives gmem and register fragments identical logical ownership.
  // This scalar/vector-agnostic teaching epilogue is correct but less coalesced
  // than the attached R2S -> shared -> S2G TiledCopy epilogue.
  copy(tDrD, tDgD);
}

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error_ = (call);                                                \
    if (error_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(error_));                                 \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

bool valid_shape(int m, int n, int k) {
  return m > 0 && n > 0 && k >= int(BK{}) && m % int(BM{}) == 0 &&
         n % int(BN{}) == 0 && k % int(BK{}) == 0;
}

void launch(const Element* a, const Element* b, Element* d,
            int m, int n, int k) {
  if (!valid_shape(m, n, k)) {
    std::fprintf(stderr,
                 "shape must satisfy M%%128=0, N%%128=0, K%%32=0 and K>=32\n");
    std::exit(EXIT_FAILURE);
  }

  constexpr int threads = decltype(size(TiledMma{}))::value;
  constexpr int smem_elements = cosize(SmemLayoutA{}) + cosize(SmemLayoutB{});
  constexpr int smem_bytes = smem_elements * sizeof(Element);
  dim3 grid(n / int(BN{}), m / int(BM{}));
  cute_hgemm_kernel<TiledMma, G2SCopy, SmemLayoutA, SmemLayoutB>
      <<<grid, threads, smem_bytes>>>(a, b, d, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace cute_example

int main(int argc, char** argv) {
  using namespace cute;
  using namespace cute_example;
  const int m = argc > 1 ? std::atoi(argv[1]) : 256;
  const int n = argc > 2 ? std::atoi(argv[2]) : 256;
  const int k = argc > 3 ? std::atoi(argv[3]) : 256;

  if (!valid_shape(m, n, k)) {
    std::fprintf(stderr,
                 "shape must satisfy M%%128=0, N%%128=0, K%%32=0 and K>=32\n");
    return EXIT_FAILURE;
  }

  std::printf("CuTe TN HGEMM: D[%d,%d] = A[%d,%d] * B[%d,%d]^T\n",
              m, n, m, k, n, k);
  std::printf("TiledMMA threads: %d, dynamic shared memory: %zu bytes\n",
              int(size(TiledMma{})),
              size_t(cosize(SmemLayoutA{}) + cosize(SmemLayoutB{})) *
                  sizeof(Element));
  std::printf("TiledMMA: ");
  print(TiledMma{});
  std::printf("\nG2SCopy: ");
  print(G2SCopy{});
  std::printf("\nSmemLayoutA: ");
  print(SmemLayoutA{});
  std::printf("\n");

  std::vector<Element> h_a(size_t(m) * k);
  std::vector<Element> h_b(size_t(n) * k);
  std::vector<Element> h_d(size_t(m) * n);
  for (size_t i = 0; i < h_a.size(); ++i) {
    h_a[i] = Element(float(int(i % 13) - 6) / 16.0f);
  }
  for (size_t i = 0; i < h_b.size(); ++i) {
    h_b[i] = Element(float(int(i % 11) - 5) / 16.0f);
  }

  Element *d_a = nullptr, *d_b = nullptr, *d_d = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_d, h_d.size() * sizeof(Element)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(Element),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(Element),
                        cudaMemcpyHostToDevice));

  launch(d_a, d_b, d_d, m, n, k);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_d.data(), d_d, h_d.size() * sizeof(Element),
                        cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float expected = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        expected += float(h_a[size_t(row) * k + kk]) *
                    float(h_b[size_t(col) * k + kk]);
      }
      max_error = std::max(
          max_error, std::abs(expected - float(h_d[size_t(row) * n + col])));
    }
  }
  std::printf("max abs error: %.6f (%s)\n", max_error,
              max_error < 0.5f ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_d));
  return max_error < 0.5f ? EXIT_SUCCESS : EXIT_FAILURE;
}
