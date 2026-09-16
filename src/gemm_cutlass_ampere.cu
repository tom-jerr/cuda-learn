#include "ffi_common.h"

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include <climits>

using tvm::ffi::TensorView;

namespace {

using namespace cute;

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using GmemLayout = cutlass::layout::RowMajor;

// CUTLASS 3.x represents a GEMM as ProblemShape + CollectiveMainloop +
// CollectiveEpilogue. SM80 has no CollectiveBuilder specialization, so all
// CuTe atoms below are intentionally explicit rather than falling back to the
// backward-compatible CUTLASS 2.x device::Gemm interface.
using ProblemShape = Shape<int, int, int, int>; // M, N, K, batch
using TileShape = Shape<_128, _128, _32>;
using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<3>;

using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>,
    Layout<Shape<_2, _2, _1>>, Tile<_32, _32, _16>>;

// A is row-major in logical (M,K) coordinates.
using SmemLayoutAtomA = decltype(composition(
    Swizzle<2, 3, 3>{},
    Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
using GmemTiledCopyA = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
    Layout<Shape<_32, _4>, Stride<_4, _1>>{},
    Layout<Shape<_1, _8>>{}));
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

// CUTLASS exposes B as logical (N,K). A row-major B[K,N] is therefore
// N-major: stride(n)=1 and stride(k)=N. The transposed LDSM atom produces the
// B fragment layout expected by mma.sync without materializing B^T in HBM.
using SmemLayoutAtomB = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
using GmemTiledCopyB = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
    Layout<Shape<_16, _8>, Stride<_1, _16>>{},
    Layout<Shape<_8, _1>>{}));
using SmemCopyAtomB = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

using StrideA = cutlass::gemm::TagToStrideA_t<GmemLayout>;
using StrideB = cutlass::gemm::TagToStrideB_t<GmemLayout>;
using StrideC = cutlass::gemm::TagToStrideC_t<GmemLayout>;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy, TileShape, Element, StrideA, Element, StrideB, TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;

// DefaultEpilogue is the CUTLASS 3.x collective boundary. The SM80 default
// implementation visits the accumulator fragment with scalar predication;
// alpha/beta computation stays FP32 and the final conversion is BF16.
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    Element, 1, Accumulator, float>;
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    Element, StrideC, StrideC, EpilogueOp,
    cutlass::gemm::EpilogueDefault>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Cutlass3AmpereGemm =
    cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

static_assert(DispatchPolicy::Stages == 3);
static_assert(size(TiledMma{}) == 128);
static_assert(GemmKernel::MaxThreadsPerBlock == 128);
static_assert(GemmKernel::SharedStorageSize == 48 * 1024);

void check_bf16_matrix(const TensorView &tensor, const char *name) {
  if (tensor.data_ptr() == nullptr || tensor.ndim() != 2 ||
      tensor.dtype().code != kDLBfloat || tensor.dtype().bits != 16 ||
      tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected a non-null 2D CUDA bfloat16 tensor";
  }
}

void check_cutlass(cutlass::Status status, const char *where) {
  if (status != cutlass::Status::kSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "gemm_cutlass_ampere: " << where << " failed: "
        << cutlassGetStatusString(status);
  }
}

} // namespace

void gemm_cutlass_ampere(TensorView a, TensorView b, TensorView c) {
  check_bf16_matrix(a, "a");
  check_bf16_matrix(b, "b");
  check_bf16_matrix(c, "c");

  const int64_t m64 = dim(a, 0);
  const int64_t k64 = dim(a, 1);
  const int64_t n64 = dim(b, 1);
  if (m64 <= 0 || n64 <= 0 || k64 <= 0 || dim(b, 0) != k64 ||
      dim(c, 0) != m64 || dim(c, 1) != n64) {
    TVM_FFI_THROW(RuntimeError)
        << "gemm_cutlass_ampere: expected non-empty a[M,K] @ b[K,N] -> "
           "c[M,N]";
  }
  if (a.device().device_id != b.device().device_id ||
      a.device().device_id != c.device().device_id) {
    TVM_FFI_THROW(RuntimeError)
        << "gemm_cutlass_ampere: all tensors must be on the same CUDA device";
  }
  if ((k64 % 8) != 0 || (n64 % 8) != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "gemm_cutlass_ampere: K and N must be multiples of 8 for aligned "
           "128-bit BF16 accesses, got K="
        << k64 << ", N=" << n64;
  }
  if (m64 > INT_MAX || n64 > INT_MAX || k64 > INT_MAX) {
    TVM_FFI_THROW(RuntimeError)
        << "gemm_cutlass_ampere: dimensions must fit in a 32-bit int";
  }

  const int m = static_cast<int>(m64);
  const int n = static_cast<int>(n64);
  const int k = static_cast<int>(k64);
  const int64_t mk = m64 * k64;
  const int64_t kn = k64 * n64;
  const int64_t mn = m64 * n64;
  const ProblemShape problem{m, n, k, 1};
  const StrideA stride_a{int64_t(k), _1{}, mk};
  const StrideB stride_b{_1{}, int64_t(n), kn};
  const StrideC stride_c{int64_t(n), _1{}, mn};

  const auto *a_ptr = static_cast<const Element *>(a.data_ptr());
  const auto *b_ptr = static_cast<const Element *>(b.data_ptr());
  auto *c_ptr = static_cast<Element *>(c.data_ptr());
  typename Cutlass3AmpereGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {a_ptr, stride_a, b_ptr, stride_b},
      {{1.0f, 0.0f}, c_ptr, stride_c, c_ptr, stride_c}};

  check_cutlass(Cutlass3AmpereGemm::can_implement(arguments),
                "can_implement");
  Cutlass3AmpereGemm operation;
  check_cutlass(operation(arguments, nullptr, get_stream(a)), "launch");
}

CUDA_LEARN_REGISTER("cuda_learn.gemm_cutlass_ampere", gemm_cutlass_ampere);
