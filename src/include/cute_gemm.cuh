#pragma once
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

// C[M,N] = A[M,K] * BT[N,K]^T，三者均为 row-major。
// 本配置用于32x32入门变体；128x128全half教程见docs/cute_gemm_81920.md。
namespace cute_tutorial {
using namespace cute;
using Input = half_t;
using Output = float;
constexpr int BlockM = 32, BlockN = 32, BlockK = 32, Threads = 128;

enum class Lesson {
  ScalarLoad = 2,       // 普通 shared load -> MMA -> global
  MatrixLoad = 3,       // ldmatrix -> MMA -> global
  SharedOutput = 4,     // ldmatrix -> MMA -> shared -> global
  DoubleBuffer = 5,     // 两个 shared stage，预取下一 tile
  SwizzledPipeline = 6  // 双缓冲 + 输入 shared 地址 swizzle
};
using TileShape = Shape<_32, _32>;
using OutputLayout = Layout<TileShape, Stride<_32, _1>>;
using TiledMma =
    decltype(make_tiled_mma(MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{}, Layout<Shape<_2, _2, _1>>{}));
using InputCopyThreads = Layout<Shape<_32, _4>, Stride<_4, _1>>;
using InputCopyValues = Layout<Shape<_1, _8>>;
using GlobalToShared =
    decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Input>{},
                             InputCopyThreads{}, InputCopyValues{}));
using MatrixLoadA = Copy_Atom<SM75_U32x4_LDSM_N, Input>;
using MatrixLoadB = Copy_Atom<SM75_U32x2_LDSM_N, Input>;
using OutputScatter = Copy_Atom<UniversalCopy<Output>, Output>;
using OutputCopy =
    decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Output>{},
                             Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _4>>{}));

// stage 是物理缓冲编号，与 global K tile 编号是不同的 mode。
template <int StageCount, bool UseSwizzle = false>
struct Config {
  static_assert(StageCount == 1 || StageCount == 2);
  static constexpr int Stages = StageCount;
  static constexpr bool Swizzled = UseSwizzle;
  using PlainLayout = Layout<Shape<_32, _32, Int<StageCount>>, Stride<_32, _1, _1024>>;
  using SwizzleAtom =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using SwizzledLayout = decltype(tile_to_shape(SwizzleAtom{}, Shape<_32, _32, Int<StageCount>>{}));
  using InputLayout = conditional_t<UseSwizzle, SwizzledLayout, PlainLayout>;
};
using SingleStage = Config<1>;
using TwoStages = Config<2>;
using TwoSwizzledStages = Config<2, true>;
static_assert(size(TiledMma{}) == Threads);
static_assert(size(GlobalToShared{}) == Threads);
static_assert(size(OutputCopy{}) == Threads);
static_assert(cosize(TwoSwizzledStages::InputLayout{}) == 2048);

// 要求正数 M/N/K 均为 32 的倍数，输入/输出指针 16-byte 对齐。
// 返回 launch 状态；调用者保证缓冲区足够大，并等待所在 stream 完成。
cudaError_t launch_gemm(Input const* a, Input const* bt, Output* c, int m, int n, int k,
                        Lesson lesson, cudaStream_t stream = nullptr);
cudaError_t launch_copy_roundtrip(Input const* src, Input* dst, cudaStream_t stream = nullptr);
}  // namespace cute_tutorial
