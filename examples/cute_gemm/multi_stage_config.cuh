#pragma once
#include <cute/tensor.hpp>

// 用户提供的 reed-lau/cute-gemm 多级 GEMM 的固定配置。
// 本版只讨论 M=81920,N=256,K=256；完整推导见 docs/cute_gemm_81920.md。
namespace half_gemm {
using namespace cute;
struct Config {
  using T = half_t;  // 输入、累加 fragment、输出都是 half。
  static constexpr int M = 81920, N = 256, K = 256;
  static constexpr int kTileM = 128, kTileN = 128, kTileK = 32;
  static constexpr int kStage = 3, kSmemLayoutCBatch = 2;

  // A/B 的逻辑 shape=(128,32,3)，共 12288 half/矩阵。
  // 先用 row-major 8x32 定义地址，再将此 atom 平铺到完整 shared。
  // Swizzle<3,3,3> 对元素偏移 x 做 x ^ ((x & 0x1c0) >> 3)。
  // 注意：BK=32 时还会改变行最低位！不能只写成“每行列号 XOR”。
  using SmemLayoutAtom =
      decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<_128, _32, _3>{}));
  using SmemLayoutB = SmemLayoutA;

  // 一条 warp MMA 为 m16n8k16，每 lane A=8 half、B=4 half、C=4 half。
  // F16F16F16F16 的顺序是 D/A/B/C，确实使用 half 累加器路径。
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  static constexpr int kMmaPM = 32, kMmaPN = 32, kMmaPK = 16;
  // 第二个参数：4 个 warp，warp_m=w%2，warp_n=w/2。
  using MMA_EU_RepeatT = Layout<Shape<_2, _2, _1>>;
  // 第三个参数：N 的 16 扩到 32，增加同一 warp 的 value 工作量。
  // 这不增加 warp；额外 8 列块在同一批线程上执行另一次 m16n8k16。
  using MMA_P_T = Tile<_32, _32, _16>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  // 32行，每行4个线程；每线程8个连续half，故基本copy tile是32x32。
  // 128x32需要沿行重复4次。这里的(4,1)是线程编号步长，不是矩阵地址步长！
  // partition后global跨32行+8192half，shared跨32行+1024half；两端shape相近但地址不同。
  using CopyThreads = Layout<Shape<_32, _4>, Stride<_4, _1>>;
  using CopyValues = Layout<Shape<_1, _8>>;
  using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>;
  using G2SCopyA = decltype(make_tiled_copy(G2SAtom{}, CopyThreads{}, CopyValues{}));
  using G2SCopyB = G2SCopyA;

  // x4 表示每 lane 收到 4 个 32-bit words，即 8 half；不是 4 half。
  // A：16x16拆成四块8x8；B：将两个8x16 N子tile合在一次copy中。
  // B fragment原为(4,8,2)，copy目的改分组为(8,4,2)，存储槽位完全不移动。
  // 跨一个B copy组是N32：原跨N16为8槽，合组后为16槽。
  using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, T>;
  using S2RCopyAtomB = S2RCopyAtomA;

  // epilogue 的 scratchpad=(32,32,2)，暂存两个 32x32 输出块。
  // Swizzle<2,3,3> 仅改变列的 bit 3..4，保留每条 8-half 向量。
  using SmemLayoutAtomC =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_32, _32>, Stride<_32, _1>>{}));
  using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{}, Shape<_32, _32, _2>{}));
  // 32-bit R2S搬两个half，不做int数值转换。
  // 32x32输出宏块内每线程有四对half；跨行8对应槽位+2，跨列16对应+16。
  // 配图与存储表见 docs/cute_gemm_81920.md 第5、8节。
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  // 128-bit 输出向量搬 8 half；这里没有反向 cp.async。
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(S2GCopyAtomC{}, CopyThreads{}, CopyValues{}));

  static constexpr int kThreadNum = size(MMA{});  // 128=4 warps
  static constexpr int shm_size_AB = cosize(SmemLayoutA{}) + cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cosize(SmemLayoutC{});
  static constexpr int kShmSize = shm_size_AB * sizeof(T);  // 49152 bytes=48 KiB
  // 输出复用 A 的一个 stage，不额外分配 32 KiB 的完整 C tile。
  static_assert(shm_size_C <= cosize(SmemLayoutA{}) / kStage);
  static_assert(kThreadNum == 128 && shm_size_C == 2048 && kShmSize == 49152);
};
}  // namespace half_gemm
