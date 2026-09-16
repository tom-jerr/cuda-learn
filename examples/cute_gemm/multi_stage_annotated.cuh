#pragma once
#include "multi_stage_config.cuh"

// 根据用户粘贴的 gemm-multi-stage.cu 加注释，保留原变量名以便对照。
// 原作者：reed-lau/cute-gemm；修改点与完整推导见 docs/cute_gemm_81920.md。
// 阅读方法：先看基本tile、再数重复次数得到shape，最后沿原存储寻找下一个tile起点。
// 配图：docs/assets/cute_g2s_tile_steps.svg、cute_mma_tile_storage.svg、cute_s2r_tile_retile.svg。
// 本文件是81920x256x256的全half版本；src/cute_gemm.cu是另一份32x32、FP32累加示例。
namespace half_gemm {
template <typename Config>
__global__ void gemm_multi_stage(void *Dptr, const void *Aptr, const void *Bptr,
                                 int m, int n, int k) {
  using namespace cute;

  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  // A: [0,24576) bytes；B: [24576,49152) bytes。每个 stage 各 8192 bytes。
  // 显式保证 cp.async / ldmatrix 的 16-byte 基址对齐。
  extern __shared__ __align__(16) unsigned char shm_bytes[];
  T *shm_data = reinterpret_cast<T *>(shm_bytes);

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  // 1. 只建立视图：A[m,k] 的地址=m*K+k，C[m,n] 的地址=m*N+n。
  // 数学 B 是 column-major [K,N]，地址=k+n*K。
  // CuTe B 使用 (n,k) 坐标，故 shape=(N,K),stride=(K,1)，仍是同一地址。
  // 这里没有数据转置，也不是把数学 B 改成 row-major。
  Tensor A = make_tensor(make_gmem_ptr((const T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((const T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // (M, N)

  // grid=(2,640)，每 CTA 128 线程负责 C 的 128x128 块。
  // gA/gB=(128,32,8)，最后一维是全局 K tile 编号，不是 K=256。
  // gD=(128,128)；gA/gB 的行 stride 仍是全局 K=256。
  Tensor gA =
      local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                 make_coord(iy, _)); // (tile_m, tile_k, k/tile_k) (128,32,8)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _)); // (128,32,8)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix)); // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK, kStage)

  // 2. 四warp各做一次atom：A覆盖32x16，B覆盖16x16，C覆盖32x16。
  // 为覆盖当前CTA，A沿M重复4次、K重复2次；B沿N重复8次、K重复2次。
  // C沿M重复4次、N重复8次。每线程基本值数分别8/4/4，shape因此为：
  // tCrA=(8,4,2)=64 half；tCrB=(4,8,2)=64 half；tCrD=(4,4,8)=128 half。
  // fragment创建存储而不加载：A同一M组的两段k16相邻，跨M32为+16槽，跨K16为+8槽；
  // B同一N组的两段k16相邻，跨N16为+8槽，跨K16为+4槽。
  // C先沿M存完四个小tile，跨M32为+4槽，跨N16为+16槽。
  // 对比：partition_A(gA)仍指向global，跨M32是+8192half，不是寄存器的+16槽。
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)

  // 清零一次；后续 8 个 K tile、每 tile 两个 k16 都累加到这份 half C。
  clear(tCrD);

  // 3. 按需要的子tile组织x4：A的一个16x16正好拆四块8x8，提供8half/lane。
  // B的一个8x16只有4half/lane；将两个N子tile一起装入，正好也是四块8x8。
  // 所以B的8个N repetitions两两合组：(4,8,2) -> copy目的(8,4,2)。
  // 源视图为(8,4,2,3)：每条源行描述8half、4组、2段k16、3个shared stage。
  // A retile：跨M32仍+16槽，跨K16仍+8槽。
  // B retile：两组4half的起点相隔8槽；跨一对N组(N32)为+16槽，跨K16仍+4槽。
  // shared源行首：跨32行+1024half，跨stage+4096；跨K16在swizzle后可能+16或-16。
  // partition_S描述地址提供者，retile_D描述结果接收者；二者不是逐lane逐元素赋值。
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA =
      s2r_thr_copy_a.partition_S(sA); // (8,4,2,3): (CPY,CPY_M,CPY_K,STAGE)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (8,4,2): (CPY,CPY_M,CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB =
      s2r_thr_copy_b.partition_S(sB); // (8,4,2,3): (CPY,CPY_N,CPY_K,STAGE)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (8,4,2): (CPY,CPY_N,CPY_K)

  // 4. G2S基本tile=32行x(每行4线程*8half)=32x32。
  // 当前128x32输入要向下重复4次，向右不重复；global有8个K
  // tiles，shared有3个stage。
  // 因此global源shape=(8,4,1,8)，shared目的shape=(8,4,1,3)。
  // 源partition保持全局行宽256：向下跨32行地址+8192，下一K
  // tile向右32列地址+32。
  // 目的partition：向下跨32行地址+1024，下一stage跳整个128x32缓冲，地址+4096。
  // 每线程每矩阵重复4条16-byte
  // cp.async，A+B共8条；末维的kt和stage由循环分别选。
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (8,4,1,8): 最后是 K_TILE
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (8,4,1,3): 最后是 STAGE

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (8,4,1,8): 最后是 K_TILE
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (8,4,1,3): 最后是 STAGE

  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

  // 5. prologue：tile0->stage0，tile1->stage1，保留 stage2 作为未来写槽。
  // 固定 K=256 有 8 tiles，因此两次无 predicate 预取合法。
  // 改成更短的 K 时必须重新设计 prologue，不能仅改 main 的数字。
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // 每线程允许最多 1 个最新 group 尚未完成，故较老的 tile0 已完成。
  // wait 只管本线程；barrier 后才能读取其他线程负责搬入的行。
  cp_async_wait<kStage - 2>();
  __syncthreads();

  int ik = 0;
  // smem -> reg
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // 6. shared 有 3 槽，寄存器有 2 个 k16 槽，两套流水互相配合。
  // 初始 r[0] 已就绪。ik=0 先装当前 tile 的 r[1]，再计算 r[0]；
  // ik=1 先切到下一 shared tile、装其 r[0]，再计算当前 tile 的 r[1]。
  // 这让下一次 shared load 与当前 MMA 在指令调度上有重叠机会。
  int ntile = k / kTileK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // 原代码最后仍会预取一份不再使用的旧数据；这里跳过最终那次加载。
      // 普通 ik=0 的 r[1] 加载，以及非末 tile 的下一 r[0] 加载，都照常执行。
      if (ik + 1 < nk || itile + 1 < ntile) {
        cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                   tCrA_view(_, _, ik_next));
        cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                   tCrB_view(_, _, ik_next));
      }

      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        // 尾部没有新 global tile 时仍提交空 group，以维持 wait<1> 的推进。
        // 不能直接把 fence 一起挪进上面的 if(itile_to_read<ntile)。
        cp_async_fence();
      }

      // 一次调用遍历 4 个 MMA_M 与 8 个 MMA_N：每 warp 32 次 atom。
      // ik 有两轮；每 CTA 每 K tile 为 4*32*2=256 条 warp MMA。
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  } // itile

  // 7. 全部 K 计算结束，明确排空输入搬运后才将 shared A 改作输出 scratch。
  // 额外 drain/barrier 使复用边界可审查，不依赖尾部旧数据预取来解释安全性。
  cp_async_wait<0>();
  __syncthreads();
  // sC=(32,32,2)=2048 half=4096 bytes，小于 A 的一槽 8192 bytes。
  // ismem_read 此时为 8%3=2；.data() 取该槽基址，再用 C 的 swizzle 建新视图。
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);  // (CPY, CPY_M, CPY_N)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD); // (CPY, CPY_M, CPY_N)

  // 8. 把两个32x16输出小tile合成32x32宏块：每线程8个值，宏块网格4x4。
  // R2S retile=(8,4,4)：源值没搬动，跨M32仍+4槽，跨N32为+32槽。
  // 同一宏块内8个值分两段：首宏块占0..3和16..19槽，不是连续0..7。
  // S2G global虽然也为(8,4,4)，但跨M32是+8192half，跨N32是+32half。
  // shared只有两个32x32槽，partition得到(8,1,1,2)，换槽+1024half。
  // group_modes<1,3> 合并 mode 1、2（右端 3 不包含），得到 (8,16)。
  // p=im+4*in，其中 im/in=0..3，是 32x32 宏块坐标；M 最先变化。
  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY_, CPY_MN)
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY_, CPY_MN)

  // scratch 的 batch=2，与输入 stage=3 完全不同。
  // 16 个输出宏块每批放 2 个，故外层循环 8 次。
  int step = size<3>(tCsC_r2s); // 2
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // 本配置累加和输出同为 half，这里无需数值类型转换。
      // 保留原代码的临时 fragment 以便对应阅读；改 accumulator 类型时，
      // 真正的转换发生在下面这次 copy，而不是 UniversalCopy<int> 中。
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);

      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    // 读者线程与写者可能不同；上面的 barrier 保证全部 R2S 已完成。
    // 普通 shared->global copy 实际经寄存器中转：每线程一条 8-half 向量。
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }

    __syncthreads(); // 所有读者结束后，下批才可覆盖同一 scratch。
  }
}

} // namespace half_gemm
