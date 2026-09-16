#include <cstdint>

#include "include/cute_gemm.cuh"

// 本文件是32x32 tile、FP32累加的入门变体。
// 81920x256x256全half规格的kernel和tile/stride注释位于
// examples/cute_gemm/multi_stage_annotated.cuh；文档为docs/cute_gemm_81920.md。
// 参考 reed-lau/cute-gemm/gemm-multi-stage.cu 的数据流与 stage 思路，
// 重新组织成小tile教学实现；配置与本次128x128全half教程不同。
namespace cute_tutorial {
namespace {

template <class Cfg>
struct SharedInputs {
  alignas(16) Input a[cosize(typename Cfg::InputLayout{})];
  alignas(16) Input bt[cosize(typename Cfg::InputLayout{})];
};

// MMA ownership -> shared 坐标 -> 连续向量 ownership。
template <class Accumulator, class GlobalTile>
__device__ void store_via_shared(Accumulator const& accum, GlobalTile global_c,
                                 Output* shared_storage, int tid) {
  auto shared_c = make_tensor(make_smem_ptr(shared_storage), OutputLayout{});
  auto r2s = make_tiled_copy_C(OutputScatter{}, TiledMma{});
  auto scatter_thread = r2s.get_slice(tid);
  auto accum_for_store = scatter_thread.retile_S(accum);
  auto shared_for_store = scatter_thread.partition_D(shared_c);
  copy(r2s, accum_for_store, shared_for_store);
  __syncthreads();  // 连续向量的读者会读取其他线程写入的结果。

  OutputCopy output_copy;
  auto output_thread = output_copy.get_slice(tid);
  auto shared_for_output = output_thread.partition_S(shared_c);
  auto global_for_output = output_thread.partition_D(global_c);
  auto output_regs = make_fragment_like(shared_for_output);
  copy(output_copy, shared_for_output, output_regs);  // shared -> register
  copy(output_copy, output_regs, global_for_output);  // register -> global
}

template <class Cfg, bool UseMatrixLoad, bool UseSharedOutput>
__global__ void gemm_kernel(Input const* a, Input const* bt, Output* c, int m, int n, int k) {
  using InputLayout = typename Cfg::InputLayout;
  __shared__ SharedInputs<Cfg> storage;
  const int tid = threadIdx.x;

  // 1. 完整矩阵 -> 当前 CTA tile。以下操作只建立视图。
  auto matrix_a = make_tensor(make_gmem_ptr(a), make_shape(m, k), make_stride(k, _1{}));
  auto matrix_b = make_tensor(make_gmem_ptr(bt), make_shape(n, k), make_stride(k, _1{}));
  auto matrix_c = make_tensor(make_gmem_ptr(c), make_shape(m, n), make_stride(n, _1{}));
  auto global_a = local_tile(matrix_a, TileShape{}, make_coord(blockIdx.y, _));
  auto global_b = local_tile(matrix_b, TileShape{}, make_coord(blockIdx.x, _));
  auto global_c = local_tile(matrix_c, TileShape{}, make_coord(blockIdx.y, blockIdx.x));
  auto shared_a = make_tensor(make_smem_ptr(storage.a), InputLayout{});
  auto shared_b = make_tensor(make_smem_ptr(storage.bt), InputLayout{});

  // 2. G2S ownership：global 末维为 K tile，shared 末维为 stage。
  GlobalToShared input_copy;
  auto input_thread = input_copy.get_slice(tid);
  auto global_a_for_copy = input_thread.partition_S(global_a);
  auto global_b_for_copy = input_thread.partition_S(global_b);
  auto shared_a_for_copy = input_thread.partition_D(shared_a);
  auto shared_b_for_copy = input_thread.partition_D(shared_b);
  auto enqueue_tile = [&](int k_tile, int stage) {
    copy(input_copy, global_a_for_copy(_, _, _, k_tile), shared_a_for_copy(_, _, _, stage));
    copy(input_copy, global_b_for_copy(_, _, _, k_tile), shared_b_for_copy(_, _, _, stage));
    cp_async_fence();  // 一个 group 包含当前线程的 A 和 B 搬运。
  };

  // 3. MMA ownership：创建本线程 fragment，但尚未加载 A/B。
  TiledMma mma;
  auto mma_thread = mma.get_slice(tid);
  auto regs_a = mma_thread.partition_fragment_A(shared_a(_, _, 0));
  auto regs_b = mma_thread.partition_fragment_B(shared_b(_, _, 0));
  auto accum = mma_thread.partition_fragment_C(global_c);
  clear(accum);

  // 4. 从 MMA 反推 ldmatrix 源地址；retile 只重解释已有寄存器。
  auto matrix_copy_a = make_tiled_copy_A(MatrixLoadA{}, mma);
  auto matrix_copy_b = make_tiled_copy_B(MatrixLoadB{}, mma);
  auto matrix_thread_a = matrix_copy_a.get_slice(tid);
  auto matrix_thread_b = matrix_copy_b.get_slice(tid);
  auto shared_a_for_mma = matrix_thread_a.partition_S(shared_a);
  auto shared_b_for_mma = matrix_thread_b.partition_S(shared_b);
  auto regs_a_for_copy = matrix_thread_a.retile_D(regs_a);
  auto regs_b_for_copy = matrix_thread_b.retile_D(regs_b);

  // 5. 双缓冲先装 tile 0；单缓冲在每轮开始时装当前 tile。
  const int k_tiles = k / BlockK;
  if constexpr (Cfg::Stages == 2) enqueue_tile(0, 0);
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    const int read_stage = k_tile % Cfg::Stages;
    if constexpr (Cfg::Stages == 1) enqueue_tile(k_tile, 0);
    cp_async_wait<0>();
    __syncthreads();  // 所有线程搬入的当前 tile 均已准备好。

    // 下一 tile 写入另一槽，其旧读者已在上一轮末尾 barrier 结束。
    if constexpr (Cfg::Stages == 2) {
      if (k_tile + 1 < k_tiles) enqueue_tile(k_tile + 1, 1 - read_stage);
    }
    if constexpr (UseMatrixLoad) {
      copy(matrix_copy_a, shared_a_for_mma(_, _, _, read_stage), regs_a_for_copy);
      copy(matrix_copy_b, shared_b_for_mma(_, _, _, read_stage), regs_b_for_copy);
    } else {
      copy(mma_thread.partition_A(shared_a(_, _, read_stage)), regs_a);
      copy(mma_thread.partition_B(shared_b(_, _, read_stage)), regs_b);
    }
#pragma unroll
    for (int k_atom = 0; k_atom < size<2>(regs_a); ++k_atom) {
      gemm(mma, regs_a(_, _, k_atom), regs_b(_, _, k_atom), accum);
    }
    __syncthreads();  // 当前 tile 所有读者完成，下一轮可复用此槽。
  }

  // 6. 全部 K 已累加完。输出只改变存放位置与线程归属。
  if constexpr (UseSharedOutput) {
    __shared__ __align__(16) Output output_storage[cosize(OutputLayout{})];
    store_via_shared(accum, global_c, output_storage, tid);
  } else {
    copy(accum, mma_thread.partition_C(global_c));
  }
}

__global__ void copy_roundtrip_kernel(Input const* src, Input* dst) {
  __shared__ __align__(16) Input storage[1024];
  auto global_src = make_tensor(make_gmem_ptr(src), OutputLayout{});
  auto global_dst = make_tensor(make_gmem_ptr(dst), OutputLayout{});
  auto shared = make_tensor(make_smem_ptr(storage), OutputLayout{});
  GlobalToShared load;
  auto load_thread = load.get_slice(threadIdx.x);
  copy(load, load_thread.partition_S(global_src), load_thread.partition_D(shared));
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  auto store = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Input>{}, InputCopyThreads{},
                               InputCopyValues{});
  auto store_thread = store.get_slice(threadIdx.x);
  auto shared_src = store_thread.partition_S(shared);
  auto regs = make_fragment_like(shared_src);
  copy(store, shared_src, regs);
  copy(store, regs, store_thread.partition_D(global_dst));
}

bool aligned(void const* p) {
  return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}
}  // namespace

cudaError_t launch_gemm(Input const* a, Input const* bt, Output* c, int m, int n, int k,
                        Lesson lesson, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || m % BlockM || n % BlockN || k % BlockK || !aligned(a) ||
      !aligned(bt) || !aligned(c))
    return cudaErrorInvalidValue;
  dim3 grid(n / BlockN, m / BlockM);
  switch (lesson) {
    case Lesson::ScalarLoad:
      gemm_kernel<SingleStage, false, false><<<grid, Threads, 0, stream>>>(a, bt, c, m, n, k);
      break;
    case Lesson::MatrixLoad:
      gemm_kernel<SingleStage, true, false><<<grid, Threads, 0, stream>>>(a, bt, c, m, n, k);
      break;
    case Lesson::SharedOutput:
      gemm_kernel<SingleStage, true, true><<<grid, Threads, 0, stream>>>(a, bt, c, m, n, k);
      break;
    case Lesson::DoubleBuffer:
      gemm_kernel<TwoStages, true, true><<<grid, Threads, 0, stream>>>(a, bt, c, m, n, k);
      break;
    case Lesson::SwizzledPipeline:
      gemm_kernel<TwoSwizzledStages, true, true><<<grid, Threads, 0, stream>>>(a, bt, c, m, n, k);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t launch_copy_roundtrip(Input const* src, Input* dst, cudaStream_t stream) {
  if (!aligned(src) || !aligned(dst)) return cudaErrorInvalidValue;
  copy_roundtrip_kernel<<<1, Threads, 0, stream>>>(src, dst);
  return cudaGetLastError();
}
}  // namespace cute_tutorial
