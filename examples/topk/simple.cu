#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// 面试极简版只面向 small-K：一个 block 处理至多 8192 个元素，256 个线程
// 各自维护局部 Top-K，然后线程 0 合并出 block Top-K。
// 一个 block 的工作：
//   1. 负责 input[blockIdx.x * 8192 : min(..., n)]；
//   2. 256 个线程 stride 扫描，每线程在 shared memory 中维护一个 Top-K；
//   3. 同步后由线程 0 合并 256 份线程 Top-K；
//   4. 向 partial 写出这一块的 k 个候选。
//
// 正确性依据：若元素没进入自己分组的 Top-K，则仅在该分组内就已有至少 k
// 个元素不小于它，因此它也不可能进入全局 Top-K。于是：
// TopK(所有元素) == TopK(各分组 Top-K 的并集)。
constexpr int kBlockThreads = 256;
constexpr int kMaxK = 4;
constexpr int kItemsPerBlock = 8192;

// top[0:k] 始终按降序排列。插入一个新值并只保留最大的 k 个，最坏 O(k)。
__device__ __forceinline__ void insert_topk(float *top, int k, float value) {
  if (value <= top[k - 1])
    return;
  int pos = k - 1;

  while (pos > 0 && value > top[pos - 1]) {
    top[pos] = top[pos - 1];
    --pos;
  }
  top[pos] = value;
}

__global__ void topk_kernel(const float *input, float *partial, int N, int k) {
  const int bid = blockIdx.x;
  const int b_begin = bid * kItemsPerBlock;
  const int b_end = min(b_begin + kItemsPerBlock, N);
  __shared__ float thread_top[kBlockThreads][kMaxK];

  for (int i = 0; i < k; ++i)
    thread_top[threadIdx.x][i] = -FLT_MAX;
  for (int idx = b_begin + threadIdx.x; idx < b_end; idx += kBlockThreads) {
    insert_topk(thread_top[threadIdx.x], k, input[idx]);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float block_top[kMaxK];
    for (int i = 0; i < k; ++i)
      block_top[i] = -FLT_MAX;

    for (int i = 0; i < kBlockThreads; ++i) {
      for (int j = 0; j < k; ++j) {
        insert_topk(block_top, k, thread_top[i][j]);
      }
    }
    for (int i = 0; i < k; ++i)
      partial[bid * k + i] = block_top[i];
  }
}

extern "C" void solve(const float *input, float *output, int n, int k) {
  // 该面试接口没有错误返回值，非法参数直接打印并返回。
  if (input == nullptr || output == nullptr || n <= 0 || k <= 0 || k > kMaxK ||
      k > n) {
    std::fprintf(stderr, "topk: require n > 0 and 1 <= k <= min(%d, n)\n",
                 kMaxK);
    return;
  }

  int num_blocks = (n + kItemsPerBlock - 1) / kItemsPerBlock;
  float *current = nullptr;
  CUDA_CHECK(cudaMalloc(&current,
                        static_cast<size_t>(num_blocks) * k * sizeof(float)));

  // 第一层：n 个输入 -> num_blocks * k 个候选。
  topk_kernel<<<num_blocks, kBlockThreads>>>(input, current, n, k);
  CUDA_CHECK(cudaGetLastError());

  // 保留各层 allocation 到所有 kernel 完成后再统一释放，避免在循环中
  // cudaFree 引入逐层同步。候选每层至少缩小约 8192 / k 倍，所以额外空间小。
  std::vector<float *> allocations{current};
  int current_n = num_blocks * k;
  bool output_written = false;

  while (current_n > k) {
    const int next_blocks = (current_n + kItemsPerBlock - 1) / kItemsPerBlock;

    if (next_blocks == 1) {
      // 最后一层直接写 output，避免再分配一个只有 k 个元素的 buffer。
      topk_kernel<<<1, kBlockThreads>>>(current, output, current_n, k);
      CUDA_CHECK(cudaGetLastError());
      output_written = true;
      break;
    }

    float *next = nullptr;
    CUDA_CHECK(cudaMalloc(&next, static_cast<size_t>(next_blocks) * k *
                                     sizeof(float)));
    allocations.push_back(next);
    topk_kernel<<<next_blocks, kBlockThreads>>>(current, next, current_n, k);
    CUDA_CHECK(cudaGetLastError());

    current = next;
    current_n = next_blocks * k;
  }

  // n 本来就由一个 block 处理时，第一层结果仍在 current 中。
  if (!output_written) {
    CUDA_CHECK(cudaMemcpy(output, current,
                          static_cast<size_t>(k) * sizeof(float),
                          cudaMemcpyDeviceToDevice));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  for (float *ptr : allocations)
    CUDA_CHECK(cudaFree(ptr));
}

// ---------------------------------------------------------------------------
// 适用情况：大 N、非常小的 k（这里限制 k <= 4）、只需要值、不要求完整排序。
// 相比 bitonic 全排序，它无需把 n 补到 2 的幂，通常只读一遍原输入，并把每块
// 8192 个数压缩成 k 个候选；递归层数很少。
//
// 仍有的问题（面试时应主动说明）：
//   1. 每线程 Top-K 放在 shared memory，插入时会有 bank conflict；生产版应使用
//      编译期固定 K 的寄存器数组。
//   2. block 合并完全由线程 0 串行完成；生产版应使用 warp shuffle/bitonic
//      merge，再分层合并各 warp 的结果。
//   3. 插入复杂度 O(k)，整个算法只适合 small-K；大 K 应考虑 radix-select、
//      CUB DeviceTopK 或更完整的分块选择算法。
//   4. 只返回 value，不返回原始 index；相等元素也没有稳定下标语义。
//   5. 假设输入为有限 float；NaN 会破坏这里的普通大小比较。
//   6. solve 使用默认 stream、cudaMalloc 和最终全设备同步，不适合作为异步库
//      接口；生产版应传入 stream 并复用 workspace/cudaMallocAsync。
// ---------------------------------------------------------------------------

int main() {
  constexpr int n = 20003; // 跨越多个 8192-element block
  constexpr int k = 4;

  std::vector<float> input(n);
  for (int i = 0; i < n; ++i) {
    input[i] = static_cast<float>((i * 17 + i / 7) % 997 - 498);
  }

  float *device_input = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, k * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(),
                        input.size() * sizeof(float), cudaMemcpyHostToDevice));

  solve(device_input, device_output, n, k);

  std::vector<float> output(k);
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, k * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::partial_sort(input.begin(), input.begin() + k, input.end(),
                    std::greater<float>());
  for (int rank = 0; rank < k; ++rank) {
    if (output[rank] != input[rank]) {
      std::fprintf(stderr, "mismatch at %d: gpu=%g cpu=%g\n", rank,
                   output[rank], input[rank]);
      return EXIT_FAILURE;
    }
  }

  std::printf("hierarchical Top-K: PASS\nresult: ");
  for (int rank = 0; rank < k; ++rank) {
    std::printf("%g%s", output[rank], rank + 1 == k ? "\n" : " ");
  }

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return 0;
}
