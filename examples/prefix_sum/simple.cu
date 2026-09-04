#include <cuda_runtime.h>

#include <algorithm>
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

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 256;

// Baseline：单线程顺序 inclusive scan，O(n) work，无 GPU 并行度。
__global__ void prefix_sum_baseline_kernel(const int *input, int *output,
                                           int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += input[i];
    output[i] = sum;
  }
}

void prefix_sum_baseline(const int *input, int *output, int n) {
  if (n > 0)
    prefix_sum_baseline_kernel<<<1, 1>>>(input, output, n);
}

__device__ __forceinline__ int warp_inclusive_scan(int val) {
  const int lane = threadIdx.x % 32;
  for (int offset = 1; offset < 32; offset <<= 1) {
    int other = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset)
      val += other;
  }
  return val;
}

__global__ void block_scan_kernel(const int *input, int *output, int *block,
                                  int N) {
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int kNumWarp = kBlockThreads / kWarpSize;
  __shared__ int warp_sum[kNumWarp];

  const int index = blockIdx.x * kBlockThreads + tid;
  const int value = index < N ? input[index] : 0;
  int warp_prefix = warp_inclusive_scan(value);
  if (lane == kWarpSize - 1) {
    warp_sum[warp] = warp_prefix;
  }
  __syncthreads();

  if (warp == 0) {
    int value = lane < kNumWarp ? warp_sum[lane] : 0;
    value = warp_inclusive_scan(value);
    if (lane < kNumWarp)
      warp_sum[lane] = value;
  }
  __syncthreads();

  const int prev_warp_prefix = warp == 0 ? 0 : warp_sum[warp - 1];
  if (index < N)
    output[index] = prev_warp_prefix + warp_prefix;

  if (block && tid == kBlockThreads - 1) {
    block[blockIdx.x] = warp_sum[kNumWarp - 1];
  }
}

__global__ void add_kernel(int *output, int *block, int N) {
  const int bid = blockIdx.x;
  if (bid == 0)
    return;
  const int index = bid * kBlockThreads + threadIdx.x;
  if (index < N) {
    output[index] += block[bid - 1];
  }
}

void prefix_sum_warp(const int *input, int *output, int n) {
  if (n <= 0)
    return;

  struct ScanLevel {
    int *data;
    int *block_sums;
    int size;
  };
  std::vector<ScanLevel> levels;

  // 向上：扫描当前层，并生成下一层的 block sums。
  const int *current_input = input;
  int *current_output = output;
  int current_size = n;
  while (true) {
    const int num_blocks =
        (current_size + kBlockThreads - 1) / kBlockThreads;
    int *next_block_sums = nullptr;
    if (num_blocks > 1) {
      CUDA_CHECK(cudaMalloc(&next_block_sums, num_blocks * sizeof(int)));
    }

    block_scan_kernel<<<num_blocks, kBlockThreads>>>(
        current_input, current_output, next_block_sums, current_size);
    CUDA_CHECK(cudaGetLastError());

    if (num_blocks == 1)
      break;

    levels.push_back({current_output, next_block_sums, current_size});

    // 下一层直接原地扫描 block sums，省掉 scanned_block_sums 缓冲区。
    current_input = next_block_sums;
    current_output = next_block_sums;
    current_size = num_blocks;
  }

  // 向下：高层前缀已经算好，依次加回低层。
  for (int level = static_cast<int>(levels.size()) - 1; level >= 0; --level) {
    const int num_blocks =
        (levels[level].size + kBlockThreads - 1) / kBlockThreads;
    add_kernel<<<num_blocks, kBlockThreads>>>(
        levels[level].data, levels[level].block_sums, levels[level].size);
    CUDA_CHECK(cudaGetLastError());
  }

  for (const ScanLevel &level : levels) {
    CUDA_CHECK(cudaFree(level.block_sums));
  }
}

int main() {
  constexpr int n = 1'000'003; // 跨多层且不是 block size 的倍数
  std::vector<int> input(n);
  for (int i = 0; i < n; ++i)
    input[i] = i % 7 - 3;

  int *d_input = nullptr, *d_baseline = nullptr, *d_warp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_baseline, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_warp, n * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(int),
                        cudaMemcpyHostToDevice));

  prefix_sum_baseline(d_input, d_baseline, n);
  prefix_sum_warp(d_input, d_warp, n);
  CUDA_CHECK(cudaGetLastError());

  std::vector<int> baseline(n), warp(n);
  CUDA_CHECK(cudaMemcpy(baseline.data(), d_baseline, n * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(warp.data(), d_warp, n * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    if (baseline[i] != warp[i]) {
      std::fprintf(stderr, "mismatch at %d: baseline=%d, warp=%d\n", i,
                   baseline[i], warp[i]);
      return EXIT_FAILURE;
    }
  }
  std::printf("Prefix sum baseline == warp: PASS (n=%d, last=%d)\n", n,
              warp.back());

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_baseline));
  CUDA_CHECK(cudaFree(d_warp));
  return 0;
}
