#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 256;
constexpr int kItemsPerBlock = 2 * kBlockThreads;

// Baseline：单线程顺序 inclusive scan，O(n) work，无 GPU 并行度。
__global__ void prefix_sum_baseline_kernel(const int* input, int* output,
                                           int n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += input[i];
    output[i] = sum;
  }
}

__device__ __forceinline__ int warp_inclusive_scan(int value) {
  const int lane = threadIdx.x % kWarpSize;
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, value, offset);
    if (lane >= offset) value += other;
  }
  return value;
}

// 每个 block 扫描连续 512 项。每个线程持有相邻的两个元素：
// 先 scan 线程的 pair sum，再把线程前缀加回 pair 内部。
__global__ void block_scan_kernel(const int* input, int* output,
                                  int* block_sums, int n) {
  constexpr int kNumWarps = kBlockThreads / kWarpSize;
  __shared__ int warp_sums[kNumWarps];

  const int tid = threadIdx.x;
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;
  const int i0 = blockIdx.x * kItemsPerBlock + 2 * tid;
  const int i1 = i0 + 1;
  const int x0 = i0 < n ? input[i0] : 0;
  const int x1 = i1 < n ? input[i1] : 0;
  const int thread_sum = x0 + x1;

  const int warp_prefix = warp_inclusive_scan(thread_sum);
  if (lane == kWarpSize - 1) warp_sums[warp] = warp_prefix;
  __syncthreads();

  // 第一个 warp 扫描最多 8 个 warp totals。
  if (warp == 0) {
    int value = lane < kNumWarps ? warp_sums[lane] : 0;
    value = warp_inclusive_scan(value);
    if (lane < kNumWarps) warp_sums[lane] = value;
  }
  __syncthreads();

  const int previous_warps = warp == 0 ? 0 : warp_sums[warp - 1];
  const int previous_threads = warp_prefix - thread_sum;
  const int thread_offset = previous_warps + previous_threads;
  if (i0 < n) output[i0] = thread_offset + x0;
  if (i1 < n) output[i1] = thread_offset + x0 + x1;

  if (block_sums != nullptr && tid == kBlockThreads - 1) {
    block_sums[blockIdx.x] = warp_sums[kNumWarps - 1];
  }
}

__global__ void uniform_add_kernel(int* data, const int* scanned_block_sums,
                                   int n) {
  const int block = blockIdx.x;
  if (block == 0) return;
  const int offset = scanned_block_sums[block - 1];
  const int i0 = block * kItemsPerBlock + 2 * threadIdx.x;
  const int i1 = i0 + 1;
  if (i0 < n) data[i0] += offset;
  if (i1 < n) data[i1] += offset;
}

void prefix_sum_baseline(const int* input, int* output, int n) {
  if (n > 0) prefix_sum_baseline_kernel<<<1, 1>>>(input, output, n);
}

// 分层 scan：先做各 block 的局部 scan，再递归 scan block sums，最后 uniform add。
// 总 work O(n)，支持任意 n，而不是只处理一个 block 的演示片段。
void prefix_sum_warp(const int* input, int* output, int n) {
  if (n <= 0) return;
  const int num_blocks = (n + kItemsPerBlock - 1) / kItemsPerBlock;
  int* block_sums = nullptr;
  if (num_blocks > 1) {
    CUDA_CHECK(cudaMalloc(&block_sums, num_blocks * sizeof(int)));
  }

  block_scan_kernel<<<num_blocks, kBlockThreads>>>(input, output, block_sums, n);
  CUDA_CHECK(cudaGetLastError());

  if (num_blocks > 1) {
    int* scanned_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&scanned_block_sums, num_blocks * sizeof(int)));
    prefix_sum_warp(block_sums, scanned_block_sums, num_blocks);
    uniform_add_kernel<<<num_blocks, kBlockThreads>>>(
        output, scanned_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(scanned_block_sums));
    CUDA_CHECK(cudaFree(block_sums));
  }
}

int main() {
  constexpr int n = 1'000'003;  // 跨多层且不是 block size 的倍数
  std::vector<int> input(n);
  for (int i = 0; i < n; ++i) input[i] = i % 7 - 3;

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
  CUDA_CHECK(cudaMemcpy(warp.data(), d_warp, n * sizeof(int),
                        cudaMemcpyDeviceToHost));

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
