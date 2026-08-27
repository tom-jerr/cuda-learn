#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
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

// 面试版约束：每行独立做 Top-K，输入为有限 float，1 <= k <= 32。
// 相同 value 时选择更小的 index，因此 baseline/warp 版结果完全一致。
constexpr int kMaxK = 32;
constexpr int kWarpSize = 32;

__device__ __forceinline__ bool better(float lhs_value, int lhs_index,
                                       float rhs_value, int rhs_index) {
  return lhs_index >= 0 &&
         (rhs_index < 0 || lhs_value > rhs_value ||
          (lhs_value == rhs_value && lhs_index < rhs_index));
}

// Baseline：一行一个线程，在寄存器数组中维护有序 Top-K。
// 复杂度 O(cols * k)，并行度只有 rows。
__global__ void topk_baseline_kernel(const float* input, float* values,
                                     int* indices, int rows, int cols, int k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  float best_values[kMaxK];
  int best_indices[kMaxK];
#pragma unroll
  for (int j = 0; j < kMaxK; ++j) {
    best_values[j] = -FLT_MAX;
    best_indices[j] = -1;
  }

  const float* row_input = input + row * cols;
  for (int col = 0; col < cols; ++col) {
    const float candidate = row_input[col];
    if (!better(candidate, col, best_values[k - 1], best_indices[k - 1])) {
      continue;
    }

    int pos = k - 1;
    while (pos > 0 && better(candidate, col, best_values[pos - 1],
                             best_indices[pos - 1])) {
      best_values[pos] = best_values[pos - 1];
      best_indices[pos] = best_indices[pos - 1];
      --pos;
    }
    best_values[pos] = candidate;
    best_indices[pos] = col;
  }

  for (int j = 0; j < k; ++j) {
    values[row * k + j] = best_values[j];
    indices[row * k + j] = best_indices[j];
  }
}

// Warp argmax：value 大者胜；相等时 index 小者胜。
__device__ __forceinline__ void warp_argmax(float& value, int& index) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const float other_value =
        __shfl_down_sync(0xffffffffu, value, offset);
    const int other_index = __shfl_down_sync(0xffffffffu, index, offset);
    if (better(other_value, other_index, value, index)) {
      value = other_value;
      index = other_index;
    }
  }
}

// 优化版：一个 warp 处理一行。第 t 轮 32 lanes 并行扫描未选元素，
// shuffle 做 argmax；lane 0 记录 winner。复杂度 O(k * cols / 32)。
// 这是适合面试手写的 small-k 版本；大规模通用 Top-K 通常用 radix-select、
// 分块候选 + merge 或 CUB/Thrust，而不是反复扫描 k 次。
__global__ void topk_warp_kernel(const float* input, float* values,
                                 int* indices, int rows, int cols, int k) {
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int row = blockIdx.x * warps_per_block + warp_in_block;
  if (row >= rows) return;

  // 每个 warp 有 kMaxK 个槽，保存已经选过的原始下标。
  extern __shared__ int selected[];
  int* warp_selected = selected + warp_in_block * kMaxK;
  const float* row_input = input + row * cols;

  for (int rank = 0; rank < k; ++rank) {
    float local_value = -FLT_MAX;
    int local_index = -1;

    for (int col = lane; col < cols; col += kWarpSize) {
      bool already_selected = false;
      for (int j = 0; j < rank; ++j) {
        already_selected |= (warp_selected[j] == col);
      }
      if (!already_selected &&
          better(row_input[col], col, local_value, local_index)) {
        local_value = row_input[col];
        local_index = col;
      }
    }

    warp_argmax(local_value, local_index);
    if (lane == 0) {
      warp_selected[rank] = local_index;
      values[row * k + rank] = local_value;
      indices[row * k + rank] = local_index;
    }
    // 下一轮所有 lane 都会读 lane 0 刚写入的 selected[rank]。
    __syncwarp();
  }
}

void topk_baseline(const float* input, float* values, int* indices, int rows,
                   int cols, int k) {
  constexpr int threads = 128;
  topk_baseline_kernel<<<(rows + threads - 1) / threads, threads>>>(
      input, values, indices, rows, cols, k);
}

void topk_warp(const float* input, float* values, int* indices, int rows,
               int cols, int k) {
  constexpr int threads = 128;  // 4 rows/block
  constexpr int warps_per_block = threads / kWarpSize;
  const int blocks = (rows + warps_per_block - 1) / warps_per_block;
  const size_t shared_bytes = warps_per_block * kMaxK * sizeof(int);
  topk_warp_kernel<<<blocks, threads, shared_bytes>>>(input, values, indices,
                                                      rows, cols, k);
}

int main() {
  constexpr int rows = 7;
  constexpr int cols = 1003;  // 故意不是 32 的倍数
  constexpr int k = 8;
  static_assert(k <= kMaxK);

  std::vector<float> input(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    // 含重复值，用来检查 tie-break。
    input[i] = static_cast<float>((i * 17 + i / 11) % 101 - 50);
  }

  float *d_input = nullptr, *d_baseline_values = nullptr,
        *d_warp_values = nullptr;
  int *d_baseline_indices = nullptr, *d_warp_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_baseline_values, rows * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_warp_values, rows * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_baseline_indices, rows * k * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_warp_indices, rows * k * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  topk_baseline(d_input, d_baseline_values, d_baseline_indices, rows, cols, k);
  topk_warp(d_input, d_warp_values, d_warp_indices, rows, cols, k);
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> baseline_values(rows * k), warp_values(rows * k);
  std::vector<int> baseline_indices(rows * k), warp_indices(rows * k);
  CUDA_CHECK(cudaMemcpy(baseline_values.data(), d_baseline_values,
                        rows * k * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(warp_values.data(), d_warp_values,
                        rows * k * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(baseline_indices.data(), d_baseline_indices,
                        rows * k * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(warp_indices.data(), d_warp_indices,
                        rows * k * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < rows * k; ++i) {
    if (baseline_values[i] != warp_values[i] ||
        baseline_indices[i] != warp_indices[i]) {
      std::fprintf(stderr,
                   "mismatch at %d: baseline=(%g,%d), warp=(%g,%d)\n", i,
                   baseline_values[i], baseline_indices[i], warp_values[i],
                   warp_indices[i]);
      return EXIT_FAILURE;
    }
  }

  std::printf("Top-K baseline == warp: PASS\nrow 0: ");
  for (int j = 0; j < k; ++j) {
    std::printf("(%g, %d)%s", warp_values[j], warp_indices[j],
                j + 1 == k ? "\n" : " ");
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_baseline_values));
  CUDA_CHECK(cudaFree(d_warp_values));
  CUDA_CHECK(cudaFree(d_baseline_indices));
  CUDA_CHECK(cudaFree(d_warp_indices));
  return 0;
}
