/*
RoPE（Rotary Position Embedding）的极简 CUDA 面试版

输入：
  x[tokens, heads, head_dim]，row-major 连续存储

本例采用 split-half 配对方式：
  (x[i], x[i + rotary_dim / 2]), 0 <= i < rotary_dim / 2

令：
  inv_freq_i = 10000^(-2i / rotary_dim)
  angle       = token * inv_freq_i

则：
  y[i]        = x[i] * cos(angle) - x[i + half] * sin(angle)
  y[i + half] = x[i] * sin(angle) + x[i + half] * cos(angle)

launch: <<<dim3(tokens, heads), round_up(rotary_dim / 2, 32)>>>

并行分工：
  grid：覆盖所有 (token, head)，共 tokens * heads 个 block。

  block：blockIdx.x 选择一个 token，blockIdx.y 选择一个 head。因此，一个
         block 只负责 x[token, head, :] 的 rotary 部分；不同 block 之间没有
         数据依赖，不需要原子操作或跨 block 同步。

  warp：CUDA 每 32 个 thread 组成一个 warp。若 rotary_dim=128，则 half=64，
        一个 block 含 64 个 thread，也就是 2 个 warp。每个 warp 处理 32 个
        二维旋转对。warp 内对 x[i] 的访问连续，对 x[i+half] 的访问也连续，
        因而可以形成合并的全局内存访问。

  thread：threadIdx.x=i 的线程独占一对 (i, i+half)，完成两次读取、角度
          计算和两次写回。不同线程处理的地址不重叠，所以原地更新是安全的，
          也不需要 __syncthreads()。

说明：
  - rotary_dim 必须是偶数，且不能超过 head_dim。
  - [rotary_dim, head_dim) 的非 rotary 部分保持不变。
  - token 在这里直接充当 position id；batch/packed sequence 场景通常会显式
    传入 position_ids，而不能简单使用全局 token 下标。
  - 为突出线程映射，本例在 kernel 内直接计算 sin/cos。生产实现通常会缓存
    cos/sin，并支持 FP16/BF16、向量化加载以及同时处理 Q/K。
*/

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                          \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void rope_kernel(float *x, int tokens, int heads, int head_dim,
                            int rotary_dim) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int i = threadIdx.x;

  const int half = rotary_dim / 2;
  if (i >= half) return;

  // x shape: [tokens, heads, head_dim]
  const size_t base =
      (static_cast<size_t>(token) * heads + head) * head_dim;

  const float x1 = x[base + i];
  const float x2 = x[base + i + half];

  // theta_i = 10000^(-2i / rotary_dim)
  const float inv_freq = powf(10000.0f, -2.0f * i / rotary_dim);
  const float angle = token * inv_freq;

  float s;
  float c;
  sincosf(angle, &s, &c);

  x[base + i] = x1 * c - x2 * s;
  x[base + i + half] = x1 * s + x2 * c;
}

void rope(float *x, int tokens, int heads, int head_dim, int rotary_dim,
          cudaStream_t stream = 0) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > head_dim || rotary_dim % 2 != 0) {
    std::fprintf(stderr,
                 "invalid shape: tokens=%d heads=%d head_dim=%d "
                 "rotary_dim=%d\n",
                 tokens, heads, head_dim, rotary_dim);
    std::exit(EXIT_FAILURE);
  }

  const int half = rotary_dim / 2;
  const int threads = ((half + 31) / 32) * 32;
  if (threads > 1024) {
    std::fprintf(stderr,
                 "this minimal kernel requires rotary_dim / 2 <= 1024\n");
    std::exit(EXIT_FAILURE);
  }

  const dim3 grid(tokens, heads);
  rope_kernel<<<grid, threads, 0, stream>>>(x, tokens, heads, head_dim,
                                            rotary_dim);
}

int main() {
  constexpr int tokens = 4;
  constexpr int heads = 2;
  constexpr int head_dim = 12;
  constexpr int rotary_dim = 8;
  constexpr int half = rotary_dim / 2;
  constexpr int size = tokens * heads * head_dim;

  std::vector<float> input(size);
  for (int i = 0; i < size; ++i) {
    input[i] = (i % 17 - 8) * 0.125f;
  }
  std::vector<float> expected = input;
  std::vector<float> output(size);

  // CPU reference：只旋转每个 head 的前 rotary_dim 个元素。
  for (int token = 0; token < tokens; ++token) {
    for (int head = 0; head < heads; ++head) {
      const int base = (token * heads + head) * head_dim;
      for (int i = 0; i < half; ++i) {
        const float x1 = input[base + i];
        const float x2 = input[base + i + half];
        const float inv_freq =
            std::pow(10000.0f, -2.0f * i / rotary_dim);
        const float angle = token * inv_freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        expected[base + i] = x1 * c - x2 * s;
        expected[base + i + half] = x1 * s + x2 * c;
      }
    }
  }

  float *d_x = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, input.data(), size * sizeof(float),
                        cudaMemcpyHostToDevice));

  rope(d_x, tokens, heads, head_dim, rotary_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_x, size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (int i = 0; i < size; ++i) {
    max_error = std::max(max_error, std::abs(output[i] - expected[i]));
  }

  std::printf("RoPE split-half: %s (max error = %.8g)\n",
              max_error < 1e-5f ? "PASS" : "FAIL", max_error);
  std::printf("token=1, head=0, output: ");
  const int example_base = heads * head_dim;
  for (int i = 0; i < head_dim; ++i) {
    std::printf("%.5f%s", output[example_base + i],
                i + 1 == head_dim ? "\n" : " ");
  }

  CUDA_CHECK(cudaFree(d_x));
  return max_error < 1e-5f ? EXIT_SUCCESS : EXIT_FAILURE;
}
