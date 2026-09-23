/*
SwiGLU（SiLU-gated Linear Unit）的极简 CUDA 面试版

输入（vLLM / LLaMA 约定，gate 与 up 在最后一维拼在一起）：
  x[row, 2 * hidden]，row-major 连续存储
  gate = x[row, 0 : hidden]
  up   = x[row, hidden : 2 * hidden]

输出：
  y[row, hidden]
  y[row, j] = silu(gate[j]) * up[j]
            = gate[j] / (1 + exp(-gate[j])) * up[j]

launch: <<<dim3(ceil(hidden / (threads * 4)), rows), threads>>>

并行分工：
  - grid.y：一个 block 只负责一行，因此行内的 gate / up / out 基址固定，
    不会出现跨行读写的越界问题。
  - thread：线程 x 负责行内第 4*x 开始的连续 4 个输出元素。gate 与 up 相距
    hidden 个元素，一个 float4 只能覆盖同一侧的 4 个值，所以每个线程做两次
    float4 读（gate 段一次、up 段一次）和一次 float4 写。这样把 16 次标量
    访存压缩成 3 次 128-bit 访存，是这类 elementwise 融合算子最直接的优化。
  - hidden 不是 4 的倍数（或基址无法 4 对齐）时退还标量 kernel，保证通用。

说明：
  - 相比 examples/fusion/silu_and_mul.cu（gate/up 是两个独立指针），本例的
    gate 与 up 来自同一段内存，这是算子融合后最常见的布局。
  - 生产实现会支持 FP16/BF16（用 __half2 / bf16x2 拿到更高带宽），并直接用
    x 和 x + x.size(-1)/2 两个 torch 视图，无需额外拷贝。
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

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// 标量版：一个线程一个输出元素，hidden 任意取值都安全。
__global__ void swiglu_kernel(const float *x, float *y, int hidden) {
  const long long row = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= hidden)
    return;

  const float *gate = x + row * 2 * hidden;
  y[row * hidden + column] = silu(gate[column]) * gate[hidden + column];
}

// float4 版：要求 hidden % 4 == 0，此时行基址与段内偏移都天然 16 字节对齐。
__global__ void swiglu_vec4_kernel(const float *x, float *y, int hidden) {
  const long long row = blockIdx.y;
  const int column = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (column >= hidden)
    return;

  const float *gate = x + row * 2 * hidden;
  const float *up = gate + hidden;
  float *out = y + row * hidden;

  const float4 g = *reinterpret_cast<const float4 *>(gate + column);
  const float4 u = *reinterpret_cast<const float4 *>(up + column);
  float4 result;
  result.x = silu(g.x) * u.x;
  result.y = silu(g.y) * u.y;
  result.z = silu(g.z) * u.z;
  result.w = silu(g.w) * u.w;
  *reinterpret_cast<float4 *>(out + column) = result;
}

void swiglu(const float *x, float *y, int rows, int hidden,
            cudaStream_t stream = 0) {
  if (rows <= 0 || hidden <= 0) {
    std::fprintf(stderr, "invalid shape: rows=%d hidden=%d\n", rows, hidden);
    std::exit(EXIT_FAILURE);
  }

  constexpr int threads = 256;
  constexpr int vec = 4;
  const int work = (hidden % vec == 0) ? hidden / vec : hidden;
  const dim3 grid((work + threads - 1) / threads, rows);
  if (hidden % vec == 0) {
    swiglu_vec4_kernel<<<grid, threads, 0, stream>>>(x, y, hidden);
  } else {
    swiglu_kernel<<<grid, threads, 0, stream>>>(x, y, hidden);
  }
}

bool run_case(int rows, int hidden) {
  const size_t size = static_cast<size_t>(rows) * hidden;

  std::vector<float> input(static_cast<size_t>(rows) * 2 * hidden);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 37) - 18) * 0.125f;
  }

  std::vector<float> expected(size);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < hidden; ++column) {
      const size_t base = static_cast<size_t>(row) * 2 * hidden;
      const float gate = input[base + column];
      const float up = input[base + hidden + column];
      expected[static_cast<size_t>(row) * hidden + column] =
          gate / (1.0f + std::exp(-gate)) * up;
    }
  }

  std::vector<float> output(size);
  float *d_x = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  swiglu(d_x, d_y, rows, hidden);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), d_y, size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    max_error = std::max(max_error, std::abs(output[i] - expected[i]));
  }
  std::printf("SwiGLU rows=%d hidden=%d (%s): %s (max error = %.8g)\n", rows,
              hidden, hidden % 4 == 0 ? "float4" : "scalar",
              max_error < 1e-5f ? "PASS" : "FAIL", max_error);

  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_x));
  return max_error < 1e-5f;
}

int main() {
  bool ok = run_case(4, 1024);  // 走 float4 路径
  ok = run_case(3, 1002) && ok; // hidden 非 4 的倍数，走标量路径
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
