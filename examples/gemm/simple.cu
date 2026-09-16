#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

// Row-major SGEMM: C[M, N] = A[M, K] * B[K, N].
//
// 这个文件故意只使用 CUDA Runtime，不依赖 cuBLAS，包含：
//   1. naive kernel：一个线程计算一个 C 元素；
//   2. tiled kernel：shared-memory 分块 + 每线程 4x4 寄存器分块；
//   3. 非方阵/非 tile 整数倍下的正确性检查；
//   4. CUDA Event benchmark 和 GFLOP/s 指标。

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(status_));                                 \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

__global__ void naive_gemm(const float *__restrict__ A,
                           const float *__restrict__ B,
                           float *__restrict__ C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc = fmaf(A[row * K + k], B[k * N + col], acc);
  }
  C[row * N + col] = acc;
}

// 一个 CTA 计算 BM x BN 个输出；K 维每轮处理 BK。
// block=(BN/TN, BM/TM)=(8,8)，共 64 个线程。
// 每线程计算相邻的 TM x TN=4x4 个输出，16 个 accumulator 常驻寄存器。
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int BLOCK_X = BN / TN;
constexpr int BLOCK_Y = BM / TM;
constexpr int THREADS_PER_BLOCK = BLOCK_X * BLOCK_Y;

static_assert(BM % TM == 0 && BN % TN == 0);

__global__ void tiled_gemm(const float *__restrict__ A,
                           const float *__restrict__ B,
                           float *__restrict__ C, int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int thread_row = threadIdx.y * TM;
  const int thread_col = threadIdx.x * TN;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  float acc[TM][TN] = {};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // 64 个线程协作加载 A[BM,BK] 和 B[BK,BN]；越界元素补零。
    for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
      const int smem_row = idx / BK;
      const int smem_col = idx % BK;
      const int global_row = block_row + smem_row;
      const int global_col = k0 + smem_col;
      As[smem_row][smem_col] =
          (global_row < M && global_col < K)
              ? A[global_row * K + global_col]
              : 0.0f;
    }
    for (int idx = tid; idx < BK * BN; idx += THREADS_PER_BLOCK) {
      const int smem_row = idx / BN;
      const int smem_col = idx % BN;
      const int global_row = k0 + smem_row;
      const int global_col = block_col + smem_col;
      Bs[smem_row][smem_col] =
          (global_row < K && global_col < N)
              ? B[global_row * N + global_col]
              : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_reg[TM];
      float b_reg[TN];
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        a_reg[tm] = As[thread_row + tm][kk];
      }
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        b_reg[tn] = Bs[kk][thread_col + tn];
      }
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          acc[tm][tn] = fmaf(a_reg[tm], b_reg[tn], acc[tm][tn]);
        }
      }
    }
    // 下一轮会覆盖 As/Bs，必须等所有线程读完当前 tile。
    __syncthreads();
  }

#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      const int row = block_row + thread_row + tm;
      const int col = block_col + thread_col + tn;
      if (row < M && col < N) {
        C[row * N + col] = acc[tm][tn];
      }
    }
  }
}

using LaunchFn = void (*)(const float *, const float *, float *, int, int, int);

void launch_naive(const float *A, const float *B, float *C, int M, int N,
                  int K) {
  const dim3 block(16, 16);
  const dim3 grid((N + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);
  naive_gemm<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_tiled(const float *A, const float *B, float *C, int M, int N,
                  int K) {
  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  tiled_gemm<<<grid, block>>>(A, B, C, M, N, K);
}

struct Timing {
  float min_ms;
  float median_ms;
  float mean_ms;
};

Timing benchmark(LaunchFn launch, const float *A, const float *B, float *C,
                 int M, int N, int K, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    launch(A, B, C, M, N, K);
  }
  CUDA_CHECK(cudaGetLastError());

  std::vector<cudaEvent_t> starts(iters);
  std::vector<cudaEvent_t> stops(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventCreate(&starts[i]));
    CUDA_CHECK(cudaEventCreate(&stops[i]));
    CUDA_CHECK(cudaEventRecord(starts[i]));
    launch(A, B, C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stops[i]));
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(stops.back()));

  std::vector<float> samples(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventElapsedTime(&samples[i], starts[i], stops[i]));
    CUDA_CHECK(cudaEventDestroy(starts[i]));
    CUDA_CHECK(cudaEventDestroy(stops[i]));
  }
  const float mean =
      std::accumulate(samples.begin(), samples.end(), 0.0f) / iters;
  std::sort(samples.begin(), samples.end());
  const float median = iters % 2
                           ? samples[iters / 2]
                           : 0.5f * (samples[iters / 2 - 1] + samples[iters / 2]);
  return {samples.front(), median, mean};
}

void print_metrics(const char *name, const Timing &t, int M, int N, int K) {
  // GEMM 约定一次乘法和一次加法算 2 FLOPs。
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  const double gflops = flops / (t.median_ms * 1.0e6);

  // 这是理想情况下 A/B 各读一次、C 写一次的最低“逻辑字节数”，不是实测 DRAM
  // traffic。真实 traffic 应用 Nsight Compute 的 dram__bytes 指标测量。
  const double logical_bytes =
      sizeof(float) * (static_cast<double>(M) * K +
                       static_cast<double>(K) * N +
                       static_cast<double>(M) * N);
  const double logical_gbs = logical_bytes / (t.median_ms * 1.0e6);

  std::printf("%-8s min/median/mean = %8.4f / %8.4f / %8.4f ms, "
              "%9.2f GFLOP/s, logical %7.2f GB/s\n",
              name, t.min_ms, t.median_ms, t.mean_ms, gflops, logical_gbs);
}

bool validate(const std::vector<float> &A, const std::vector<float> &B,
              const std::vector<float> &reference,
              const std::vector<float> &actual, int M, int N, int K) {
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t worst = 0;
  bool finite = true;
  const size_t count = static_cast<size_t>(M) * N;
  for (size_t i = 0; i < count; ++i) {
    const double abs_err =
        std::abs(static_cast<double>(actual[i]) - reference[i]);
    const double rel_err =
        abs_err / (std::abs(static_cast<double>(reference[i])) + 1e-6);
    if (!std::isfinite(actual[i])) {
      finite = false;
    }
    if (abs_err > max_abs) {
      max_abs = abs_err;
      worst = i;
    }
    max_rel = std::max(max_rel, rel_err);
  }

  // 再抽样使用 double CPU dot product，避免两个 GPU kernel 犯同一种错误。
  const double atol = 2e-5 * K;
  const double rtol = 2e-4;
  double sampled_max_abs = 0.0;
  bool sampled_ok = true;
  constexpr int kSamples = 32;
  for (int s = 0; s < kSamples; ++s) {
    const int row = (s * 104729) % M;
    const int col = (s * 130363 + 17) % N;
    double expected = 0.0;
    for (int k = 0; k < K; ++k) {
      expected += static_cast<double>(A[row * K + k]) * B[k * N + col];
    }
    const double abs_err =
        std::abs(expected - static_cast<double>(actual[row * N + col]));
    sampled_max_abs = std::max(sampled_max_abs, abs_err);
    sampled_ok = sampled_ok && abs_err <= atol + rtol * std::abs(expected);
  }

  // 对随机 [-0.5, 0.5] 输入，误差随 K 增长；混合绝对/相对容差更稳健。
  bool ok = finite && sampled_ok;
  for (size_t i = 0; i < count && ok; ++i) {
    const double abs_err =
        std::abs(static_cast<double>(actual[i]) - reference[i]);
    ok = abs_err <=
         atol + rtol * std::abs(static_cast<double>(reference[i]));
  }
  std::printf("check    %s; max_abs=%.3e, max_rel=%.3e at (%zu,%zu), "
              "CPU-sampled max_abs=%.3e\n",
              ok ? "PASS" : "FAIL", max_abs, max_rel, worst / N, worst % N,
              sampled_max_abs);
  return ok;
}

int parse_positive(const char *text, const char *name) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value <= 0 || value > 1'000'000) {
    std::fprintf(stderr, "%s must be an integer in [1, 1000000]\n", name);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

int main(int argc, char **argv) {
  if (argc != 1 && argc != 5) {
    std::fprintf(stderr, "usage: %s [M N K iterations]\n", argv[0]);
    return EXIT_FAILURE;
  }
  const int M = argc == 5 ? parse_positive(argv[1], "M") : 513;
  const int N = argc == 5 ? parse_positive(argv[2], "N") : 509;
  const int K = argc == 5 ? parse_positive(argv[3], "K") : 515;
  const int iters = argc == 5 ? parse_positive(argv[4], "iterations") : 50;
  constexpr int warmup = 10;

  // Kernel 使用 int 做 row-major 地址计算，拒绝会导致乘积溢出的输入。
  const uint64_t max_index = std::numeric_limits<int>::max();
  if (static_cast<uint64_t>(M) * K > max_index ||
      static_cast<uint64_t>(K) * N > max_index ||
      static_cast<uint64_t>(M) * N > max_index) {
    std::fprintf(stderr, "matrix element count exceeds 32-bit kernel indexing\n");
    return EXIT_FAILURE;
  }

  const size_t a_count = static_cast<size_t>(M) * K;
  const size_t b_count = static_cast<size_t>(K) * N;
  const size_t c_count = static_cast<size_t>(M) * N;
  std::vector<float> h_A(a_count), h_B(b_count), h_naive(c_count),
      h_tiled(c_count);
  // 固定、无需随机数库且同时包含正负数的输入。
  for (size_t i = 0; i < a_count; ++i) {
    h_A[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 101) - 50) /
             101.0f;
  }
  for (size_t i = 0; i < b_count; ++i) {
    h_B[i] = static_cast<float>(static_cast<int>((i * 29 + 7) % 103) - 51) /
             103.0f;
  }

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C_naive = nullptr;
  float *d_C_tiled = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, a_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, b_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C_naive, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C_tiled, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_count * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_count * sizeof(float),
                        cudaMemcpyHostToDevice));

  launch_naive(d_A, d_B, d_C_naive, M, N, K);
  launch_tiled(d_A, d_B, d_C_tiled, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_naive.data(), d_C_naive, c_count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_tiled.data(), d_C_tiled, c_count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::printf("SGEMM C[%d,%d] = A[%d,%d] * B[%d,%d], warmup=%d, iters=%d\n",
              M, N, M, K, K, N, warmup, iters);
  const bool ok = validate(h_A, h_B, h_naive, h_tiled, M, N, K);

  const Timing naive = benchmark(launch_naive, d_A, d_B, d_C_naive, M, N, K,
                                 warmup, iters);
  const Timing tiled = benchmark(launch_tiled, d_A, d_B, d_C_tiled, M, N, K,
                                 warmup, iters);
  print_metrics("naive", naive, M, N, K);
  print_metrics("tiled", tiled, M, N, K);
  std::printf("speedup  naive/tiled median = %.2fx\n",
              naive.median_ms / tiled.median_ms);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C_naive));
  CUDA_CHECK(cudaFree(d_C_tiled));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
