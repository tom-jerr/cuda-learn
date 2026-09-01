#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

// 极简 W4A16：A/C 为 FP16，weight 为 group-wise UINT4。
// AWQ: real_w = scale * (q - zero_point)
// GPTQ: real_w = scale * (q - 8)，8 是隐式 symmetric bias。
constexpr int kBits = 4;
constexpr int kPack = 32 / kBits;
constexpr int kBM = 16;
constexpr int kBN = 16;
constexpr int kBK = 32;
constexpr int kThreads = kBM * kBN;

struct W4Weights {
  // checkpoint_qweight:
  //   AWQ  [K, N/8]，沿 N pack，且 nibble 次序 interleaved；
  //   GPTQ [K/8, N]，沿 K pack，标准 nibble 次序。
  std::vector<uint32_t> checkpoint_qweight;
  std::vector<uint32_t> checkpoint_qzeros;  // AWQ [K/G, N/8]

  // 教学 kernel layout：[K/8, N]。同一个 k-pack 下相邻线程读取连续 N，
  // 因而 weight word load 合并；真实 Marlin 会继续重排成 MMA tile-major。
  std::vector<uint32_t> kernel_qweight;
  std::vector<half> scales;       // [K/G, N]
  std::vector<uint8_t> zeros;     // AWQ [K/G, N]；GPTQ 不使用
  std::vector<float> dequantized; // [K, N]，只用于 CPU reference
};

int clamp_int(int x, int lo, int hi) {
  return std::max(lo, std::min(hi, x));
}

// AWQ 对每 8 个连续输出通道的 interleave：bit positions 中依次存放
// logical lanes [0,2,4,6,1,3,5,7]。这是 checkpoint/dequantizer 格式，
// 不是后续 GEMM 最自然的 K-major 格式。
constexpr int kAwqPackOrder[kPack] = {0, 2, 4, 6, 1, 3, 5, 7};
constexpr int kAwqBitPosition[kPack] = {0, 4, 1, 5, 2, 6, 3, 7};

uint32_t pack_awq_lanes(const uint8_t* values) {
  uint32_t word = 0;
  for (int bit_pos = 0; bit_pos < kPack; ++bit_pos) {
    word |= static_cast<uint32_t>(values[kAwqPackOrder[bit_pos]])
            << (bit_pos * kBits);
  }
  return word;
}

uint8_t unpack_awq_lane(uint32_t word, int logical_lane) {
  return (word >> (kAwqBitPosition[logical_lane] * kBits)) & 0xF;
}

W4Weights quantize_awq(const std::vector<float>& weight, int K, int N,
                       int group_size) {
  const int groups = K / group_size;
  W4Weights result;
  result.checkpoint_qweight.resize(K * (N / kPack));
  result.checkpoint_qzeros.resize(groups * (N / kPack));
  result.kernel_qweight.assign((K / kPack) * N, 0);
  result.scales.resize(groups * N);
  result.zeros.resize(groups * N);
  std::vector<uint8_t> q(K * N);

  for (int g = 0; g < groups; ++g) {
    for (int n = 0; n < N; ++n) {
      float w_min = 0.0f;
      float w_max = 0.0f;
      for (int k = g * group_size; k < (g + 1) * group_size; ++k) {
        w_min = std::min(w_min, weight[k * N + n]);
        w_max = std::max(w_max, weight[k * N + n]);
      }
      const float scale_fp32 =
          w_max > w_min ? (w_max - w_min) / 15.0f : 1.0f;
      const half scale_fp16 = __float2half(scale_fp32);
      const float scale = __half2float(scale_fp16);
      const int zp = clamp_int(static_cast<int>(std::nearbyint(-w_min / scale)),
                               0, 15);
      result.scales[g * N + n] = scale_fp16;
      result.zeros[g * N + n] = static_cast<uint8_t>(zp);
      for (int k = g * group_size; k < (g + 1) * group_size; ++k) {
        q[k * N + n] = static_cast<uint8_t>(clamp_int(
            static_cast<int>(std::nearbyint(weight[k * N + n] / scale)) + zp,
            0, 15));
      }
    }
  }

  // 模拟 AWQ checkpoint：qweight/qzeros 都沿输出 N pack。
  for (int k = 0; k < K; ++k) {
    for (int nb = 0; nb < N / kPack; ++nb) {
      result.checkpoint_qweight[k * (N / kPack) + nb] =
          pack_awq_lanes(&q[k * N + nb * kPack]);
    }
  }
  for (int g = 0; g < groups; ++g) {
    for (int nb = 0; nb < N / kPack; ++nb) {
      result.checkpoint_qzeros[g * (N / kPack) + nb] =
          pack_awq_lanes(&result.zeros[g * N + nb * kPack]);
    }
  }

  // 离线/加载时 repack：撤销 AWQ interleave，并改成沿 K pack。
  std::fill(result.zeros.begin(), result.zeros.end(), 0);
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      const uint32_t src =
          result.checkpoint_qweight[k * (N / kPack) + n / kPack];
      const uint8_t value = unpack_awq_lane(src, n % kPack);
      result.kernel_qweight[(k / kPack) * N + n] |=
          static_cast<uint32_t>(value) << ((k % kPack) * kBits);
    }
  }
  for (int g = 0; g < groups; ++g) {
    for (int n = 0; n < N; ++n) {
      const uint32_t src =
          result.checkpoint_qzeros[g * (N / kPack) + n / kPack];
      result.zeros[g * N + n] = unpack_awq_lane(src, n % kPack);
    }
  }

  result.dequantized.resize(K * N);
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      const uint32_t word = result.kernel_qweight[(k / kPack) * N + n];
      const int value = (word >> ((k % kPack) * kBits)) & 0xF;
      const int g = k / group_size;
      result.dequantized[k * N + n] =
          __half2float(result.scales[g * N + n]) *
          (value - static_cast<int>(result.zeros[g * N + n]));
    }
  }
  return result;
}

W4Weights quantize_gptq(const std::vector<float>& weight, int K, int N,
                        int group_size) {
  const int groups = K / group_size;
  W4Weights result;
  result.checkpoint_qweight.assign((K / kPack) * N, 0);
  result.scales.resize(groups * N);

  for (int g = 0; g < groups; ++g) {
    for (int n = 0; n < N; ++n) {
      float w_min = 0.0f;
      float w_max = 0.0f;
      for (int k = g * group_size; k < (g + 1) * group_size; ++k) {
        w_min = std::min(w_min, weight[k * N + n]);
        w_max = std::max(w_max, weight[k * N + n]);
      }
      // signed range [-8, 7]，分别用两侧端点求 scale，存储时加 bias 8。
      const float scale_candidate = std::max(std::fabs(w_max / 7.0f),
                                             std::fabs(w_min / -8.0f));
      const float scale_fp32 = scale_candidate > 0.0f ? scale_candidate : 1.0f;
      result.scales[g * N + n] = __float2half(scale_fp32);
      const float scale = __half2float(result.scales[g * N + n]);
      for (int k = g * group_size; k < (g + 1) * group_size; ++k) {
        const int signed_q = clamp_int(
            static_cast<int>(std::nearbyint(weight[k * N + n] / scale)), -8,
            7);
        const uint32_t stored = static_cast<uint32_t>(signed_q + 8);
        result.checkpoint_qweight[(k / kPack) * N + n] |=
            stored << ((k % kPack) * kBits);
      }
    }
  }

  // 无 desc_act 的标准 GPTQ pack 已是 K-packed；真实 Marlin 仍会把它
  // repack 到 Tensor Core tile-major。这里复制到教学 kernel layout。
  result.kernel_qweight = result.checkpoint_qweight;
  result.dequantized.resize(K * N);
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      const uint32_t word = result.kernel_qweight[(k / kPack) * N + n];
      const int stored = (word >> ((k % kPack) * kBits)) & 0xF;
      result.dequantized[k * N + n] =
          __half2float(result.scales[(k / group_size) * N + n]) *
          (stored - 8);
    }
  }
  return result;
}

template <bool kHasZeroPoint>
__global__ void w4a16_gemm_kernel(const half* __restrict__ a,
                                  const uint32_t* __restrict__ packed_w,
                                  const half* __restrict__ scales,
                                  const uint8_t* __restrict__ zero_points,
                                  half* __restrict__ c, int M, int N, int K,
                                  int group_size) {
  __shared__ half a_tile[kBM][kBK];
  __shared__ half w_tile[kBK][kBN];

  const int tid = threadIdx.x;
  const int local_m = tid / kBN;
  const int local_n = tid % kBN;
  const int global_m = blockIdx.y * kBM + local_m;
  const int global_n = blockIdx.x * kBN + local_n;
  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += kBK) {
    // A tile: 256 threads each load two FP16 values.
    for (int idx = tid; idx < kBM * kBK; idx += kThreads) {
      const int row = idx / kBK;
      const int col = idx % kBK;
      const int gm = blockIdx.y * kBM + row;
      const int gk = k0 + col;
      a_tile[row][col] =
          (gm < M && gk < K) ? a[gm * K + gk] : __float2half(0.0f);
    }

    // 每个 active thread 读取一个 uint32，再展开其中 8 个 UINT4 到 smem。
    // 同一 k-pack 的线程沿 N 连续，global load 可合并。
    constexpr int kPackedWordsPerTile = (kBK / kPack) * kBN;
    if (tid < kPackedWordsPerTile) {
      const int local_k_pack = tid / kBN;
      const int wn = tid % kBN;
      const int gn = blockIdx.x * kBN + wn;
      const int global_k_pack = k0 / kPack + local_k_pack;
      const uint32_t word =
          gn < N ? packed_w[global_k_pack * N + gn] : 0u;
#pragma unroll
      for (int lane = 0; lane < kPack; ++lane) {
        const int local_k = local_k_pack * kPack + lane;
        const int global_k = k0 + local_k;
        const int stored = (word >> (lane * kBits)) & 0xF;
        float dequant = 0.0f;
        if (global_k < K && gn < N) {
          const int group = global_k / group_size;
          const float scale = __half2float(scales[group * N + gn]);
          if constexpr (kHasZeroPoint) {
            dequant = scale *
                      (stored - static_cast<int>(zero_points[group * N + gn]));
          } else {
            dequant = scale * (stored - 8);
          }
        }
        w_tile[local_k][wn] = __float2half(dequant);
      }
    }
    __syncthreads();

    if (global_m < M && global_n < N) {
#pragma unroll
      for (int kk = 0; kk < kBK; ++kk) {
        acc += __half2float(a_tile[local_m][kk]) *
               __half2float(w_tile[kk][local_n]);
      }
    }
    __syncthreads();
  }

  if (global_m < M && global_n < N) {
    c[global_m * N + global_n] = __float2half(acc);
  }
}

template <bool kHasZeroPoint>
void launch_w4a16(const half* a, const uint32_t* packed_w,
                  const half* scales, const uint8_t* zero_points, half* c,
                  int M, int N, int K, int group_size) {
  dim3 block(kThreads);
  dim3 grid((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
  w4a16_gemm_kernel<kHasZeroPoint><<<grid, block>>>(
      a, packed_w, scales, zero_points, c, M, N, K, group_size);
}

std::vector<float> cpu_gemm(const std::vector<half>& a,
                            const std::vector<float>& w, int M, int N, int K) {
  std::vector<float> c(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += __half2float(a[m * K + k]) * w[k * N + n];
      }
      c[m * N + n] = acc;
    }
  }
  return c;
}

struct ErrorStats {
  float max_abs = 0.0f;
  double mse = 0.0;
};

ErrorStats compare(const std::vector<half>& actual,
                   const std::vector<float>& expected) {
  ErrorStats stats;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float error = __half2float(actual[i]) - expected[i];
    stats.max_abs = std::max(stats.max_abs, std::fabs(error));
    stats.mse += static_cast<double>(error) * error;
  }
  stats.mse /= actual.size();
  return stats;
}

ErrorStats compare_fp32(const std::vector<float>& actual,
                        const std::vector<float>& expected) {
  ErrorStats stats;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float error = actual[i] - expected[i];
    stats.max_abs = std::max(stats.max_abs, std::fabs(error));
    stats.mse += static_cast<double>(error) * error;
  }
  stats.mse /= actual.size();
  return stats;
}

template <bool kHasZeroPoint>
float benchmark(const half* d_a, const uint32_t* d_w, const half* d_scales,
                const uint8_t* d_zeros, half* d_c, int M, int N, int K,
                int group_size) {
  constexpr int warmup = 20;
  constexpr int iterations = 200;
  for (int i = 0; i < warmup; ++i) {
    launch_w4a16<kHasZeroPoint>(d_a, d_w, d_scales, d_zeros, d_c, M, N, K,
                                group_size);
  }
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch_w4a16<kHasZeroPoint>(d_a, d_w, d_scales, d_zeros, d_c, M, N, K,
                                group_size);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / iterations;
}

template <bool kHasZeroPoint>
bool run_case(const char* name, const std::vector<half>& a,
              const std::vector<float>& fp_weight, const W4Weights& qweight,
              int M, int N, int K, int group_size) {
  half *d_a = nullptr, *d_scales = nullptr, *d_c = nullptr;
  uint32_t* d_w = nullptr;
  uint8_t* d_zeros = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_w, qweight.kernel_qweight.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_scales, qweight.scales.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_c, M * N * sizeof(half)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, qweight.kernel_qweight.data(),
                        qweight.kernel_qweight.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scales, qweight.scales.data(),
                        qweight.scales.size() * sizeof(half),
                        cudaMemcpyHostToDevice));
  if constexpr (kHasZeroPoint) {
    CUDA_CHECK(cudaMalloc(&d_zeros, qweight.zeros.size() * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_zeros, qweight.zeros.data(),
                          qweight.zeros.size() * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
  }

  launch_w4a16<kHasZeroPoint>(d_a, d_w, d_scales, d_zeros, d_c, M, N, K,
                              group_size);
  CUDA_CHECK(cudaGetLastError());
  std::vector<half> output(M * N);
  CUDA_CHECK(cudaMemcpy(output.data(), d_c, output.size() * sizeof(half),
                        cudaMemcpyDeviceToHost));

  const std::vector<float> quant_ref = cpu_gemm(a, qweight.dequantized, M, N, K);
  const std::vector<float> fp_ref = cpu_gemm(a, fp_weight, M, N, K);
  const ErrorStats kernel_error = compare(output, quant_ref);
  const ErrorStats quant_error = compare_fp32(quant_ref, fp_ref);
  const float elapsed_ms = benchmark<kHasZeroPoint>(
      d_a, d_w, d_scales, d_zeros, d_c, M, N, K, group_size);
  const double tflops = 2.0 * M * N * K / (elapsed_ms * 1.0e9);

  std::printf(
      "%s: kernel-vs-dequant max=%g rmse=%g | quant-vs-fp rmse=%g | "
      "%g ms, %.3f TFLOP/s\n",
      name, kernel_error.max_abs, std::sqrt(kernel_error.mse),
      std::sqrt(quant_error.mse), elapsed_ms, tflops);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_scales));
  CUDA_CHECK(cudaFree(d_c));
  if (d_zeros != nullptr) CUDA_CHECK(cudaFree(d_zeros));
  return kernel_error.max_abs < 0.05f;
}

int main() {
  constexpr int M = 8;    // decode 阶段常见 small-M
  constexpr int N = 256;
  constexpr int K = 512;
  constexpr int group_size = 32;
  static_assert(K % group_size == 0 && K % kBK == 0 && N % kBN == 0);

  std::vector<half> a(M * K);
  std::vector<float> weight(K * N);
  for (int i = 0; i < M * K; ++i) {
    a[i] = __float2half(std::sin(i * 0.017f) * 0.7f);
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      // channel-dependent range + small offset，使 AWQ/GPTQ 两种 qparam 都有意义。
      const float channel_scale = 0.2f + (n % 31) * 0.025f;
      weight[k * N + n] =
          channel_scale * std::sin(k * 0.031f + n * 0.007f) +
          0.04f * std::cos(k * 0.013f);
    }
  }

  const W4Weights awq = quantize_awq(weight, K, N, group_size);
  const W4Weights gptq = quantize_gptq(weight, K, N, group_size);
  std::printf("AWQ checkpoint qweight [%d,%d] -> kernel [%d,%d]\n", K,
              N / kPack, K / kPack, N);
  std::printf("GPTQ checkpoint qweight [%d,%d] -> kernel [%d,%d]\n", K / kPack,
              N, K / kPack, N);

  const bool awq_ok =
      run_case<true>("AWQ ", a, weight, awq, M, N, K, group_size);
  const bool gptq_ok =
      run_case<false>("GPTQ", a, weight, gptq, M, N, K, group_size);
  std::printf("W4A16 minimal GEMM: %s\n", awq_ok && gptq_ok ? "PASS" : "FAIL");
  return awq_ok && gptq_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
