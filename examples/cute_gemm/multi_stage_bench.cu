#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "multi_stage_annotated.cuh"

// 教学入口：--layout 先比较global/shared/register之间的tile步长，再逐lane验证覆盖。
// kernel及copy注释在multi_stage_annotated.cuh，示例图见docs/cute_gemm_81920.md第4~6节。
namespace half_gemm {
void verify_layouts();
}
namespace {
using half_gemm::Config;
using T = Config::T;
void check(cudaError_t status, char const* expr) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", expr, cudaGetErrorString(status));
    std::exit(1);
  }
}
void check(cublasStatus_t status, char const* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s: cuBLAS status %d\n", expr, int(status));
    std::exit(1);
  }
}
#define CHECK(expr) check((expr), #expr)
void require(bool condition, char const* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}
template <class E>
struct Buffer {
  E* p = nullptr;
  explicit Buffer(size_t n) { CHECK(cudaMalloc(&p, n * sizeof(E))); }
  ~Buffer() { cudaFree(p); }
  Buffer(Buffer const&) = delete;
  Buffer& operator=(Buffer const&) = delete;
};
float sample(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return (int(x % 65) - 32) / 64.f;  // [-0.5,0.5]，都能被 half 精确表示。
}
// 只计同一 stream 上的重复 GEMM，分配、memcpy、校验和预热均在事件之外。
template <class F>
float time_ms(F launch, int repeats) {
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) launch();
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  float elapsed = 0;
  CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return elapsed / repeats;
}
void report(char const* label, std::vector<float> times) {
  std::sort(times.begin(), times.end());
  double ms = times[times.size() / 2];
  double flops = 2.0 * Config::M * Config::N * Config::K;
  std::printf("%s median_ms=%.6f min_ms=%.6f max_ms=%.6f TFLOP/s=%.3f\n", label, ms, times.front(),
              times.back(), flops / (ms * 1e9));
}
}  // namespace
int main(int argc, char** argv) {
  if (argc > 2 ||
      (argc == 2 && std::strcmp(argv[1], "--layout") && std::strcmp(argv[1], "--check"))) {
    std::fprintf(stderr, "usage: %s [--layout|--check] (default: check and benchmark)\n", argv[0]);
    return 1;
  }
  half_gemm::verify_layouts();
  if (argc == 2 && !std::strcmp(argv[1], "--layout")) return 0;  // 无 GPU 也能检查 layout。
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  require(prop.major >= 8, "SM80+ required");
  std::printf("GPU=%s SM%d%d SMs=%d\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  int runtime = 0, driver = 0;
  CHECK(cudaRuntimeGetVersion(&runtime));
  CHECK(cudaDriverGetVersion(&driver));
  cublasHandle_t handle;
  CHECK(cublasCreate(&handle));
  int version = 0;
  CHECK(cublasGetVersion(handle, &version));
  // 明确 cuBLAS 的计算类型；不使用 FP32 accumulate 来替换题目的 half 规格。
  CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  std::printf("CUDA runtime=%d driver=%d cuBLAS=%d compute=CUBLAS_COMPUTE_16F\n", runtime, driver,
              version);
  constexpr int M = Config::M, N = Config::N, K = Config::K;
  size_t na = size_t(M) * K, nb = size_t(K) * N, nc = size_t(M) * N;
  std::vector<T> a(na), b(nb), got(nc), reference(nc);
  for (size_t i = 0; i < na; ++i) a[i] = T(sample(unsigned(i) + 17));
  // 数学 B[k,n] 的 column-major 地址=n*K+k，与 kernel 的 B(n,k) 一致。
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) b[n * K + k] = T(float(n == k));
  Buffer<T> da(na), db(nb), dc(nc), dr(nc);
  CHECK(cudaMemcpy(da.p, a.data(), na * sizeof(T), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db.p, b.data(), nb * sizeof(T), cudaMemcpyHostToDevice));
  CHECK(cudaFuncSetAttribute(half_gemm::gemm_multi_stage<Config>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, Config::kShmSize));
  cudaFuncAttributes attributes{};
  CHECK(cudaFuncGetAttributes(&attributes, half_gemm::gemm_multi_stage<Config>));
  int active_blocks = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, half_gemm::gemm_multi_stage<Config>, 128, Config::kShmSize));
  std::printf(
      "registers/thread=%d local_bytes/thread=%zu dynamic_shared/CTA=%d active_CTAs/SM=%d\n",
      attributes.numRegs, attributes.localSizeBytes, Config::kShmSize, active_blocks);
  auto launch = [&] {
    half_gemm::gemm_multi_stage<Config>
        <<<dim3(N / 128, M / 128), 128, Config::kShmSize>>>(dc.p, da.p, db.p, M, N, K);
    CHECK(cudaGetLastError());
  };
  T alpha(1), beta(0);
  auto blas = [&] {
    // cuBLAS 使用 column-major，因此计算 C^T[N,M]=B^T[N,K]*A^T[K,M]。
    // Bptr 本来是 column-major KxN，需 OP_T；Aptr 被看作 column-major KxM，需 OP_N。
    // 输出 column-major NxM 的存储，就是用户需要的 row-major MxN。
    CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, db.p, CUDA_R_16F, K, da.p,
                       CUDA_R_16F, K, &beta, dr.p, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
                       CUBLAS_GEMM_DEFAULT));
  };
  // B=I：精确验证所有 20971520 个输出，能检查最后 CTA、全部 epilogue 批次。
  CHECK(cudaMemset(dc.p, 0xff, nc * sizeof(T)));
  launch();
  CHECK(cudaMemcpy(got.data(), dc.p, nc * sizeof(T), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nc; ++i) require(float(got[i]) == float(a[i]), "full identity output");
  std::puts("PASS identity: all 20971520 output elements");

  for (size_t i = 0; i < nb; ++i) b[i] = T(sample(unsigned(i) + 1000003));
  CHECK(cudaMemcpy(db.p, b.data(), nb * sizeof(T), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dc.p, 0xff, nc * sizeof(T)));
  CHECK(cudaMemset(dr.p, 0xff, nc * sizeof(T)));
  launch();
  blas();
  CHECK(cudaMemcpy(got.data(), dc.p, nc * sizeof(T), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(reference.data(), dr.p, nc * sizeof(T), cudaMemcpyDeviceToHost));
  double max_error = 0, square_error = 0;
  size_t errors = 0;
  for (size_t i = 0; i < nc; ++i) {
    double x = float(got[i]), r = float(reference[i]), e = std::abs(x - r);
    if (!std::isfinite(x) || !std::isfinite(r) || e > 0.02 + 0.02 * std::abs(r)) ++errors;
    max_error = std::max(max_error, e);
    square_error += e * e;
  }
  std::printf("cuBLAS comparison: count=%zu max_abs=%.8g RMSE=%.8g outside_tolerance=%zu\n", nc,
              max_error, std::sqrt(square_error / nc), errors);
  require(errors == 0, "cuBLAS comparison tolerance");
  std::printf("cuBLAS bitwise_equal=%s\n",
              std::memcmp(got.data(), reference.data(), nc * sizeof(T)) == 0 ? "yes" : "no");
  // 独立 CPU double 点积，包含首末元素、内部位置；不是把 cuBLAS 当作精确实数。
  double cpu_max = 0, blas_cpu_max = 0;
  for (unsigned i = 0; i < 1024; ++i) {
    int row = i == 0 ? 0 : i == 1 ? M - 1 : int((i * 7919u) % M);
    int col = i == 0 ? 0 : i == 1 ? N - 1 : int((i * 73u) % N);
    double exact = 0;
    for (int q = 0; q < K; ++q)
      exact += double(float(a[size_t(row) * K + q])) * float(b[col * K + q]);
    double custom_err = std::abs(double(float(got[size_t(row) * N + col])) - exact);
    double blas_err = std::abs(double(float(reference[size_t(row) * N + col])) - exact);
    cpu_max = std::max(cpu_max, custom_err);
    blas_cpu_max = std::max(blas_cpu_max, blas_err);
    require(custom_err <= 0.02 + 0.02 * std::abs(exact), "CPU double sample");
  }
  std::printf("PASS CPU double: 1024 positions custom_max_abs=%.8g cuBLAS_max_abs=%.8g\n", cpu_max,
              blas_cpu_max);
  std::printf("M=%d N=%d K=%d grid=(2,640) threads=128 shared=%d FLOPs=%.0f\n", M, N, K,
              Config::kShmSize, 2.0 * M * N * K);
  if (argc == 1) {
    // 预热 10 次，5 轮 x 50 次，每轮交替先测哪个实现，减少顺序带来的影响。
    for (int i = 0; i < 10; ++i) {
      launch();
      blas();
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<float> custom_times, blas_times;
    for (int round = 0; round < 5; ++round) {
      if (round % 2 == 0) {
        custom_times.push_back(time_ms(launch, 50));
        blas_times.push_back(time_ms(blas, 50));
      } else {
        blas_times.push_back(time_ms(blas, 50));
        custom_times.push_back(time_ms(launch, 50));
      }
    }
    report("CuTe annotated", custom_times);
    report("cuBLAS 16F", blas_times);
  }
  CHECK(cublasDestroy(handle));
}
