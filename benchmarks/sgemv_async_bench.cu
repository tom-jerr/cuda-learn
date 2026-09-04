#include "sgemv_async.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    const cudaError_t status_ = (expr);                                        \
    if (status_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(status_));                               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    const cublasStatus_t status_ = (expr);                                     \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                    \
      std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                   static_cast<int>(status_));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace {

enum class Kernel {
  kAll,
  kAsync,
  kCoalesced,
  kRegister2,
  kRegister4,
  kRegister8,
  kRegister16,
  kCublas
};

struct Options {
  int rows = 11008; // Llama-style gated MLP projection: 4096 -> 11008.
  int cols = 4096;
  int warmup = 20;
  int iterations = 100;
  bool check = true;
  Kernel kernel = Kernel::kAll;
};

void usage(const char *program) {
  std::printf(
      "Usage: %s [--rows N] [--cols K] [--warmup N] [--iters N]\n"
      "          [--kernel all|async|coalesced|register2|register4|register8|register16|cublas]\n"
      "          [--skip-check]\n",
      program);
}

int parse_positive(const char *text, const char *option) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value <= 0 || value > (1L << 30)) {
    std::fprintf(stderr, "invalid value for %s: %s\n", option, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

int parse_nonnegative(const char *text, const char *option) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value < 0 || value > (1L << 30)) {
    std::fprintf(stderr, "invalid value for %s: %s\n", option, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
      options.rows = parse_positive(argv[++i], "--rows");
    } else if (std::strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
      options.cols = parse_positive(argv[++i], "--cols");
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      options.warmup = parse_nonnegative(argv[++i], "--warmup");
    } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      options.iterations = parse_positive(argv[++i], "--iters");
    } else if (std::strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
      const std::string kernel = argv[++i];
      if (kernel == "all") {
        options.kernel = Kernel::kAll;
      } else if (kernel == "async") {
        options.kernel = Kernel::kAsync;
      } else if (kernel == "coalesced") {
        options.kernel = Kernel::kCoalesced;
      } else if (kernel == "register2") {
        options.kernel = Kernel::kRegister2;
      } else if (kernel == "register4") {
        options.kernel = Kernel::kRegister4;
      } else if (kernel == "register8") {
        options.kernel = Kernel::kRegister8;
      } else if (kernel == "register16") {
        options.kernel = Kernel::kRegister16;
      } else if (kernel == "cublas") {
        options.kernel = Kernel::kCublas;
      } else {
        std::fprintf(stderr, "unknown kernel: %s\n", kernel.c_str());
        std::exit(EXIT_FAILURE);
      }
    } else if (std::strcmp(argv[i], "--skip-check") == 0) {
      options.check = false;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  return options;
}

template <class Launch>
float benchmark_ms(Launch launch, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return elapsed_ms / iterations;
}

void print_result(const char *name, float milliseconds, int rows, int cols) {
  const double seconds = milliseconds * 1.0e-3;
  const double bytes =
      (static_cast<double>(rows) * cols + cols + rows) * sizeof(float);
  const double flops = 2.0 * rows * static_cast<double>(cols);
  std::printf("%-12s %8.4f ms  %8.2f GB/s  %7.3f TFLOP/s\n", name,
              milliseconds, bytes / seconds / 1.0e9,
              flops / seconds / 1.0e12);
}

} // namespace

int main(int argc, char **argv) {
  const Options options = parse_options(argc, argv);
  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 8) {
    std::fprintf(stderr, "cp.async needs compute capability 8.0 or newer\n");
    return EXIT_FAILURE;
  }

  const size_t weight_elements =
      static_cast<size_t>(options.rows) * options.cols;
  std::vector<float> h_weight(weight_elements);
  std::vector<float> h_x(options.cols);
  for (size_t i = 0; i < weight_elements; ++i) {
    h_weight[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 256.0f;
  }
  for (int i = 0; i < options.cols; ++i) {
    h_x[i] = static_cast<float>((i % 17) - 8) / 16.0f;
  }

  float *d_weight = nullptr;
  float *d_x = nullptr;
  float *d_y = nullptr;
  float *d_reference = nullptr;
  CUDA_CHECK(cudaMalloc(&d_weight, weight_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x, options.cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, options.rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_reference, options.rows * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                        weight_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), options.cols * sizeof(float),
                        cudaMemcpyHostToDevice));

  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto cublas_launch = [&] {
    // Row-major [rows, cols] is column-major [cols, rows] in the cuBLAS view.
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, options.cols, options.rows,
                            &alpha, d_weight, options.cols, d_x, 1, &beta,
                            d_reference, 1));
  };
  auto async_launch = [&] {
    launch_sgemv_async(d_weight, d_x, d_y, options.rows, options.cols);
  };
  auto coalesced_launch = [&] {
    launch_sgemv_coalesced(d_weight, d_x, d_y, options.rows, options.cols);
  };
  auto register2_launch = [&] {
    launch_sgemv_register_reuse(d_weight, d_x, d_y, options.rows, options.cols,
                                2);
  };
  auto register4_launch = [&] {
    launch_sgemv_register_reuse(d_weight, d_x, d_y, options.rows, options.cols,
                                4);
  };
  auto register8_launch = [&] {
    launch_sgemv_register_reuse(d_weight, d_x, d_y, options.rows, options.cols,
                                8);
  };
  auto register16_launch = [&] {
    launch_sgemv_register_reuse(d_weight, d_x, d_y, options.rows, options.cols,
                                16);
  };

  if (options.check) {
    cublas_launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(options.rows);
    std::vector<float> reference(options.rows);
    CUDA_CHECK(cudaMemcpy(reference.data(), d_reference,
                          options.rows * sizeof(float),
                          cudaMemcpyDeviceToHost));
    auto check_one = [&](const char *name, auto launch) {
      launch();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(actual.data(), d_y, options.rows * sizeof(float),
                            cudaMemcpyDeviceToHost));
      float max_abs_error = 0.0f;
      float max_rel_error = 0.0f;
      for (int row = 0; row < options.rows; ++row) {
        const float abs_error = std::fabs(actual[row] - reference[row]);
        const float rel_error =
            abs_error / std::max(std::fabs(reference[row]), 1.0e-5f);
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
      }
      if (max_abs_error > 2.0e-3f && max_rel_error > 2.0e-3f) {
        std::fprintf(
            stderr,
            "%s correctness FAILED: max abs error=%g, max rel error=%g\n",
            name, max_abs_error, max_rel_error);
        std::exit(EXIT_FAILURE);
      }
      std::printf("%-12s correctness: PASS (max abs %.3g, max rel %.3g)\n",
                  name, max_abs_error, max_rel_error);
    };
    if (options.kernel == Kernel::kAll || options.kernel == Kernel::kAsync) {
      check_one("async", async_launch);
    }
    if (options.kernel == Kernel::kAll || options.kernel == Kernel::kCoalesced) {
      check_one("coalesced", coalesced_launch);
    }
    if (options.kernel == Kernel::kAll || options.kernel == Kernel::kRegister2) {
      check_one("register x2", register2_launch);
    }
    if (options.kernel == Kernel::kAll || options.kernel == Kernel::kRegister4) {
      check_one("register x4", register4_launch);
    }
    if (options.kernel == Kernel::kAll || options.kernel == Kernel::kRegister8) {
      check_one("register x8", register8_launch);
    }
    if (options.kernel == Kernel::kAll ||
        options.kernel == Kernel::kRegister16) {
      check_one("register x16", register16_launch);
    }
    if (options.kernel == Kernel::kCublas) {
      std::printf("cuBLAS       correctness: reference\n");
    }
  }

  std::printf("GPU: %s (sm_%d%d)\n", properties.name, properties.major,
              properties.minor);
  int memory_clock_khz = 0;
  int memory_bus_width = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz,
                                    cudaDevAttrMemoryClockRate, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&memory_bus_width,
                                    cudaDevAttrGlobalMemoryBusWidth, device));
  const double peak_dram_gb_s = 2.0 * memory_clock_khz * 1000.0 *
                                (memory_bus_width / 8.0) / 1.0e9;
  const SgemvKernelInfo kernel_info = get_sgemv_async_kernel_info();
  const SgemvKernelInfo register_info = get_sgemv_register_kernel_info();
  const double occupancy =
      100.0 * kernel_info.active_blocks_per_sm * kernel_info.threads_per_block /
      properties.maxThreadsPerMultiProcessor;
  std::printf("Peak DRAM: %.1f GB/s (%d-bit, %.0f MHz); async kernel: %d regs, "
              "%d B smem, %.0f%% theoretical occupancy\n",
              peak_dram_gb_s, memory_bus_width, memory_clock_khz / 1000.0,
              kernel_info.registers_per_thread,
              kernel_info.static_smem_bytes, occupancy);
  const double register_occupancy =
      100.0 * register_info.active_blocks_per_sm *
      register_info.threads_per_block / properties.maxThreadsPerMultiProcessor;
  std::printf("Register-reuse kernel: %d threads, %d regs, %d B smem, %.0f%% "
              "theoretical occupancy\n",
              register_info.threads_per_block,
              register_info.registers_per_thread,
              register_info.static_smem_bytes, register_occupancy);
  std::printf("SGEMV: y[%d] = W[%d,%d] * x[%d], %.1f MiB weights\n",
              options.rows, options.rows, options.cols, options.cols,
              weight_elements * sizeof(float) / 1048576.0);
  if (options.kernel == Kernel::kAll || options.kernel == Kernel::kAsync) {
    print_result("async", benchmark_ms(async_launch, options.warmup,
                                       options.iterations),
                 options.rows, options.cols);
  }
  if (options.kernel == Kernel::kAll || options.kernel == Kernel::kCoalesced) {
    print_result("coalesced", benchmark_ms(coalesced_launch, options.warmup,
                                           options.iterations),
                 options.rows, options.cols);
  }
  if (options.kernel == Kernel::kAll || options.kernel == Kernel::kRegister2) {
    print_result("register x2", benchmark_ms(register2_launch, options.warmup,
                                              options.iterations),
                 options.rows, options.cols);
  }
  if (options.kernel == Kernel::kAll || options.kernel == Kernel::kRegister4) {
    print_result("register x4", benchmark_ms(register4_launch, options.warmup,
                                              options.iterations),
                 options.rows, options.cols);
  }
  if (options.kernel == Kernel::kAll || options.kernel == Kernel::kRegister8) {
    print_result("register x8", benchmark_ms(register8_launch, options.warmup,
                                              options.iterations),
                 options.rows, options.cols);
  }
  if (options.kernel == Kernel::kAll || options.kernel == Kernel::kRegister16) {
    print_result("register x16", benchmark_ms(register16_launch, options.warmup,
                                               options.iterations),
                 options.rows, options.cols);
  }
  if (options.kernel == Kernel::kAll || options.kernel == Kernel::kCublas) {
    print_result("cuBLAS", benchmark_ms(cublas_launch, options.warmup,
                                        options.iterations),
                 options.rows, options.cols);
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_reference));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_weight));
  return EXIT_SUCCESS;
}
