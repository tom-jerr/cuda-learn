#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int kWarpSize = 32;
constexpr int kVectorWidth = 4;

__device__ __forceinline__ float warp_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <int RowsPerWrap, int WarpsPerBlock>
__global__ void sgemv_kernel(const float *__restrict__ weight,
                             const float *__restrict__ x, float *__restrict__ y,
                             int rows, int cols) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int first_row = (blockIdx.x * WarpsPerBlock + warp) * RowsPerWrap;
  if (first_row >= rows)
    return;

  float sums[RowsPerWrap] = {};
  for (int col = lane * kVectorWidth; col < cols;
       col += kWarpSize * kVectorWidth) {
    const float4 xv = *reinterpret_cast<const float4 *>(x + col);
#pragma unroll
    for (int r = 0; r < RowsPerWrap; ++r) {
      const int row = first_row + r;
      if (row < rows) {
        const float4 w =
            *reinterpret_cast<const float4 *>(weight + row * cols + col);
        sums[r] = fmaf(w.x, xv.x, sums[r]);
        sums[r] = fmaf(w.y, xv.y, sums[r]);
        sums[r] = fmaf(w.z, xv.z, sums[r]);
        sums[r] = fmaf(w.w, xv.w, sums[r]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RowsPerWrap; ++r) {
    sums[r] = warp_sum(sums[r]);
    if (lane == 0 && first_row + r < rows) {
      y[first_row + r] = sums[r];
    }
  }
}

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

template <int RowsPerWrap, int WarpsPerBlock>
void launch_sgemv(const float *weight, const float *x, float *y, int rows,
                  int cols) {
  const int rows_per_block = RowsPerWrap * WarpsPerBlock;
  const int blocks = (rows + rows_per_block - 1) / rows_per_block;
  sgemv_kernel<RowsPerWrap, WarpsPerBlock>
      <<<blocks, WarpsPerBlock * kWarpSize>>>(weight, x, y, rows, cols);
}

bool run_precision_test(int rows, int cols, unsigned int seed) {
  // float4 loads require every row to start on a 16-byte boundary.
  if (rows <= 0 || cols <= 0 || cols % kVectorWidth != 0) {
    std::fprintf(stderr, "invalid shape: rows=%d, cols=%d\n", rows, cols);
    return false;
  }

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> weight(static_cast<size_t>(rows) * cols);
  std::vector<float> x(cols);
  std::vector<float> y(rows);
  std::vector<double> reference(rows, 0.0);

  for (float &value : weight)
    value = distribution(generator);
  for (float &value : x)
    value = distribution(generator);

  // Accumulate in double precision so the reference error is much smaller
  // than the error of the float kernel being measured.
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      reference[row] += static_cast<double>(weight[row * cols + col]) *
                        static_cast<double>(x[col]);
    }
  }

  float *d_weight = nullptr;
  float *d_x = nullptr;
  float *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, y.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  constexpr int kRowsPerWarp = 4;
  constexpr int kWarpsPerBlock = 4;
  launch_sgemv<kRowsPerWarp, kWarpsPerBlock>(d_weight, d_x, d_y, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  constexpr double kAbsoluteTolerance = 2.0e-4;
  constexpr double kRelativeTolerance = 2.0e-5;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  int worst_row = 0;
  bool passed = true;
  for (int row = 0; row < rows; ++row) {
    const double absolute_error =
        std::abs(static_cast<double>(y[row]) - reference[row]);
    const double relative_error =
        absolute_error / std::max(std::abs(reference[row]), 1.0e-12);
    if (absolute_error > max_absolute_error) {
      max_absolute_error = absolute_error;
      worst_row = row;
    }
    max_relative_error = std::max(max_relative_error, relative_error);
    if (absolute_error >
        kAbsoluteTolerance + kRelativeTolerance * std::abs(reference[row])) {
      passed = false;
    }
  }

  std::printf(
      "rows=%4d cols=%4d: %s, max_abs=%.3e, max_rel=%.3e "
      "(worst row %d: GPU=%.8g, reference=%.8g)\n",
      rows, cols, passed ? "PASS" : "FAIL", max_absolute_error,
      max_relative_error, worst_row, y[worst_row], reference[worst_row]);
  return passed;
}

int main() {
  // These shapes cover a tiny input, a multi-iteration column loop, and a
  // partial final block in the row dimension.
  bool passed = true;
  passed &= run_precision_test(1, 4, 1);
  passed &= run_precision_test(17, 132, 2);
  passed &= run_precision_test(1003, 1024, 3);

  std::printf("SGEMV precision test: %s\n", passed ? "PASS" : "FAIL");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
