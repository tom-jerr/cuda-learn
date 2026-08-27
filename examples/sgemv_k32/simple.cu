#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 4;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffu, val, offset);
  }
  return val;
}

// SGEMV: Warp SGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(ceil(M/4)), block(32,4)，blockDim.x=warp size，blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k32(const float *a, const float *x, float *y, int M,
                          int K) {
  const int tid = threadIdx.x;          // 0~31
  const int row_in_block = threadIdx.y; // 0~3
  const int row = blockIdx.x * blockDim.y + row_in_block;
  if (row >= M)
    return; // 整个 warp 走相同分支，可以安全 return

  float sum = 0.0f;
  for (int k = tid; k < K; k += kWarpSize) {
    // 同一次循环中，32 lanes 读取 a 的连续 32 项，合并访存。
    sum = fmaf(a[row * K + k], x[k], sum);
  }

  sum = warp_reduce_sum(sum);
  if (tid == 0)
    y[row] = sum;
}

void launch_sgemv_k32(const float *a, const float *x, float *y, int M, int K) {
  const dim3 block(kWarpSize, kRowsPerBlock);
  const dim3 grid((M + kRowsPerBlock - 1) / kRowsPerBlock);
  sgemv_k32<<<grid, block>>>(a, x, y, M, K);
}

int main() {
  constexpr int M = 1003; // 检查最后一个不完整 block
  constexpr int K = 320;
  static_assert(K % kWarpSize == 0);

  std::vector<float> a(M * K), x(K), y(M), reference(M, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      a[m * K + k] = static_cast<float>((m * 13 + k * 7) % 29 - 14) / 8.0f;
    }
  }
  for (int k = 0; k < K; ++k) {
    x[k] = static_cast<float>(k % 9 - 4) / 4.0f;
  }
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      reference[m] = fmaf(a[m * K + k], x[k], reference[m]);
    }
  }

  float *d_a = nullptr, *d_x = nullptr, *d_y = nullptr;
  cudaMalloc(&d_a, a.size() * sizeof(float));
  cudaMalloc(&d_x, x.size() * sizeof(float));
  cudaMalloc(&d_y, y.size() * sizeof(float));
  cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

  launch_sgemv_k32(d_a, d_x, d_y, M, K);

  cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for (int m = 0; m < M; ++m) {
    if (std::fabs(y[m] - reference[m]) > 1e-5f) {
      std::fprintf(stderr, "mismatch row %d: GPU=%g, CPU=%g\n", m, y[m],
                   reference[m]);
      return EXIT_FAILURE;
    }
  }
  std::printf("Warp SGEMV K32: PASS (M=%d, K=%d)\n", M, K);
  cudaFree(d_a);
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}
