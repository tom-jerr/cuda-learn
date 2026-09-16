#include <cuda_runtime.h>

constexpr int kBlockThreads = 256;

__device__ float warp_reduce_sum(float x) {
  for (int offset = 16; offset > 0; offset >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, offset);
  return x;
}

// 固定 256 线程：warp 内归约 -> warp 间归约 -> 广播给整个 block。
__device__ float block_reduce_sum(float x) {
  __shared__ float sums[kBlockThreads / 32];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;

  x = warp_reduce_sum(x);
  if (lane == 0)
    sums[warp] = x;
  __syncthreads();

  if (warp == 0) {
    x = lane < kBlockThreads / 32 ? sums[lane] : 0.0f;
    x = warp_reduce_sum(x);
    if (lane == 0)
      sums[0] = x;
  }
  __syncthreads();
  return sums[0];
}

// x/y: [rows, cols]，weight: [cols]，cols > 0。
// 公式：y[row, i] = x[row, i] * weight[i]
//                  / sqrt((1 / cols) * sum_j(x[row, j]^2) + eps)。
__global__ void rmsnorm_kernel(const float *x, const float *weight, float *y,
                               int cols, float eps) {
  const int bid = blockIdx.x;

  float sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    sum += x[bid * cols + i] * x[bid * cols + i];

  // block_reduce_sum 得到整行平方和，/ cols 将它变成平方均值。
  // RMS = sqrt(mean(x^2))（均方根）；不除 cols 就成了 L2 范数。
  // rsqrtf(z) = 1 / sqrt(z)，eps 避免全零输入时除以零。
  float inv_rms = rsqrtf(block_reduce_sum(sum) / cols + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    y[bid * cols + i] = x[bid * cols + i] * inv_rms * weight[i];
}

void rmsnorm(const float *x, const float *weight, float *y, int rows, int cols,
             float eps = 1e-6f) {
  rmsnorm_kernel<<<rows, kBlockThreads>>>(x, weight, y, cols, eps);
}
