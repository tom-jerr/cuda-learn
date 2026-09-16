#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 256;
// inclusive scan
__device__ __forceinline__ float warp_prefix(float val) {
  int lane = threadIdx.x % kWarpSize;
  for (int offset = 1; offset < 32; offset <<= 1) {
    float other = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset)
      val += other;
  }
  return val;
}
// exclusive scan
__global__ void block_prefix(const float *__restrict__ x, float *__restrict__ y,
                             int n) {
  constexpr int kNumWarps = kBlockThreads / kWarpSize;
  __shared__ float warp_sums[kNumWarps];
  int tid = threadIdx.x;
  int lane = tid % kWarpSize;
  int warp = tid / kWarpSize;

  float val = (tid < n) ? x[tid] : 0.0f;
  float orig = val;
  val = warp_prefix(val);
  if (lane == kWarpSize - 1)
    warp_sums[warp] = val;
  __syncthreads();

  if (warp == 0) {
    float sum = (lane < kNumWarps) ? warp_sums[lane] : 0.0f;
    sum = warp_prefix(sum);
    if (lane < kNumWarps)
      warp_sums[lane] = sum;
  }
  __syncthreads();

  float warp_prefix_val = (warp == 0) ? 0.0f : warp_sums[warp - 1];
  float result = warp_prefix_val + val - orig;
  if (tid < n)
    y[tid] = result;
}