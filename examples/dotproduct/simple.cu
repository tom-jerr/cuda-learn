#include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}
// block 内求和
// 假设 blockDim.x <= 1024
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float warp_sums[32];

  int tid = threadIdx.x;
  int lane = tid & 31;
  int wid = tid >> 5;

  // 第一层：每个 warp 内 reduce
  val = warp_reduce_sum(val);

  // 每个 warp 的 lane0 写入 shared memory
  if (lane == 0) {
    warp_sums[wid] = val;
  }

  __syncthreads();

  // warp 数量，支持 blockDim.x 非 32 整数倍
  int num_warps = (blockDim.x + 31) / 32;

  // 第二层：warp0 reduce 所有 warp 的结果
  float block_sum = 0.0f;

  if (wid == 0) {
    block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
  }

  // 最终只有 warp0 lane0 的 block_sum 有效
  return block_sum;
}
__global__ void dot_reduce(const float *__restrict__ a,
                           const float *__restrict__ b, float *__restrict__ out,
                           int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  float sum = 0.0f;

  if (idx < n)
    sum = a[idx] * b[idx];

  sum = block_reduce_sum(sum);

  if (tid == 0)
    out[blockIdx.x] = sum;
}
__global__ void reduce(const float *__restrict__ in, float *__restrict__ out,
                       int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  float sum = (idx < n) ? in[idx] : 0.0f;

  sum = block_reduce_sum(sum);

  if (tid == 0)
    out[blockIdx.x] = sum;
}
void launch_dot_product(const float *d_a, const float *d_b, float *d_result,
                        int n) {
  constexpr int BLOCK = 256;

  int num_blocks = (n + BLOCK - 1) / BLOCK;

  float *buf1;
  float *buf2;

  cudaMalloc(&buf1, num_blocks * sizeof(float));
  cudaMalloc(&buf2, num_blocks * sizeof(float));

  // 第一轮：
  // n -> ceil(n / BLOCK)
  dot_reduce<<<num_blocks, BLOCK>>>(d_a, d_b, buf1, n);

  int cur_n = num_blocks;

  float *in = buf1;
  float *out = buf2;

  // 后续不断：
  // cur_n -> ceil(cur_n / BLOCK)
  while (cur_n > 1) {
    int blocks = (cur_n + BLOCK - 1) / BLOCK;

    reduce<<<blocks, BLOCK>>>(in, out, cur_n);

    cur_n = blocks;

    float *tmp = in;
    in = out;
    out = tmp;
  }

  // 最后 in[0] 就是结果
  cudaMemcpy(d_result, in, sizeof(float), cudaMemcpyDeviceToDevice);

  cudaFree(buf1);
  cudaFree(buf2);
}