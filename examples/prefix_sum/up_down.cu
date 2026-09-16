#include <cuda_runtime.h>

// 单 block 的 Blelloch exclusive scan；固定 256 线程，0 <= n <= 256。
// 输入 [1, 2, 3, 4] -> 输出 [0, 1, 3, 6]。
__global__ void prefix_sum_up_down(const int* x, int* y, int n) {
  __shared__ int s[256];
  int tid = threadIdx.x;
  s[tid] = tid < n ? x[tid] : 0;  // 补零到 2 的幂
  __syncthreads();

  // Up-sweep：左右子树的和合并到右端点，最后 s[255] 是总和。
  for (int stride = 1; stride < 256; stride *= 2) {
    int r = (tid + 1) * 2 * stride - 1;
    if (r < 256) s[r] += s[r - stride];
    __syncthreads();
  }

  // 根的前缀为 0：整棵树之前没有元素。
  if (tid == 0) s[255] = 0;
  __syncthreads();

  // Down-sweep：左孩子拿父前缀，右孩子拿父前缀 + 左子树总和。
  for (int stride = 128; stride > 0; stride /= 2) {
    int r = (tid + 1) * 2 * stride - 1;
    if (r < 256) {
      int left_sum = s[r - stride];
      s[r - stride] = s[r];
      s[r] += left_sum;
    }
    __syncthreads();
  }

  if (tid < n) y[tid] = s[tid];
}

// 调用：prefix_sum_up_down<<<1, 256>>>(x, y, n);
