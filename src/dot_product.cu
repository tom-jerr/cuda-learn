#include "ffi_common.h"

using tvm::ffi::TensorView;

__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void dot_product_kernel(const float *a, const float *b, float *result,
                                   int n) {
  __shared__ float shared[32]; // assuming blockDim.x <= 1024
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  int lane = tid % 32;
  int warp_id = tid / 32;

  float sum = 0.0f;

  // grid-stride loop to handle large arrays
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    sum += a[i] * b[i];
  }

  sum = warp_reduce_sum(sum);

  if (lane == 0) {
    shared[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    sum = shared[lane];
    sum = warp_reduce_sum(sum);
    if (tid == 0) {
      atomicAdd(result, sum);
    }
  }
}

void dot_product(TensorView a, TensorView b, TensorView result) {
  check_tensor(a, "a", 1);
  check_tensor(b, "b", 1);
  check_tensor(result, "result", 1);
  int n = static_cast<int>(dim(a, 0));
  if (dim(b, 0) != n || dim(result, 0) != 1) {
    TVM_FFI_THROW(RuntimeError) << "dot_product: length mismatch";
  }
  cudaStream_t stream = get_stream(a);
  // 累加结果先清零（与原 main 中 *result = 0.0f 对应，改为 stream 上的异步 memset）
  CUDA_LEARN_CHECK(cudaMemsetAsync(result.data_ptr(), 0, sizeof(float), stream));

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  dot_product_kernel<<<grid_size, block_size, 0, stream>>>(
      static_cast<const float *>(a.data_ptr()),
      static_cast<const float *>(b.data_ptr()),
      static_cast<float *>(result.data_ptr()), n);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.dot_product", dot_product);
