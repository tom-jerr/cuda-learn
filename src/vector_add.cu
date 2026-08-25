#include "ffi_common.h"

// per thread calc one element
__global__ void vector_add_kernel(const float *a, const float *b, float *out,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

// grid-loop
__global__ void vector_add_grid_loop_kernel(const float *a, const float *b,
                                            float *out, int n) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    out[idx] = a[idx] + b[idx];
  }
}

// vectorized version: each thread calc 4 elements
__global__ void vector_add_vectorized_kernel(const float *a, const float *b,
                                             float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vector_size = 4;
  int start_idx = idx * vector_size;
  if (start_idx + 3 < n) {
    float4 a_vec = reinterpret_cast<const float4 *>(a)[idx];
    float4 b_vec = reinterpret_cast<const float4 *>(b)[idx];
    float4 out_vec;
    out_vec.x = a_vec.x + b_vec.x;
    out_vec.y = a_vec.y + b_vec.y;
    out_vec.z = a_vec.z + b_vec.z;
    out_vec.w = a_vec.w + b_vec.w;
    reinterpret_cast<float4 *>(out)[idx] = out_vec;
  } else {
    for (int i = start_idx; i < n; ++i) {
      out[i] = a[i] + b[i];
    }
  }
}

using tvm::ffi::TensorView;

namespace {
// 公共 launch 配置与参数校验。
void vector_add_common(const TensorView& a, const TensorView& b,
                       const TensorView& out, int vector_size) {
  check_tensor(a, "a", 1);
  check_tensor(b, "b", 1);
  check_tensor(out, "out", 1);
  int n = static_cast<int>(dim(a, 0));
  if (dim(b, 0) != n || dim(out, 0) != n) {
    TVM_FFI_THROW(RuntimeError) << "vector_add: length mismatch";
  }
  cudaStream_t stream = get_stream(a);
  int block_size = 256;
  int grid_size = (n + block_size * vector_size - 1) / (block_size * vector_size);
  vector_add_kernel<<<grid_size, block_size, 0, stream>>>(
      static_cast<float *>(a.data_ptr()), static_cast<float *>(b.data_ptr()),
      static_cast<float *>(out.data_ptr()), n);
  CUDA_LEARN_CHECK(cudaGetLastError());
}
}  // namespace

void vector_add(TensorView a, TensorView b, TensorView out) {
  vector_add_common(a, b, out, /*vector_size=*/1);
}

void vector_add_grid_loop(TensorView a, TensorView b, TensorView out) {
  check_tensor(a, "a", 1);
  check_tensor(b, "b", 1);
  check_tensor(out, "out", 1);
  int n = static_cast<int>(dim(a, 0));
  if (dim(b, 0) != n || dim(out, 0) != n) {
    TVM_FFI_THROW(RuntimeError) << "vector_add_grid_loop: length mismatch";
  }
  int block_size = 256;
  // grid-stride 版本用固定小 grid，任意 n 都适用
  int grid_size = (n + block_size - 1) / block_size;
  vector_add_grid_loop_kernel<<<grid_size, block_size, 0, get_stream(a)>>>(
      static_cast<float *>(a.data_ptr()), static_cast<float *>(b.data_ptr()),
      static_cast<float *>(out.data_ptr()), n);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

void vector_add_vectorized(TensorView a, TensorView b, TensorView out) {
  vector_add_common(a, b, out, /*vector_size=*/4);
}

CUDA_LEARN_REGISTER("cuda_learn.vector_add", vector_add);
CUDA_LEARN_REGISTER("cuda_learn.vector_add_grid_loop", vector_add_grid_loop);
CUDA_LEARN_REGISTER("cuda_learn.vector_add_vectorized", vector_add_vectorized);
