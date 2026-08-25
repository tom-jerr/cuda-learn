#include "include/ffi_common.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cudnn.h>

using tvm::ffi::TensorView;

#define CUDA_LEARN_CHECK_CUBLAS(call)                                          \
  do {                                                                         \
    cublasStatus_t check_status = (call);                                      \
    if (check_status != CUBLAS_STATUS_SUCCESS) {                               \
      TVM_FFI_THROW(RuntimeError)                                              \
          << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": status "  \
          << static_cast<int>(check_status);                                   \
    }                                                                          \
  } while (0)

#define CUDA_LEARN_CHECK_CUDNN(call)                                           \
  do {                                                                         \
    cudnnStatus_t check_status = (call);                                       \
    if (check_status != CUDNN_STATUS_SUCCESS) {                                \
      TVM_FFI_THROW(RuntimeError)                                              \
          << "cuDNN error at " << __FILE__ << ":" << __LINE__ << ": "          \
          << cudnnGetErrorString(check_status);                                \
    }                                                                          \
  } while (0)

// benchmark 基线用 handle。首次创建发生在正确性检查阶段，不会进入计时窗口。
// 本学习项目的 benchmark 为单线程、单 GPU，因此一个进程级 handle 足够。
cublasHandle_t cublas_handle() {
  static cublasHandle_t handle = [] {
    cublasHandle_t value = nullptr;
    CUDA_LEARN_CHECK_CUBLAS(cublasCreate(&value));
    return value;
  }();
  return handle;
}

cudnnHandle_t cudnn_handle() {
  static cudnnHandle_t handle = [] {
    cudnnHandle_t value = nullptr;
    CUDA_LEARN_CHECK_CUDNN(cudnnCreate(&value));
    return value;
  }();
  return handle;
}

void cublas_gemm(TensorView a, TensorView b, TensorView c) {
  check_tensor(a, "a", 2);
  check_tensor(b, "b", 2);
  check_tensor(c, "c", 2);
  int m = static_cast<int>(dim(a, 0));
  int k = static_cast<int>(dim(a, 1));
  int n = static_cast<int>(dim(b, 1));
  if (dim(b, 0) != k || dim(c, 0) != m || dim(c, 1) != n) {
    TVM_FFI_THROW(RuntimeError)
        << "cublas_gemm: shapes must be a[M,K] @ b[K,N] -> c[M,N]";
  }

  cublasHandle_t handle = cublas_handle();
  CUDA_LEARN_CHECK_CUBLAS(cublasSetStream(handle, get_stream(a)));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  // cuBLAS 使用 column-major。对 row-major 输入交换 A/B，计算
  // C^T[N,M] = B^T[N,K] @ A^T[K,M]，内存布局即为所需 C[M,N]。
  CUDA_LEARN_CHECK_CUBLAS(
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                  static_cast<const float *>(b.data_ptr()), n,
                  static_cast<const float *>(a.data_ptr()), k, &beta,
                  static_cast<float *>(c.data_ptr()), n));
}

void cublas_gemm_bf16(TensorView a, TensorView b, TensorView c) {
  auto check_bf16 = [](const TensorView &tensor, const char *name) {
    if (tensor.ndim() != 2 || tensor.dtype().code != kDLBfloat ||
        tensor.dtype().bits != 16 || tensor.device().device_type != kDLCUDA) {
      TVM_FFI_THROW(RuntimeError)
          << name << ": expected a 2D CUDA bfloat16 tensor";
    }
  };
  check_bf16(a, "a");
  check_bf16(b, "b");
  check_bf16(c, "c");
  int m = static_cast<int>(dim(a, 0));
  int k = static_cast<int>(dim(a, 1));
  int n = static_cast<int>(dim(b, 1));
  if (dim(b, 0) != k || dim(c, 0) != m || dim(c, 1) != n) {
    TVM_FFI_THROW(RuntimeError)
        << "cublas_gemm_bf16: shapes must be a[M,K] @ b[K,N] -> c[M,N]";
  }

  cublasHandle_t handle = cublas_handle();
  CUDA_LEARN_CHECK_CUBLAS(cublasSetStream(handle, get_stream(a)));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  CUDA_LEARN_CHECK_CUBLAS(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b.data_ptr(),
      CUDA_R_16BF, n, a.data_ptr(), CUDA_R_16BF, k, &beta, c.data_ptr(),
      CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void cudnn_softmax(TensorView input, TensorView output) {
  check_tensor(input, "input", 1);
  check_tensor(output, "output", 1);
  int n = static_cast<int>(dim(input, 0));
  if (dim(output, 0) != n) {
    TVM_FFI_THROW(RuntimeError) << "cudnn_softmax: length mismatch";
  }

  cudnnHandle_t handle = cudnn_handle();
  CUDA_LEARN_CHECK_CUDNN(cudnnSetStream(handle, get_stream(input)));
  cudnnTensorDescriptor_t desc = nullptr;
  CUDA_LEARN_CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
  CUDA_LEARN_CHECK_CUDNN(cudnnSetTensor4dDescriptor(
      desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, n));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  cudnnStatus_t status = cudnnSoftmaxForward(
      handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
      input.data_ptr(), &beta, desc, output.data_ptr());
  // 即使 softmax 失败也先释放 descriptor，再向 Python 报错。
  cudnnStatus_t destroy_status = cudnnDestroyTensorDescriptor(desc);
  CUDA_LEARN_CHECK_CUDNN(status);
  CUDA_LEARN_CHECK_CUDNN(destroy_status);
}

CUDA_LEARN_REGISTER("cuda_learn.cublas_gemm", cublas_gemm);
CUDA_LEARN_REGISTER("cuda_learn.cublas_gemm_bf16", cublas_gemm_bf16);
CUDA_LEARN_REGISTER("cuda_learn.cudnn_softmax", cudnn_softmax);
