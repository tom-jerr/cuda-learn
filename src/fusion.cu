#include "ffi_common.h"

using tvm::ffi::TensorView;

__global__ void silu_and_mul_kernel(const float* gate, const float* up,
                                    float* output, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    float value = gate[index];
    output[index] = value / (1.0f + expf(-value)) * up[index];
  }
}

// residual = x + residual
// output   = RMSNorm(residual) * weight
__global__ void rmsnorm_and_add_kernel(const float* x, float* residual,
                                       const float* weight, float* output,
                                       int hidden_size, float epsilon) {
  extern __shared__ float sum[];

  int row = blockIdx.x;
  int tid = threadIdx.x;
  float square_sum = 0.0f;

  for (int column = tid; column < hidden_size; column += blockDim.x) {
    int index = row * hidden_size + column;
    float value = x[index] + residual[index];
    residual[index] = value;
    square_sum += value * value;
  }

  sum[tid] = square_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) sum[tid] += sum[tid + offset];
    __syncthreads();
  }

  float inverse_rms = rsqrtf(sum[0] / hidden_size + epsilon);
  for (int column = tid; column < hidden_size; column += blockDim.x) {
    int index = row * hidden_size + column;
    output[index] = residual[index] * inverse_rms * weight[column];
  }
}

void silu_and_mul(TensorView gate, TensorView up, TensorView output) {
  check_tensor(gate, "gate", 1);
  check_tensor(up, "up", 1);
  check_tensor(output, "output", 1);
  int size = static_cast<int>(dim(gate, 0));
  if (dim(up, 0) != size || dim(output, 0) != size) {
    TVM_FFI_THROW(RuntimeError) << "silu_and_mul: length mismatch";
  }
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  silu_and_mul_kernel<<<blocks, threads, 0, get_stream(gate)>>>(
      static_cast<const float*>(gate.data_ptr()),
      static_cast<const float*>(up.data_ptr()),
      static_cast<float*>(output.data_ptr()), size);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

void rmsnorm_and_add(TensorView x, TensorView residual, TensorView weight,
                     TensorView output, double epsilon) {
  check_tensor(x, "x", 2);
  check_tensor(residual, "residual", 2);
  check_tensor(weight, "weight", 1);
  check_tensor(output, "output", 2);
  int rows = static_cast<int>(dim(x, 0));
  int hidden_size = static_cast<int>(dim(x, 1));
  if (dim(residual, 0) != rows || dim(residual, 1) != hidden_size ||
      dim(output, 0) != rows || dim(output, 1) != hidden_size ||
      dim(weight, 0) != hidden_size) {
    TVM_FFI_THROW(RuntimeError) << "rmsnorm_and_add: shape mismatch";
  }
  int threads = 256;
  rmsnorm_and_add_kernel<<<rows, threads, threads * sizeof(float),
                           get_stream(x)>>>(
      static_cast<const float*>(x.data_ptr()),
      static_cast<float*>(residual.data_ptr()),
      static_cast<const float*>(weight.data_ptr()),
      static_cast<float*>(output.data_ptr()), hidden_size,
      static_cast<float>(epsilon));
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.silu_and_mul", silu_and_mul);
CUDA_LEARN_REGISTER("cuda_learn.rmsnorm_and_add", rmsnorm_and_add);
