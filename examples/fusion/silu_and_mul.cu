#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                 \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

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

void silu_and_mul(const float* gate, const float* up, float* output, int size,
                  cudaStream_t stream = 0) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  silu_and_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, output, size);
}

void rmsnorm_and_add(const float* x, float* residual, const float* weight,
                     float* output, int rows, int hidden_size, float epsilon,
                     cudaStream_t stream = 0) {
  int threads = 256;
  rmsnorm_and_add_kernel<<<rows, threads, threads * sizeof(float), stream>>>(
      x, residual, weight, output, hidden_size, epsilon);
}

int main() {
  constexpr int rows = 4;
  constexpr int hidden_size = 1024;
  constexpr int size = rows * hidden_size;
  constexpr float epsilon = 1e-5f;

  std::vector<float> gate(size), up(size), x(size), residual(size);
  std::vector<float> weight(hidden_size), output(size), residual_output(size);
  std::vector<float> reference(size), residual_reference(size);

  for (int i = 0; i < size; ++i) {
    gate[i] = (i % 17 - 8) * 0.1f;
    up[i] = (i % 13 - 6) * 0.2f;
    x[i] = (i % 11 - 5) * 0.1f;
    residual[i] = (i % 7 - 3) * 0.2f;
  }
  for (int i = 0; i < hidden_size; ++i) weight[i] = 1.0f + (i % 5) * 0.01f;

  float *d_gate, *d_up, *d_x, *d_residual, *d_weight, *d_output;
  CHECK_CUDA(cudaMalloc(&d_gate, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_up, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_x, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_residual, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_weight, hidden_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, size * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_gate, gate.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_up, up.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  silu_and_mul(d_gate, d_up, d_output, size);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; ++i)
    reference[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
  float max_error = 0.0f;
  for (int i = 0; i < size; ++i)
    max_error = std::max(max_error, std::abs(output[i] - reference[i]));
  std::printf("silu_and_mul max error: %.8f\n", max_error);

  CHECK_CUDA(cudaMemcpy(d_x, x.data(), size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_residual, residual.data(), size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_weight, weight.data(), hidden_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  rmsnorm_and_add(d_x, d_residual, d_weight, d_output, rows, hidden_size, epsilon);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(residual_output.data(), d_residual, size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int row = 0; row < rows; ++row) {
    float square_sum = 0.0f;
    for (int column = 0; column < hidden_size; ++column) {
      int index = row * hidden_size + column;
      residual_reference[index] = x[index] + residual[index];
      square_sum += residual_reference[index] * residual_reference[index];
    }
    float inverse_rms = 1.0f / std::sqrt(square_sum / hidden_size + epsilon);
    for (int column = 0; column < hidden_size; ++column) {
      int index = row * hidden_size + column;
      reference[index] = residual_reference[index] * inverse_rms * weight[column];
    }
  }

  max_error = 0.0f;
  float residual_error = 0.0f;
  for (int i = 0; i < size; ++i) {
    max_error = std::max(max_error, std::abs(output[i] - reference[i]));
    residual_error = std::max(
        residual_error, std::abs(residual_output[i] - residual_reference[i]));
  }
  std::printf("rmsnorm_and_add max error: %.8f, residual error: %.8f\n",
              max_error, residual_error);

  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaFree(d_weight));
  CHECK_CUDA(cudaFree(d_residual));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_up));
  CHECK_CUDA(cudaFree(d_gate));
  return 0;
}
