// Minimal LLM-style PDL example: RMSNorm -> Linear for one decode token.
// Build (Hopper or newer):
//   nvcc -O3 -std=c++17 -arch=sm_90 pdl_rmsnorm_linear.cu -o pdl_demo
//
// The example is intentionally simple, not a competitive GEMM.  Its purpose is
// to show where the dependency wait belongs in a real inference kernel pair.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(error__));                                \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

constexpr int kHidden = 256;
constexpr int kOutput = 128;
constexpr int kThreads = 256;
constexpr float kEpsilon = 1.0e-5f;

template <bool UsePdl>
__global__ void rmsnorm_kernel(const float *__restrict__ x,
                               float *__restrict__ normalized) {
  __shared__ float scratch[kThreads];
  __shared__ float inv_rms;

  const int lane = threadIdx.x;
  const float value = x[lane];
  scratch[lane] = value * value;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) scratch[lane] += scratch[lane + stride];
    __syncthreads();
  }
  if (lane == 0) inv_rms = rsqrtf(scratch[0] / kHidden + kEpsilon);
  __syncthreads();

  if constexpr (UsePdl) {
    // One call per CTA is sufficient.  This says the dependent grid may start;
    // it is deliberately before the stores to normalized[].
    if (lane == 0) cudaTriggerProgrammaticLaunchCompletion();
  }

  // Tail of the primary kernel: this can overlap the secondary's weight load.
  normalized[lane] = value * inv_rms;
}

template <bool UsePdl>
__global__ void linear_kernel(const float *__restrict__ normalized,
                              const float *__restrict__ weight,
                              float *__restrict__ output) {
  __shared__ float weight_tile[kHidden];
  __shared__ float partial[kThreads];

  const int lane = threadIdx.x;
  const int out = blockIdx.x;

  // Independent prologue: weights are persistent model state, not an output of
  // RMSNorm, so they may be fetched before the dependency is resolved.
  weight_tile[lane] = weight[out * kHidden + lane];
  __syncthreads();

  if constexpr (UsePdl) {
    // After this point all writes by the preceding RMSNorm grid are visible.
    cudaGridDependencySynchronize();
  }

  partial[lane] = weight_tile[lane] * normalized[lane];
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) partial[lane] += partial[lane + stride];
    __syncthreads();
  }
  if (lane == 0) output[out] = partial[0];
}

void launch_pipeline(bool use_pdl, const float *x, float *normalized,
                     const float *weight, float *output, cudaStream_t stream) {
  if (!use_pdl) {
    rmsnorm_kernel<false><<<1, kThreads, 0, stream>>>(x, normalized);
    linear_kernel<false><<<kOutput, kThreads, 0, stream>>>(normalized, weight,
                                                           output);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  rmsnorm_kernel<true><<<1, kThreads, 0, stream>>>(x, normalized);
  CUDA_CHECK(cudaGetLastError());

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kOutput);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&config, linear_kernel<true>, normalized, weight,
                                output));
}

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 9) {
    std::printf("%s has compute capability %d.%d; PDL requires 9.0+.\n",
                properties.name, properties.major, properties.minor);
    return 0;
  }

  std::vector<float> h_x(kHidden);
  std::vector<float> h_weight(kOutput * kHidden);
  for (int i = 0; i < kHidden; ++i) h_x[i] = std::sin(0.1f * i);
  for (int i = 0; i < kOutput * kHidden; ++i)
    h_weight[i] = std::cos(0.01f * i) / kHidden;

  float *d_x = nullptr;
  float *d_normalized = nullptr;
  float *d_weight = nullptr;
  float *d_baseline = nullptr;
  float *d_pdl = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, kHidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normalized, kHidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, h_weight.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_baseline, kOutput * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_pdl, kOutput * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), kHidden * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(),
                        h_weight.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  launch_pipeline(false, d_x, d_normalized, d_weight, d_baseline, stream);
  launch_pipeline(true, d_x, d_normalized, d_weight, d_pdl, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> h_baseline(kOutput), h_pdl(kOutput);
  CUDA_CHECK(cudaMemcpy(h_baseline.data(), d_baseline,
                        kOutput * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_pdl.data(), d_pdl, kOutput * sizeof(float),
                        cudaMemcpyDeviceToHost));
  float max_error = 0.0f;
  for (int i = 0; i < kOutput; ++i)
    max_error = std::max(max_error, std::abs(h_baseline[i] - h_pdl[i]));
  std::printf("baseline vs PDL max error: %.3e\n", max_error);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_pdl));
  CUDA_CHECK(cudaFree(d_baseline));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_normalized));
  CUDA_CHECK(cudaFree(d_x));
  return max_error < 1.0e-5f ? 0 : 1;
}
