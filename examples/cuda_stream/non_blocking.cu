#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

constexpr int kBlockSize = 256;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                              \
      std::cerr << __FILE__ << ':' << __LINE__ << " CUDA error: "           \
                << cudaGetErrorString(error) << '\n';                        \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void square(const float *input, float *squared, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    squared[index] = input[index] * input[index];
  }
}

__global__ void produce_bias(float *bias, float value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *bias = value;
  }
}

__global__ void add_bias(const float *squared, const float *bias, float *output,
                         int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = squared[index] + *bias;
  }
}

int main() {
  constexpr int kElementCount = 1 << 20;
  constexpr size_t kBytes = kElementCount * sizeof(float);
  constexpr float kBias = 3.0F;
  constexpr int kNumBlocks =
      (kElementCount + kBlockSize - 1) / kBlockSize;

  float *host_input = nullptr;
  float *host_output = nullptr;
  CUDA_CHECK(cudaMallocHost(&host_input, kBytes));
  CUDA_CHECK(cudaMallocHost(&host_output, kBytes));

  for (int i = 0; i < kElementCount; ++i) {
    host_input[i] = static_cast<float>(i % 1000) * 0.25F;
  }

  float *device_input = nullptr;
  float *device_squared = nullptr;
  float *device_bias = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, kBytes));
  CUDA_CHECK(cudaMalloc(&device_squared, kBytes));
  CUDA_CHECK(cudaMalloc(&device_bias, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, kBytes));

  cudaStream_t producer_stream;
  cudaStream_t consumer_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&producer_stream,
                                       cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&consumer_stream,
                                       cudaStreamNonBlocking));

  cudaEvent_t squared_ready;
  cudaEvent_t bias_ready;
  cudaEvent_t output_ready;
  CUDA_CHECK(cudaEventCreateWithFlags(&squared_ready, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&bias_ready, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&output_ready, cudaEventDisableTiming));

  // producer_stream is non-blocking, so this work has no implicit dependency
  // on work submitted to the legacy default stream below.
  CUDA_CHECK(cudaMemcpyAsync(device_input, host_input, kBytes,
                             cudaMemcpyHostToDevice, producer_stream));
  square<<<kNumBlocks, kBlockSize, 0, producer_stream>>>(
      device_input, device_squared, kElementCount);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(squared_ready, producer_stream));

  // No stream argument means the legacy default stream. Because the two named
  // streams are non-blocking, this launch does not implicitly wait for them,
  // and they do not implicitly wait for this launch.
  produce_bias<<<1, 1>>>(device_bias, kBias);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(bias_ready, 0));

  // add_bias consumes results from both producer_stream and the default stream.
  // Non-blocking streams provide no implicit ordering, so express both edges.
  CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, squared_ready, 0));
  CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, bias_ready, 0));

  add_bias<<<kNumBlocks, kBlockSize, 0, consumer_stream>>>(
      device_squared, device_bias, device_output, kElementCount);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(host_output, device_output, kBytes,
                             cudaMemcpyDeviceToHost, consumer_stream));
  CUDA_CHECK(cudaEventRecord(output_ready, consumer_stream));

  // This is the only host-side wait. All cross-stream synchronization above is
  // performed on the GPU through events.
  CUDA_CHECK(cudaEventSynchronize(output_ready));

  for (int i = 0; i < kElementCount; ++i) {
    const float expected = host_input[i] * host_input[i] + kBias;
    if (std::fabs(host_output[i] - expected) > 1e-5F) {
      std::cerr << "Mismatch at index " << i << ": GPU=" << host_output[i]
                << ", CPU=" << expected << '\n';
      return EXIT_FAILURE;
    }
  }

  std::cout << "Non-blocking stream example PASSED\n"
            << "  producer_stream: H2D(input) -> square -> squared_ready\n"
            << "  default stream: produce_bias -> bias_ready\n"
            << "  consumer_stream: wait for both -> add_bias -> D2H(output)\n";

  CUDA_CHECK(cudaEventDestroy(output_ready));
  CUDA_CHECK(cudaEventDestroy(bias_ready));
  CUDA_CHECK(cudaEventDestroy(squared_ready));
  CUDA_CHECK(cudaStreamDestroy(consumer_stream));
  CUDA_CHECK(cudaStreamDestroy(producer_stream));

  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_bias));
  CUDA_CHECK(cudaFree(device_squared));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFreeHost(host_output));
  CUDA_CHECK(cudaFreeHost(host_input));

  return EXIT_SUCCESS;
}
