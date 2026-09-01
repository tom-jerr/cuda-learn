#include <cuda_runtime.h>

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

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

// Two streams with a host-side dependency:
//
// stream_a: H2D(A) ------------------------------> kernel(A, B) -> D2H(C)
// stream_b: H2D(B) -> cudaStreamSynchronize() ----^
//                       CPU waits here
//
// The kernel runs in stream_a. Its dependency on A is automatic because A was
// copied in the same stream. The B copy is in another stream, so the host must
// wait for stream_b before it is allowed to enqueue the kernel.
void launch_vector_add_stream_sync(const float *host_a, const float *host_b,
                                   float *host_c, float *device_a,
                                   float *device_b, float *device_c, int n,
                                   cudaStream_t stream_a,
                                   cudaStream_t stream_b) {
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  const int num_blocks = (n + kBlockSize - 1) / kBlockSize;

  CUDA_CHECK(cudaMemcpyAsync(device_a, host_a, bytes, cudaMemcpyHostToDevice,
                             stream_a));
  CUDA_CHECK(cudaMemcpyAsync(device_b, host_b, bytes, cudaMemcpyHostToDevice,
                             stream_b));

  // Host-side cross-stream synchronization. Without this call, stream_a has no
  // knowledge that the kernel must wait for the B copy in stream_b.
  CUDA_CHECK(cudaStreamSynchronize(stream_b));

  vector_add<<<num_blocks, kBlockSize, 0, stream_a>>>(device_a, device_b,
                                                       device_c, n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(host_c, device_c, bytes, cudaMemcpyDeviceToHost,
                             stream_a));

  // The function promises that host_c is ready when it returns.
  CUDA_CHECK(cudaStreamSynchronize(stream_a));
}

// Two streams with a device-side event dependency:
//
// stream_a: H2D(A) -> wait(B-ready) -> kernel(A, B) -> D2H(C) -> result-ready
//                         ^
// stream_b: H2D(B) -> record(B-ready)
//
// cudaStreamWaitEvent does not block the CPU. It inserts a dependency into
// stream_a, allowing the CPU to enqueue the rest of the pipeline immediately.
void launch_vector_add_event_sync(const float *host_a, const float *host_b,
                                  float *host_c, float *device_a,
                                  float *device_b, float *device_c, int n,
                                  cudaStream_t stream_a,
                                  cudaStream_t stream_b) {
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  const int num_blocks = (n + kBlockSize - 1) / kBlockSize;

  cudaEvent_t b_ready;
  cudaEvent_t result_ready;
  CUDA_CHECK(cudaEventCreateWithFlags(&b_ready, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&result_ready, cudaEventDisableTiming));

  CUDA_CHECK(cudaMemcpyAsync(device_a, host_a, bytes, cudaMemcpyHostToDevice,
                             stream_a));

  CUDA_CHECK(cudaMemcpyAsync(device_b, host_b, bytes, cudaMemcpyHostToDevice,
                             stream_b));
  CUDA_CHECK(cudaEventRecord(b_ready, stream_b));

  // This is the real cross-stream dependency: later work in stream_a cannot
  // start until stream_b reaches b_ready.
  CUDA_CHECK(cudaStreamWaitEvent(stream_a, b_ready, 0));

  // H2D(A) precedes the wait in stream_a, and the wait guarantees H2D(B), so
  // both inputs are ready when the kernel starts.
  vector_add<<<num_blocks, kBlockSize, 0, stream_a>>>(device_a, device_b,
                                                       device_c, n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(host_c, device_c, bytes, cudaMemcpyDeviceToHost,
                             stream_a));
  CUDA_CHECK(cudaEventRecord(result_ready, stream_a));

  // cudaStreamWaitEvent would not wait for the CPU. Because main reads host_c
  // immediately after this function returns, use cudaEventSynchronize here.
  CUDA_CHECK(cudaEventSynchronize(result_ready));

  CUDA_CHECK(cudaEventDestroy(result_ready));
  CUDA_CHECK(cudaEventDestroy(b_ready));
}

void verify(const float *a, const float *b, const float *c, int n,
            const char *version) {
  for (int i = 0; i < n; ++i) {
    if (c[i] != a[i] + b[i]) {
      std::cerr << version << " failed at index " << i << ": " << c[i]
                << " != " << a[i] + b[i] << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  std::cout << version << " PASSED\n";
}

int main() {
  constexpr int kElementCount = 1 << 20;
  constexpr size_t kBytes = kElementCount * sizeof(float);

  // Pinned host memory is needed for reliable asynchronous host/device copies.
  float *host_a = nullptr;
  float *host_b = nullptr;
  float *host_c = nullptr;
  CUDA_CHECK(cudaMallocHost(&host_a, kBytes));
  CUDA_CHECK(cudaMallocHost(&host_b, kBytes));
  CUDA_CHECK(cudaMallocHost(&host_c, kBytes));

  for (int i = 0; i < kElementCount; ++i) {
    host_a[i] = static_cast<float>(i);
    host_b[i] = static_cast<float>(2 * i);
  }

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  CUDA_CHECK(cudaMalloc(&device_a, kBytes));
  CUDA_CHECK(cudaMalloc(&device_b, kBytes));
  CUDA_CHECK(cudaMalloc(&device_c, kBytes));

  cudaStream_t stream_a;
  cudaStream_t stream_b;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking));

  launch_vector_add_stream_sync(host_a, host_b, host_c, device_a, device_b,
                                device_c, kElementCount, stream_a, stream_b);
  verify(host_a, host_b, host_c, kElementCount, "stream synchronization");

  launch_vector_add_event_sync(host_a, host_b, host_c, device_a, device_b,
                               device_c, kElementCount, stream_a, stream_b);
  verify(host_a, host_b, host_c, kElementCount, "event synchronization");

  CUDA_CHECK(cudaStreamDestroy(stream_b));
  CUDA_CHECK(cudaStreamDestroy(stream_a));

  CUDA_CHECK(cudaFree(device_c));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_a));

  CUDA_CHECK(cudaFreeHost(host_c));
  CUDA_CHECK(cudaFreeHost(host_b));
  CUDA_CHECK(cudaFreeHost(host_a));

  return EXIT_SUCCESS;
}
