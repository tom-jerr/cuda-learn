#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                 \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void kernel_a() { printf("kernel A: origin stream\n"); }
__global__ void kernel_b() { printf("kernel B: origin stream\n"); }
__global__ void kernel_c() { printf("kernel C: forked stream\n"); }
__global__ void kernel_d() { printf("kernel D: joined origin stream\n"); }

int main() {
  cudaStream_t stream1, stream2;
  cudaEvent_t fork_event, join_event;
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;

  CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
  CHECK_CUDA(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
  CHECK_CUDA(cudaEventCreateWithFlags(&join_event, cudaEventDisableTiming));

  // stream1 是源流，捕获必须从它开始，也必须由它结束。
  CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

  kernel_a<<<1, 1, 0, stream1>>>();

  // Fork：stream2 等待 stream1 中记录的事件，从而加入同一次捕获。
  CHECK_CUDA(cudaEventRecord(fork_event, stream1));
  CHECK_CUDA(cudaStreamWaitEvent(stream2, fork_event));

  kernel_b<<<1, 1, 0, stream1>>>();
  kernel_c<<<1, 1, 0, stream2>>>();

  // Join：让源流等待 stream2，重新连接后才能结束捕获。
  CHECK_CUDA(cudaEventRecord(join_event, stream2));
  CHECK_CUDA(cudaStreamWaitEvent(stream1, join_event));

  kernel_d<<<1, 1, 0, stream1>>>();

  CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));
  CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, 0));
  CHECK_CUDA(cudaGraphLaunch(graph_exec, stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream1));

  CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
  CHECK_CUDA(cudaGraphDestroy(graph));
  CHECK_CUDA(cudaEventDestroy(join_event));
  CHECK_CUDA(cudaEventDestroy(fork_event));
  CHECK_CUDA(cudaStreamDestroy(stream2));
  CHECK_CUDA(cudaStreamDestroy(stream1));
  return 0;
}
