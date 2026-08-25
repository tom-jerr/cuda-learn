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

__global__ void kernel_a() { printf("kernel A\n"); }
__global__ void kernel_b() { printf("kernel B\n"); }
__global__ void kernel_c() { printf("kernel C\n"); }
__global__ void kernel_d() { printf("kernel D\n"); }

int main() {
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  cudaGraphNode_t a, b, c, d;

  CHECK_CUDA(cudaGraphCreate(&graph, 0));

  cudaKernelNodeParams params{};
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);

  params.func = reinterpret_cast<void*>(kernel_a);
  CHECK_CUDA(cudaGraphAddKernelNode(&a, graph, nullptr, 0, &params));

  params.func = reinterpret_cast<void*>(kernel_b);
  CHECK_CUDA(cudaGraphAddKernelNode(&b, graph, &a, 1, &params));

  params.func = reinterpret_cast<void*>(kernel_c);
  CHECK_CUDA(cudaGraphAddKernelNode(&c, graph, &a, 1, &params));

  cudaGraphNode_t d_dependencies[] = {b, c};
  params.func = reinterpret_cast<void*>(kernel_d);
  CHECK_CUDA(cudaGraphAddKernelNode(&d, graph, d_dependencies, 2, &params));

  CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, 0));
  CHECK_CUDA(cudaGraphLaunch(graph_exec, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
  CHECK_CUDA(cudaGraphDestroy(graph));
  return 0;
}
