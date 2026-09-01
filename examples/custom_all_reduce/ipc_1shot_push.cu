// Linux, two processes, two GPUs.
//
// This is the multi-process counterpart of simple.cu. Each process owns one
// rank and exchanges cudaIpcMemHandle_t over a Unix socket. Only the symmetric
// push workspace is exported; input, output, stream, and epoch counters remain
// rank-local, matching the ownership split in SGLang.

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr int kWorld = 2;
constexpr int kThreads = 256;

void check_cuda(cudaError_t status, const char *expression, const char *file,
                int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": " + expression + ": " +
                             cudaGetErrorString(status));
  }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

void write_all(int fd, const void *data, size_t bytes) {
  const char *cursor = static_cast<const char *>(data);
  while (bytes != 0) {
    const ssize_t count = ::write(fd, cursor, bytes);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("socket write failed");
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
}

void read_all(int fd, void *data, size_t bytes) {
  char *cursor = static_cast<char *>(data);
  while (bytes != 0) {
    const ssize_t count = ::read(fd, cursor, bytes);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("socket peer exited early");
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
}

template <typename T> T exchange(int socket, const T &local) {
  static_assert(std::is_trivially_copyable_v<T>);
  T peer{};
  // socketpair is full duplex. Both ranks can write first without a protocol
  // leader because these control messages are tiny.
  write_all(socket, &local, sizeof(local));
  read_all(socket, &peer, sizeof(peer));
  return peer;
}

void process_barrier(int socket) {
  const char local = 'B';
  (void)exchange(socket, local);
}

struct alignas(16) Vec16 {
  float x, y, z, w;
};
union VecBits {
  Vec16 vec;
  uint4 bits;
};
static_assert(sizeof(Vec16) == 16);

__device__ __forceinline__ void store_relaxed_sys(Vec16 *base, int index,
                                                  Vec16 value) {
  const VecBits raw{value};
  Vec16 *address = base + index;
  asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(raw.bits.x), "r"(raw.bits.y), "r"(raw.bits.z),
                 "r"(raw.bits.w), "l"(address)
               : "memory");
}

__device__ __forceinline__ Vec16 load_relaxed_sys(const Vec16 *base,
                                                  int index) {
  VecBits raw;
  const Vec16 *address = base + index;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(raw.bits.x), "=r"(raw.bits.y), "=r"(raw.bits.z),
                 "=r"(raw.bits.w)
               : "l"(address)
               : "memory");
  return raw.vec;
}

__device__ __forceinline__ void clear_slot(Vec16 *base, int index) {
  Vec16 *address = base + index;
  asm volatile("st.global.v4.b32 [%0], {0, 0, 0, 0};"
               :
               : "l"(address)
               : "memory");
}

__device__ __forceinline__ Vec16 encode(Vec16 value) {
  VecBits raw{value};
  if (raw.bits.x == 0)
    raw.bits.x = 0x80000000u;
  if (raw.bits.y == 0)
    raw.bits.y = 0x80000000u;
  if (raw.bits.z == 0)
    raw.bits.z = 0x80000000u;
  if (raw.bits.w == 0)
    raw.bits.w = 0x80000000u;
  return raw.vec;
}

__device__ __forceinline__ bool is_empty(const Vec16 &value) {
  const VecBits raw{value};
  return raw.bits.x == 0 || raw.bits.y == 0 || raw.bits.z == 0 ||
         raw.bits.w == 0;
}

struct PushParams {
  const Vec16 *input;
  Vec16 *output;
  Vec16 *workspace[kWorld];
  unsigned int *epochs;
  int vectors;
  int rank;
};

__global__ __launch_bounds__(kThreads) void ipc_push_kernel(PushParams params) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const unsigned int epoch = params.epochs[blockIdx.x] & 1u;

  Vec16 *destinations[kWorld];
#pragma unroll
  for (int dst = 0; dst < kWorld; ++dst) {
    destinations[dst] =
        params.workspace[dst] + (epoch * kWorld + params.rank) * params.vectors;
  }
  for (int vid = tid; vid < params.vectors; vid += stride) {
    const Vec16 value = encode(params.input[vid]);
#pragma unroll
    for (int dst = 0; dst < kWorld; ++dst) {
      store_relaxed_sys(destinations[dst], vid, value);
    }
  }

  Vec16 *sources[kWorld];
#pragma unroll
  for (int src = 0; src < kWorld; ++src) {
    sources[src] =
        params.workspace[params.rank] + (epoch * kWorld + src) * params.vectors;
  }
  for (int vid = tid; vid < params.vectors; vid += stride) {
    Vec16 values[kWorld];
    do {
#pragma unroll
      for (int src = 0; src < kWorld; ++src) {
        values[src] = load_relaxed_sys(sources[src], vid);
      }
    } while (is_empty(values[0]) || is_empty(values[1]));

    params.output[vid] = {values[0].x + values[1].x, values[0].y + values[1].y,
                          values[0].z + values[1].z, values[0].w + values[1].w};
#pragma unroll
    for (int src = 0; src < kWorld; ++src)
      clear_slot(sources[src], vid);
  }
  __syncthreads();
  if (threadIdx.x == 0)
    params.epochs[blockIdx.x] = epoch ^ 1u;
}

int parse_positive(const char *text, const char *name) {
  const long value = std::strtol(text, nullptr, 10);
  if (value <= 0 || value > (1L << 30)) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
  return static_cast<int>(value);
}

int run_rank(int rank, int socket, int elements, int iterations) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < kWorld) {
    if (rank == 0) {
      std::cout << "SKIP: ipc_1shot_push needs two GPUs; found " << device_count
                << ".\n";
    }
    return 0;
  }
  if (elements % 4 != 0) {
    throw std::runtime_error("element count must be a multiple of four");
  }

  const int peer_rank = rank ^ 1;
  CUDA_CHECK(cudaSetDevice(rank));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, rank));
  if (properties.major < 7) {
    throw std::runtime_error("system-scope PTX accesses require SM70+");
  }
  int can_access = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, rank, peer_rank));
  if (!can_access)
    throw std::runtime_error("the two GPUs are not P2P capable");

  const int vectors = elements / 4;
  const int blocks =
      std::max(1, std::min(32, (vectors + kThreads - 1) / kThreads));
  const size_t tensor_bytes = static_cast<size_t>(elements) * sizeof(float);
  const size_t workspace_bytes = 2ULL * kWorld * tensor_bytes;

  cudaStream_t stream{};
  Vec16 *input = nullptr;
  Vec16 *output = nullptr;
  Vec16 *local_workspace = nullptr;
  Vec16 *peer_workspace = nullptr;
  unsigned int *epochs = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&input, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&output, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&local_workspace, workspace_bytes));
  CUDA_CHECK(cudaMalloc(&epochs, blocks * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemsetAsync(local_workspace, 0, workspace_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(epochs, 0, blocks * sizeof(unsigned int), stream));

  std::vector<float> host_input(elements);
  for (int i = 0; i < elements; ++i) {
    host_input[i] =
        static_cast<float>(rank) + static_cast<float>(i % 29) * 0.03125f;
  }
  CUDA_CHECK(cudaMemcpyAsync(input, host_input.data(), tensor_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // The opaque handle, not local_workspace's pointer value, crosses the process
  // boundary. cudaIpcOpenMemHandle returns a mapping valid in this process.
  cudaIpcMemHandle_t local_handle{};
  CUDA_CHECK(cudaIpcGetMemHandle(&local_handle, local_workspace));
  const cudaIpcMemHandle_t peer_handle = exchange(socket, local_handle);
  CUDA_CHECK(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&peer_workspace),
                                  peer_handle, cudaIpcMemLazyEnablePeerAccess));

  PushParams params{};
  params.input = input;
  params.output = output;
  params.workspace[rank] = local_workspace;
  params.workspace[peer_rank] = peer_workspace;
  params.epochs = epochs;
  params.vectors = vectors;
  params.rank = rank;

  // Both memset operations and both IPC mappings must exist before either rank
  // starts pushing into the other process's allocation.
  process_barrier(socket);
  for (int i = 0; i < iterations; ++i) {
    ipc_push_kernel<<<blocks, kThreads, 0, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> actual(elements);
  CUDA_CHECK(
      cudaMemcpy(actual.data(), output, tensor_bytes, cudaMemcpyDeviceToHost));
  bool passed = true;
  for (int i = 0; i < elements; ++i) {
    const float unit = static_cast<float>(i % 29) * 0.03125f;
    const float expected = unit + (1.0f + unit);
    if (std::abs(actual[i] - expected) > 1e-5f) {
      std::cerr << "rank " << rank << ", element " << i << ": expected "
                << expected << ", got " << actual[i] << '\n';
      passed = false;
      break;
    }
  }

  // The exporter must keep its allocation alive until the peer has stopped
  // using and closed the imported mapping.
  process_barrier(socket);
  CUDA_CHECK(cudaIpcCloseMemHandle(peer_workspace));
  process_barrier(socket);
  CUDA_CHECK(cudaFree(epochs));
  CUDA_CHECK(cudaFree(local_workspace));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::cout << "rank " << rank << ": " << (passed ? "PASS" : "FAIL")
            << " (IPC 1shot_push, " << iterations << " iterations)\n";
  return passed ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  const int elements =
      argc > 1 ? parse_positive(argv[1], "elements") : (1 << 20);
  const int iterations = argc > 2 ? parse_positive(argv[2], "iterations") : 10;

  int sockets[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    std::perror("socketpair");
    return 1;
  }
  // Fork before either process initializes the CUDA runtime.
  const pid_t child = ::fork();
  if (child < 0) {
    std::perror("fork");
    return 1;
  }
  if (child == 0) {
    ::close(sockets[0]);
    try {
      const int result = run_rank(1, sockets[1], elements, iterations);
      ::close(sockets[1]);
      return result;
    } catch (const std::exception &error) {
      std::cerr << "rank 1 error: " << error.what() << '\n';
      return 1;
    }
  }

  ::close(sockets[1]);
  int parent_result = 1;
  try {
    parent_result = run_rank(0, sockets[0], elements, iterations);
  } catch (const std::exception &error) {
    std::cerr << "rank 0 error: " << error.what() << '\n';
  }
  ::close(sockets[0]);
  int child_status = 0;
  (void)::waitpid(child, &child_status, 0);
  const bool child_ok =
      WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0;
  return parent_result == 0 && child_ok ? 0 : 1;
}
