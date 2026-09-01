#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// A standalone, CUTLASS-free sketch of SGLang V2's 1shot_push all-reduce.
//
// One host thread controls every GPU to keep the example small. Production
// SGLang normally has one process per rank and exchanges peer pointers through
// CUDA IPC / symmetric memory. Once the pointers reach the kernel, the data
// path is the same: a rank writes its local vector into every destination
// rank's P2P-visible workspace, then polls and reduces only local memory.

namespace {

constexpr int kMaxRanks = 8;

void check_cuda(cudaError_t status, const char *expression, const char *file,
                int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": " + expression +
                             " failed: " + cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

struct alignas(16) Vec16 {
  float x;
  float y;
  float z;
  float w;
};
static_assert(sizeof(Vec16) == 16);

union VecBits {
  Vec16 vec;
  uint4 bits;
};

// `.sys` is essential: `.gpu` only scopes the operation to the issuing GPU,
// while these stores are consumed by a different GPU.
__device__ __forceinline__ void store_relaxed_sys(Vec16 *base, int index,
                                                  const Vec16 &value) {
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

__device__ __forceinline__ void store_local(Vec16 *base, int index,
                                            const Vec16 &value) {
  const VecBits raw{value};
  Vec16 *address = base + index;
  asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(raw.bits.x), "r"(raw.bits.y), "r"(raw.bits.z),
                 "r"(raw.bits.w), "l"(address)
               : "memory");
}

// The workspace is initially all +0.0. A producer changes payload +0.0 to
// -0.0 (same arithmetic value, different bits), making an all-zero word a
// per-word "not arrived" marker. This is the Lamport-style protocol used by
// SGLang V2's push kernel and removes a separate flag/fence round trip.
__device__ __forceinline__ Vec16 encode_payload(Vec16 value) {
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

__device__ __forceinline__ bool contains_empty_marker(const Vec16 &value) {
  const VecBits raw{value};
  return raw.bits.x == 0 || raw.bits.y == 0 || raw.bits.z == 0 ||
         raw.bits.w == 0;
}

__device__ __forceinline__ Vec16 add(Vec16 a, const Vec16 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

template <int World> struct PushParams {
  const Vec16 *input;
  Vec16 *output;
  Vec16 *workspaces[World];
  unsigned int *block_epochs;
  int vector_count;
  int rank;
};

template <int World>
__global__
__launch_bounds__(256) void one_shot_push_all_reduce(PushParams<World> params) {
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_threads = gridDim.x * blockDim.x;
  const unsigned int epoch = params.block_epochs[blockIdx.x] & 1u;

  // Layout on each destination:
  //   [epoch 0: World source slots][epoch 1: World source slots]
  // Each slot is vector_count * 16 bytes.
  Vec16 *destinations[World];
#pragma unroll
  for (int destination = 0; destination < World; ++destination) {
    destinations[destination] =
        params.workspaces[destination] +
        (epoch * World + params.rank) * params.vector_count;
  }

  // Push this rank's input to every peer (and to itself). The rank that owns a
  // workspace later performs only local polling loads from it.
  for (int vector_id = global_thread; vector_id < params.vector_count;
       vector_id += grid_threads) {
    const Vec16 value = encode_payload(params.input[vector_id]);
#pragma unroll
    for (int destination = 0; destination < World; ++destination) {
      store_relaxed_sys(destinations[destination], vector_id, value);
    }
  }

  const Vec16 empty{0.0f, 0.0f, 0.0f, 0.0f};
  const Vec16 *sources[World];
#pragma unroll
  for (int source = 0; source < World; ++source) {
    sources[source] = params.workspaces[params.rank] +
                      (epoch * World + source) * params.vector_count;
  }

  // Poll local memory until every source vector has arrived. A vector store
  // may be observed piecemeal; that is safe because any not-yet-written word
  // remains +0 and keeps the loop spinning.
  for (int vector_id = global_thread; vector_id < params.vector_count;
       vector_id += grid_threads) {
    Vec16 values[World];
    while (true) {
      bool waiting = false;
#pragma unroll
      for (int source = 0; source < World; ++source) {
        values[source] = load_relaxed_sys(sources[source], vector_id);
        waiting |= contains_empty_marker(values[source]);
      }
      if (!waiting)
        break;
    }

    Vec16 sum = values[0];
#pragma unroll
    for (int source = 1; source < World; ++source) {
      sum = add(sum, values[source]);
    }
    params.output[vector_id] = sum;

    // Restore the empty marker so this half can be reused two calls later.
#pragma unroll
    for (int source = 0; source < World; ++source) {
      store_local(const_cast<Vec16 *>(sources[source]), vector_id, empty);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    params.block_epochs[blockIdx.x] = epoch ^ 1u;
  }
}

template <int World>
void launch_rank(int rank, const std::vector<Vec16 *> &inputs,
                 const std::vector<Vec16 *> &outputs,
                 const std::vector<Vec16 *> &workspaces,
                 const std::vector<unsigned int *> &epochs, int vector_count,
                 int blocks, cudaStream_t stream) {
  PushParams<World> params{};
  params.input = inputs[rank];
  params.output = outputs[rank];
  params.block_epochs = epochs[rank];
  params.vector_count = vector_count;
  params.rank = rank;
  for (int i = 0; i < World; ++i)
    params.workspaces[i] = workspaces[i];
  one_shot_push_all_reduce<World><<<blocks, 256, 0, stream>>>(params);
  CUDA_CHECK(cudaGetLastError());
}

void launch(int world, int rank, const std::vector<Vec16 *> &inputs,
            const std::vector<Vec16 *> &outputs,
            const std::vector<Vec16 *> &workspaces,
            const std::vector<unsigned int *> &epochs, int vector_count,
            int blocks, cudaStream_t stream) {
#define LAUNCH_CASE(n)                                                         \
  case n:                                                                      \
    launch_rank<n>(rank, inputs, outputs, workspaces, epochs, vector_count,    \
                   blocks, stream);                                            \
    break
  switch (world) {
    LAUNCH_CASE(2);
    LAUNCH_CASE(3);
    LAUNCH_CASE(4);
    LAUNCH_CASE(5);
    LAUNCH_CASE(6);
    LAUNCH_CASE(7);
    LAUNCH_CASE(8);
  default:
    throw std::runtime_error("world size must be in [2, 8]");
  }
#undef LAUNCH_CASE
}

void enable_full_peer_access(int world) {
  for (int source = 0; source < world; ++source) {
    CUDA_CHECK(cudaSetDevice(source));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, source));
    if (properties.major < 7) {
      throw std::runtime_error("the scoped PTX accesses require SM 7.0+");
    }
    for (int destination = 0; destination < world; ++destination) {
      if (source == destination)
        continue;
      int can_access = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, source, destination));
      if (!can_access) {
        throw std::runtime_error(
            "GPU " + std::to_string(source) + " cannot access GPU " +
            std::to_string(destination) + "; a full P2P clique is required");
      }
      const cudaError_t status = cudaDeviceEnablePeerAccess(destination, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
      } else {
        CUDA_CHECK(status);
      }
    }
  }
}

int parse_positive(const char *value, const char *name) {
  const long parsed = std::strtol(value, nullptr, 10);
  if (parsed <= 0 || parsed > (1L << 30)) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
  return static_cast<int>(parsed);
}

} // namespace

int main(int argc, char **argv) {
  try {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
      std::cout
          << "SKIP: this example needs at least two peer-accessible GPUs; "
             "found "
          << device_count << ".\n";
      return 0;
    }

    const int element_count =
        argc > 1 ? parse_positive(argv[1], "element_count") : (1 << 20);
    const int world = argc > 2 ? parse_positive(argv[2], "world_size")
                               : std::min(device_count, kMaxRanks);
    const int iterations =
        argc > 3 ? parse_positive(argv[3], "iterations") : 10;
    if (world > device_count || world > kMaxRanks || world < 2) {
      throw std::runtime_error(
          "world_size must be between 2 and min(8, GPU count)");
    }
    if (element_count % 4 != 0) {
      throw std::runtime_error(
          "element_count must be a multiple of four (16 bytes)");
    }

    enable_full_peer_access(world);

    const int vector_count = element_count / 4;
    const size_t tensor_bytes =
        static_cast<size_t>(element_count) * sizeof(float);
    const size_t workspace_bytes = 2ULL * world * tensor_bytes;
    const int blocks = std::max(1, std::min(32, (vector_count + 255) / 256));

    std::vector<Vec16 *> inputs(world);
    std::vector<Vec16 *> outputs(world);
    std::vector<Vec16 *> workspaces(world);
    std::vector<unsigned int *> epochs(world);
    std::vector<cudaStream_t> streams(world);

    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking));
      CUDA_CHECK(cudaMalloc(&inputs[rank], tensor_bytes));
      CUDA_CHECK(cudaMalloc(&outputs[rank], tensor_bytes));
      CUDA_CHECK(cudaMalloc(&workspaces[rank], workspace_bytes));
      CUDA_CHECK(cudaMalloc(&epochs[rank], blocks * sizeof(unsigned int)));
      CUDA_CHECK(
          cudaMemsetAsync(workspaces[rank], 0, workspace_bytes, streams[rank]));
      CUDA_CHECK(cudaMemsetAsync(epochs[rank], 0, blocks * sizeof(unsigned int),
                                 streams[rank]));

      std::vector<float> host_input(element_count);
      for (int i = 0; i < element_count; ++i) {
        // rank 0 deliberately contains +0.0 values, exercising marker encoding.
        host_input[i] =
            static_cast<float>(rank) + static_cast<float>(i % 29) * 0.03125f;
      }
      CUDA_CHECK(cudaMemcpyAsync(inputs[rank], host_input.data(), tensor_bytes,
                                 cudaMemcpyHostToDevice, streams[rank]));
    }

    // This is the host-side equivalent of SGLang's synchronize + process-group
    // barrier after zeroing symmetric slabs. Without it, rank 0 could push to
    // rank 1 while rank 1's initialization memset is still erasing that slab.
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    }

    // All ranks enqueue the same collective calls in the same order. Reusing
    // one communicator concurrently from unrelated streams is intentionally
    // unsupported by this simple Lamport protocol (and is unsafe in SGLang V2).
    for (int iteration = 0; iteration < iterations; ++iteration) {
      for (int rank = 0; rank < world; ++rank) {
        CUDA_CHECK(cudaSetDevice(rank));
        launch(world, rank, inputs, outputs, workspaces, epochs, vector_count,
               blocks, streams[rank]);
      }
    }

    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    }

    std::vector<float> actual(element_count);
    bool passed = true;
    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaMemcpy(actual.data(), outputs[rank], tensor_bytes,
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < element_count; ++i) {
        float expected = 0.0f;
        for (int source = 0; source < world; ++source) {
          expected += static_cast<float>(source) +
                      static_cast<float>(i % 29) * 0.03125f;
        }
        if (std::abs(actual[i] - expected) > 1e-5f) {
          std::cerr << "rank " << rank << ", element " << i << ": expected "
                    << expected << ", got " << actual[i] << '\n';
          passed = false;
          break;
        }
      }
    }

    for (int rank = 0; rank < world; ++rank) {
      CUDA_CHECK(cudaSetDevice(rank));
      CUDA_CHECK(cudaFree(epochs[rank]));
      CUDA_CHECK(cudaFree(workspaces[rank]));
      CUDA_CHECK(cudaFree(outputs[rank]));
      CUDA_CHECK(cudaFree(inputs[rank]));
      CUDA_CHECK(cudaStreamDestroy(streams[rank]));
    }

    std::cout
        << (passed ? "PASS" : "FAIL") << ": " << world << " GPUs, "
        << element_count << " float32 elements, " << iterations
        << " iterations, 1shot_push (plain CUDA + inline PTX, no CUTLASS)\n";
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
