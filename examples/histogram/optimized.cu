// Histogram optimization study.
//
// The teaching kernel in simple.cu issues one *global* atomicAdd per element:
//
//   __global__ void histogram_native_kernel(int *a, int *y, int N) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < N) atomicAdd(&(y[a[idx]]), 1);
//   }
//
// Every one of the N additions is a round trip to the same handful of L2
// addresses, so the kernel is completely serialized on L2 atomic throughput.
// This file adds one idea at a time and measures each step on the machine it
// runs on:
//
//   native             the kernel above, verbatim
//   native_gs          same global atomics, but a fixed grid + grid-stride loop
//   smem               + per-block privatization in shared memory
//   smem_repl          + one private copy per warp (kills inter-warp conflicts)
//   smem_repl_agg      + intra-warp aggregation with __match_any_sync
//   smem_repl_agg_vec4 + int4 loads (4 elements per load instruction)
//
// Build:  nvcc -std=c++17 -O2 -arch=sm_89 optimized.cu -o optimized
// Run:    ./optimized [N] [num_bins] [blocks_per_sm]
//         ./optimized 67108864 256     # 256 MB of input, DRAM-bound
//         ./optimized 4194304 256      # 16 MB, fits in L2

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error = (call);                                              \
    if (error != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(error));                               \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int kThreads = 256;

// ---------------------------------------------------------------------------
// 1. Baseline: one thread per element, one global atomicAdd per element.
// ---------------------------------------------------------------------------
__global__ void histogram_native(const int *__restrict__ a,
                                 int *__restrict__ y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    atomicAdd(&(y[a[idx]]), 1);
}

// 2. The same global-atomic scheme, but the grid is small and fixed and each
//    thread walks several elements. This isolates "how much of the baseline's
//    cost is the launch/config" from "how much is the atomic itself".
__global__ void histogram_native_grid_stride(const int *__restrict__ a,
                                             int *__restrict__ y, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride)
    atomicAdd(&(y[a[idx]]), 1);
}

// 3. Privatization. Each block accumulates into a private histogram in shared
//    memory and merges it into the global one exactly once at the end. Global
//    atomic traffic drops from N to gridDim.x * num_bins.
__global__ void histogram_smem(const int *__restrict__ a, int *__restrict__ y,
                               int n, int num_bins) {
  extern __shared__ unsigned int hist[];
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    hist[i] = 0u;
  __syncthreads();

  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride)
    atomicAdd(&hist[a[idx]], 1u);
  __syncthreads();

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    atomicAdd(&y[i], static_cast<int>(hist[i]));
}

// 3b. Counter-example: the same privatization, but keeping the original
//     one-thread-per-element grid. This is what shows the grid-stride loop in
//     histogram_smem() is load-bearing rather than cosmetic -- see the merge
//     loop at the bottom: it costs gridDim.x * num_bins global atomics, and
//     with N/256 = 262144 blocks that is 67M atomics, as many as the baseline
//     ever issued.
__global__ void histogram_smem_biggrid(const int *__restrict__ a,
                                       int *__restrict__ y, int n,
                                       int num_bins) {
  extern __shared__ unsigned int hist[];
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    hist[i] = 0u;
  __syncthreads();

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    atomicAdd(&hist[a[idx]], 1u);
  __syncthreads();

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    atomicAdd(&y[i], static_cast<int>(hist[i]));
}

// 4. + replication. Shared-memory atomics from different warps to the same bin
//    serialize, so give each warp its own copy of the histogram. Shared memory
//    atomics are per-SM traffic and are pipelined, so this is far cheaper than
//    the L2 round trips it replaces.
__global__ void histogram_smem_repl(const int *__restrict__ a,
                                    int *__restrict__ y, int n, int num_bins,
                                    int reps) {
  extern __shared__ unsigned int hist[];
  const int warp = threadIdx.x >> 5;
  unsigned int *mine = hist + (warp % reps) * num_bins;

  for (int i = threadIdx.x; i < reps * num_bins; i += blockDim.x)
    hist[i] = 0u;
  __syncthreads();

  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride)
    atomicAdd(&mine[a[idx]], 1u);
  __syncthreads();

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    unsigned int total = 0u;
    for (int r = 0; r < reps; ++r)
      total += hist[r * num_bins + i];
    atomicAdd(&y[i], static_cast<int>(total));
  }
}

// 5. + intra-warp aggregation. Lanes of one warp that landed on the same bin
//    elect the lowest-numbered lane and let it add popc(peers) in a single
//    shared atomic. Up to 32 shared atomics collapse into one. This is worth
//    most when the input distribution is skewed.
__global__ void histogram_smem_repl_agg(const int *__restrict__ a,
                                        int *__restrict__ y, int n,
                                        int num_bins, int reps) {
  extern __shared__ unsigned int hist[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  unsigned int *mine = hist + (warp % reps) * num_bins;

  for (int i = threadIdx.x; i < reps * num_bins; i += blockDim.x)
    hist[i] = 0u;
  __syncthreads();

  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    const unsigned int bin = static_cast<unsigned int>(a[idx]);
    // __activemask() keeps the tail iteration (where some lanes have already
    // left the loop) from handing __match_any_sync a mask with dead lanes.
    const unsigned int active = __activemask();
    const unsigned int peers = __match_any_sync(active, bin);
    if (lane == __ffs(peers) - 1)
      atomicAdd(&mine[bin], static_cast<unsigned int>(__popc(peers)));
  }
  __syncthreads();

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    unsigned int total = 0u;
    for (int r = 0; r < reps; ++r)
      total += hist[r * num_bins + i];
    atomicAdd(&y[i], static_cast<int>(total));
  }
}

// 6. + vectorized loads. One LDG.128 replaces four LDG.32, so the load pipe is
//    free to keep up with the (now much cheaper) shared atomics.
//    Requires n % 4 == 0; the driver asserts that.
__global__ void histogram_smem_repl_agg_vec4(const int *__restrict__ a,
                                             int *__restrict__ y, int n,
                                             int num_bins, int reps) {
  extern __shared__ unsigned int hist[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  unsigned int *mine = hist + (warp % reps) * num_bins;

  for (int i = threadIdx.x; i < reps * num_bins; i += blockDim.x)
    hist[i] = 0u;
  __syncthreads();

  const int4 *a4 = reinterpret_cast<const int4 *>(a);
  const int n4 = n >> 2;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n4;
       idx += stride) {
    const int4 v = a4[idx];
    const unsigned int bins[4] = {static_cast<unsigned int>(v.x),
                                  static_cast<unsigned int>(v.y),
                                  static_cast<unsigned int>(v.z),
                                  static_cast<unsigned int>(v.w)};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const unsigned int active = __activemask();
      const unsigned int peers = __match_any_sync(active, bins[k]);
      if (lane == __ffs(peers) - 1)
        atomicAdd(&mine[bins[k]], static_cast<unsigned int>(__popc(peers)));
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    unsigned int total = 0u;
    for (int r = 0; r < reps; ++r)
      total += hist[r * num_bins + i];
    atomicAdd(&y[i], static_cast<int>(total));
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

void histogram_cpu(const std::vector<int> &a, std::vector<long long> &y) {
  for (int value : a)
    ++y[value];
}

std::vector<int> make_input(int n, int num_bins, bool skewed) {
  // Fixed seed: every variant sees byte-identical input.
  std::mt19937 rng(20260920u);
  std::vector<int> a(n);
  // With skew, 80% of the elements land in the first num_bins/16 bins, which
  // is the case where warp-level aggregation pays off most.
  const int hot_bins = skewed ? std::max(1, num_bins / 16) : 0;
  std::uniform_int_distribution<int> hot(0, hot_bins - 1);
  std::uniform_int_distribution<int> cold(0, num_bins - 1);
  for (int i = 0; i < n; ++i)
    a[i] = skewed && (rng() % 100) < 80 ? hot(rng) : cold(rng);
  return a;
}

// Time `iters` repetitions of (zero y, launch). The memset is common to every
// variant; its cost is measured once and subtracted by time_memset().
template <typename LaunchFn>
double time_kernel(LaunchFn launch, int *d_y, size_t y_bytes, int iters) {
  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaMemsetAsync(d_y, 0, y_bytes));
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaMemsetAsync(d_y, 0, y_bytes));
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

double time_memset(int *d_y, size_t y_bytes, int iters) {
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    CUDA_CHECK(cudaMemsetAsync(d_y, 0, y_bytes));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

// One entry per kernel variant. `iterations` adapts so that slow baselines do
// not make the whole sweep take forever while fast variants still get enough
// samples to be stable.
struct Variant {
  const char *name;
  std::function<void()> launch;
  int iterations = 0;
  double ms = 0.0;
  int bins_ok = 0;  // number of y[] entries that matched the CPU reference
};

// Launch `launch` once, compare every bin against the CPU reference.
int check_correctness(const std::function<void()> &launch, int *d_y,
                      size_t y_bytes, const std::vector<long long> &h_ref,
                      std::vector<int> &h_y) {
  CUDA_CHECK(cudaMemset(d_y, 0, y_bytes));
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));
  int ok = 0;
  for (size_t b = 0; b < h_ref.size(); ++b)
    if (h_y[b] == static_cast<int>(h_ref[b]))
      ++ok;
  return ok;
}

int main(int argc, char **argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : (4 << 20);
  const int num_bins = argc > 2 ? std::atoi(argv[2]) : 256;
  const int blocks_per_sm = argc > 3 ? std::atoi(argv[3]) : 8;
  if (n <= 0 || n % 4 != 0) {
    std::fprintf(stderr, "N must be positive and a multiple of 4\n");
    return EXIT_FAILURE;
  }
  if (num_bins <= 0) {
    std::fprintf(stderr, "num_bins must be positive\n");
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s (sm_%d%d, %d SMs, %d KiB shared/SM)\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              static_cast<int>(prop.sharedMemPerMultiprocessor / 1024));

  // Replicas: one per warp, but never more than fits in 32 KiB of shared
  // memory, so occupancy stays high across small and large bin counts.
  const int warps = kThreads / 32;
  int reps = 32768 / (num_bins * static_cast<int>(sizeof(unsigned int)));
  reps = std::max(1, std::min(reps, warps));

  // Grid-stride variants use a fixed grid; `native` uses one thread per
  // element, exactly like the original kernel.
  const int gs_blocks =
      std::min((n + kThreads - 1) / kThreads,
               prop.multiProcessorCount * blocks_per_sm);
  const int native_blocks = (n + kThreads - 1) / kThreads;

  const size_t y_bytes = static_cast<size_t>(num_bins) * sizeof(int);
  const size_t repl_bytes =
      static_cast<size_t>(reps) * num_bins * sizeof(unsigned int);

  std::printf("N = %d, num_bins = %d, threads = %d\n", n, num_bins, kThreads);
  std::printf("  native grid: %d blocks | grid-stride grid: %d blocks\n",
              native_blocks, gs_blocks);
  std::printf("  replicas = %d, shared memory for the private histogram = "
              "%zu bytes\n\n",
              reps, static_cast<size_t>(repl_bytes));

  int *d_a = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, static_cast<size_t>(n) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_y, y_bytes));

  const double memset_ms = time_memset(d_y, y_bytes, 100);

  for (int skew = 0; skew <= 1; ++skew) {
    std::printf("distribution: %s\n", skew ? "skewed (80% in 1/16 of bins)"
                                           : "uniform");
    std::printf("  %-22s %10s %10s %9s\n", "kernel", "time", "bandwidth",
                "speedup");

    const std::vector<int> h_a = make_input(n, num_bins, skew != 0);
    std::vector<long long> h_ref(num_bins, 0);
    histogram_cpu(h_a, h_ref);
    std::vector<int> h_y(num_bins, 0);

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), static_cast<size_t>(n) * sizeof(int),
                          cudaMemcpyHostToDevice));

    const std::vector<Variant> variants = {
        {"native", [&] { histogram_native<<<native_blocks, kThreads>>>(d_a, d_y, n); }},
        {"native_gs", [&] {
           histogram_native_grid_stride<<<gs_blocks, kThreads>>>(d_a, d_y, n);
         }},
        {"smem", [&] {
           histogram_smem<<<gs_blocks, kThreads, y_bytes>>>(d_a, d_y, n,
                                                            num_bins);
         }},
        {"smem_biggrid", [&] {
           histogram_smem_biggrid<<<native_blocks, kThreads, y_bytes>>>(
               d_a, d_y, n, num_bins);
         }},
        {"smem_repl", [&] {
           histogram_smem_repl<<<gs_blocks, kThreads, repl_bytes>>>(
               d_a, d_y, n, num_bins, reps);
         }},
        {"smem_repl_agg", [&] {
           histogram_smem_repl_agg<<<gs_blocks, kThreads, repl_bytes>>>(
               d_a, d_y, n, num_bins, reps);
         }},
        {"smem_repl_agg_vec4", [&] {
           histogram_smem_repl_agg_vec4<<<gs_blocks, kThreads, repl_bytes>>>(
               d_a, d_y, n, num_bins, reps);
         }},
    };

    double best = 1e30;
    std::vector<double> times(variants.size());
    std::vector<int> correctness(variants.size());
    for (size_t i = 0; i < variants.size(); ++i) {
      // Pick the iteration count from one untimed run so that every variant
      // gets a comparable total measurement window.
      const double once = time_kernel(variants[i].launch, d_y, y_bytes, 1);
      const int iters =
          std::max(3, std::min(50, static_cast<int>(200.0 / once)));
      times[i] = std::max(0.0, time_kernel(variants[i].launch, d_y, y_bytes,
                                           iters) -
                                   memset_ms);
      correctness[i] =
          check_correctness(variants[i].launch, d_y, y_bytes, h_ref, h_y);
      best = std::min(best, times[i]);
    }

    for (size_t i = 0; i < variants.size(); ++i) {
      const double ms = times[i];
      const double gbps = static_cast<double>(n) * sizeof(int) /
                          (ms * 1e-3) / 1e9;
      std::printf("  %-22s %7.4f ms %7.1f GB/s %8.2fx   %s (%d/%d bins)\n",
                  variants[i].name, ms, gbps, best / ms,
                  correctness[i] == num_bins ? "ok" : "WRONG", correctness[i],
                  num_bins);
    }
    std::printf("\n");
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_y));
  return EXIT_SUCCESS;
}
