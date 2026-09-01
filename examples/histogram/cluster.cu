#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error = (call);                                              \
    if (error != cudaSuccess) {                                              \
      std::cerr << __FILE__ << ':' << __LINE__                               \
                << " CUDA error: " << cudaGetErrorString(error) << '\n';    \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Each block owns bins_per_block counters in shared memory. Together, the
// blocks in a cluster hold one complete histogram in distributed shared memory
// (DSM). The cluster size is supplied at launch time with cudaLaunchKernelEx.
__global__ void dsm_histogram(const int *__restrict__ input,
                              unsigned int *__restrict__ global_bins, int n,
                              int num_bins, int bins_per_block) {
  extern __shared__ unsigned int block_bins[];

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int block_rank = cluster.block_rank();

  // Initialize the part of the cluster histogram owned by this block.
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
    block_bins[i] = 0;
  }

  // Besides acting as a barrier, this guarantees that every block and every
  // shared-memory segment in the cluster now exists and is initialized.
  cluster.sync();

  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_stride = gridDim.x * blockDim.x;

  for (int i = global_thread; i < n; i += grid_stride) {
    const int bin = input[i];
    if (bin < 0 || bin >= num_bins) {
      continue;
    }

    // Select the block that owns this bin, then map the local shared-memory
    // pointer to the corresponding address in that block.
    const unsigned int destination_rank = bin / bins_per_block;
    const int destination_offset = bin % bins_per_block;
    unsigned int *destination_bins =
        cluster.map_shared_rank(block_bins, destination_rank);

    // This may be a local or a remote shared-memory atomic operation.
    atomicAdd(destination_bins + destination_offset, 1U);
  }

  // All DSM atomics must finish before any owner reads its counters or any
  // block exits and destroys its shared memory.
  cluster.sync();

  // Every cluster has a private DSM histogram. Merge each block's owned slice
  // into the single global result. Different clusters can update the same bin,
  // so this final merge still needs global atomics.
  const int first_bin = block_rank * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
    const int global_bin = first_bin + i;
    if (global_bin < num_bins) {
      atomicAdd(global_bins + global_bin, block_bins[i]);
    }
  }
}

void histogram_cpu(const std::vector<int> &input,
                   std::vector<unsigned int> &bins) {
  for (int value : input) {
    if (value >= 0 && value < static_cast<int>(bins.size())) {
      ++bins[value];
    }
  }
}

int main() {
  constexpr int kElementCount = 1 << 20;
  constexpr int kNumBins = 4096;
  constexpr int kThreadsPerBlock = 256;
  constexpr int kClusterSize = 4;
  constexpr int kNumClusters = 8;
  constexpr int kNumBlocks = kClusterSize * kNumClusters;

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  std::cout << "GPU: " << properties.name << " (compute capability "
            << properties.major << '.' << properties.minor << ")\n";

  if (properties.major < 9) {
    std::cout << "Skipped: thread block clusters require compute capability "
                 "9.0 or newer.\n";
    return EXIT_SUCCESS;
  }

  // ceil(kNumBins / kClusterSize); extra counters in the final slice, if any,
  // are simply ignored during the global merge.
  constexpr int kBinsPerBlock =
      (kNumBins + kClusterSize - 1) / kClusterSize;
  constexpr size_t kSharedBytesPerBlock =
      kBinsPerBlock * sizeof(unsigned int);

  std::vector<int> host_input(kElementCount);
  for (int i = 0; i < kElementCount; ++i) {
    // 37 is coprime with 4096, so this also gives a convenient uniform test.
    host_input[i] = (i * 37 + 11) % kNumBins;
  }

  std::vector<unsigned int> expected(kNumBins, 0);
  std::vector<unsigned int> actual(kNumBins, 0);
  histogram_cpu(host_input, expected);

  int *device_input = nullptr;
  unsigned int *device_bins = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, kElementCount * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device_bins, kNumBins * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        kElementCount * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_bins, 0, kNumBins * sizeof(unsigned int)));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kNumBlocks, 1, 1); // Dimensions are still in blocks.
  config.blockDim = dim3(kThreadsPerBlock, 1, 1);
  config.dynamicSmemBytes = kSharedBytesPerBlock; // Per block, not per cluster.

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = kClusterSize;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;

  int maximum_cluster_size = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(
      &maximum_cluster_size, dsm_histogram, &config));
  if (maximum_cluster_size < kClusterSize) {
    std::cerr << "Requested cluster size " << kClusterSize
              << ", but this launch configuration supports at most "
              << maximum_cluster_size << ".\n";
    CUDA_CHECK(cudaFree(device_bins));
    CUDA_CHECK(cudaFree(device_input));
    return EXIT_FAILURE;
  }

  int maximum_active_clusters = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveClusters(
      &maximum_active_clusters, dsm_histogram, &config));

  CUDA_CHECK(cudaLaunchKernelEx(&config, dsm_histogram, device_input,
                                device_bins, kElementCount, kNumBins,
                                kBinsPerBlock));
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), device_bins,
                        kNumBins * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  for (int bin = 0; bin < kNumBins; ++bin) {
    if (actual[bin] != expected[bin]) {
      std::cerr << "Mismatch at bin " << bin << ": GPU=" << actual[bin]
                << ", CPU=" << expected[bin] << '\n';
      CUDA_CHECK(cudaFree(device_bins));
      CUDA_CHECK(cudaFree(device_input));
      return EXIT_FAILURE;
    }
  }

  std::cout << "Histogram is correct.\n"
            << "  blocks: " << kNumBlocks << '\n'
            << "  blocks per cluster: " << kClusterSize << '\n'
            << "  bins per block: " << kBinsPerBlock << '\n'
            << "  shared memory per block: " << kSharedBytesPerBlock
            << " bytes\n"
            << "  DSM per cluster: "
            << kClusterSize * kSharedBytesPerBlock << " bytes\n"
            << "  maximum active clusters for this configuration: "
            << maximum_active_clusters << '\n';

  CUDA_CHECK(cudaFree(device_bins));
  CUDA_CHECK(cudaFree(device_input));
  return EXIT_SUCCESS;
}
