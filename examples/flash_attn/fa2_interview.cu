#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Interview-friendly FlashAttention-2 forward kernel.
//
// Input layout: [batch, heads, seqlen, kHeadDim].
// One CUDA block owns one [kBlockM, kHeadDim] query tile and streams over K/V
// in kBlockN-row tiles.  No seqlen x seqlen attention matrix is materialized.
// It intentionally uses scalar FP32 math instead of Tensor Cores/cp.async so
// the QK^T -> online softmax -> PV dataflow remains short enough to explain in
// an interview; this is a teaching kernel, not a performance implementation.
constexpr int kWarpSize = 32;
constexpr int kHeadDim = 64;
constexpr int kBlockM = 4;
constexpr int kBlockN = 32;
constexpr int kThreads = kBlockM * kWarpSize;

static_assert(kBlockN == kWarpSize);
static_assert(kHeadDim % kWarpSize == 0);

struct SharedStorage {
  float q[kBlockM][kHeadDim];
  float k[kBlockN][kHeadDim];
  float v[kBlockN][kHeadDim];
  float p[kBlockM][kBlockN];
};

__device__ __forceinline__ float warp_allreduce_max(float value) {
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
  }
  return value;
}

__device__ __forceinline__ float warp_allreduce_sum(float value) {
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask);
  }
  return value;
}
/**
 * @brief 每个 block 处理 Br=4 行 Query
          每个 warp 负责一行 Query
          以 Bc=32 分块遍历 K/V
 */
__global__ __launch_bounds__(kThreads) void flash_attention_v2(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ output, int heads,
    int seqlen, bool causal) {
  __shared__ SharedStorage shared;

  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int query_start = blockIdx.x * kBlockM;
  const int query_idx = query_start + warp;
  const bool query_valid = query_idx < seqlen;
  const size_t bh_offset =
      (static_cast<size_t>(batch_idx) * heads + head_idx) * seqlen * kHeadDim;

  // Load Q once.  This tile stays resident while the block visits every K/V
  // tile, which is the query-block outer-loop structure used by FA2.
  for (int index = tid; index < kBlockM * kHeadDim; index += kThreads) {
    const int row = index / kHeadDim;
    const int dim = index % kHeadDim;
    const int global_row = query_start + row;
    shared.q[row][dim] =
        global_row < seqlen ? q[bh_offset + global_row * kHeadDim + dim] : 0.0f;
  }
  __syncthreads();

  // Online-softmax state for this query row:
  //
  //   m = max_j S_ij
  //   l = sum_j exp(S_ij - m)
  //   o[d] = sum_j exp(S_ij - m) * V_jd
  //
  // Each warp owns one query row.  Every lane holds the same m/l, while lane d
  // owns output dimensions d and d + 32 in registers.
  float row_max = -FLT_MAX;
  float row_sum = 0.0f;
  float out_acc[kHeadDim / kWarpSize] = {0.0f};
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  // A causal query tile only needs K/V rows up to its final query row.
  const int kv_end = causal ? min(seqlen, query_start + kBlockM) : seqlen;

  for (int key_start = 0; key_start < kv_end; key_start += kBlockN) {
    // Cooperative global-to-shared load of one K/V tile.  Tail rows are zero
    // filled and later masked out of the softmax.
    for (int index = tid; index < kBlockN * kHeadDim; index += kThreads) {
      const int row = index / kHeadDim;
      const int dim = index % kHeadDim;
      const int key_idx = key_start + row;
      const bool valid = key_idx < seqlen;
      shared.k[row][dim] =
          valid ? k[bh_offset + key_idx * kHeadDim + dim] : 0.0f;
      shared.v[row][dim] =
          valid ? v[bh_offset + key_idx * kHeadDim + dim] : 0.0f;
    }
    __syncthreads();

    // Tiled GEMM 1: S_tile = scale * Q_tile * K_tile^T.
    // Warp `warp` computes one S row; lane `lane` computes one S column.
    const int key_idx = key_start + lane;
    const bool pair_valid =
        query_valid && key_idx < seqlen && (!causal || key_idx <= query_idx);
    float score = -FLT_MAX;
    if (pair_valid) {
      score = 0.0f;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        score += shared.q[warp][dim] * shared.k[lane][dim];
      }
      score *= scale;
    }

    // Merge this score tile into the running online-softmax state.  If
    // m_new = max(m_old, tile_max), then the old state must be rescaled by
    // alpha = exp(m_old - m_new):
    //
    //   l_new = alpha * l_old + sum_j exp(S_ij - m_new)
    //   o_new = alpha * o_old + sum_j exp(S_ij - m_new) * V_j
    const float tile_max = warp_allreduce_max(score);
    const float new_max = fmaxf(row_max, tile_max);
    const float alpha = expf(row_max - new_max);
    const float probability = pair_valid ? expf(score - new_max) : 0.0f;
    const float tile_sum = warp_allreduce_sum(probability);

    shared.p[warp][lane] = probability;
    __syncthreads();

    // Tiled GEMM 2: O_tile += P_tile * V_tile.
    // Each lane owns two output columns, so the numerator remains in registers
    // across the complete K/V loop.
#pragma unroll
    for (int item = 0; item < kHeadDim / kWarpSize; ++item) {
      const int dim = lane + item * kWarpSize;
      float pv = 0.0f;
#pragma unroll
      for (int key = 0; key < kBlockN; ++key) {
        pv += shared.p[warp][key] * shared.v[key][dim];
      }
      out_acc[item] = alpha * out_acc[item] + pv;
    }

    row_max = new_max;
    row_sum = alpha * row_sum + tile_sum;
    __syncthreads(); // K/V/P may now be reused by the next tile.
  }

  if (query_valid) {
#pragma unroll
    for (int item = 0; item < kHeadDim / kWarpSize; ++item) {
      const int dim = lane + item * kWarpSize;
      output[bh_offset + query_idx * kHeadDim + dim] = out_acc[item] / row_sum;
    }
  }
}

void attention_reference(const std::vector<float> &q,
                         const std::vector<float> &k,
                         const std::vector<float> &v,
                         std::vector<float> &output, int batch, int heads,
                         int seqlen, bool causal) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  std::vector<double> weights(seqlen);

  for (int batch_idx = 0; batch_idx < batch; ++batch_idx) {
    for (int head_idx = 0; head_idx < heads; ++head_idx) {
      const size_t base = (static_cast<size_t>(batch_idx) * heads + head_idx) *
                          seqlen * kHeadDim;
      for (int query_idx = 0; query_idx < seqlen; ++query_idx) {
        const int key_end = causal ? query_idx + 1 : seqlen;
        double max_score = -INFINITY;

        for (int key_idx = 0; key_idx < key_end; ++key_idx) {
          double score = 0.0;
          for (int dim = 0; dim < kHeadDim; ++dim) {
            score += static_cast<double>(q[base + query_idx * kHeadDim + dim]) *
                     k[base + key_idx * kHeadDim + dim];
          }
          weights[key_idx] = score * scale;
          max_score = std::max(max_score, weights[key_idx]);
        }

        double sum = 0.0;
        for (int key_idx = 0; key_idx < key_end; ++key_idx) {
          weights[key_idx] = std::exp(weights[key_idx] - max_score);
          sum += weights[key_idx];
        }

        for (int dim = 0; dim < kHeadDim; ++dim) {
          double numerator = 0.0;
          for (int key_idx = 0; key_idx < key_end; ++key_idx) {
            numerator += weights[key_idx] * v[base + key_idx * kHeadDim + dim];
          }
          output[base + query_idx * kHeadDim + dim] =
              static_cast<float>(numerator / sum);
        }
      }
    }
  }
}

float verify_case(const std::vector<float> &q, const std::vector<float> &k,
                  const std::vector<float> &v, float *d_q, float *d_k,
                  float *d_v, float *d_output, int batch, int heads, int seqlen,
                  bool causal) {
  std::vector<float> output(q.size());
  std::vector<float> reference(q.size());
  const dim3 grid((seqlen + kBlockM - 1) / kBlockM, heads, batch);

  flash_attention_v2<<<grid, kThreads>>>(d_q, d_k, d_v, d_output, heads, seqlen,
                                         causal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  attention_reference(q, k, v, reference, batch, heads, seqlen, causal);
  float max_error = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    max_error = std::max(max_error, std::fabs(output[i] - reference[i]));
  }
  return max_error;
}

int main() {
  constexpr int batch = 2;
  constexpr int heads = 2;
  constexpr int seqlen = 67; // Exercises both query and K/V tail tiles.
  const size_t elements =
      static_cast<size_t>(batch) * heads * seqlen * kHeadDim;
  const size_t bytes = elements * sizeof(float);

  std::vector<float> q(elements);
  std::vector<float> k(elements);
  std::vector<float> v(elements);
  for (size_t i = 0; i < elements; ++i) {
    q[i] = 0.5f * std::sin(0.013f * static_cast<float>(i));
    k[i] = 0.5f * std::cos(0.017f * static_cast<float>(i));
    v[i] = std::sin(0.019f * static_cast<float>(i));
  }

  float *d_q = nullptr;
  float *d_k = nullptr;
  float *d_v = nullptr;
  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, bytes));
  CUDA_CHECK(cudaMalloc(&d_k, bytes));
  CUDA_CHECK(cudaMalloc(&d_v, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_q, q.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, k.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, v.data(), bytes, cudaMemcpyHostToDevice));

  const float noncausal_error = verify_case(q, k, v, d_q, d_k, d_v, d_output,
                                            batch, heads, seqlen, false);
  const float causal_error =
      verify_case(q, k, v, d_q, d_k, d_v, d_output, batch, heads, seqlen, true);

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_output));

  std::printf("FA2 non-causal max error: %.3e\n", noncausal_error);
  std::printf("FA2 causal max error:     %.3e\n", causal_error);
  if (noncausal_error > 1e-5f || causal_error > 1e-5f) {
    std::fprintf(stderr, "verification failed\n");
    return EXIT_FAILURE;
  }
  std::printf("verification passed\n");
  return EXIT_SUCCESS;
}
