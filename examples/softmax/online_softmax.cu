#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
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

struct state {
  float m_val;
  float l_val;
};

__device__ __forceinline__ state combine(const state *a, const state *b) {
  state res;
  res.m_val = fmaxf(a->m_val, b->m_val);
  res.l_val = a->l_val * expf(a->m_val - res.m_val) +
              b->l_val * expf(b->m_val - res.m_val);
  return res;
}

// 对一个 warp 内的 32 个 State 做 all-reduce。
// 使用 xor shuffle 后，最终每个 lane 都持有相同的 warp state。
__device__ __forceinline__ state warp_softmax(state s) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    state other;
    other.m_val = __shfl_xor_sync(0xffffffffu, s.m_val, offset);
    other.l_val = __shfl_xor_sync(0xffffffffu, s.l_val, offset);
    s = combine(&s, &other);
  }
  return s;
}

__global__ void online_softmax_2d(const float *__restrict__ input,
                                  float *__restrict__ output, int rows,
                                  int cols) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x >> 5;
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= rows)
    return;

  __shared__ state smem[8];
  // ---- 1) 线程内在线 softmax ----
  state local;
  local.m_val = -FLT_MAX;
  local.l_val = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    float x = input[row * cols + col];
    float new_m = fmaxf(local.m_val, x);
    local.l_val = local.l_val * expf(local.m_val - new_m) + expf(x - new_m);
    local.m_val = new_m;
  }

  local = warp_softmax(local);
  if (lane == 0)
    smem[warp] = local;
  __syncthreads();

  if (warp == 0) {
    state block_state;
    block_state.m_val = lane < 8 ? smem[lane].m_val : -FLT_MAX;
    block_state.l_val = lane < 8 ? smem[lane].l_val : 0.0f;

    block_state = warp_softmax(block_state);
    if (lane == 0)
      smem[0] = block_state;
  }
  __syncthreads();
  const state row_state = smem[0];

  for (int col = tid; col < cols; col += blockDim.x) {
    const float x = input[row * cols + col];
    output[row * cols + col] = expf(x - row_state.m_val) / row_state.l_val;
  }
}
int main() {
  constexpr int rows = 7;
  constexpr int cols = 1003; // Deliberately not divisible by the block size.
  constexpr int threads = 256;
  const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

  std::vector<float> input(rows * cols);
  std::vector<float> output(rows * cols);
  std::vector<float> reference(rows * cols);

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      // A wide value range makes an unstabilized softmax overflow easily.
      input[row * cols + col] =
          20.0f * std::sin(0.01f * static_cast<float>(row * cols + col)) +
          static_cast<float>((col % 17) - 8);
    }
  }

  for (int row = 0; row < rows; ++row) {
    float max_value = -FLT_MAX;
    for (int col = 0; col < cols; ++col) {
      max_value = fmaxf(max_value, input[row * cols + col]);
    }

    double normalizer = 0.0;
    for (int col = 0; col < cols; ++col) {
      normalizer +=
          std::exp(static_cast<double>(input[row * cols + col]) - max_value);
    }
    for (int col = 0; col < cols; ++col) {
      reference[row * cols + col] = static_cast<float>(
          std::exp(static_cast<double>(input[row * cols + col]) - max_value) /
          normalizer);
    }
  }

  float *d_input = nullptr;
  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

  online_softmax_2d<<<rows, threads>>>(d_input, d_output, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(
      cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    max_error = fmaxf(max_error, std::fabs(output[i] - reference[i]));
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  std::printf("online softmax: rows=%d, cols=%d, max error=%.3e\n", rows, cols,
              max_error);
  if (max_error > 1e-5f) {
    std::fprintf(stderr, "verification failed\n");
    return EXIT_FAILURE;
  }
  std::printf("verification passed\n");
  return EXIT_SUCCESS;
}
