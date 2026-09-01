/*
KV cache scatter store：把当前 step 的 K/V 按 indices 写入 cache。

输入：
  k[L, D], v[L, D]                 当前 step 的 K/V
  indices[L]                       token 对应的 cache 物理行号
输出：
  k_cache[cache_capacity, D]
  v_cache[cache_capacity, D]

公式：
  k_cache[indices[token], d] = k[token, d]
  v_cache[indices[token], d] = v[token, d]

并行分工（launch: <<<L, threads>>>）：
  grid  ：整个 grid 完成 L 个 token 的 scatter store。
  block ：blockIdx.x 对应一个 token，相当于 Triton 版本的一个 program。
  thread：线程负责 d = threadIdx.x, threadIdx.x + blockDim.x, ...。
          因此 D 大于 blockDim.x 时也能处理，D 不是 2 的幂时也无需额外 padding。

与参考 Triton kernel 的对应关系：
  tl.program_id(0)       <=> blockIdx.x
  tl.arange(0, BLOCK_D)  <=> threadIdx.x（以及 thread-stride loop）
  mask_d                 <=> d < D

约束：
  - stride 的单位是元素，不是字节；K、V、K cache、V cache 分别传 stride。
  - indices 越界时跳过该 token，防止越界写。
  - indices 应互不重复；若两个 token 写同一物理行，会发生数据竞争，最终值未定义。
  - 这是单层、单个 KV head 被展平为 [L, D] 的教学版。多层/多 head 只需把
    batch、layer、head 对应的偏移加入指针或扩展 grid 维度。
*/

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                          \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

template <typename T>
__global__ void store_kv_cache_kernel(
    const T *__restrict__ k, const T *__restrict__ v,
    T *__restrict__ k_cache, T *__restrict__ v_cache,
    const int64_t *__restrict__ indices, int64_t stride_k_l,
    int64_t stride_k_d, int64_t stride_v_l, int64_t stride_v_d,
    int64_t stride_k_cache_x, int64_t stride_k_cache_d,
    int64_t stride_v_cache_x, int64_t stride_v_cache_d, int L, int D,
    int cache_capacity) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= L) {
    return;
  }

  const int64_t target = indices[token];
  if (target < 0 || target >= cache_capacity) {
    return;
  }

  for (int d = static_cast<int>(threadIdx.x); d < D;
       d += static_cast<int>(blockDim.x)) {
    const int64_t k_input_offset =
        static_cast<int64_t>(token) * stride_k_l + d * stride_k_d;
    const int64_t v_input_offset =
        static_cast<int64_t>(token) * stride_v_l + d * stride_v_d;
    const int64_t k_cache_offset =
        target * stride_k_cache_x + d * stride_k_cache_d;
    const int64_t v_cache_offset =
        target * stride_v_cache_x + d * stride_v_cache_d;

    k_cache[k_cache_offset] = k[k_input_offset];
    v_cache[v_cache_offset] = v[v_input_offset];
  }
}

// 参数顺序与题目中的 Python wrapper 一致：cache、indices、当前 step 的 K/V。
template <typename T>
void store_kv_cache(T *k_cache, T *v_cache, const int64_t *indices,
                    const T *k, const T *v, int L, int D,
                    int cache_capacity, int64_t stride_k_l,
                    int64_t stride_k_d, int64_t stride_v_l,
                    int64_t stride_v_d, int64_t stride_k_cache_x,
                    int64_t stride_k_cache_d, int64_t stride_v_cache_x,
                    int64_t stride_v_cache_d, cudaStream_t stream = nullptr) {
  if (L == 0 || D == 0) {
    return;
  }

  // head_dim 常见为 64/128/256。取不超过 256 的 2 次幂线程数；D 很大时由
  // thread-stride loop 继续处理。至少一个 warp，避免为很小的 D 发射零散线程。
  int threads = 32;
  while (threads < D && threads < 256) {
    threads *= 2;
  }

  store_kv_cache_kernel<<<L, threads, 0, stream>>>(
      k, v, k_cache, v_cache, indices, stride_k_l, stride_k_d, stride_v_l,
      stride_v_d, stride_k_cache_x, stride_k_cache_d, stride_v_cache_x,
      stride_v_cache_d, L, D, cache_capacity);
}

int main() {
  constexpr int L = 5;
  constexpr int D = 70; // 非 2 的幂，用来覆盖边界判断。
  constexpr int cache_capacity = 11;
  constexpr float sentinel = -9999.0f;

  // 每个矩阵故意使用不同的 leading dimension，验证 kernel 没有错误地假设
  // K/V 或两个 cache 的 stride 相同。
  constexpr int64_t stride_k_l = D + 3;
  constexpr int64_t stride_v_l = D + 5;
  constexpr int64_t stride_k_cache_x = D + 7;
  constexpr int64_t stride_v_cache_x = D + 9;
  constexpr int64_t stride_d = 1;

  std::vector<float> h_k(L * stride_k_l, sentinel);
  std::vector<float> h_v(L * stride_v_l, sentinel);
  std::vector<float> h_k_cache(cache_capacity * stride_k_cache_x, sentinel);
  std::vector<float> h_v_cache(cache_capacity * stride_v_cache_x, sentinel);
  const std::vector<int64_t> h_indices = {7, 2, 10, 0, 5};

  for (int token = 0; token < L; ++token) {
    for (int d = 0; d < D; ++d) {
      h_k[token * stride_k_l + d] = token * 1000.0f + d;
      h_v[token * stride_v_l + d] = 10000.0f + token * 1000.0f + d;
    }
  }

  float *d_k = nullptr;
  float *d_v = nullptr;
  float *d_k_cache = nullptr;
  float *d_v_cache = nullptr;
  int64_t *d_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k_cache, h_k_cache.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v_cache, h_v_cache.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_indices, h_indices.size() * sizeof(int64_t)));

  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(),
                        h_k_cache.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(),
                        h_v_cache.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(),
                        h_indices.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice));

  store_kv_cache(d_k_cache, d_v_cache, d_indices, d_k, d_v, L, D,
                 cache_capacity, stride_k_l, stride_d, stride_v_l, stride_d,
                 stride_k_cache_x, stride_d, stride_v_cache_x, stride_d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_k_cache.data(), d_k_cache,
                        h_k_cache.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_v_cache.data(), d_v_cache,
                        h_v_cache.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // CPU reference：先保持 sentinel，再 scatter 写入有效的 D 个元素。
  std::vector<float> expected_k(cache_capacity * stride_k_cache_x, sentinel);
  std::vector<float> expected_v(cache_capacity * stride_v_cache_x, sentinel);
  for (int token = 0; token < L; ++token) {
    const int64_t target = h_indices[token];
    for (int d = 0; d < D; ++d) {
      expected_k[target * stride_k_cache_x + d] =
          h_k[token * stride_k_l + d];
      expected_v[target * stride_v_cache_x + d] =
          h_v[token * stride_v_l + d];
    }
  }

  for (size_t i = 0; i < expected_k.size(); ++i) {
    if (h_k_cache[i] != expected_k[i]) {
      std::fprintf(stderr, "K cache mismatch at flat index %zu: "
                           "expected=%g, got=%g\n",
                   i, expected_k[i], h_k_cache[i]);
      return EXIT_FAILURE;
    }
  }
  for (size_t i = 0; i < expected_v.size(); ++i) {
    if (h_v_cache[i] != expected_v[i]) {
      std::fprintf(stderr, "V cache mismatch at flat index %zu: "
                           "expected=%g, got=%g\n",
                   i, expected_v[i], h_v_cache[i]);
      return EXIT_FAILURE;
    }
  }

  std::printf("KV cache scatter store: PASS\n");
  std::printf("token 0 -> cache row %lld, first 8 K values: ",
              static_cast<long long>(h_indices[0]));
  for (int d = 0; d < 8; ++d) {
    std::printf("%g%s", h_k_cache[h_indices[0] * stride_k_cache_x + d],
                d == 7 ? "\n" : " ");
  }

  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_k_cache));
  CUDA_CHECK(cudaFree(d_v_cache));
  CUDA_CHECK(cudaFree(d_indices));
  return 0;
}
