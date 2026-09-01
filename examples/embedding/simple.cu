/*
Embedding lookup（前向查表）的面试简化版

输入：
  table[V, D]：词表，row-major 存储
  ids[N]     ：N 个 token id
输出：
  output[N, D]

公式：
  output[i, j] = table[ids[i], j],  0 <= ids[i] < V
  output[i, j] = 0,                 ids[i] 越界时

并行分工（launch: <<<N, 128>>>）：
  grid  ：整个 grid 完成 N 个 token 的查表。
  block ：blockIdx.x 对应第 i 个 token，整个 block 复制 table[ids[i], :]。
  warp  ：block 内每个 warp 每轮协作复制连续 32 个 embedding 元素；同一
          warp 的全局内存访问连续，便于合并访问。warp 之间处理不同的列。
  thread：threadIdx.x = t 负责列 j = t, t + blockDim.x, ...；每个线程只做
          若干次 load + store，不需要 shared memory、原子操作或同步。
          vec4 版中线程 t 负责从 j = 4*t 开始的连续 4 个 float。

说明：
  - 这是 embedding forward，不含 backward。反向会把相同 id 的梯度累加回
    table，需要 atomicAdd、按 id 排序/分段归约等额外处理。
  - 重复 id 没有写冲突：它们读取同一行，但写入 output 的不同行。
  - 该 kernel 主要受显存带宽限制；生产实现还会考虑向量化读写、数据类型、
    padding_idx、超大 batch/grid-stride 等问题。
*/

#include <cuda_runtime.h>

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

__global__ void embedding_kernel(const float *table, const int *ids,
                                 float *output, int vocab_size,
                                 int embedding_dim) {
  const int token = blockIdx.x;
  const int id = ids[token];
  float *destination = output + static_cast<size_t>(token) * embedding_dim;

  // 同一个 block 看到相同的 id，因此该分支在 block 内不会产生分歧。
  if (id < 0 || id >= vocab_size) {
    for (int column = threadIdx.x; column < embedding_dim;
         column += blockDim.x) {
      destination[column] = 0.0f;
    }
    return;
  }

  const float *source = table + static_cast<size_t>(id) * embedding_dim;
  for (int column = threadIdx.x; column < embedding_dim; column += blockDim.x) {
    destination[column] = source[column];
  }
}

__global__ void embedding_vec4_kernel(const float *table, const int *ids,
                                      float *output, int vocab_size,
                                      int embedding_dim) {
  const int token = blockIdx.x;
  const int id = ids[token];
  float *destination = output + static_cast<size_t>(token) * embedding_dim;

  // 越界 id 仍然输出 0。这个判断对整个 block 一致，不会产生线程分歧。
  if (id < 0 || id >= vocab_size) {
    for (int column = threadIdx.x; column < embedding_dim;
         column += blockDim.x) {
      destination[column] = 0.0f;
    }
    return;
  }

  const float *source = table + static_cast<size_t>(id) * embedding_dim;

  // cudaMalloc 返回的首地址满足 float4 的 16-byte 对齐要求；当 D % 4 == 0
  // 时，每一行的起始地址仍然对齐，可以安全地使用 float4 load/store。
  if (embedding_dim % 4 == 0) {
    const float4 *source4 = reinterpret_cast<const float4 *>(source);
    float4 *destination4 = reinterpret_cast<float4 *>(destination);
    const int num_vec4 = embedding_dim / 4;

    for (int index = threadIdx.x; index < num_vec4; index += blockDim.x) {
      destination4[index] = source4[index];
    }
    return;
  }
}

void embedding(const float *table, const int *ids, float *output,
               int num_tokens, int vocab_size, int embedding_dim) {
  constexpr int threads = 128;
  embedding_kernel<<<num_tokens, threads>>>(table, ids, output, vocab_size,
                                            embedding_dim);
}

void embedding_vec4(const float *table, const int *ids, float *output,
                    int num_tokens, int vocab_size, int embedding_dim) {
  constexpr int threads = 128;
  embedding_vec4_kernel<<<num_tokens, threads>>>(table, ids, output, vocab_size,
                                                 embedding_dim);
}

int main() {
  constexpr int vocab_size = 8;
  constexpr int num_tokens = 6;
  // 4 的倍数保证走 float4 路径；不是 128*4 的倍数，覆盖线程边界判断。
  constexpr int embedding_dim = 512;

  std::vector<float> table(vocab_size * embedding_dim);
  for (int row = 0; row < vocab_size; ++row) {
    for (int column = 0; column < embedding_dim; ++column) {
      table[row * embedding_dim + column] = row * 1000.0f + column;
    }
  }

  // 包含重复 id，并用 -1 / vocab_size 检查越界 id 输出为 0。
  const std::vector<int> ids = {3, 1, 3, 7, -1, vocab_size};
  std::vector<float> output(num_tokens * embedding_dim);

  float *d_table = nullptr;
  int *d_ids = nullptr;
  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_table, table.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ids, ids.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  embedding_vec4(d_table, d_ids, d_output, num_tokens, vocab_size,
                 embedding_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int token = 0; token < num_tokens; ++token) {
    for (int column = 0; column < embedding_dim; ++column) {
      const int id = ids[token];
      const float expected = (id >= 0 && id < vocab_size)
                                 ? table[id * embedding_dim + column]
                                 : 0.0f;
      const float actual = output[token * embedding_dim + column];
      if (actual != expected) {
        std::fprintf(stderr,
                     "mismatch at token=%d, column=%d: expected=%g, got=%g\n",
                     token, column, expected, actual);
        return EXIT_FAILURE;
      }
    }
  }

  std::printf("Embedding vec4 lookup: PASS\n");
  std::printf("token 0 (id=%d), first 8 values: ", ids[0]);
  for (int column = 0; column < 8; ++column) {
    std::printf("%g%s", output[column], column == 7 ? "\n" : " ");
  }

  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_ids));
  CUDA_CHECK(cudaFree(d_output));
  return 0;
}
