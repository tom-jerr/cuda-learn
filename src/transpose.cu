#include "ffi_common.h"

// per thread transpose one element in matrix
__global__ void matrix_transpose_kernel(const float *input, float *output, int M,
                                        int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < N && row < M) {
    output[col * M + row] = input[row * N + col];
  }
}

// tile transpose: each thread transpose one tile and use smem to reduce global
// memory accesses
__global__ void matrix_transpose_tile_kernel(const float *input, float *output,
                                             int M, int N) {
  __shared__ float tile[16][16 + 1]; // +1 to avoid bank conflicts
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < N && row < M) {
    tile[threadIdx.y][threadIdx.x] = input[row * N + col];
  }
  __syncthreads();
  int new_col = blockIdx.y * blockDim.y + threadIdx.x;
  int new_row = blockIdx.x * blockDim.x + threadIdx.y;
  if (new_col < M && new_row < N) {
    output[new_row * M + new_col] = tile[threadIdx.x][threadIdx.y];
  }
}

using tvm::ffi::TensorView;

namespace {
void transpose_common(const TensorView& input, const TensorView& output,
                      bool tile) {
  check_tensor(input, "input", 2);
  check_tensor(output, "output", 2);
  int M = static_cast<int>(dim(input, 0));
  int N = static_cast<int>(dim(input, 1));
  if (dim(output, 0) != N || dim(output, 1) != M) {
    TVM_FFI_THROW(RuntimeError) << "transpose: output shape must be (N, M)";
  }
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  cudaStream_t stream = get_stream(input);
  if (tile) {
    matrix_transpose_tile_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float *>(input.data_ptr()),
        static_cast<float *>(output.data_ptr()), M, N);
  } else {
    matrix_transpose_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float *>(input.data_ptr()),
        static_cast<float *>(output.data_ptr()), M, N);
  }
  CUDA_LEARN_CHECK(cudaGetLastError());
}
}  // namespace

void transpose(TensorView input, TensorView output) {
  transpose_common(input, output, /*tile=*/false);
}

void transpose_tile(TensorView input, TensorView output) {
  transpose_common(input, output, /*tile=*/true);
}

CUDA_LEARN_REGISTER("cuda_learn.transpose", transpose);
CUDA_LEARN_REGISTER("cuda_learn.transpose_tile", transpose_tile);
