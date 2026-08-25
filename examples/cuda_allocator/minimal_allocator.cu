#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t error = (call);                                                 \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                \
                   cudaGetErrorString(error));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// 显式 pool 本身只提供 stream-ordered API；同步版本就是在调用后等待 stream。
void* alloc_sync(cudaMemPool_t pool, size_t bytes, cudaStream_t stream) {
  void* pointer = nullptr;
  CHECK(cudaMallocFromPoolAsync(&pointer, bytes, pool, stream));
  CHECK(cudaStreamSynchronize(stream));
  return pointer;
}

void free_sync(void* pointer, cudaStream_t stream) {
  CHECK(cudaFreeAsync(pointer, stream));
  CHECK(cudaStreamSynchronize(stream));
}

int main() {
  constexpr size_t bytes = 16 << 20;
  constexpr int device = 0;

  CHECK(cudaSetDevice(device));

  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = device;
  props.handleTypes = cudaMemHandleTypeNone;

  cudaMemPool_t pool;
  cudaStream_t stream;
  CHECK(cudaMemPoolCreate(&pool, &props));
  CHECK(cudaStreamCreate(&stream));

  // 同步 alloc -> use -> free：每一步都等待完成。
  void* sync_pointer = alloc_sync(pool, bytes, stream);
  CHECK(cudaMemsetAsync(sync_pointer, 1, bytes, stream));
  CHECK(cudaStreamSynchronize(stream));
  free_sync(sync_pointer, stream);
  std::puts("sync alloc/free: PASS");

  // 异步 alloc -> use -> free：全部进入同一个 stream，最后只同步一次。
  void* async_pointer = nullptr;
  CHECK(cudaMallocFromPoolAsync(&async_pointer, bytes, pool, stream));
  CHECK(cudaMemsetAsync(async_pointer, 2, bytes, stream));
  CHECK(cudaFreeAsync(async_pointer, stream));
  CHECK(cudaStreamSynchronize(stream));
  std::puts("async alloc/free: PASS");

  CHECK(cudaMemPoolDestroy(pool));
  CHECK(cudaStreamDestroy(stream));
  return 0;
}
