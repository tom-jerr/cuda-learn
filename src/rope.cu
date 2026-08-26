#include "ffi_common.h"

#include <cstdint>
#include <limits>

using tvm::ffi::TensorView;

namespace {

// Apply two independent split-half rotations. The two half2 lanes correspond
// to frequencies i and i + 1; they are not the two coordinates of one pair.
__device__ __forceinline__ void rotate_half2(half *x, int i, int rotary_half,
                                             half2 cos2, half2 sin2) {
  const half2 x0 = *reinterpret_cast<const half2 *>(x + i);
  const half2 x1 =
      *reinterpret_cast<const half2 *>(x + rotary_half + i);

  const half2 y0 = __hfma2(x0, cos2, __hneg2(__hmul2(x1, sin2)));
  const half2 y1 = __hfma2(x0, sin2, __hmul2(x1, cos2));

  *reinterpret_cast<half2 *>(x + i) = y0;
  *reinterpret_cast<half2 *>(x + rotary_half + i) = y1;
}

__global__ void rope_neox_fp16_kernel(
    half *__restrict__ q, half *__restrict__ k,
    const half *__restrict__ cos_cache, const half *__restrict__ sin_cache,
    const int *__restrict__ position_ids, int num_tokens, int num_q_heads,
    int num_kv_heads, int head_dim, int rotary_half, int max_position) {
  const int token = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  if (token >= num_tokens) {
    return;
  }

  const int position = position_ids[token];
  // Validating device values in the host wrapper would synchronize the stream.
  // Keep malformed input memory-safe here; valid positions remain an API
  // contract because an invalid token is deliberately left unchanged.
  if (position < 0 || position >= max_position) {
    return;
  }

  const size_t cache_offset =
      static_cast<size_t>(position) * static_cast<size_t>(rotary_half);
  const half *cos_row = cos_cache + cache_offset;
  const half *sin_row = sin_cache + cache_offset;

  half *q_head = nullptr;
  half *k_head = nullptr;
  if (head < num_q_heads) {
    const size_t q_offset =
        (static_cast<size_t>(token) * static_cast<size_t>(num_q_heads) +
         static_cast<size_t>(head)) *
        static_cast<size_t>(head_dim);
    q_head = q + q_offset;
  }
  if (head < num_kv_heads) {
    const size_t k_offset =
        (static_cast<size_t>(token) * static_cast<size_t>(num_kv_heads) +
         static_cast<size_t>(head)) *
        static_cast<size_t>(head_dim);
    k_head = k + k_offset;
  }

  // One thread processes two frequencies. With rotary_dim=128 this maps one
  // complete head to one warp and produces contiguous 4-byte transactions.
  for (int i = static_cast<int>(threadIdx.x) * 2; i < rotary_half;
       i += static_cast<int>(blockDim.x) * 2) {
    const half2 cos2 = *reinterpret_cast<const half2 *>(cos_row + i);
    const half2 sin2 = *reinterpret_cast<const half2 *>(sin_row + i);
    if (q_head != nullptr) {
      rotate_half2(q_head, i, rotary_half, cos2, sin2);
    }
    if (k_head != nullptr) {
      rotate_half2(k_head, i, rotary_half, cos2, sin2);
    }
  }
}

void check_cuda_tensor(const TensorView &tensor, const char *name, int ndim,
                       uint8_t dtype_code, uint8_t dtype_bits) {
  if (tensor.data_ptr() == nullptr) {
    TVM_FFI_THROW(RuntimeError) << "rope_neox: " << name << " is null";
  }
  if (tensor.ndim() != ndim) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: " << name << " must have " << ndim << " dimensions";
  }
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != dtype_code || dtype.bits != dtype_bits ||
      dtype.lanes != 1) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: " << name << " has an unsupported dtype";
  }
  if (tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: " << name << " must be a CUDA tensor";
  }
}

void check_same_device(const TensorView &tensor, const TensorView &q,
                       const char *name) {
  if (tensor.device().device_id != q.device().device_id) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: " << name << " must be on the same CUDA device as q";
  }
}

void check_half2_alignment(const TensorView &tensor, const char *name) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  if (address % alignof(half2) != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: " << name << " must be 4-byte aligned";
  }
}

void check_rope_args(const TensorView &q, const TensorView &k,
                     const TensorView &cos_cache,
                     const TensorView &sin_cache,
                     const TensorView &position_ids) {
  check_cuda_tensor(q, "q", 3, kDLFloat, 16);
  check_cuda_tensor(k, "k", 3, kDLFloat, 16);
  check_cuda_tensor(cos_cache, "cos_cache", 2, kDLFloat, 16);
  check_cuda_tensor(sin_cache, "sin_cache", 2, kDLFloat, 16);
  check_cuda_tensor(position_ids, "position_ids", 1, kDLInt, 32);

  check_same_device(k, q, "k");
  check_same_device(cos_cache, q, "cos_cache");
  check_same_device(sin_cache, q, "sin_cache");
  check_same_device(position_ids, q, "position_ids");

  const int64_t num_tokens = dim(q, 0);
  const int64_t num_q_heads = dim(q, 1);
  const int64_t num_kv_heads = dim(k, 1);
  const int64_t head_dim = dim(q, 2);
  const int64_t max_position = dim(cos_cache, 0);
  const int64_t rotary_half = dim(cos_cache, 1);

  if (num_tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 ||
      head_dim <= 0 || max_position <= 0 || rotary_half <= 0) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: all tensor dimensions must be positive";
  }
  if (dim(k, 0) != num_tokens || dim(k, 2) != head_dim) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: q and k must have matching token and head dimensions";
  }
  if (dim(sin_cache, 0) != max_position ||
      dim(sin_cache, 1) != rotary_half) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: cos_cache and sin_cache must have identical shapes";
  }
  if (dim(position_ids, 0) != num_tokens) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: position_ids length must equal the token count";
  }
  if (rotary_half > head_dim / 2) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: rotary_dim inferred from the cache exceeds head_dim";
  }
  if (rotary_half % 2 != 0 || head_dim % 2 != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: rotary_dim must be divisible by 4 and head_dim by 2";
  }
  if (num_tokens > std::numeric_limits<int>::max() ||
      num_q_heads > 65535 || num_kv_heads > 65535 ||
      head_dim > std::numeric_limits<int>::max() ||
      max_position > std::numeric_limits<int>::max() ||
      rotary_half > std::numeric_limits<int>::max()) {
    TVM_FFI_THROW(RuntimeError)
        << "rope_neox: tensor dimensions exceed CUDA launch/index limits";
  }

  check_half2_alignment(q, "q");
  check_half2_alignment(k, "k");
  check_half2_alignment(cos_cache, "cos_cache");
  check_half2_alignment(sin_cache, "sin_cache");

  if (q.data_ptr() == k.data_ptr()) {
    TVM_FFI_THROW(RuntimeError) << "rope_neox: q and k must not alias";
  }
}

} // namespace

void rope_neox(TensorView q, TensorView k, TensorView cos_cache,
               TensorView sin_cache, TensorView position_ids) {
  check_rope_args(q, k, cos_cache, sin_cache, position_ids);

  const int num_tokens = static_cast<int>(dim(q, 0));
  const int num_q_heads = static_cast<int>(dim(q, 1));
  const int num_kv_heads = static_cast<int>(dim(k, 1));
  const int head_dim = static_cast<int>(dim(q, 2));
  const int max_position = static_cast<int>(dim(cos_cache, 0));
  const int rotary_half = static_cast<int>(dim(cos_cache, 1));
  const int heads = num_q_heads > num_kv_heads ? num_q_heads : num_kv_heads;

  const int useful_threads = rotary_half / 2;
  int threads = ((useful_threads + 31) / 32) * 32;
  if (threads > 256) {
    threads = 256;
  }

  const dim3 grid(num_tokens, heads);
  rope_neox_fp16_kernel<<<grid, threads, 0, get_stream(q)>>>(
      static_cast<half *>(q.data_ptr()), static_cast<half *>(k.data_ptr()),
      static_cast<const half *>(cos_cache.data_ptr()),
      static_cast<const half *>(sin_cache.data_ptr()),
      static_cast<const int *>(position_ids.data_ptr()), num_tokens,
      num_q_heads, num_kv_heads, head_dim, rotary_half, max_position);
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.rope_neox", rope_neox);
