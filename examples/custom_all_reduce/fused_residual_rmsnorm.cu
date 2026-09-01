#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// CUTLASS-free, FP32 teaching example:
//
//   baseline: 1-shot push AllReduce -> residual add + RMSNorm
//   fused:    1-shot push AllReduce + residual add + RMSNorm
//
// The fused path never materializes the AllReduce-only result. It keeps each
// reduced float4 in registers, adds the residual, participates in a row-wise
// sum-of-squares reduction, then applies rsqrt and the affine RMSNorm weight.
//
// For readability this example fixes world_size=2 and one 1024-element row.
// With one physical GPU it runs two logical ranks in separate streams, which
// makes the protocol and numerical check executable on a development laptop.

namespace {

constexpr int kWorld = 2;
constexpr int kHidden = 1024;
constexpr int kVecCount = kHidden / 4;
constexpr float kEps = 1.0e-5f;

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

// +0 means "empty". Turn a payload's exact +0 words into -0 so that all
// arithmetic values, including zero, can be distinguished from an empty slot.
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

__device__ __forceinline__ bool contains_empty(const Vec16 &value) {
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

__device__ __forceinline__ float sum_squares(const Vec16 &a) {
  return a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
}

struct PushParams {
  const Vec16 *input;
  Vec16 *output;
  Vec16 *workspaces[kWorld];
  unsigned int *epoch;
  int rank;
};

__device__ __forceinline__ Vec16 push_then_poll(const PushParams &params,
                                                int vector_id,
                                                unsigned int phase) {
  const Vec16 encoded = encode_payload(params.input[vector_id]);

#pragma unroll
  for (int destination = 0; destination < kWorld; ++destination) {
    Vec16 *slot = params.workspaces[destination] +
                  (phase * kWorld + params.rank) * kVecCount;
    store_relaxed_sys(slot, vector_id, encoded);
  }

  const Vec16 *sources[kWorld];
#pragma unroll
  for (int source = 0; source < kWorld; ++source) {
    sources[source] = params.workspaces[params.rank] +
                      (phase * kWorld + source) * kVecCount;
  }

  Vec16 values[kWorld];
  do {
#pragma unroll
    for (int source = 0; source < kWorld; ++source)
      values[source] = load_relaxed_sys(sources[source], vector_id);
  } while (contains_empty(values[0]) || contains_empty(values[1]));

  const Vec16 empty{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int source = 0; source < kWorld; ++source)
    store_local(const_cast<Vec16 *>(sources[source]), vector_id, empty);

  return add(values[0], values[1]);
}

// Baseline stage 1: materialize the pure AllReduce output in global memory.
__global__ __launch_bounds__(kVecCount)
void one_shot_push_all_reduce(PushParams params) {
  const int vector_id = threadIdx.x;
  const unsigned int phase = *params.epoch & 1u;
  params.output[vector_id] = push_then_poll(params, vector_id, phase);

  __syncthreads();
  if (threadIdx.x == 0)
    *params.epoch = phase ^ 1u;
}

// A single grid is used for one-GPU protocol emulation. Separate spinning
// kernels in different streams are not guaranteed to execute concurrently.
__global__ __launch_bounds__(kVecCount)
void emulated_one_shot_push_all_reduce(PushParams rank0, PushParams rank1) {
  const PushParams params = blockIdx.x == 0 ? rank0 : rank1;
  const int vector_id = threadIdx.x;
  const unsigned int phase = *params.epoch & 1u;
  params.output[vector_id] = push_then_poll(params, vector_id, phase);

  __syncthreads();
  if (threadIdx.x == 0)
    *params.epoch = phase ^ 1u;
}

// Baseline stage 2: read the materialized AR tensor and do the post-ops.
__global__ __launch_bounds__(kVecCount)
void residual_rmsnorm(const Vec16 *all_reduce_out, const Vec16 *residual,
                      const Vec16 *weight, Vec16 *residual_out,
                      Vec16 *norm_out) {
  __shared__ float squares[kVecCount];
  const int vector_id = threadIdx.x;

  const Vec16 pre_norm = add(all_reduce_out[vector_id], residual[vector_id]);
  residual_out[vector_id] = pre_norm;
  squares[vector_id] = sum_squares(pre_norm);
  __syncthreads();

  for (int stride = kVecCount / 2; stride > 0; stride >>= 1) {
    if (vector_id < stride)
      squares[vector_id] += squares[vector_id + stride];
    __syncthreads();
  }

  const float inverse_rms = rsqrtf(squares[0] / kHidden + kEps);
  const Vec16 gamma = weight[vector_id];
  norm_out[vector_id] =
      Vec16{pre_norm.x * inverse_rms * gamma.x,
            pre_norm.y * inverse_rms * gamma.y,
            pre_norm.z * inverse_rms * gamma.z,
            pre_norm.w * inverse_rms * gamma.w};
}

// Fused path: the AllReduce-only value exists only in `pre_norm` registers.
// One CTA owns the whole row, so the row-wise RMS reduction needs only shared
// memory and __syncthreads(), never a second global kernel or grid barrier.
__global__ __launch_bounds__(kVecCount)
void one_shot_push_residual_rmsnorm(PushParams params,
                                    const Vec16 *residual,
                                    const Vec16 *weight,
                                    Vec16 *residual_out, Vec16 *norm_out) {
  __shared__ float squares[kVecCount];
  const int vector_id = threadIdx.x;
  const unsigned int phase = *params.epoch & 1u;

  const Vec16 reduced = push_then_poll(params, vector_id, phase);
  const Vec16 pre_norm = add(reduced, residual[vector_id]);
  residual_out[vector_id] = pre_norm;
  squares[vector_id] = sum_squares(pre_norm);
  __syncthreads();

  for (int stride = kVecCount / 2; stride > 0; stride >>= 1) {
    if (vector_id < stride)
      squares[vector_id] += squares[vector_id + stride];
    __syncthreads();
  }

  const float inverse_rms = rsqrtf(squares[0] / kHidden + kEps);
  const Vec16 gamma = weight[vector_id];
  norm_out[vector_id] =
      Vec16{pre_norm.x * inverse_rms * gamma.x,
            pre_norm.y * inverse_rms * gamma.y,
            pre_norm.z * inverse_rms * gamma.z,
            pre_norm.w * inverse_rms * gamma.w};

  __syncthreads();
  if (threadIdx.x == 0)
    *params.epoch = phase ^ 1u;
}

__global__ __launch_bounds__(kVecCount)
void emulated_one_shot_push_residual_rmsnorm(
    PushParams rank0, PushParams rank1, const Vec16 *residual0,
    const Vec16 *residual1, const Vec16 *weight0, const Vec16 *weight1,
    Vec16 *residual_out0, Vec16 *residual_out1, Vec16 *norm_out0,
    Vec16 *norm_out1) {
  __shared__ float squares[kVecCount];
  const int rank = blockIdx.x;
  const PushParams params = rank == 0 ? rank0 : rank1;
  const Vec16 *residual = rank == 0 ? residual0 : residual1;
  const Vec16 *weight = rank == 0 ? weight0 : weight1;
  Vec16 *residual_out = rank == 0 ? residual_out0 : residual_out1;
  Vec16 *norm_out = rank == 0 ? norm_out0 : norm_out1;
  const int vector_id = threadIdx.x;
  const unsigned int phase = *params.epoch & 1u;

  const Vec16 reduced = push_then_poll(params, vector_id, phase);
  const Vec16 pre_norm = add(reduced, residual[vector_id]);
  residual_out[vector_id] = pre_norm;
  squares[vector_id] = sum_squares(pre_norm);
  __syncthreads();

  for (int stride = kVecCount / 2; stride > 0; stride >>= 1) {
    if (vector_id < stride)
      squares[vector_id] += squares[vector_id + stride];
    __syncthreads();
  }

  const float inverse_rms = rsqrtf(squares[0] / kHidden + kEps);
  const Vec16 gamma = weight[vector_id];
  norm_out[vector_id] =
      Vec16{pre_norm.x * inverse_rms * gamma.x,
            pre_norm.y * inverse_rms * gamma.y,
            pre_norm.z * inverse_rms * gamma.z,
            pre_norm.w * inverse_rms * gamma.w};

  __syncthreads();
  if (threadIdx.x == 0)
    *params.epoch = phase ^ 1u;
}

PushParams make_params(int rank, const std::vector<Vec16 *> &inputs,
                       const std::vector<Vec16 *> &outputs,
                       const std::vector<Vec16 *> &workspaces,
                       const std::vector<unsigned int *> &epochs) {
  PushParams params{};
  params.input = inputs[rank];
  params.output = outputs.empty() ? nullptr : outputs[rank];
  params.epoch = epochs[rank];
  params.rank = rank;
  for (int i = 0; i < kWorld; ++i)
    params.workspaces[i] = workspaces[i];
  return params;
}

void require_peer_access(const int devices[kWorld]) {
  for (int rank = 0; rank < kWorld; ++rank) {
    CUDA_CHECK(cudaSetDevice(devices[rank]));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, devices[rank]));
    if (prop.major < 7)
      throw std::runtime_error("system-scope vector PTX requires SM 7.0+");

    const int peer = devices[1 - rank];
    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, devices[rank], peer));
    if (!can_access)
      throw std::runtime_error("the two GPUs do not form a P2P clique");
    const cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled)
      (void)cudaGetLastError();
    else
      CUDA_CHECK(status);
  }
}

float max_error(const std::vector<float> &a, const std::vector<float> &b) {
  float result = 0.0f;
  for (int i = 0; i < kHidden; ++i)
    result = std::max(result, std::abs(a[i] - b[i]));
  return result;
}

} // namespace

int main() {
  try {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
      std::cout << "SKIP: no CUDA device\n";
      return 0;
    }

    const bool emulated = device_count < kWorld;
    const int devices[kWorld] = {0, emulated ? 0 : 1};
    if (emulated) {
      std::cout << "Only one GPU found: running two logical ranks on GPU 0.\n";
    } else {
      require_peer_access(devices);
    }

    constexpr size_t data_bytes = kHidden * sizeof(float);
    constexpr size_t workspace_bytes =
        2 * kWorld * kVecCount * sizeof(Vec16);

    std::vector<std::vector<float>> host_inputs(
        kWorld, std::vector<float>(kHidden));
    std::vector<float> host_residual(kHidden);
    std::vector<float> host_weight(kHidden);
    for (int i = 0; i < kHidden; ++i) {
      for (int rank = 0; rank < kWorld; ++rank) {
        host_inputs[rank][i] =
            std::sin(0.013f * i + 0.37f * rank) + 0.1f * (rank + 1);
      }
      host_residual[i] = 0.25f * std::cos(0.017f * i);
      host_weight[i] = 0.75f + 0.25f * std::sin(0.007f * i);
    }

    std::vector<Vec16 *> inputs(kWorld), residuals(kWorld), weights(kWorld);
    std::vector<Vec16 *> baseline_ar(kWorld), baseline_residual(kWorld),
        baseline_norm(kWorld), fused_residual(kWorld), fused_norm(kWorld);
    std::vector<Vec16 *> baseline_ws(kWorld), fused_ws(kWorld);
    std::vector<unsigned int *> baseline_epochs(kWorld), fused_epochs(kWorld);
    std::vector<cudaStream_t> streams(kWorld);

    for (int rank = 0; rank < kWorld; ++rank) {
      CUDA_CHECK(cudaSetDevice(devices[rank]));
      CUDA_CHECK(cudaStreamCreateWithFlags(&streams[rank],
                                           cudaStreamNonBlocking));
#define ALLOC_DATA(name) CUDA_CHECK(cudaMalloc(&name[rank], data_bytes))
      ALLOC_DATA(inputs);
      ALLOC_DATA(residuals);
      ALLOC_DATA(weights);
      ALLOC_DATA(baseline_ar);
      ALLOC_DATA(baseline_residual);
      ALLOC_DATA(baseline_norm);
      ALLOC_DATA(fused_residual);
      ALLOC_DATA(fused_norm);
#undef ALLOC_DATA
      CUDA_CHECK(cudaMalloc(&baseline_ws[rank], workspace_bytes));
      CUDA_CHECK(cudaMalloc(&fused_ws[rank], workspace_bytes));
      CUDA_CHECK(cudaMalloc(&baseline_epochs[rank], sizeof(unsigned int)));
      CUDA_CHECK(cudaMalloc(&fused_epochs[rank], sizeof(unsigned int)));

      CUDA_CHECK(cudaMemcpy(inputs[rank], host_inputs[rank].data(), data_bytes,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(residuals[rank], host_residual.data(), data_bytes,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(weights[rank], host_weight.data(), data_bytes,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(baseline_ws[rank], 0, workspace_bytes));
      CUDA_CHECK(cudaMemset(fused_ws[rank], 0, workspace_bytes));
      CUDA_CHECK(cudaMemset(baseline_epochs[rank], 0, sizeof(unsigned int)));
      CUDA_CHECK(cudaMemset(fused_epochs[rank], 0, sizeof(unsigned int)));
    }

    // Baseline: two kernels and one materialized AllReduce-only tensor.
    if (emulated) {
      const PushParams rank0 = make_params(0, inputs, baseline_ar, baseline_ws,
                                           baseline_epochs);
      const PushParams rank1 = make_params(1, inputs, baseline_ar, baseline_ws,
                                           baseline_epochs);
      emulated_one_shot_push_all_reduce<<<kWorld, kVecCount, 0, streams[0]>>>(
          rank0, rank1);
      for (int rank = 0; rank < kWorld; ++rank) {
        residual_rmsnorm<<<1, kVecCount, 0, streams[0]>>>(
            baseline_ar[rank], residuals[rank], weights[rank],
            baseline_residual[rank], baseline_norm[rank]);
      }
      CUDA_CHECK(cudaGetLastError());
    } else {
      for (int rank = 0; rank < kWorld; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        const PushParams params = make_params(rank, inputs, baseline_ar,
                                              baseline_ws, baseline_epochs);
        one_shot_push_all_reduce<<<1, kVecCount, 0, streams[rank]>>>(params);
        residual_rmsnorm<<<1, kVecCount, 0, streams[rank]>>>(
            baseline_ar[rank], residuals[rank], weights[rank],
            baseline_residual[rank], baseline_norm[rank]);
        CUDA_CHECK(cudaGetLastError());
      }
    }
    for (int rank = 0; rank < kWorld; ++rank) {
      CUDA_CHECK(cudaSetDevice(devices[rank]));
      CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    }

    // Fused: communication, residual add, row reduction and affine transform
    // are one kernel. `params.output` is deliberately unused.
    if (emulated) {
      const PushParams rank0 =
          make_params(0, inputs, {}, fused_ws, fused_epochs);
      const PushParams rank1 =
          make_params(1, inputs, {}, fused_ws, fused_epochs);
      emulated_one_shot_push_residual_rmsnorm<<<kWorld, kVecCount, 0,
                                               streams[0]>>>(
          rank0, rank1, residuals[0], residuals[1], weights[0], weights[1],
          fused_residual[0], fused_residual[1], fused_norm[0], fused_norm[1]);
      CUDA_CHECK(cudaGetLastError());
    } else {
      for (int rank = 0; rank < kWorld; ++rank) {
        CUDA_CHECK(cudaSetDevice(devices[rank]));
        const PushParams params =
            make_params(rank, inputs, {}, fused_ws, fused_epochs);
        one_shot_push_residual_rmsnorm<<<1, kVecCount, 0, streams[rank]>>>(
            params, residuals[rank], weights[rank], fused_residual[rank],
            fused_norm[rank]);
        CUDA_CHECK(cudaGetLastError());
      }
    }
    for (int rank = 0; rank < kWorld; ++rank) {
      CUDA_CHECK(cudaSetDevice(devices[rank]));
      CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    }

    std::vector<float> reference_residual(kHidden), reference_norm(kHidden);
    float square_sum = 0.0f;
    for (int i = 0; i < kHidden; ++i) {
      reference_residual[i] = host_inputs[0][i] + host_inputs[1][i] +
                              host_residual[i];
      square_sum += reference_residual[i] * reference_residual[i];
    }
    const float inverse_rms = 1.0f / std::sqrt(square_sum / kHidden + kEps);
    for (int i = 0; i < kHidden; ++i)
      reference_norm[i] =
          reference_residual[i] * inverse_rms * host_weight[i];

    float worst_reference_error = 0.0f;
    float worst_fusion_error = 0.0f;
    for (int rank = 0; rank < kWorld; ++rank) {
      std::vector<float> got_baseline_norm(kHidden), got_fused_norm(kHidden),
          got_baseline_residual(kHidden), got_fused_residual(kHidden);
      CUDA_CHECK(cudaSetDevice(devices[rank]));
      CUDA_CHECK(cudaMemcpy(got_baseline_norm.data(), baseline_norm[rank],
                            data_bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(got_fused_norm.data(), fused_norm[rank], data_bytes,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(got_baseline_residual.data(),
                            baseline_residual[rank], data_bytes,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(got_fused_residual.data(), fused_residual[rank],
                            data_bytes, cudaMemcpyDeviceToHost));

      worst_reference_error =
          std::max({worst_reference_error,
                    max_error(got_fused_norm, reference_norm),
                    max_error(got_fused_residual, reference_residual)});
      worst_fusion_error =
          std::max({worst_fusion_error,
                    max_error(got_fused_norm, got_baseline_norm),
                    max_error(got_fused_residual, got_baseline_residual)});
    }

    std::cout << "fused vs CPU max error: " << worst_reference_error << '\n'
              << "fused vs split baseline max error: " << worst_fusion_error
              << '\n';
    if (worst_reference_error > 3.0e-5f || worst_fusion_error > 1.0e-6f)
      throw std::runtime_error("numerical validation failed");
    std::cout << "PASS: FP32 1-shot push + residual + RMSNorm\n";

    for (int rank = 0; rank < kWorld; ++rank) {
      CUDA_CHECK(cudaSetDevice(devices[rank]));
      CUDA_CHECK(cudaStreamDestroy(streams[rank]));
#define FREE_DATA(name) CUDA_CHECK(cudaFree(name[rank]))
      FREE_DATA(inputs);
      FREE_DATA(residuals);
      FREE_DATA(weights);
      FREE_DATA(baseline_ar);
      FREE_DATA(baseline_residual);
      FREE_DATA(baseline_norm);
      FREE_DATA(fused_residual);
      FREE_DATA(fused_norm);
      FREE_DATA(baseline_ws);
      FREE_DATA(fused_ws);
      FREE_DATA(baseline_epochs);
      FREE_DATA(fused_epochs);
#undef FREE_DATA
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
