#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Minimal Hopper primitives used by flash_attn3_hopper.cu.  This header is
// intentionally self-contained: it uses CUDA/PTX directly and has no
// CUTLASS/CuTe dependency.
namespace hopper {

struct alignas(8) MBarrier {
  uint64_t value;
};

struct Accum64x64 {
  // wgmma.m64n64 with FP32 accumulators owns 32 scalar outputs per thread.
  float x[32];
};

__device__ __forceinline__ uint32_t shared_u32(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init(MBarrier *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(shared_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbarrier_expect_tx(MBarrier *bar,
                                                   uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(shared_u32(bar)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(MBarrier *bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
               :
               : "r"(shared_u32(bar))
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait(MBarrier *bar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "L_wait_%=: \n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra.uni L_wait_%=;\n"
      "}\n"
      :
      : "r"(shared_u32(bar)), "r"(phase)
      : "memory");
}

__device__ __forceinline__ void tma_load_5d(void *dst, const CUtensorMap *map,
                                            int c0, int c1, int c2, int c3,
                                            int c4, MBarrier *bar) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile."
               "mbarrier::complete_tx::bytes "
               "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
               :
               : "r"(shared_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(c2),
                 "r"(c3), "r"(c4), "r"(shared_u32(bar))
               : "memory");
}

__device__ __forceinline__ void tma_store_5d(const CUtensorMap *map, int c0,
                                             int c1, int c2, int c3, int c4,
                                             const void *src) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group "
               "[%0, {%2, %3, %4, %5, %6}], [%1];"
               :
               : "l"(map), "r"(shared_u32(src)), "r"(c0), "r"(c1), "r"(c2),
                 "r"(c3), "r"(c4)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

template <int Registers> __device__ __forceinline__ void setmaxnreg_inc() {
  static_assert(Registers % 8 == 0, "setmaxnreg requires a multiple of 8");
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(Registers));
}

template <int Registers> __device__ __forceinline__ void setmaxnreg_dec() {
  static_assert(Registers % 8 == 0, "setmaxnreg requires a multiple of 8");
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(Registers));
}

__device__ __forceinline__ uint64_t descriptor_encode(uint64_t bytes) {
  return (bytes & 0x3ffffULL) >> 4;
}

__device__ __forceinline__ uint64_t matrix_descriptor(const void *base,
                                                      uint32_t leading_bytes,
                                                      uint32_t stride_bytes,
                                                      uint32_t swizzle) {
  return descriptor_encode(reinterpret_cast<uint64_t>(base)) |
         (descriptor_encode(leading_bytes) << 16) |
         (descriptor_encode(stride_bytes) << 32) |
         (static_cast<uint64_t>(swizzle) << 62);
}

// A 64x64 BF16 tile loaded with a 128-byte TMA swizzle.  For K-major Q/K,
// four consecutive K=16 chunks lie 32 bytes apart inside one swizzle atom.
__device__ __forceinline__ uint64_t k_major_descriptor(const void *base,
                                                       int k_chunk) {
  return matrix_descriptor(base, 16, 1024, 1) +
         descriptor_encode(static_cast<uint64_t>(k_chunk) * 32);
}

// V is consumed as a KxN B operand.  In MN-major mode each K=16 chunk advances
// one 2 KiB core matrix inside the 64x64 swizzled tile.
__device__ __forceinline__ uint64_t mn_major_descriptor(const void *base,
                                                        int k_chunk) {
  return matrix_descriptor(base, 8192, 1024, 1) +
         descriptor_encode(static_cast<uint64_t>(k_chunk) * 2048);
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void wgmma_fence(Accum64x64 &accum) {
  // Tie the C++ register values to the asynchronous-proxy fence.  Without
  // these constraints nvcc may move ordinary ALU accesses to the accumulator
  // across the fence because a generic memory clobber does not name registers.
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    asm volatile("" : "+f"(accum.x[i]) : : "memory");
  }
  wgmma_fence();
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int Pending> __device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" : : "n"(Pending) : "memory");
}

#define HOPPER_ACCUM_OUTPUTS(D)                                                \
  "+f"((D).x[0]), "+f"((D).x[1]), "+f"((D).x[2]), "+f"((D).x[3]),              \
      "+f"((D).x[4]), "+f"((D).x[5]), "+f"((D).x[6]), "+f"((D).x[7]),          \
      "+f"((D).x[8]), "+f"((D).x[9]), "+f"((D).x[10]), "+f"((D).x[11]),        \
      "+f"((D).x[12]), "+f"((D).x[13]), "+f"((D).x[14]), "+f"((D).x[15]),      \
      "+f"((D).x[16]), "+f"((D).x[17]), "+f"((D).x[18]), "+f"((D).x[19]),      \
      "+f"((D).x[20]), "+f"((D).x[21]), "+f"((D).x[22]), "+f"((D).x[23]),      \
      "+f"((D).x[24]), "+f"((D).x[25]), "+f"((D).x[26]), "+f"((D).x[27]),      \
      "+f"((D).x[28]), "+f"((D).x[29]), "+f"((D).x[30]), "+f"((D).x[31])

// D = A(shared) @ B(shared)^T for one K=16 chunk.
__device__ __forceinline__ void wgmma_ss_qk(Accum64x64 &d, uint64_t a_desc,
                                            uint64_t b_desc, bool accumulate) {
  const int scale_d = accumulate ? 1 : 0;
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
               "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
               "%26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 1;\n"
               "}\n"
               : HOPPER_ACCUM_OUTPUTS(d)
               : "l"(a_desc), "l"(b_desc), "r"(scale_d));
}

// D += A(register BF16) @ B(shared) for one K=16 chunk.
__device__ __forceinline__ void wgmma_rs_pv(Accum64x64 &d, const uint32_t a[4],
                                            uint64_t b_desc) {
  const int scale_d = 1;
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
               "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, "
               "%26, %27, %28, %29, %30, %31}, "
               "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n"
               "}\n"
               : HOPPER_ACCUM_OUTPUTS(d)
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b_desc),
                 "r"(scale_d));
}

#undef HOPPER_ACCUM_OUTPUTS

} // namespace hopper
