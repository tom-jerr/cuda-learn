#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Small, explicit wrappers around the three Ampere instructions used by the
// examples. Keeping them here makes the kernels read like a data-flow graph:
//
//   HBM --cp.async--> shared memory --ldmatrix--> registers --mma.sync--> regs

namespace ampere {

__device__ __forceinline__ uint32_t shared_u32(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Copy exactly 16 bytes from global memory to shared memory. On sm_80 this is
// an asynchronous copy: the values do not pass through programmer-visible
// registers. cg means cache in L2, bypassing L1 for this transfer.
__device__ __forceinline__ void cp_async_16(uint32_t dst_smem,
                                            const void *src_gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(dst_smem), "l"(src_gmem));
}

// Predicated 16-byte copy. A zero source size suppresses the global-memory
// access and zero-fills the whole shared-memory destination. This keeps tiled
// kernels vectorized while handling a final partial tile safely.
__device__ __forceinline__ void cp_async_16_zfill(uint32_t dst_smem,
                                                  const void *src_gmem,
                                                  bool valid) {
  const int src_bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(dst_smem), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}

// One warp cooperatively loads four 8x8 half matrices (256 half values total)
// from shared memory into 4 x 32-bit registers per lane. The four 8x8 pieces
// form the 16x16 A fragment consumed by mma.m16n8k16.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&dst)[4],
                                            uint32_t src_smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
               "{%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
               : "r"(src_smem));
}

// Load two 8x8 half matrices. This is the natural load for a Kx8 B fragment.
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&dst)[2],
                                            uint32_t src_smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(dst[0]), "=r"(dst[1])
               : "r"(src_smem));
}

// Same physical load, but transpose every 8x8 matrix while moving it to the
// register fragment. This is useful when B is stored row-major in shared
// memory while mma.sync expects its B operand in column-major fragment form.
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&dst)[2],
                                                  uint32_t src_smem) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(dst[0]), "=r"(dst[1])
      : "r"(src_smem));
}

// Warp-level matrix multiply:
//   A fragment: 16x16 fp16, 4 registers/lane
//   B fragment: 16x8  fp16, 2 registers/lane
//   C/D:        16x8  fp32, 4 registers/lane
// Across 32 lanes this performs 16*8*16 fused multiply-adds.
__device__ __forceinline__ void mma_m16n8k16_f32(float (&d)[4],
                                                 const uint32_t (&a)[4],
                                                 const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

// BF16 variant with the same fragment/register layout.
__device__ __forceinline__ void mma_m16n8k16_bf16_f32(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  return value;
}

} // namespace ampere
