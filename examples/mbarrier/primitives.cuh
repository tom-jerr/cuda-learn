#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Minimal PTX wrappers shared by the mbarrier and TMA lessons.
// mbarrier is opaque: only instructions below may access it while valid.
namespace barrier_demo {
struct alignas(8) Barrier { uint64_t opaque; };

__device__ __forceinline__ uint32_t shared_address(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void init(Barrier* b, uint32_t arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(shared_address(b)), "r"(arrivals) : "memory");
}
// One arrival, not one byte. Does not itself wait for phase completion.
__device__ __forceinline__ void arrive(Barrier* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :: "r"(shared_address(b)) : "memory");
}
__device__ __forceinline__ bool test_wait(Barrier* b, uint32_t phase) {
  uint32_t done;
  asm volatile("{ .reg .pred p;\n"
               "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "selp.u32 %0, 1, 0, p; }"
               : "=r"(done) : "r"(shared_address(b)), "r"(phase) : "memory");
  return done;
}
__device__ __forceinline__ void wait(Barrier* b, uint32_t phase) {
  // Wait for completion of the supplied phase. Never read the opaque bitfield.
  uint32_t done;
  do {
#if __CUDA_ARCH__ >= 900
    asm volatile("{ .reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(shared_address(b)), "r"(phase) : "memory");
#else
    done = test_wait(b, phase); // test_wait is available on Ampere/Ada too.
#endif
  } while (!done);
}
// Only after all users and outstanding async transfers have finished.
__device__ __forceinline__ void invalidate(Barrier* b) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
               :: "r"(shared_address(b)) : "memory");
}

#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void expect_tx(Barrier* b, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
               :: "r"(shared_address(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void arrive_expect_tx(Barrier* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(shared_address(b)), "r"(bytes) : "memory");
}
// Accounting experiment only: no bytes are copied by this instruction.
// Real TMA complete_tx::bytes updates the barrier itself; don't double-count.
__device__ __forceinline__ void complete_tx(Barrier* b, uint32_t bytes) {
  asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
               :: "r"(shared_address(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void async_shared_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
#endif
} // namespace barrier_demo
