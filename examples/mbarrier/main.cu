// Read alongside docs/tma_descriptor_mbarrier.md.
// Inspired by reed-lau/cute-gemm/mbarrier at 37f3a01; uses supported operations,
// not the reference's invalidate-then-inspect experiment on opaque storage.
#include "primitives.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#define CUDA_CHECK(call) do { auto e=(call); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(1); } } while(0)

// CPU conceptual counters, explicitly NOT a decoder/emulator of hardware bits.
struct Model {
  int expected, pending, tx=0, phase=0;
  explicit Model(int n): expected(n),pending(n) {}
  void update(int arrivals, int expected_bytes, int completed_bytes) {
    if(arrivals>pending || completed_bytes>tx+expected_bytes)
      throw std::runtime_error("invalid teaching-model operation");
    pending-=arrivals; tx+=expected_bytes-completed_bytes;
    if(pending==0 && tx==0) { phase^=1; pending=expected; }
  }
  void print(const char* event) const {
    std::printf("%-26s pending=%d tx=%4d active_phase=%d\n",event,pending,tx,phase);
  }
};
void print_model() {
  Model m(3); m.print("init(3)");
  m.update(0,1024,0); m.print("expect_tx(1024)");
  m.update(1,0,0); m.print("arrive()");
  m.update(0,0,256); m.print("complete_tx(256)");
  m.update(2,0,0); m.print("two arrive() calls");
  m.update(0,0,512); m.print("complete_tx(512)");
  m.update(0,0,256); m.print("complete_tx(256)");
  if(m.phase!=1 || m.pending!=3 || m.tx!=0) throw std::runtime_error("model phase 0");
  m.update(1,512,0); m.print("arrive_expect_tx(512)");
  m.update(2,0,0); m.print("two arrive() calls");
  m.update(0,0,512); m.print("complete_tx(512)");
  if(m.phase!=0 || m.pending!=3 || m.tx!=0) throw std::runtime_error("model phase 1");
  std::puts("CPU MODEL PASS (does not test GPU transaction instructions)");
}

__global__ void arrival_visibility(int* sums) {
  __shared__ barrier_demo::Barrier ready;
  __shared__ int values[128];
  if(threadIdx.x==0) barrier_demo::init(&ready,128);
  __syncthreads(); // Publish initialization before any thread uses the barrier.
  unsigned phase=0;
  for(int round=0;round<6;++round) {
    values[threadIdx.x]=round*1000+int(threadIdx.x);
    barrier_demo::arrive(&ready); // Exactly 128 arrivals; no transaction bytes.
    barrier_demo::wait(&ready,phase);
    // Arrival release + successful wait acquire publishes every values[] store.
    if(threadIdx.x==0) {
      int sum=0;
      for(int i=0;i<128;++i) sum+=values[i];
      sums[round]=sum;
    }
    __syncthreads(); // The reader finished: next round may overwrite values[].
    phase^=1; // Change only after waiting for the old phase to complete.
  }
  if(threadIdx.x==0) barrier_demo::invalidate(&ready);
}

__global__ void transaction_accounting(int* states) {
#if __CUDA_ARCH__ >= 900
  __shared__ barrier_demo::Barrier ready;
  barrier_demo::init(&ready,3); // Launched with ONE thread, expected count is 3.
  barrier_demo::expect_tx(&ready,1024);
  barrier_demo::arrive(&ready);
  barrier_demo::complete_tx(&ready,256);
  states[0]=barrier_demo::test_wait(&ready,0); // still missing 2 arrivals/768 B
  barrier_demo::arrive(&ready); barrier_demo::arrive(&ready);
  states[1]=barrier_demo::test_wait(&ready,0); // arrivals done, bytes still pending
  barrier_demo::complete_tx(&ready,512);
  states[2]=barrier_demo::test_wait(&ready,0); // still missing 256 B
  barrier_demo::complete_tx(&ready,256);
  barrier_demo::wait(&ready,0); states[3]=1;  // phase 0 completed
  // Count automatically resets to 3; next active phase is 1.
  barrier_demo::arrive_expect_tx(&ready,512);
  barrier_demo::arrive(&ready); barrier_demo::arrive(&ready);
  states[4]=barrier_demo::test_wait(&ready,1); // missing 512 B
  barrier_demo::complete_tx(&ready,512);
  barrier_demo::wait(&ready,1); states[5]=1;
  barrier_demo::invalidate(&ready); // No more wait/arrive on this object.
#endif
}

int main(int argc,char** argv) {
  if(argc>2 || (argc==2 && std::strcmp(argv[1],"--model")!=0)) {
    std::fprintf(stderr,"Usage: %s [--model]\n",argv[0]); return 1;
  }
  print_model();
  if(argc==2) return 0;
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  if(prop.major<8) { std::puts("SKIP: arrival demo requires SM80+"); return 77; }
  int* result=nullptr; CUDA_CHECK(cudaMalloc(&result,6*sizeof(int)));
  arrival_visibility<<<1,128>>>(result);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  int host[6]; CUDA_CHECK(cudaMemcpy(host,result,sizeof(host),cudaMemcpyDeviceToHost));
  for(int round=0;round<6;++round) {
    if(host[round]!=128*round*1000+127*128/2) {
      std::fprintf(stderr,"FAIL: arrival visibility round %d\n",round); return 1;
    }
  }
  std::puts("GPU ARRIVAL PASS: 128 threads, 6 phases, shared-memory visibility");
  if(prop.major<9) {
    std::puts("SKIP: expect_tx/complete_tx experiment requires SM90+ (arrival check passed)");
  } else {
    transaction_accounting<<<1,1>>>(result);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host,result,sizeof(host),cudaMemcpyDeviceToHost));
    const int expected[6]={0,0,0,1,0,1};
    for(int i=0;i<6;++i) if(host[i]!=expected[i]) {
      std::fprintf(stderr,"FAIL: transaction observation %d\n",i); return 1;
    }
    std::puts("GPU TRANSACTION PASS: arrivals and bytes gate completion independently");
  }
  CUDA_CHECK(cudaFree(result));
}
