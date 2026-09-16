// End-to-end SM90 lesson: global -> TMA -> shared -> elementwise transform
// -> TMA -> global. One buffer / one mbarrier is reused for twelve phases.
#include "config.cuh"
#include "../mbarrier/primitives.cuh"
#include <vector>
#include <cstring>
#include <algorithm>
using namespace tma_demo;

#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void load_2d(float* dst,const CUtensorMap* map,
                                      int col,int row,barrier_demo::Barrier* ready) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
               "[%0], [%1, {%2, %3}], [%4];"
               :: "r"(barrier_demo::shared_address(dst)), "l"(map), "r"(col), "r"(row),
                  "r"(barrier_demo::shared_address(ready)) : "memory");
}
__device__ __forceinline__ void store_2d(const CUtensorMap* map,int col,int row,const float* src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
               "[%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(barrier_demo::shared_address(src)), "r"(col), "r"(row) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  // Full completion (not just .read): simple and intentionally conservative.
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
#endif

template<bool Sw128>
__global__ void copy_tiles(__grid_constant__ const CUtensorMap input,
                           __grid_constant__ const CUtensorMap output) {
#if __CUDA_ARCH__ >= 900
  __shared__ __align__(1024) float tile[TileElements];
  __shared__ barrier_demo::Barrier ready;
  if(threadIdx.x==0) {
    barrier_demo::init(&ready,1); // ONE issuer arrives; 128 consumers may wait.
    barrier_demo::async_shared_fence(); // Publish initialization to async proxy.
  }
  __syncthreads();
  unsigned phase=0;
  for(int tile_id=0;tile_id<Tiles;++tile_id) {
    int row=(tile_id/(N/TN))*TM, col=(tile_id%(N/TN))*TN;
    if(threadIdx.x==0) {
      // Reserve 8*32*4=1024 transaction bytes and consume the one arrival.
      // This must not accidentally count 128 arrivals or 128*1024 bytes.
      barrier_demo::arrive_expect_tx(&ready,TileBytes);
      load_2d(tile,&input,col,row,&ready);
    }
    barrier_demo::wait(&ready,phase); // TMA itself completes all 1024 bytes.
    phase^=1;
    for(int i=threadIdx.x;i<TileElements;i+=blockDim.x) {
      int r=i/TN,c=i%TN;
      int physical=shared_offset(r,c,Sw128);
      // Coordinate-dependent transform detects wrong swizzle indexing: applying
      // a uniform +1 everywhere would pass even if logical coordinates were wrong.
      tile[physical]+=float(1+3*r+c);
    }
    // All writers publish their generic shared stores, then synchronize before
    // thread 0 issues a store through the async proxy.
    barrier_demo::async_shared_fence();
    __syncthreads();
    if(threadIdx.x==0) store_2d(&output,col,row,tile);
    // Issuer finished the TMA store; no consumer/async reader still uses tile.
    // This is the buffer-free condition for the next iteration's TMA load.
    __syncthreads();
  }
  if(threadIdx.x==0) barrier_demo::invalidate(&ready);
#endif
}

int main(int argc,char** argv) {
  if(argc>2 || (argc==2 && std::strcmp(argv[1],"--layout")!=0)) {
    std::fprintf(stderr,"Usage: %s [--layout]\n",argv[0]); return 1;
  }
  if(argc==2) { explain_layout(); return 0; }
  cudaDeviceProp prop{}; cuda_check(cudaGetDeviceProperties(&prop,0),"cudaGetDeviceProperties");
  if(prop.major<9) {
    std::printf("SKIP: TMA needs SM90+, current GPU %s is SM%d%d\n",prop.name,prop.major,prop.minor);
    return 77;
  }
  driver_check(cuInit(0),"cuInit");
  constexpr float sentinel=-12345.f;
  std::vector<float> source(M*LD,-999.f),out(M*LD,sentinel);
  for(int r=0;r<M;++r) for(int c=0;c<N;++c) source[r*LD+c]=float(r*1000+c);
  float *in_device=nullptr,*out_device=nullptr;
  cuda_check(cudaMalloc(&in_device,source.size()*sizeof(float)),"cudaMalloc input");
  cuda_check(cudaMalloc(&out_device,out.size()*sizeof(float)),"cudaMalloc output");
  cuda_check(cudaMemcpy(in_device,source.data(),source.size()*sizeof(float),cudaMemcpyHostToDevice),"copy input");
  for(bool sw128:{false,true}) {
    std::fill(out.begin(),out.end(),sentinel);
    cuda_check(cudaMemcpy(out_device,out.data(),out.size()*sizeof(float),cudaMemcpyHostToDevice),"init output");
    alignas(64) CUtensorMap input{},output{};
    driver_check(encode(input,in_device,sw128),"encode input");
    driver_check(encode(output,out_device,sw128),"encode output");
    if(sw128) copy_tiles<true><<<1,128>>>(input,output);
    else copy_tiles<false><<<1,128>>>(input,output);
    cuda_check(cudaGetLastError(),"copy_tiles launch");
    cuda_check(cudaDeviceSynchronize(),"copy_tiles synchronize");
    cuda_check(cudaMemcpy(out.data(),out_device,out.size()*sizeof(float),cudaMemcpyDeviceToHost),"read output");
    for(int r=0;r<M;++r) for(int c=0;c<LD;++c) {
      float expected=c<N ? source[r*LD+c]+float(1+3*(r%TM)+c%TN) : sentinel;
      if(out[r*LD+c]!=expected) {
        std::fprintf(stderr,"FAIL %s (%d,%d): got %g expected %g\n",sw128?"SW128":"NONE",r,c,out[r*LD+c],expected);
        return 1;
      }
    }
    std::printf("GPU TMA PASS: %s, 12 tiles/phases, 3072 values, 512 padding values unchanged\n",sw128?"SW128":"NONE");
  }
  cuda_check(cudaFree(in_device),"free input"); cuda_check(cudaFree(out_device),"free output");
}
