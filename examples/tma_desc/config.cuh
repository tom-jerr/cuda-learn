#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <array>

namespace tma_demo {
// Logical rows/columns differ from the allocated row pitch. The padding makes
// accidentally passing N*sizeof(float) as the row stride observable.
constexpr int M=32, N=96, LD=112, TM=8, TN=32;
constexpr int TileElements=TM*TN, TileBytes=TileElements*sizeof(float);
constexpr int Tiles=(M/TM)*(N/TN);
static_assert(M%TM==0 && N%TN==0 && LD>=N);
static_assert(LD*sizeof(float)%16==0 && TN*sizeof(float)==128);

inline void cuda_check(cudaError_t e,const char* call) {
  if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",call,cudaGetErrorString(e)); std::exit(1); }
}
inline void driver_check(CUresult e,const char* call) {
  if(e!=CUDA_SUCCESS) {
    const char* message=nullptr; cuGetErrorString(e,&message);
    std::fprintf(stderr,"%s: %s (%d)\n",call,message?message:"unknown",int(e)); std::exit(1);
  }
}

// This formula requires a 1024-byte-aligned shared tile base and 32 float/row.
// SW128 permutes eight 16-byte groups; it never permutes within a 4-float group.
__host__ __device__ constexpr int shared_offset(int row,int col,bool sw128) {
  return row*TN+(sw128 ? (col^(4*(row%8))) : col);
}

inline void explain_layout() {
  std::printf("Logical A(row,col)=(%d,%d), float strides=(%d,1)\n",M,N,LD);
  std::printf("TMA order=(col,row): globalDim={%d,%d}, globalStrides={%zu} bytes\n",N,M,LD*sizeof(float));
  std::printf("boxDim={%d,%d}, elementStrides={1,1}, tile=%d bytes, tiles=%d\n",TN,TM,TileBytes,Tiles);
  std::puts("Global: +1 col = +4 B; +1 row = +448 B; +8 rows = +3584 B.");
  std::puts("Shared NONE: +1 col = +4 B; +1 row = +128 B. Reuse one 1024 B tile.");
  std::puts("SW128: logical row -> physical 16-byte group order:");
  for(int row=0;row<TM;++row) {
    std::printf("row %d:",row);
    for(int group=0;group<8;++group) std::printf(" %d",shared_offset(row,4*group,true)%TN/4);
    std::puts("");
  }
  // Enumerate all physical locations and all global tile coordinates.
  std::array<int,TileElements> smem_seen{};
  std::array<int,M*LD> global_seen{};
  for(int row=0;row<TM;++row) for(int col=0;col<TN;++col) {
    int p=shared_offset(row,col,true);
    if(p<0 || p>=TileElements) std::abort();
    ++smem_seen[p];
  }
  for(int tile=0;tile<Tiles;++tile) {
    int row0=(tile/(N/TN))*TM, col0=(tile%(N/TN))*TN;
    std::printf("tile %2d: TMA coordinate=(%2d,%2d), wait phase=%d\n",tile,col0,row0,tile%2);
    for(int row=0;row<TM;++row) for(int col=0;col<TN;++col)
      ++global_seen[(row0+row)*LD+col0+col];
  }
  for(int n:smem_seen) if(n!=1) std::abort();
  for(int r=0;r<M;++r) for(int c=0;c<LD;++c)
    if(global_seen[r*LD+c]!=(c<N ? 1 : 0)) std::abort();
  std::puts("CPU LAYOUT PASS: swizzle bijection, all logical elements once, padding untouched");
}

// CUtensorMap is 128 opaque bytes, aligned to at least 64 bytes at the API call.
// It is NOT the 64-bit shared-memory descriptor consumed by WGMMA.
inline CUresult encode(CUtensorMap& map,void* base,bool sw128) {
  const cuuint64_t dims[2]={N,M};              // ELEMENTS; contiguous dim first
  const cuuint64_t strides[1]={LD*sizeof(float)}; // BYTES; dim0 stride is implicit
  const cuuint32_t box[2]={TN,TM};             // ELEMENTS in one transfer
  const cuuint32_t element_steps[2]={1,1};     // sampling steps, not byte pitches
  return cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,
      base,dims,strides,box,element_steps,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      sw128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace tma_demo
