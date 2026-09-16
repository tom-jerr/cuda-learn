// Descriptor parameters and layout: CPU inspection, optional official encoding.
#include "config.cuh"
#include <cstring>

int main(int argc,char** argv) {
  if(argc>2 || (argc==2 && std::strcmp(argv[1],"--encode")!=0 && std::strcmp(argv[1],"--layout")!=0)) {
    std::fprintf(stderr,"Usage: %s [--layout|--encode]\n",argv[0]); return 1;
  }
  tma_demo::explain_layout();
  if(argc==1 || std::strcmp(argv[1],"--layout")==0) return 0;
  // Official encoding on a real allocation. Successful encoding does not prove
  // that the current GPU can execute TMA; the separate copy demo checks SM90.
  tma_demo::driver_check(cuInit(0),"cuInit");
  float* data=nullptr;
  tma_demo::cuda_check(cudaMalloc(&data,tma_demo::M*tma_demo::LD*sizeof(float)),"cudaMalloc");
  for(bool sw128:{false,true}) {
    alignas(64) CUtensorMap map{};
    CUresult status=tma_demo::encode(map,data,sw128);
    if(status==CUDA_ERROR_NOT_SUPPORTED) {
      tma_demo::cuda_check(cudaFree(data),"cudaFree");
      std::puts("SKIP: this driver/device does not support cuTensorMapEncodeTiled (801); CPU layout passed");
      return 77;
    }
    tma_demo::driver_check(status,"cuTensorMapEncodeTiled");
    std::printf("ENCODE PASS: %s, sizeof(CUtensorMap)=%zu (not a TMA execution test)\n",
                sw128?"SW128":"NONE",sizeof(map));
  }
  tma_demo::cuda_check(cudaFree(data),"cudaFree");
}
