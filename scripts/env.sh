# 运行 cuda_learn 前 source 此文件：
#   source scripts/env.sh
# 作用：让动态加载器能找到系统 CUDA 13 的 libcudart.so.13
#（我们的 libcuda_learn.so 由 nvcc 13 编译；torch 自带的是 cudart 12.8）。
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
