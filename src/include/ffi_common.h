// ffi_common.h — TVM FFI 绑定的公共设施。
//
// TVM 0.26 起 FFI 独立为 tvm_ffi 包，经典 TVM_REGISTER_GLOBAL / packed_func.h
// 已不存在。新机制（见 tvm_ffi/include/tvm/ffi/reflection/registry.h）：
//   refl::GlobalDef().def("name", func)  +  TVM_FFI_STATIC_INIT_BLOCK()
// 后者展开为 __attribute__((constructor))，dlopen 插件时自动执行注册。
// 进程内所有插件共享 libtvm_ffi.so 中唯一的全局注册表，Python 侧
// tvm.get_global_func("name") 即可取到。
//
// 函数形参用 tvm::ffi::TensorView（按值传递的 DLTensor 视图，官方
// cubin_launcher 示例同款），标量参数用 double / int64_t。
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/base_details.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/reflection/registry.h>

// CUDA 调用检查（沿用原示例的 CHECK_CUDA 风格）。
#define CUDA_LEARN_CHECK(call)                                                 \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      TVM_FFI_THROW(RuntimeError)                                              \
          << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "           \
          << cudaGetErrorString(error);                                        \
    }                                                                          \
  } while (0)

// 参数校验：必须是连续、fp32、指定 device 的 tensor。
// torch 侧经 __dlpack__ 传入的张量天然满足（分配器 512B 对齐 >= 64B 要求），
// 连续性是 tvm_ffi from_dlpack 的硬性检查，非连续张量在 Python 侧已
// .contiguous()。
inline void check_tensor(const tvm::ffi::TensorView &t, const char *name,
                         int ndim, DLDeviceType dev_type = kDLCUDA) {
  if (t.data_ptr() == nullptr) {
    TVM_FFI_THROW(RuntimeError) << name << ": null tensor";
  }
  if (t.ndim() != ndim) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected " << ndim << " dims, got " << t.ndim();
  }
  if (t.dtype().code != kDLFloat || t.dtype().bits != 32) {
    TVM_FFI_THROW(RuntimeError) << name << ": expected float32";
  }
  if (t.device().device_type != dev_type) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected device type " << dev_type << ", got "
        << static_cast<int>(t.device().device_type);
  }
}

// 取 torch 当前流：Python 侧 `with tvm_ffi.use_torch_stream():` 把 torch 当前流
// 写入 FFI 线程局部环境，这里取回，保证 kernel 与 torch 算子/Event 计时同流。
inline cudaStream_t get_stream(const tvm::ffi::TensorView &t) {
  return reinterpret_cast<cudaStream_t>(
      TVMFFIEnvGetStream(t.device().device_type, t.device().device_id));
}

inline int64_t dim(const tvm::ffi::TensorView &t, int i) {
  return t.shape()[i];
}

// 复刻经典 TVM_REGISTER_GLOBAL 形态的注册宏。
// 每个 op 文件末尾：CUDA_LEARN_REGISTER("cuda_learn.gemm", gemm);
#define CUDA_LEARN_REGISTER(NAME, FN)                                          \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                \
    ::tvm::ffi::reflection::GlobalDef().def(NAME, FN);                         \
  }
