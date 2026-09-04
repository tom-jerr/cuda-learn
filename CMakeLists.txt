cmake_minimum_required(VERSION 3.22)
project(cuda_learn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 生成编译数据库供 clangd 等工具使用（build/compile_commands.json）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# RTX 4060 Laptop (sm_89)
set(CMAKE_CUDA_ARCHITECTURES 89)
option(CUDA_LEARN_ENABLE_HOPPER_FA3
       "Build the raw-PTX FlashAttention-3 reproduction for sm_90a" OFF)
option(CUDA_LEARN_ENABLE_CUTE_EXAMPLE
       "Configure the standalone CuTe Ampere teaching examples" ON)
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH
    "CUTLASS source directory (its include/ directory must contain cute/tensor.hpp)")
find_package(CUDAToolkit REQUIRED)

# ---- 定位 tvm_ffi（随 apache-tvm 一起安装，提供 FFI 头文件与 libtvm_ffi.so）----
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TVM_FFI_CMAKE_DIR
  RESULT_VARIABLE TVM_FFI_CONFIG_RC)
if(NOT TVM_FFI_CONFIG_RC EQUAL 0)
  message(FATAL_ERROR
    "tvm_ffi not found for ${Python_EXECUTABLE}; install it with: pip install apache-tvm")
endif()
find_package(tvm_ffi CONFIG REQUIRED
  PATHS "${TVM_FFI_CMAKE_DIR}" NO_DEFAULT_PATH)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --libfiles
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TVM_FFI_LIB_FILE)
get_filename_component(TVM_FFI_LIB_DIR "${TVM_FFI_LIB_FILE}" DIRECTORY)

# ---- 定位 cuDNN（优先使用当前 Python/torch 环境附带的 nvidia-cudnn）----
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "print(__import__('sysconfig').get_path('purelib') + '/nvidia/cudnn', end='')"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYTHON_CUDNN_ROOT)
find_path(CUDNN_INCLUDE_DIR cudnn.h
  HINTS "${PYTHON_CUDNN_ROOT}/include" "/usr/local/cuda/include")
if(EXISTS "${PYTHON_CUDNN_ROOT}/lib/libcudnn.so.9")
  set(CUDNN_LIBRARY "${PYTHON_CUDNN_ROOT}/lib/libcudnn.so.9")
elseif(EXISTS "${PYTHON_CUDNN_ROOT}/lib/libcudnn.so.8")
  set(CUDNN_LIBRARY "${PYTHON_CUDNN_ROOT}/lib/libcudnn.so.8")
else()
  find_library(CUDNN_LIBRARY cudnn
    HINTS "${PYTHON_CUDNN_ROOT}/lib" "/usr/local/cuda/lib64")
endif()
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
  message(FATAL_ERROR
    "cuDNN headers/library not found; install nvidia-cudnn in ${Python_EXECUTABLE}'s environment")
endif()
get_filename_component(CUDNN_LIB_DIR "${CUDNN_LIBRARY}" DIRECTORY)

include_directories("include")

# ---- 算子库：所有 kernel + FFI 注册 → libcuda_learn.so ----
file(GLOB_RECURSE CUDA_LEARN_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(cuda_learn SHARED ${CUDA_LEARN_SOURCES})
target_include_directories(cuda_learn PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
  "${CUDNN_INCLUDE_DIR}")
set_target_properties(cuda_learn PROPERTIES
  OUTPUT_NAME "cuda_learn"
  # nvcc emits includes inherited from imported targets as -isystem=<path>.
  # clangd's CUDA driver treats the '=' as part of the directory name, so it
  # cannot find dlpack/dlpack.h and reports cascading errors for all TVM types.
  # Regular -I<path> flags are understood by both nvcc and clangd.
  NO_SYSTEM_FROM_IMPORTED ON
  # libtvm_ffi.so 通常在 import tvm 时已被 RTLD_GLOBAL 加载，
  # RUNPATH 只作为兜底（先 dlopen 插件后 import tvm 的顺序）；
  # /usr/local/cuda/lib64 提供 nvcc 13 依赖的 libcudart.so.13。
  BUILD_RPATH "${TVM_FFI_LIB_DIR};${CUDNN_LIB_DIR};/usr/local/cuda/lib64")
target_link_libraries(cuda_learn PRIVATE
  tvm_ffi::shared CUDA::cudart CUDA::cublas "${CUDNN_LIBRARY}")

# Standalone correctness/performance driver for the coalesced + cp.async
# SGEMV implementation.  It is opt-in so the normal shared-library build stays
# lightweight:
#   cmake --build build --target sgemv_async_bench
add_executable(sgemv_async_bench EXCLUDE_FROM_ALL
  src/sgemv_async.cu
  benchmarks/sgemv_async_bench.cu)
target_compile_definitions(sgemv_async_bench PRIVATE CUDA_LEARN_SGEMV_NO_FFI=1)
target_include_directories(sgemv_async_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/include")
target_link_libraries(sgemv_async_bench PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(sgemv_async_bench PROPERTIES
  CUDA_ARCHITECTURES 89
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")

# CMake 3.22 does not understand the architecture suffix in "90a".  Keep the
# normal sm_89 build untouched and use nvcc's native flag for a separate Hopper
# build directory.  WGMMA requires the architecture-specific sm_90a target,
# not the forward-compatible sm_90 target.
if(CUDA_LEARN_ENABLE_HOPPER_FA3)
  set_target_properties(cuda_learn PROPERTIES CUDA_ARCHITECTURES OFF)
  target_compile_options(cuda_learn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_90a,code=sm_90a>)
  target_compile_definitions(cuda_learn PRIVATE CUDA_LEARN_HAS_SM90A=1)
  target_link_libraries(cuda_learn PRIVATE CUDA::cuda_driver)
endif()

# CuTe is the header-only layout/tensor layer shipped in the CUTLASS submodule.
# Configuring this target emits the exact include path into compile_commands.json
# so clangd can parse the CUDA templates.  The executable is excluded from the
# default build and remains an explicit teaching target:
#   cmake --build build-cute --target cute_hgemm cute_fa2
if(CUDA_LEARN_ENABLE_CUTE_EXAMPLE)
  find_path(CUTLASS_INCLUDE_DIR cute/tensor.hpp
    HINTS "${CUTLASS_DIR}/include"
    NO_DEFAULT_PATH)
  if(NOT CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR
      "cute/tensor.hpp not found. Initialize the dependency with: "
      "git submodule update --init --recursive")
  endif()

  add_executable(cute_hgemm EXCLUDE_FROM_ALL examples/cute_gemm/simple.cu)
  target_include_directories(cute_hgemm PRIVATE
    "${CUTLASS_INCLUDE_DIR}"
    # CUDA 13 moved libcudacxx/CUB/Thrust under include/cccl. nvcc searches it
    # implicitly; spelling it out is required by clangd's CUDA parser.
    "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  target_link_libraries(cute_hgemm PRIVATE CUDA::cudart)
  set_target_properties(cute_hgemm PROPERTIES
    CUDA_ARCHITECTURES 89
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/cute_gemm")

  add_executable(cute_fa2 EXCLUDE_FROM_ALL examples/cute_fa2/simple.cu)
  target_include_directories(cute_fa2 PRIVATE
    "${CUTLASS_INCLUDE_DIR}"
    "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  target_link_libraries(cute_fa2 PRIVATE CUDA::cudart)
  set_target_properties(cute_fa2 PROPERTIES
    CUDA_ARCHITECTURES 89
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/cute_fa2")

  add_executable(cute_fa2_4096 EXCLUDE_FROM_ALL
    examples/cute_fa2/fa2_4096.cu)
  target_include_directories(cute_fa2_4096 PRIVATE
    "${CUTLASS_INCLUDE_DIR}"
    "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  target_link_libraries(cute_fa2_4096 PRIVATE CUDA::cudart)
  set_target_properties(cute_fa2_4096 PROPERTIES
    CUDA_ARCHITECTURES 89
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/cute_fa2")
endif()
