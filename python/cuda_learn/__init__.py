"""cuda_learn — CUDA 学习 kernels，经 TVM FFI 暴露为统一 pytorch binding。

import cuda_learn 时自动 dlopen build/libcuda_learn.so 并注册全部
"cuda_learn.<op>" 全局函数。使用方式：

    import torch
    from cuda_learn import ops

    a = torch.rand(1024, 1024, device="cuda")
    b = torch.rand(1024, 1024, device="cuda")
    c = ops.gemm(a, b)   # 与 torch.matmul 同形态的调用
"""

from . import _ffi  # noqa: F401  # 副作用：加载 libcuda_learn.so
from . import ops
from .ops import (  # noqa: F401
    cublas_gemm,
    cublas_gemm_bf16,
    cudnn_softmax,
    dot_product,
    flash_attn,
    flash_attn_optimized,
    flash_attn_swizzled,
    gemm,
    rmsnorm_and_add,
    silu_and_mul,
    softmax,
    transpose,
    transpose_tile,
    vector_add,
    vector_add_grid_loop,
    vector_add_vectorized,
)

__all__ = [
    "ops",
    "cublas_gemm",
    "cublas_gemm_bf16",
    "cudnn_softmax",
    "vector_add",
    "vector_add_grid_loop",
    "vector_add_vectorized",
    "transpose",
    "transpose_tile",
    "dot_product",
    "gemm",
    "flash_attn",
    "flash_attn_optimized",
    "flash_attn_swizzled",
    "silu_and_mul",
    "rmsnorm_and_add",
    "softmax",
]
