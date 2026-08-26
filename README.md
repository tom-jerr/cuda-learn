# cuda_learn

CUDA kernel 学习仓库：手写 kernels 通过 **TVM FFI** 暴露为统一的 **PyTorch binding**，
以测试函数名为 key 一键完成正确性校验与性能 benchmark。

## 架构

```
手写 CUDA kernel (src/*.cu)
      │  每文件末尾 CUDA_LEARN_REGISTER("cuda_learn.<op>", fn)
      ▼
libcuda_learn.so ── GlobalDef().def() + constructor ──► libtvm_ffi.so 全局注册表
      │  tvm.runtime.load_module() dlopen（import cuda_learn 时自动）
      ▼
tvm.get_global_func("cuda_learn.<op>")
      │  torch.Tensor ──__dlpack__──► DLPack 零拷贝；use_torch_stream 同步流
      ▼
cuda_learn.ops.<op>(a, b) -> torch.Tensor   （统一 pytorch binding 形式）
      │
      ▼
tests/test_<op>() + @bench 装饰器 ──► python -m cuda_learn.bench test_gemm
```

关键点（TVM 0.26 FFI 新架构，与网上旧教程的 `TVM_REGISTER_GLOBAL`/`tvm.nd` 不同）：

- FFI 独立为 pip 包 `tvm_ffi`（apache-tvm 自动安装）；头文件随包提供，无需检出 TVM 源码；
- C++ 注册：`tvm::ffi::reflection::GlobalDef().def(name, fn)` + `TVM_FFI_STATIC_INIT_BLOCK()`
  （`__attribute__((constructor))`，dlopen 时执行），封装见 [src/ffi_common.h](src/ffi_common.h) 的
  `CUDA_LEARN_REGISTER` 宏；
- 函数形参用 `tvm::ffi::TensorView`（DLTensor 视图，按值传），标量用 `double`/`int64_t`；
- stream：Python `tvm_ffi.use_torch_stream()` 写入 FFI 线程局部环境，C++ `TVMFFIEnvGetStream`
  取回，kernel 与 torch 算子/Event 计时同流；
- 只链接 `libtvm_ffi.so`（不链接 libtvm_runtime*.so）；构建后 `nm -D` 检查插件 UND 符号
  应只含 `TVMFFI*` C API 与系统常规符号。

## 环境准备

- GPU：RTX 4060 Laptop (sm_89)；nvcc 13.0；cmake ≥ 3.22
- venv（本项目用 `~/.python/miniinfer`，已有 torch 2.10+cu128）：

```bash
source ~/.python/miniinfer/bin/activate
pip install apache-tvm==0.26.0          # 自动带上 tvm_ffi>=0.1.13.post2
pip install -e python                    # 安装 cuda_learn Python 包
```

## 构建

```bash
cmake -B build && cmake --build build -j   # 产物 build/libcuda_learn.so
```

`CMakeLists.txt` 通过 `python -m tvm_ffi.config` 定位头文件与 libtvm_ffi.so，
CUDA arch 固定为 89（改架构时改 `CMAKE_CUDA_ARCHITECTURES`）。

## Benchmark（按测试函数名驱动）

```bash
source scripts/env.sh                     # LD_LIBRARY_PATH 加 /usr/local/cuda/lib64（cudart13）
source ~/.python/miniinfer/bin/activate

python -m cuda_learn.bench --list         # 列出全部测试
python -m cuda_learn.bench test_gemm      # 单个：正确性 + 计时
python -m cuda_learn.bench gemm           # 支持唯一前缀/子串匹配
python -m cuda_learn.bench --check-only test_gemm      # 只校验正确性
python -m cuda_learn.bench --warmup 20 --iters 200 test_gemm
python -m cuda_learn.bench                # 无参数 = 跑全部
```

机制：`tests/test_ops.py` 中每个 `test_<op>()` 用 `@bench(make_inputs=..., ref=...,
flops=..., rtol=..., atol=...)` 注册元数据；测试函数体只做 op 调用。运行器先跑一次
正确性（与 torch 参考实现 `ref` 对比，在计时窗之外），再用 CUDA Event 计时
（warmup + iters，与 kernel 同流、一次 sync），输出 min/median/mean 与 GFLOPS/TFLOPS。
GEMM 会在相同输入下对比 FP32 `cublasSgemm`，BF16 MMA 会对比
`cublasGemmEx`，softmax 会对比 FP32
`cudnnSoftmaxForward`，flash_attn 会对比稀疏检出的 flash-attention 2.8.4
FP16/D64 forward kernel，并输出 `cuda_learn speedup`（大于 1 表示手写 kernel 更快）。
GEMM/softmax 库基线都直接调用 NVIDIA API，而不是用 PyTorch 算子名称推断后端。
flash-attention 基线首次运行时会把官方 D64 forward specialization 编译到 torch
extension cache；只裁剪无关 dtype/head-dim/反向编译单元，被测 kernel 源码不变。

当前 benchmark（19 个）：vector_add ×3、transpose ×2、dot_product、gemm（tiled sgemm）、
gemm_mma（BF16 Tensor Core）、flash_attn ×5（shared-memory 教学版、PAD=8 8-warp 和
XOR-swizzled 4-warp 的 non-causal/causal，FP16 [B,H,N,64]）、
silu_and_mul、rmsnorm（宽行、小 hidden、边界形状）、rmsnorm_and_add、softmax。

## 直接用 ops（pytorch binding 形式）

```python
import torch
import cuda_learn

a, b = torch.rand(1024, 1024, device="cuda"), torch.rand(1024, 1024, device="cuda")
c = cuda_learn.gemm(a, b)          # 与 torch.matmul 同形态
d = cuda_learn.vector_add(a, b)
```

约束：输入须为连续 fp32 CUDA 张量（封装已自动 `.contiguous()`）；
输出由 Python 预分配（统一 out-param 形式）。注意 import 时若 `libcuda_learn.so`
未构建或未 source env.sh（cudart13），会直接报错。

## 新增算子 checklist

1. `src/<op>.cu`：kernel（原样）+ host wrapper（`TensorView` 参数 + `check_tensor` +
   `get_stream` + launch），末尾 `CUDA_LEARN_REGISTER("cuda_learn.<op>", <op>);`；
2. 在 `CMakeLists.txt` 的 `CUDA_LEARN_SOURCES` 加该文件；
3. `python/cuda_learn/ops.py` 加薄封装（`_contig` + 预分配 out + `call(...)`），
   `__init__.py` 导出；
4. `python/cuda_learn/tests/test_ops.py` 加 `@bench(...)` 装饰的 `test_<op>()`；
5. `cmake --build build -j`，然后 `python -m cuda_learn.bench test_<op>`。

## 目录

- `docs/` — [Flash Attention 完整学习文档](docs/flash_attention.md)（背景、初始版、优化版、causal 与性能对比），
  [RMSNorm 学习文档](docs/rmsnorm.md)（与 LayerNorm 的差异、small hidden size 优化与 profiler）
- `src/` — kernels + FFI 注册（`ffi_common.h` 公共设施）
- `python/cuda_learn/` — Python 包：`ops.py`（binding）、`bench.py`（@bench + 运行器）、`tests/`
- `examples/` — 原始 standalone 演示（含 Makefile 一键重编）
- `scripts/env.sh` — 运行前 source（cudart13 搜索路径）
- `ampere_kernels/` — Ampere GEMM/GEMV/FA2 源码地图（只读学习材料）
- `third_party/` — LeetCUDA、flash-attention 稀疏检出（不动）

## 已知边界

- flash_attn kernel 是 forward-only、无 dropout，head dim 硬编码为 64，N 须为 64 的倍数；
  `flash_attn_optimized` 将 output/online-softmax state 放在寄存器，用 4-lane subgroup
  并行 softmax，使用 128×128 tile 和单-stage K/V 分阶段流水，并支持可选 causal
  tile skip/mask；输出复用 Q shared tile 做 layout conversion，并使用 128-bit global store；
- `flash_attn_swizzled` 使用 4 warp × 32 rows/warp 和 48 KiB XOR-swizzled Q/K/V layout，
  在 sm_89 上目标为 2 CTA/SM；
- 计时为端到端（含 launch 开销）；FFI env stream 是线程局部的，bench 单线程；
- cuda_allocator / cuda_graph 为运行时机制演示，非 tensor-op 形态，暂留在
  `examples/`（未纳入绑定系统）；
- `examples/gemm/mma.cu` 为空，留作 mma 迁移占位。
