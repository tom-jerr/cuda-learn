# 非 CUTLASS/CuTe 的 Hopper FlashAttention-3 复刻

对应实现：

- [`src/flash_attn3_hopper.cu`](../src/flash_attn3_hopper.cu)：attention
  mainloop、online softmax、TMA tensor map 和 TVM FFI launcher；
- [`src/include/hopper_primitives.cuh`](../src/include/hopper_primitives.cuh)：直接用
  inline PTX 封装 TMA、mbarrier、WGMMA 与 `setmaxnreg`；
- [`python/cuda_learn/ops.py`](../python/cuda_learn/ops.py)：
  `flash_attn3_hopper(q, k, v, causal=False)`；
- [`python/cuda_learn/tests/test_ops.py`](../python/cuda_learn/tests/test_ops.py)：
  causal/non-causal 正确性与性能入口。

实现不包含、不链接 CUTLASS 或 CuTe。它是一个便于逐层阅读和修改的、shape-specific
BF16 forward 复刻，不是官方 FlashAttention-3 的全功能替代品。

## 1. 已实现范围

| 项目 | 当前实现 |
|---|---|
| GPU | Hopper SM90a：H100/H800/H200 |
| dtype | Q/K/V/O 均为 BF16，softmax 与 accumulator 为 FP32 |
| layout | `[B,H,N,64]` contiguous |
| shape | `D=64`，`N>0` 且 `N%128==0` |
| attention | forward，causal/non-causal，无 dropout |
| CTA | 1 producer WG + 2 consumer WG，共 384 threads |
| Q tile | 每个 consumer 处理 `64x64`；每 CTA 处理连续 128 行 Q |
| KV tile | `64x64`，K/V 各 2-stage circular shared-memory pipeline |
| GEMM | `wgmma.mma_async.m64n64k16`；QK 为 SS，PV 为 RS |
| 搬运 | 5D TMA load/store，128B swizzle，transaction mbarrier |
| 寄存器 | producer `setmaxnreg.dec 32`；consumer `setmaxnreg.inc 160` |

尚未实现 D=128、FP16、FP8、backward、varlen、GQA/MQA、paged KV、sliding
window 与 dropout。尤其不能把当前 BF16 kernel 描述为论文的 FP8 路径：它没有 V
transpose、byte permutation、block scale 或 incoherent processing。

## 2. CTA 内调度

矩阵 tile 使用 48 KiB dynamic shared memory，另有少量 static mbarrier 状态：

```text
Q[consumer 0]  64x64 BF16  ┐
Q[consumer 1]  64x64 BF16  ├─ 16 KiB
K[stage 0..1]  2x64x64     ├─ 16 KiB
V[stage 0..1]  2x64x64     ┘  16 KiB
```

producer 单线程创建每个 stage 的 TMA transaction，`kv_ready[stage]` 在 K/V 的
16 KiB 都到达后翻转 phase。两个 consumer 完成该 stage 的 PV 后分别 arrive
`stage_empty[stage]`，producer 才能覆盖它。

两个 consumer 使用 `turn[2]` mbarrier 传递 token，交替 commit WGMMA group，而不是
完全依赖 warp scheduler 碰运气。每个 consumer 内部进一步执行两级流水：

```text
prologue:
    QK(0); wait; softmax(0)

steady state:
    issue QK(next); commit                # older WGMMA group
    issue PV(cur);  commit                # younger WGMMA group
    wgmma.wait_group<1>                   # next scores ready，PV 仍可在飞行
    softmax(next)                         # CUDA/SFU 与 PV(cur) 重叠
    wgmma.wait_group<0>
    rescale O by alpha(next)

epilogue:
    issue PV(last); wait
    normalize O and TMA store
```

online softmax 使用 `exp2`，把 `1/sqrt(64)` 与 `log2(e)` 合并为常量。causal
模式让同一 CTA 的两个 consumer 遍历相同数量的 KV tiles；第一个 consumer 的最后
一个全 future tile 被完整 mask 为 `-inf`，从而维持 stage barrier 的相同参与数，避免
条件退出造成死锁。

## 3. 构建与运行

CMake 3.22 不识别 `90a` 架构后缀，因此项目提供单独开关并直接传给 NVCC
`compute_90a -> sm_90a`。不要覆盖现有 SM89 build：

```bash
cmake -B build-hopper \
  -DCUDA_LEARN_ENABLE_HOPPER_FA3=ON \
  -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python"
cmake --build build-hopper -j
```

让 Python loader 指向 Hopper 产物后运行：

```bash
export CUDA_LEARN_LIB="$PWD/build-hopper/libcuda_learn.so"
python -m cuda_learn.bench --check-only test_flash_attn3_hopper
python -m cuda_learn.bench --check-only test_flash_attn3_hopper_causal
python -m cuda_learn.bench --warmup 20 --iters 100 test_flash_attn3_hopper
```

直接调用：

```python
import torch
from cuda_learn import flash_attn3_hopper

q, k, v = [torch.randn(2, 16, 4096, 64, device="cuda",
                       dtype=torch.bfloat16) for _ in range(3)]
o = flash_attn3_hopper(q, k, v, causal=True)
```

普通 SM89 build 仍可编译和加载；调用该 op 会明确提示使用 Hopper build。benchmark
runner 在非 compute capability 9.x GPU 上把这两个用例标为 `SKIP`。

## 4. 静态验证与 profiler 检查

即使没有 H100，也可确认生成的是目标指令而不是 `mma.sync` fallback：

```bash
cuobjdump --dump-resource-usage build-hopper/libcuda_learn.so
cuobjdump --dump-sass build-hopper/libcuda_learn.so | \
  grep -E 'HGMMA|UTMALDG|UTMASTG|SETMAXNREG|MBARRIER|WARPGROUP'
```

CUDA 13 本地离线编译结果中 kernel 为 `sm_90a`、160 registers/thread，SASS 包含
BF16 `HGMMA.64x64x16` 的 SS/RS 形态、`UTMALDG.5D` 和 `UTMASTG.5D`。由于当前开发机
是 RTX 4060/SM89，只完成了 SM90a 离线汇编验证，最终数值正确性、偶发死锁与性能必须
在 Hopper 实机上跑上述测试，并用 Nsight Compute 检查：

1. `local_load/local_store` 与 register spill；
2. Tensor Core/SFU overlap；
3. mbarrier、long scoreboard stall；
4. shared bank conflict；
5. TMA/L2/DRAM throughput；
6. 不同 N 和 causal 模式的输出误差与重复运行稳定性。

NVCC 13 目前会对 consumer/producer 的 warpgroup-uniform 分支报告可能序列化 WGMMA
的 C7520 提示。SASS 中仍能看到异步 group 与 `WARPGROUP.DEPBAR`，但在完成 H100
profiling 前，不应声称已经达到官方 FA3 或 ThunderKittens 的吞吐。

## 5. 实现依据

原语语义与 fragment/descriptor 布局以 NVIDIA
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 和
[CUDA tensor map API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
为准。CTA 角色划分、128B swizzle descriptor 和 register-tile online softmax 的设计
参考了 MIT licensed
[ThunderKittens H100 MHA](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/attention/mha_h100)，
但本项目只保留为本 kernel 所需的少量独立 PTX 封装，不引入 ThunderKittens 依赖。
