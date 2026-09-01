# AllReduce 后处理与通信-计算融合

本文讨论两种经常被统称为“通算融合”的技术：

1. **collective epilogue fusion**：`AllReduce -> residual/bias/norm/quant` 合成一个 kernel；
2. **tiled communication-computation overlap**：把 GEMM 与 `AllGather`、`ReduceScatter` 或 `AllReduce` 分块、流水和重叠。

本仓库的复现属于第一类：FP32、2 ranks、1-shot push、无 CUTLASS，将
`AllReduce -> residual add -> RMSNorm -> affine weight` 合为一个 CUDA kernel。

## 1. 为什么 AllReduce 后面的操作能融合

对 rank `r` 的输入 `x[r, i]`，sum AllReduce 的定义是：

```text
a[i] = sum_r x[r, i]
```

若后继是逐元素函数：

```text
z[i] = f(a[i], residual[i], bias[i], scale[i], ...)
```

那么 `a[i]` 一旦收齐所有 rank 的贡献，负责 `i` 的线程就可以立即计算
`f`。它不需要等待 `a[j]`，也不需要先把 `a[i]` 写入一个全局内存中间
tensor，再由下一个 kernel 读回来。这是最直接的可融合条件：

```text
拆分：remote reduce -> store a[i] -> kernel boundary -> load a[i] -> f -> store z[i]
融合：remote reduce -> keep a[i] in register -> f -> store z[i]
```

以 FP32 为例，仅 `a` 这个中间 tensor 就少了每元素 4 B 写和 4 B 读，另外
少一次 kernel launch。AllReduce 的 P2P/NVLink 通信量本身没有改变；这是本地 HBM
流量和调度开销优化，不是通信算法复杂度优化。

CUDA 把 register、shared memory 放在 SM 上，而 global memory 位于 device
memory；这正是“寄存器续接后处理”能减少全局内存往返的硬件基础。参见
[CUDA Programming Guide: memory spaces](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#gpu-device-memory-spaces)。

### RMSNorm 不是纯 element-wise，但仍然常被融合

RMSNorm 为：

```text
p[i] = a[i] + residual[i]
s    = sum_j p[j]^2
y[i] = p[i] * rsqrt(s / H + eps) * gamma[i]
```

其中 residual add 和最后的 affine multiply 是逐元素操作，但 `s` 是整行
归约。只要一个 CTA（或一个可同步的 cluster）拥有一整行，线程就可以：

1. 在寄存器中保留 `p[i]`；
2. 将每线程平方和归约到 shared memory；
3. 用 `__syncthreads()` 保证 CTA 内归约完成；
4. 计算 `rsqrt` 并乘 `gamma[i]`。

CUDA 文档明确规定 `__syncthreads()` 同步一个 block 内的线程并为参与线程
提供内存顺序。因此“一行一个 CTA”是最简单的正确映射。跨 CTA 的一行则需
cluster DSMEM、cooperative launch、原子计数器加第二阶段，或重新切分工作；不能
直接把 `__syncthreads()` 当 grid barrier。参见
[CUDA block synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#thread-block-synchronization-functions)。

## 2. 常见融合模式

| 模式 | 依赖范围 | 常见位置 | 主要收益/约束 |
|---|---|---|---|
| AR + bias / residual / scale | element | attention output projection、MLP down projection 后 | 最容易融合；辅助 tensor 的 rank 语义必须正确 |
| AR + residual + RMSNorm | row | Transformer block 边界 | 少 AR 中间 tensor；需要行归约 |
| AR + residual + LayerNorm | row | 部分 pre/post-norm 网络 | 同时归约 mean 和 variance，代价比 RMSNorm 高 |
| AR + norm + FP8/FP4/INT8 quant | row、group 或 tensor | 下一层量化 GEMM 之前 | 同时省 norm 输出重读；动态 scale 还要做 amax 归约 |
| AR + activation / gate | element 或小组 | 模型结构恰好把激活放在 TP 聚合之后时 | SiLU/GELU/clamp/cast 很适合；不能擅自跨 AR 交换顺序 |
| MoE finalize + AR + residual + norm | token/row | TP+EP MoE 输出 | 避免先物化 top-k gather/weighted-sum 结果，实现更复杂 |
| GEMM + AR | tile | 小 M、decode 等 | producer tile 完成后直接进入 collective；既可省中间访问又可重叠 |
| AG + GEMM / GEMM + RS | tile | sequence parallel、较大 prefill/training shape | 通信和 GEMM pipeline；需要精细 tile 调度与 SM 配额 |

Dropout 也能从数据依赖角度融合，但训练时还必须保持 RNG offset、mask 输出和
重计算语义，所以它远不如 inference 下的 residual/bias/norm 融合直接。

## 3. 开源实现对照

### SGLang

SGLang 当前的 Kimi K3 通信 kernel 同时提供 push 和 NVLS pull 两族实现，源码
明确说明 epilogue 可融合 residual add 或 RMSNorm；它还要求 residual 在各 rank
上相同，否则每个 rank 得到的最终复制结果会不同：

- [K3 `ar_fusion.cuh`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh)
- [K3 Python 接口与 epilogue contract](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/ops/kimi_k3/all_reduce.py)

它的 push-norm 路径把一行拆给 cluster 中多个 CTA，用 warp reduction、cluster
distributed shared memory 和 `cluster.sync()` 合并平方和；pull-norm 路径则在
`multimem.ld_reduce` 得到归约 vector 后立刻累计平方和、计算 RMS factor 并写回。
这说明“AR 后融合 norm”的关键并非 norm 是 element-wise，而是能把 norm 的
row dependency 放入同一个可同步执行域。

SGLang 也有 FlashInfer 路径的公开 benchmark，直接比较拆分路径与
`AllReduce + Residual Add + RMSNorm + optional quantization`，并同时测试 one-shot
和 two-shot：
[SGLang FlashInfer fusion benchmark](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/benchmark/kernels/flashinfer_allreduce_fusion/README.md)。

### FlashInfer / TensorRT-LLM

FlashInfer 的 pattern enum 展示了生产实现中常见的组合：

- AR + residual + RMSNorm；
- 再接 FP8 / FP4 quant；
- 保留 norm side output 的量化版本；
- dynamic per-token FP8 或 per-token-group FP8；
- MoE reduction/finalize + AR + residual + RMSNorm。

见 [FlashInfer `trtllm_ar.py`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/trtllm_ar.py)。

TensorRT-LLM 的实现则直接展示了 `int4` 128-bit vector load、bias/residual add、
block sum-of-squares、`rsqrtf`、affine weight 与最终 vector store；还包含 Gemma2
所需的 pre/post RMSNorm 变体：
[TensorRT-LLM customAllReduceKernels.cu](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/customAllReduceKernels.cu)。

## 4. 相关论文与更广义的通算融合

- **CoCoNeT, ASPLOS 2022**：把 computation 和 communication 都提升为 DSL
  的一等构造，从编译器层做 fusion 与 overlap，指出传统二者的抽象边界会丢失
  跨边界优化机会。见 [Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/breaking-the-computation-and-communication-abstraction-barrier-in-distributed-machine-learning-workloads/)。
- **FLUX, 2024**：把通信和计算过度切分成更细粒度工作，再融合进较大的 GPU
  kernel 以隐藏依赖通信延迟。论文报告单个融合 kernel 最多可重叠 96% 通信；
  见 [paper](https://arxiv.org/abs/2406.06858) 与
  [official code](https://github.com/bytedance/flux)。
- **Triton-distributed**：开源实现包括 `AllGather-GEMM`、`GEMM-ReduceScatter`
  和 `GEMM-AllReduce`。其 Qwen3 示例指出较大 shape 更适合 AG-GEMM + GEMM-RS
  流水，而小 shape 拆分通信的额外开销可能超过重叠收益，直接 GEMM-AR 更合适：
  [end-to-end dense example](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/getting-started/e2e/e2e_dense.md)。

第一类 epilogue fusion 主要消除中间 tensor 和 launch；第二类 tiled fusion 还试图
把链路等待藏在 Tensor Core 计算之后。二者可以叠加，但工程复杂度差别很大。

## 5. 本仓库 example

源码：[`examples/custom_all_reduce/fused_residual_rmsnorm.cu`](../examples/custom_all_reduce/fused_residual_rmsnorm.cu)

### 数据路径

```text
rank input
   │
   ├── 128-bit st.relaxed.sys push 到每个 destination workspace
   │
local workspace 轮询两个 source slot（+0 empty / -0 payload）
   │
   ├── FP32 reduce，结果留在 Vec16 寄存器
   ├── residual add，同时输出 residual stream
   ├── shared-memory sum(x²) + __syncthreads
   └── rsqrt + gamma，输出 norm stream
```

关键点：

- 固定 `world_size=2`、`hidden=1024`，一行由一个 256-thread CTA 处理；
- 每线程处理一个 16-byte `float4`；
- P2P store/load 使用 `.sys` scope，使内存操作可在系统内其他 GPU 上观察；CUDA
  的 `.sys` scope 对应 system thread scope，参见
  [CUDA thread scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#thread-scopes)；
- 复用 SGLang/TensorRT-LLM 风格的 `+0/-0` Lamport payload marker；
- baseline 是两个 kernel，并物化 `all_reduce_out`；fused 是一个 kernel，不分配
  也不读写纯 AR 输出；
- residual 在两个逻辑 rank 上相同，保证融合后仍得到 replicated output；
- 只有一张物理 GPU 时，程序在同一个 grid 内启动两个 logical-rank blocks 做协议
  和数值验证；有两张 GPU 时自动使用 GPU 0/1 和双向 P2P。

### 编译和运行

```bash
make -C examples custom_all_reduce/fused_residual_rmsnorm
./examples/custom_all_reduce/fused_residual_rmsnorm
```

当前单卡开发环境输出：

```text
Only one GPU found: running two logical ranks on GPU 0.
fused vs CPU max error: 3.57628e-07
fused vs split baseline max error: 0
PASS: FP32 1-shot push + residual + RMSNorm
```

生成 PTX 后可确认跨 GPU 数据通路保留了 system-scope vector 指令：

```bash
nvcc -std=c++17 -O2 -arch=sm_89 -ptx \
  examples/custom_all_reduce/fused_residual_rmsnorm.cu -o /tmp/fused.ptx
rg 'st\.relaxed\.sys|ld\.relaxed\.sys' /tmp/fused.ptx
```

## 6. 什么时候不能直接融合

1. **运算次序不等价**：通常 `f(sum_r x_r) != sum_r f(x_r)`。能把 `f` 放在
   AR 后面，不代表能把它移动到 AR 前面。
2. **rank-local 辅助输入语义不一致**：若 residual 本应 replicated，却在各 rank
   不同，融合后各 rank 最终输出也不同，破坏 AllReduce 后的 replicated contract。
3. **依赖范围超过执行域**：RMSNorm/LayerNorm/amax 如果跨 CTA，却没有合法的
   grid/cluster 同步，就会读到不完整统计量。
4. **寄存器或 shared-memory 压力过大**：后处理越多，occupancy 越可能下降；
   一个“大而全” kernel 不必然更快。
5. **后继有多个消费者**：如果其他分支确实需要纯 `all_reduce_out`，它仍要被物化；
   此时融合只能减少一部分读取或 launch。
6. **量化 scale 的粒度不匹配**：static per-tensor scale 几乎是纯 element-wise；
   dynamic per-token/group scale 需要额外 amax reduction 和 scale 输出。
7. **shape 太大或太小**：太大时一行难以由一个同步域高效处理；太小时额外的
   cluster、分块和流水控制可能压过收益。最终策略应以目标 topology 和 shape
   的 benchmark 为准。
