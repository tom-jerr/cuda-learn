# GEMM 优化记录：tiled FP32、Tensor Core MMA 与 L2 CTA swizzle

本文整理仓库中三版 GEMM 的数据流、线程映射、资源占用和实测效果，并给出下一步
double buffering / multi-stage pipeline 的实现方案。对应源码：

- FP32 tiled 初版：[`src/gemm.cu`](../src/gemm.cu)
- BF16 Tensor Core MMA：[`src/gemm_mma.cu`](../src/gemm_mma.cu)
- MMA + L2 CTA swizzle：[`src/gemm_mma_l2.cu`](../src/gemm_mma_l2.cu)
- Ampere 指令封装：[`src/include/ampere_primitives.cuh`](../src/include/ampere_primitives.cuh)
- L2 benchmark：[`benchmarks/gemm_mma_l2.py`](../benchmarks/gemm_mma_l2.py)

三版都计算 row-major 矩阵乘：

\[
C_{M\times N}=A_{M\times K}B_{K\times N}
\]

但初版使用 FP32 输入/输出和 CUDA Core，后两版使用 BF16 输入/输出、FP32 accumulator
和 Tensor Core。因此“初版到 MMA”的耗时差异不只是代码优化，也包括数据类型、内存流量
和计算单元的变化；只有普通 MMA 与 L2 版是严格同精度、同 tile 的直接对照。

## 1. 三版总览

| 版本 | dtype | CTA tile | K tile | threads | 主要指令 | static shared | registers/thread |
|---|---|---:|---:|---:|---|---:|---:|
| tiled FP32 | FP32 → FP32 | 32×32 | 16 | 64 | FP32 FMA | 4,096 B | 68 |
| MMA | BF16 → FP32 acc → BF16 | 64×64 | 32 | 128 | `cp.async`、`ldmatrix`、`mma.sync` | 9,728 B | 78 |
| MMA + L2 | 同 MMA | 64×64 | 32 | 128 | 同 MMA，改变 CTA→tile 映射 | 9,728 B | 80 |

资源数据来自：

```bash
cuobjdump --dump-resource-usage build/libcuda_learn.so
```

L2 版多出的两个左右寄存器来自 block index 到 logical tile 的映射计算；计算主体、shared
布局和输出 fragment 映射与普通 MMA 相同。

## 2. FP32 tiled 初版

### 2.1 分块和线程映射

初版已经是 shared-memory tiled GEMM，并非每个线程直接从 global memory 完成完整 dot
product 的 naive 实现：

```cpp
BM = 32, BN = 32, BK = 16
TM = 4,  TN = 4
block = dim3(8, 8)       // 64 threads，2 warps
```

一个 CTA 计算 `C` 的 `32×32` tile。每个线程持有 `4×4=16` 个 FP32 accumulator：

```text
CTA output tile: 32 × 32

thread(0,0) owns rows 0..3, cols 0..3
thread(1,0) owns rows 0..3, cols 4..7
...
thread(7,7) owns rows 28..31, cols 28..31
```

每个 K iteration 协作加载：

```text
A tile: [32,16] FP32 = 2,048 B
B tile: [16,32] FP32 = 2,048 B
shared total          = 4,096 B
```

数据路径是：

```text
global A/B → scalar cooperative load → shared As/Bs
           → thread-local a_reg[4], b_reg[4]
           → 4×4 FP32 outer product → acc[4][4]
           → global C
```

一个 `32×32×16` stage 完成 16,384 次 FMA，即 32,768 FLOPs；读取约 4 KiB
A/B，忽略最终 C store 时算术强度约为 8 FLOP/B。相同 A/B tile 在 CTA 内被重复使用，
但 CTA 间仍依赖 L2/HBM。

### 2.2 优点与瓶颈

优点：

- 支持任意 M/N/K，边界 tile 有显式 bounds check 和 zero fill；
- shared memory 消除了同一 CTA 内大量重复 global load；
- 每线程计算 4×4 tile，增加 register reuse。

主要瓶颈：

- 使用 FP32 CUDA Core，没有使用 Tensor Core；
- global→shared 是标量 load/store，没有 16-byte vector copy；
- load、同步、compute 完全串行；
- 2-warp CTA 较小，指令和同步开销占比相对更高；
- 每个 thread 的 shared load 和地址计算较多。

## 3. BF16 Tensor Core MMA 版

### 3.1 CTA/warp/MMA 层级

MMA 版扩大到 `BM=64, BN=64, BK=32`，使用四个 warp 组成 `2×2` warp grid：

```text
CTA output [64,64]

             N: 0..31          N: 32..63
M: 0..31       warp 0             warp 1
M: 32..63      warp 2             warp 3
```

每个 warp 计算 `32×32` output tile：

- M 方向：2 个 `m16` MMA tile；
- N 方向：4 个 `n8` MMA tile；
- 每个 lane 持有 `2×4×4=32` 个 FP32 accumulator。

核心指令为：

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

一次 warp-level MMA 计算 `16×8×16` FMA。一个 CTA 的一个 `BK=32` stage 共执行 64
条 warp MMA 指令，完成 262,144 FLOPs。

### 3.2 数据路径

```text
HBM/L2
  │ cp.async.cg.shared.global，16 B/thread instruction
  ▼
shared A[64][32+8]、B[32][64+8]
  │ ldmatrix.x4 / ldmatrix.x2.trans
  ▼
warp registers
  │ mma.m16n8k16 BF16×BF16→FP32
  ▼
FP32 accumulator registers
  │ __float2bfloat16_rn
  ▼
BF16 global C
```

`B` 在 global/shared 中保持 row-major，通过 `ldmatrix.x2.trans` 变为 MMA 所需的
column-major register fragment。shared 每行增加 `PAD=8` 个 BF16 元素，即 16 B，改变
相邻行的 bank 映射：

```text
A shared = 64 × (32+8) × 2 B = 5,120 B
B shared = 32 × (64+8) × 2 B = 4,608 B
total                              9,728 B
```

每个 stage 从 global memory 读取未 padding 的 A/B tile，共约 8 KiB，完成 262,144
FLOPs，算术强度约 32 FLOP/B，是初版 CTA tile 的四倍。BF16 也把相同元素数量的
global memory 流量减半。

### 3.3 当前 `cp.async` 仍是 single-stage

虽然已经使用异步 copy 指令，当前循环实际执行顺序是：

```cpp
issue_cp_async_for_current_tile();
cp_async_commit();
cp_async_wait_all();
__syncthreads();
compute_current_tile();
__syncthreads();
```

copy 后立即等待，没有让 tile `i+1` 的 global→shared copy 与 tile `i` 的 MMA 重叠。
所以这里 `cp.async` 的现有收益主要是：

- 每条指令搬运连续 16 B；
- global→shared 数据不经过程序员可见的通用寄存器；
- `.cg` 路径缓存于 L2，并绕过 L1。

它还没有发挥 software pipeline 隐藏 global-memory latency 的主要潜力。

### 3.4 边界约束

为了保持 kernel 和 fragment 映射清晰，MMA 两版当前要求：

```text
M % 64 == 0
N % 64 == 0
K % 32 == 0
```

若需要任意形状，可以将 `cp_async_16` 换成已有的 `cp_async_16_zfill`，并对 output
store 增加 predication；这会引入额外分支和地址判断，最好保留 aligned fast path。

## 4. L2 CTA swizzle 版

### 4.1 为什么普通 raster 会丢失 B reuse

普通 MMA 使用：

```cpp
tile_m = blockIdx.y;
tile_n = blockIdx.x;
```

CUDA grid 的 x 维变化最快，因此逻辑顺序近似为：

```text
(m0,n0), (m0,n1), ... (m0,n_last),
(m1,n0), (m1,n1), ...
```

同一行 CTA 能连续复用 `A[m0,:]`，但再次访问 `B[:,n0]` 前需要经过整行 N tiles。N
很大时，B 的 reuse distance 可能超过 L2 能保留的有效工作集。

### 4.2 N-cohort 映射

L2 版支持 `SWIZZLE_N=2/4/8`。以 8 为例：

```cpp
tile_m = blockIdx.x / 8;
tile_n = blockIdx.y * 8 + blockIdx.x % 8;
```

相邻 logical CTA 变为：

```text
(m0,n0)..(m0,n7), (m1,n0)..(m1,n7), ...
```

一个 cohort 内：

- A tile 在连续 8 个 N tiles 间复用；
- 一组 B tiles 只间隔 8 个 CTA 就被下一个 M tile 再次访问；
- 同时活跃的 A/B working set 比整行 raster 更紧凑。

这只是改变 `blockIdx → logical tile` 的映射，不改变矩阵布局、shared layout、bank
conflict 或单 CTA 的访存合并方式。CUDA 不保证严格按 block ID 将 CTA 分配给 SM，因此
它提高的是邻近 CTA 在相近时间共享 L2 cache line 的概率，而不是强制调度顺序。

最后一个不完整 cohort 会产生少量 padding CTA，并在 kernel 开头返回。当前合法 N 是
64 的倍数，但 N tile 数不必是 swizzle width 的倍数。

### 4.3 为什么不是所有矩阵都更快

Swizzle 同时带来收益和代价：

- 收益：缩短 B tile 的 L2 reuse distance，减少潜在 DRAM traffic；
- 代价：多几条整数除法/取模/地址计算，且把 A 的连续复用窗口从整行限制为 cohort；
- 小 working set、较短 N 或 compute-bound 场景中，额外映射成本可能大于 L2 收益；
- 最佳 cohort width 与 M/N/K、L2 容量、SM 数和 CTA 并发度有关。

因此 API 暴露 `swizzle_n=2/4/8`，而不是认为固定 8 对所有 shape 最优：

```python
from cuda_learn import ops

c = ops.gemm_mma_l2(a, b, swizzle_n=8)
```

## 5. 实测效果

### 5.1 环境与方法

以下数字测于 2026-08-28：

```text
GPU: NVIDIA GeForce RTX 4060 Laptop, sm_89, 24 SM
CUDA: 13.0
SM resources reported by runtime:
  shared memory / SM = 102,400 B
  registers / SM     = 65,536
  max threads / SM   = 1,536
```

计时使用 CUDA Event。普通 MMA 与 swizzle-2/4/8 在每轮中轮换执行顺序，避免笔记本 GPU
Boost、温度和功耗漂移长期偏向某个版本；表中使用多轮 median。

正确性方面：

- `1024³` 和 `1024×2048×1536` 下，swizzle-2/4/8 与普通 MMA 逐 bit 一致；
- BF16 MMA 与 PyTorch reference 在 `rtol=atol=1e-2` 下通过；
- FP32 初版在 `rtol=atol=1e-3` 下通过。

### 5.2 初版与 MMA：1024³

| kernel | median | throughput | 同类 cuBLAS | 相对自身 cuBLAS |
|---|---:|---:|---:|---:|
| tiled FP32 | 0.684 ms | 3.14 TFLOPS | 0.338 ms / 6.36 TFLOPS | 0.49× |
| BF16 MMA | 0.102 ms | 20.97 TFLOPS | 0.089 ms / 24.11 TFLOPS | 0.87× |

同尺寸下 MMA wall time 是 FP32 tiled 初版的约 `0.149×`，即约 `6.7×` 加速。但这个数字
包含 FP32→BF16、CUDA Core→Tensor Core、tile 增大、16-byte copy 等多项变化，不能解释为
某一条 MMA 指令单独带来的收益。更有意义的结论是：教学版 MMA 在该 shape 已达到对应
cuBLAS BF16 的约 87%，而 FP32 初版约为 cuBLAS SGEMM 的 49%。

### 5.3 普通 MMA 与 L2 swizzle：大矩阵

| M×N×K | ordinary | 最佳 swizzle | 最佳 width | speedup |
|---|---:|---:|---:|---:|
| 4096³ | 6.343 ms | 6.487 ms | 8 | 0.978× |
| 6144³ | 30.576 ms | 25.036 ms | 2 | 1.221× |
| 4096×8192×8192 | 36.394 ms | 30.705 ms | 8 | 1.185× |
| 8192×4096×8192 | 29.823 ms | 27.766 ms | 4 | 1.074× |
| 8192³ | 88.641 ms | 60.500 ms | 8 | 1.465× |

结论：

- `4096³` 没有收益，说明此时 L2 locality 还不是足以覆盖 swizzle 成本的瓶颈；
- 从 `6144³` 开始出现明显收益；
- `8192³` 的 swizzle-8 提升约 46.5%；
- `N=4096` 的矩形 case 收益较小，符合普通 raster 的 B reuse distance 随 N tiles 缩短的
  预期；
- 最佳 width 随 shape 改变，生产代码应 autotune 或使用 shape heuristic。

本机 Nsight Compute 因 `ERR_NVGPUCTRPERM` 无权读取 performance counter，所以目前确认的
是 wall-clock 收益；“收益来自更高 L2 hit rate、较少 DRAM read”仍是基于访问顺序的合理
解释，尚未由 `lts__t_sector_hit_rate` 和 `dram__bytes_read` 硬件指标直接验证。

### 5.4 复现命令

```bash
cmake --build build -j2
source scripts/env.sh
source ~/.python/miniinfer/bin/activate

# 1024³ 正确性、初版/MMA 和各自 cuBLAS 基线
python -m cuda_learn.bench \
  --warmup 20 --iters 100 test_gemm test_gemm_mma test_gemm_mma_l2

# 大矩阵，ordinary 与 swizzle-2/4/8 交错计时
PYTHONPATH=python python benchmarks/gemm_mma_l2.py \
  --warmup 10 --iters 30

# 自定义 shape，格式为 MxNxK
PYTHONPATH=python python benchmarks/gemm_mma_l2.py \
  --shapes 6144x6144x6144 4096x8192x8192 8192x4096x8192 \
  --warmup 5 --iters 16
```

## 6. 下一步：真正的 double buffering

### 6.1 目标流水线

当前 single-stage 每轮都是：

```text
time ───────────────────────────────────────────────►
      copy tile 0 | wait | MMA tile 0 | copy tile 1 | wait | MMA tile 1
```

double buffering 为 A/B 各分配两个 shared stage：

```text
time ───────────────────────────────────────────────►
copy: copy tile 0 |   copy tile 1   |   copy tile 2   | ...
mma :             | MMA tile 0      | MMA tile 1      | ...
shared stage:          0 / 1 ping-pong
```

tile `i` 做 MMA 时，tile `i+1` 通过 `cp.async` 写入另一个 shared stage，从而隐藏部分
global/L2→shared latency。

### 6.2 适配当前 kernel 的结构

伪代码如下；实际实现需要增加 `cp.async.wait_group` primitive，并仔细处理 prologue、最后
一个 tile 和 stage 覆盖前的 CTA barrier：

```cpp
__shared__ bf16 a_smem[2][BM][BK + PAD];
__shared__ bf16 b_smem[2][BK][BN + PAD];

issue_copy(/* k_tile=0, stage=0 */);
cp_async_commit_group();

for (int tile = 0; tile < num_k_tiles; ++tile) {
  int read_stage = tile & 1;
  int write_stage = read_stage ^ 1;

  cp_async_wait_group_0();       // 当前 read stage 已完成
  __syncthreads();               // 对整个 CTA 可见

  if (tile + 1 < num_k_tiles) {
    issue_copy(tile + 1, write_stage);
    cp_async_commit_group();
  }

  mma_from_shared(read_stage);   // 与 next tile copy 重叠
  __syncthreads();               // 允许后续覆盖旧 read stage
}
```

不能简单地把 `wait_all` 移到循环末尾：`ldmatrix` 读取当前 stage 前必须确认对应 async copy
已经完成；写 stage 被复用前也必须确认所有 warp 不再读取它。

### 6.3 资源代价

shared memory 将从：

```text
single stage:  9,728 B
double stage: 19,456 B
```

RTX 4060 Laptop 每 SM 有 102,400 B shared memory。只按容量估算，single-stage 最多容纳
10 个 CTA，double-stage 最多 5 个 CTA；当前 MMA 约 80 registers/thread、128
threads/CTA，寄存器上限粗略为 6 CTA/SM。因此 double buffer 可能把 shared memory 从非
瓶颈变成约 5 CTA/SM 的限制，并略微增加地址/stage 状态寄存器。

这意味着 double buffering 的判断不是“能重叠就一定更快”，而是：

```text
隐藏的 memory latency 收益
        vs
shared/register 增长导致的 resident CTA/warp 下降
```

当前 copy 使用 `cp.async.cg`，A/B load 绕过 L1。因此 L1/shared carveout 对数据缓存本身
帮助有限，主要影响 shared capacity。只有 occupancy 工具显示 shared carveout 限制了
目标 residency 时，才值得测试 `cudaFuncAttributePreferredSharedMemoryCarveout`；不应默认
固定为 MaxShared。

### 6.4 如何判断 double buffering 是否有效

至少比较以下四个版本，并像 L2 benchmark 一样轮换执行顺序：

```text
ordinary single-stage
swizzle single-stage
ordinary double-buffered
swizzle double-buffered
```

记录：

- kernel median/min duration 与 TFLOPS；
- active warps、active CTA 和 achieved occupancy；
- long scoreboard / memory dependency stall；
- Tensor Core utilization；
- L2 hit rate、L2 throughput 与 DRAM read bytes；
- shared bank conflicts；
- registers/thread、shared/CTA 和是否出现 spill。

预期最可能受益的是 K 很大、每个 stage 的 copy latency 暴露明显、同时又没有因 shared
翻倍导致 occupancy 大幅下降的 shape。若普通 MMA 已 compute-bound，或者 L2 swizzle 后
大多数 load 命中且延迟容易由现有 CTA 隐藏，double buffering 可能收益很小甚至回退。

在实际测量前，本文不为 double buffering 填写预期加速百分比。

## 7. 其他可继续验证的优化

按当前实现，建议优先级如下：

1. **Double buffering / 2-stage `cp.async`**：补全 load-compute overlap，是当前最明显缺失
   的流水线环节。
2. **Swizzle width autotune**：按 M/N/K 在 ordinary/2/4/8 中选择，避免 `4096³` 回退。
3. **向量化 epilogue**：当前 accumulator 逐元素转换并做标量 BF16 store，可尝试
   `__nv_bfloat162` 或更宽的合并 store，减少 conversion/store 指令。
4. **3/4-stage pipeline**：仅在 2-stage 有明确收益后继续；更多 stage 会线性增加 shared
   memory，并可能降低 occupancy。
5. **Tile shape autotune**：测试 `64×128`、`128×64` 等 CTA tile，平衡 A/B reuse、warp
   数、寄存器 accumulator 和 wave quantization。
6. **边界 fast/slow path**：aligned shape 继续走无 predication kernel，尾块使用
   `cp_async_16_zfill`。
7. **Persistent/Stream-K**：面向 CTA 数不足或 K 很大、M/N 较小的 shape；它解决的是
   wave utilization 和工作划分问题，不应与 L2 raster 混为同一种优化。

优化顺序应保持一次只改变一个主要变量，并保存 ordinary、cuBLAS 和上一版 kernel 三种
基线。否则 dtype、tile、pipeline、raster 同时变化时，很难判断收益来自哪里。
