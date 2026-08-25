# Flash Attention：从教学版到寄存器、分阶段流水与 causal 优化

本文是本仓库 Flash Attention 实现的单一完整文档，合并了原先分散的背景、初始版本、
优化版本和性能测试文档。对应代码：

- 初始版本：[src/flash_attn.cu](../src/flash_attn.cu)，shared-memory 教学版；
- 优化版本：[src/flash_attn_optimized.cu](../src/flash_attn_optimized.cu)，寄存器累加、
  PAD=8 shared layout、8 warp、128×128 分阶段 K/V 流水；
- XOR swizzle 版本：[src/flash_attn_swizzled.cu](../src/flash_attn_swizzled.cu)，48 KiB
  compact layout、4 warp、目标 2 CTA/SM；
- Python 调用：[python/cuda_learn/ops.py](../python/cuda_learn/ops.py)；
- 测试与 benchmark：[python/cuda_learn/tests/test_ops.py](../python/cuda_learn/tests/test_ops.py)。

三个手写 kernel 都固定为 FP16、forward-only 和 `D=64`。初始版使用 64×64 tile；
两个优化版都使用 128×128 tile。输入布局为 `[B,H,N,64]`，且 `N` 必须是 64 的倍数。
`flash_attn_optimized(...)` 是 PAD=8 版本，`flash_attn_swizzled(...)` 是独立的 XOR
swizzle 版本；两者都通过 `causal=False/True` 选择模式。

## 1. 为什么需要 Flash Attention

标准 scaled-dot-product attention 为：

```text
S = Q K^T / sqrt(D)    # [N,N]
P = softmax(S)         # [N,N]
O = P V                # [N,D]
```

把这三步拆成独立算子时，`S` 和 `P` 会完整写回 HBM 再读回。它们的空间复杂度是
`O(N^2)`，且大量时间消耗在中间矩阵的显存流量，而不是 Tensor Core 计算。

Flash Attention 的核心是：一个 CTA 持有一块 Q，流式扫过所有 K/V tile；当前的
score/probability tile 只在片上短暂存在，完整的 `[N,N]` 中间矩阵从不落 HBM。

### 1.1 Online softmax

分块后无法一次看到整行 score，因此每个 query 行维护：

- `m`：目前见过的最大值；
- `l`：以 `m` 为基准的指数和；
- `O_acc`：尚未归一化的输出分子。

处理新 tile 时：

```text
m_new = max(m_old, tile_max)
alpha = exp(m_old - m_new)
l_new = alpha * l_old + sum_j exp(x_j - m_new)
O_new = alpha * O_old + sum_j exp(x_j - m_new) * V_j
```

最后写出 `O_acc / l`。若新 tile 出现更大的最大值，只需用 `alpha` 重新缩放旧状态，
因此整个流式过程保持数值稳定。

### 1.2 本仓库采用的硬件原语

原语封装位于 [ampere_primitives.cuh](../src/include/ampere_primitives.cuh)：

- `cp.async`：16 字节 global-to-shared 异步拷贝；
- `ldmatrix`：一个 warp 协作把 shared-memory 矩阵 fragment 搬进寄存器；
- `mma.sync.m16n8k16`：Tensor Core 执行 FP16 矩阵乘、FP32 累加；
- `__shfl_xor_sync`：寄存器之间的 warp/subgroup reduction。

整体数据流是：

```text
HBM --cp.async--> shared --ldmatrix--> registers --mma--> registers
```

## 2. 初始版本：shared-memory 教学实现

初始 kernel 是 `flash_attention2_fwd_d64`。一个 CTA 负责 64 个 query 行，4 个 warp
各负责 16 行，循环处理 64 行的 K/V tile。

### 2.1 Shared memory 布局

| 区域 | 元素与类型 | 字节 | 用途 |
|---|---:|---:|---|
| `q_smem` | `64×72` FP16 | 9,216 | 当前 Q tile，循环外载入一次 |
| `kv_smem` | `64×72` FP16 | 9,216 | K/V 共用，V 会覆盖 K |
| `score_smem` | `64×64` FP32 | 16,384 | 当前 `QK^T` score |
| `prob_smem` | `64×72` FP16 | 9,216 | 当前 softmax probability |
| `out_smem` | `64×64` FP32 | 16,384 | 跨 tile 输出分子 |
| `row_m/row_l` | `2×64` FP32 | 512 | online-softmax 行状态 |
| 合计 | | **60,928 B** | 约 59.5 KiB |

`PAD=8` 让相邻 shared-memory 行在 bank 上错开，降低 `ldmatrix` bank conflict。
60,928 B 超过传统 48 KiB 动态 shared-memory 限制，因此 host wrapper使用
`cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` opt-in。

### 2.2 每个 KV tile 的五个 CTA barrier

初始版本的处理顺序为：

1. `K: HBM -> kv_smem`，等待 K 对所有 warp 可见；
2. `QK^T` 的 MMA 结果从寄存器写入 `score_smem`，等待 softmax 消费；
3. 64 个线程各自串行处理一行 softmax，并写入 `prob_smem`，等待所有 warp；
4. `V: HBM -> kv_smem` 覆盖 K，等待 K 的旧读者和 V 的新读者；
5. `PV` 累加回 `out_smem`，等待下一个 tile rescale 输出状态。

因此每个 tile 有 5 次 `__syncthreads()`。其中 score、输出和 m/l 状态本来都可以由
拥有对应行的 warp 保留在寄存器，却为了教学可见性反复经过 shared memory。

### 2.3 初始版 softmax 的瓶颈

初始版只让 `tid < 64` 的线程参与 softmax，每个线程串行完成：

```text
scan 64 scores for max
scan 64 scores for exp and sum
scan 64 output values for rescale
```

另一半线程空闲，串行依赖链也很长。这与前后高度并行的 Tensor Core MMA 不匹配。

### 2.4 初始版的价值

该版本不是为了接近库性能，而是把 `QK^T -> online softmax -> PV` 的边界全部显式化。
它便于检查 fragment 映射、shared-memory 生命周期以及每个 barrier 的必要性，是优化版的
正确性基线。

## 3. 优化版本总体结构

优化 kernel `flash_attention2_fwd_d64_registers` 同时改变 tiling、流水和状态归属：

| 项目 | 初始版 | 优化版 |
|---|---|---|
| tile | 64×64 | 128×128 |
| threads/CTA | 128（4 warp） | 256（8 warp） |
| score tile | FP32 shared memory | `s_frag[16][4]` 寄存器 |
| 输出分子 | FP32 shared memory | `o_acc[8][4]`，跨 KV tile 常驻寄存器 |
| m/l | shared-memory 数组 | 每 lane 两组标量寄存器 |
| softmax | 每行单线程串行 | 4-lane subgroup 协作 |
| K/V staging | 共用一个 buffer | K/V 各一个单-stage buffer |
| P | FP16 shared memory | FP16 register fragment |
| CTA barrier/tile | 5 | **2** |
| causal | 不支持 | 可选，整 tile 跳过 + 对角 mask |

### 3.1 输出和 online-softmax 状态常驻寄存器

每个 lane 持有：

```cpp
float o_acc[8][4];
float row_m0, row_m1;
float row_l0, row_l1;
```

`o_acc` 就是八个 `mma.m16n8k16` 输出 fragment。元素 0/1 属于 `row0`，元素 2/3
属于 `row1`。进入新 tile 时直接在寄存器里乘 `alpha`，随后 `PV` MMA 继续累加到
同一组寄存器。循环结束后乘 `1/l` 并转 FP16，再通过向量化 epilogue 写 HBM。

这样删除了初始版的 `score_smem`、`out_smem`、`row_m` 和 `row_l`，也删除了它们导致的
CTA 级生产者/消费者同步。

### 3.2 4-lane subgroup 并行 softmax

`mma.m16n8k16` 的 FP32 accumulator 映射中，每个 lane 持有两行、每行两个相邻列。
四个相邻 lane 共同覆盖相同的两行：

```text
lanes 4g..4g+3 -> rows g and g+8 -> each row has all 128 columns
```

每个 lane 先对自己持有的 32 个 score 求局部 max/sum，再通过两轮：

```cpp
__shfl_xor_sync(0xffffffffu, value, 1, 4);
__shfl_xor_sync(0xffffffffu, value, 2, 4);
```

得到整行结果。全部 256 个线程都参与 softmax，且 reduction 只经过寄存器和 shuffle。

softmax 之后不再把 P 写入 `prob_smem`。两个相邻 8-column score accumulator 直接转为
四个 packed FP16 register，恰好组成下一次 `mma.m16n8k16` 所需的 A fragment。因此
P 路径既没有 shared-memory store/load，也不再需要 `__syncwarp()`。

### 3.3 复用 Q shared memory 的向量化输出

KV 循环结束后 `q_smem` 已不再使用。直接从 MMA accumulator 写 output 会生成每线程
32 条 `STG.E.U16`；当前版本先把 FP16 output fragment 写入 `q_smem` 完成 layout
conversion，执行一次 CTA barrier，再由所有线程按连续 16 字节读取和写出。

最终 SASS 从：

```text
32 × STG.E.U16
```

变为：

```text
5 × LDS.128 + 5 × STG.E.128
```

编译器也把 fragment 到 shared 的写入合并成 `STS.E.BYPASS.128`。代价是增加一次
epilogue CTA barrier，以及寄存器从 139 增至 143/thread。

## 4. 128×128 tile 与 K/V 分阶段流水

优化版将 tile 从 64×64 扩展到 128×128。八个 warp 各负责 16 个 query row；K/V
tile 被 128 行 Q 复用一次，因此相同 attention 计算发出的 K/V global load 请求减半。

K 和 V 各自只有一个 shared-memory stage：

```text
Q[128×64] + K[128×64] + V[128×64]
```

总动态 shared memory 为：

```text
3 * (128 * 72) half = 55,296 B = 54 KiB
```

分阶段流水为：

```text
prefetch Q and K[0]

for tile t:
    wait K[t]; __syncthreads()
    async copy V[t]
    QK[t]                         # 与 V[t] copy 重叠
    wait V[t]; __syncthreads()    # 此时 K[t] 已无人读取
    async copy K[t+1]
    online softmax + PV[t]        # 与 K[t+1] copy 重叠
```

K 和 V 的生命周期错开，因此不需要 `K[2] + V[2]` 完整 ping-pong。每 tile 两次 CTA
barrier 分别保护 K 和 V 的单 buffer 重用；虽然 barrier/tile 从旧双缓冲版的 1 增为 2，
但 128×128 tile 令整个 workload 的 tile 迭代从 4096 降到 1024，所以总 CTA barrier
约从 4096 降到 2048。

对于 `[2,8,1024,64]` non-causal workload：

| 项目 | 旧 64×64 | 当前 128×128 |
|---|---:|---:|
| query CTA 数 | 256 | 128 |
| 每 CTA 的 KV tile 数 | 16 | 8 |
| CTA×tile 迭代 | 4096 | 1024 |
| K/V global load 请求量（忽略 cache） | 64 MiB | 32 MiB |

### 4.1 64-row 尾块

接口仍支持 `N%64==0`，不强制 `N%128==0`。最后不足 128 行的 Q/K/V 使用带 source-size
的 `cp.async` zero-fill；softmax 同时把越界 key 置为 `-inf`，output store 只写有效 query
row。`N=64/128/192/320/1024` 的 causal 和 non-causal 都已单独验证。

### 4.2 为什么 shared memory 仍是 54 KiB

旧双缓冲版的 54 KiB 来自 `Q + K[2] + V[2] + P`；当前同样是 54 KiB，但来自扩大后的
`Q + K + V`。若仍为 128×128 P 分配 shared tile，总量会达到 90,112 B，因此 P 必须保留
在寄存器。PAD=8 版本仍受 1 CTA/SM 限制。

### 4.3 独立的 4-warp XOR-swizzle 版本

`flash_attn_swizzled` 保留相同的分阶段 K/V 流水、寄存器 P、causal 跳块和 vector
epilogue，但把 Q/K/V 的 shared layout 改为无 padding 的 128×64。每个连续 8-half
向量的列号与低三位 row 做 XOR：

```cpp
physical_col = col ^ ((row & 7) << 3);
physical_index = row * 64 + physical_col;
```

向量内的 8 个 half 保持连续和 16-byte 对齐，因此 `cp.async`、`ldmatrix` 和 epilogue
仍能向量化；相邻行则访问不同 bank pattern。shared memory 降为：

```text
3 * (128 * 64) half = 49,152 B = 48 KiB
```

CTA 从 8 warp/256 threads 改为 4 warp/128 threads。每个 warp 同时负责两组相隔 64 行
的 16-row MMA tile：

```text
warp w, mi=0 -> rows w*16 .. w*16+15
warp w, mi=1 -> rows 64+w*16 .. 64+w*16+15
```

编译结果为 255 registers/thread；每 CTA 的 register allocation 恰好约 32 Ki registers，
两 CTA 共 65,536 registers，shared memory 两 CTA 共 96 KiB，因此 sm_89 上满足目标
2 CTA/SM。为了降低峰值寄存器压力，两个 query row group 的 PV 依次执行；最终仍有
32-byte stack frame 和每线程 72-byte spill store/load，这是与 FA2 剩余差距的来源之一。

## 5. 可选 causal：块跳过与对角 mask

调用方式：

```python
out = cuda_learn.flash_attn_optimized(q, k, v, causal=True)
```

对于第 `blockIdx.x` 个 Q tile，只需访问：

```cpp
kv_tiles = blockIdx.x + 1;
```

因此对角线右上方的 KV tile 不会加载，也不会执行 QK/PV MMA。最后一个被访问的 tile
是对角 tile，其中再按元素执行：

```text
if key_column > query_row: score = -infinity
```

其余左下方 tile 完全有效，不产生逐元素 mask 分支。这样 causal 模式大约跳过一半
QK/PV 工作，而不是先完成完整 attention 再把未来位置清零。

对 square self-attention，计入真正执行的 QK/PV FLOP：

```text
non-causal: 4 * B * H * N^2 * D
causal:     2 * B * H * N * (N+1) * D
```

## 6. Shared memory、寄存器与同步对比

`cuobjdump --dump-resource-usage` 的 sm_89 实测：

| 版本 | registers/thread | dynamic shared | local/stack spill | barrier |
|---|---:|---:|---:|---:|
| 初始版 | 80 | 60,928 B | 0 | 5/tile |
| 单-stage register 版（历史中间状态） | 102 | 36,864 B | 0 | 2/tile |
| 64×64 完整 K/V 双缓冲（历史版本） | 125 | 55,296 B | 0 | 1/tile |
| 128×128 分阶段流水、direct output（历史版本） | 139 | 55,296 B | 0 | 2/tile |
| PAD=8 8-warp + vector epilogue | 143 | 55,296 B | 0 | 2/tile + 1/CTA |
| XOR-swizzle 4-warp | 255 | 49,152 B | 32 B stack；72 B spill store/load | 2/tile + 1/CTA |

PAD 版本为 1 CTA/SM；XOR-swizzle 版本达到 2 CTA/SM。两者都是每 SM 约 8 个 resident
warp，但后者的 warp 分属两个 CTA，可以跨 CTA 隐藏 barrier、softmax SFU 和 MMA 阶段。

## 7. 完整调用链

```text
cuda_learn.flash_attn_optimized(q,k,v,causal)
    -> Python 检查 FP16 CUDA [B,H,N,64]
    -> TVM FFI: cuda_learn.flash_attn_optimized
    -> C++ TensorView/shape/device 检查
    -> torch 当前 CUDA stream
    -> flash_attention2_fwd_d64_registers<<<...>>>

cuda_learn.flash_attn_swizzled(q,k,v,causal)
    -> TVM FFI: cuda_learn.flash_attn_swizzled
    -> flash_attention2_fwd_d64_swizzled_4warp<<<...>>>
```

官方对比基线使用 vendored flash-attention 2.8.4 的 D64 FP16 causal/non-causal 官方
specialization。benchmark-only binding 只裁剪无关 dtype/head-dim/反向 translation unit，
不修改被测 kernel 源码。

## 8. 正确性和性能

测试环境：RTX 4060 Laptop（sm_89）、torch 2.10+cu128、nvcc 13.0，输入
`[B=2,H=8,N=1024,D=64]` FP16。为降低笔记本 GPU Boost 和执行顺序的影响，先对每个
kernel warmup 100 次，再交替执行多轮、每轮 500 次 CUDA Event 计时。相对差距取每一轮
配对时间比的中位数，避免两种 kernel 的频率波动错位；vector epilogue 的最终结果使用
修改后额外执行的 8 轮配对测量。

### 8.1 Non-causal

| kernel | median | TFLOPS | 相对当前 optimized |
|---|---:|---:|---:|
| 初始版（历史标准 bench） | 1.052 ms | 4.08 | 慢 3.69x |
| 64×64 双缓冲版（改动前） | 0.331 ms | 12.99 | 慢 1.16x |
| 128×128 分阶段流水、direct output | 0.286 ms | 15.00 | 慢 1.10x |
| 当前 128×128 + vector epilogue | **0.259 ms** | **16.59** | 1.00x |
| flash-attention 2.8.4 | 0.227 ms | 18.92 | 快 1.16x（paired） |

从完整 K/V 双缓冲到当前实现，non-causal 延迟从 0.331 ms 降至 0.259 ms。单独看
vector epilogue，由于笔记本动态频率，绝对 A/B 测得约 6%--12% 提升；用同轮 FA2 时间
归一化后，相对差距从 1.235x 降到 1.156x，对应约 6.4% 的延迟下降。

### 8.2 Causal

| kernel | median | 有效 TFLOPS | 相对当前 optimized |
|---|---:|---:|---:|
| 64×64 双缓冲 causal（改动前） | 0.189 ms | 11.35 | 慢 1.06x |
| 128×128 分阶段 causal、direct output | 0.179 ms | 12.00 | 慢 1.09x |
| 当前 128×128 + vector epilogue causal | **0.164 ms** | **13.11** | 1.00x |
| flash-attention 2.8.4 causal | 0.143 ms | 15.03 | 快 1.12x（paired） |

vector epilogue 的 causal 归一化差距从 1.214x 降到 1.118x，对应约 7.9% 的延迟下降。
短序列收益较小：`N=128` 约为 0%--5%，新增 epilogue barrier 会抵消部分向量 store 收益。

### 8.3 XOR swizzle vs PAD=8 vs FA2

笔记本 GPU 的功耗状态会让长批次按固定顺序执行产生十几个百分点偏差。这里使用 30 轮
短批次，轮换三种 kernel 的执行顺序，并统计每轮时间比的中位数。小于 1 表示 swizzle
版本更快：

| mode | N | swizzle / PAD=8 | swizzle / FA2 |
|---|---:|---:|---:|
| non-causal | 128 | 0.998x | 1.706x |
| non-causal | 1024 | **0.896x** | **1.030x** |
| non-causal | 2048 | **0.896x** | **1.036x** |
| causal | 128 | 1.024x | 1.652x |
| causal | 1024 | **0.918x** | **1.038x** |
| causal | 2048 | **0.920x** | **1.034x** |

对 `N>=1024`，4-warp swizzle 版本相对 PAD=8 版本快约 8%--10%，与 FA2 的差距缩小到
约 3%--4%。`N=128` 时两种自研版本基本持平，但仍比 FA2 慢约 65%--71%；此时只有一个
KV tile，双 CTA 流水优势无法摊薄 kernel 固定控制开销。non-causal 的8个 KV tile 会反复
支付 spill 成本，因此下一步应优先压缩 register live range 和拆分编译期 mask 路径。

多组 `[B,H,N,D]` 形状的正确性测试均与
`torch.nn.functional.scaled_dot_product_attention` 对齐：non-causal/causal 最大绝对误差
均不超过 `4.88e-4`，没有 NaN/Inf。仓库全部 16 个 benchmark case 正确性通过。

笔记本 GPU 会动态调整时钟和功耗，因此不同进程的绝对时间可能波动；比较时应关注同一
进程、相同输入和相同 warmup 下的 median。

## 9. 复现

```bash
source ~/.python/miniinfer/bin/activate
source scripts/env.sh
cmake --build build -j4

# 初始版 vs 官方 non-causal
MAX_JOBS=1 python -m cuda_learn.bench \
  --warmup 100 --iters 500 test_flash_attn

# 128×128 分阶段流水 vs 初始版 vs 官方 non-causal
MAX_JOBS=1 python -m cuda_learn.bench \
  --warmup 100 --iters 500 test_flash_attn_optimized

# 128×128 分阶段 causal vs 官方 causal
MAX_JOBS=1 python -m cuda_learn.bench \
  --warmup 100 --iters 500 test_flash_attn_optimized_causal

# 4-warp XOR swizzle vs PAD=8 vs FA2
MAX_JOBS=1 python -m cuda_learn.bench \
  --warmup 100 --iters 500 test_flash_attn_swizzled
MAX_JOBS=1 python -m cuda_learn.bench \
  --warmup 100 --iters 500 test_flash_attn_swizzled_causal

# 全部正确性
MAX_JOBS=1 python -m cuda_learn.bench --check-only
```

首次运行官方基线时会把 D64 causal/non-causal specialization 编译到 torch extension
cache，编译时间不进入 CUDA Event 计时窗口。

## 10. 当前边界与后续方向

- 只支持 FP16、forward、`D=64`、`N%64==0`；
- 没有 backward、dropout、GQA、variable length、bias、ALiBi 或 softcap；
- tile/warp 调度只针对当前教育实现，没有设备级 autotune；
- XOR-swizzle 版本已经达到 4-warp、48 KiB 和 2 CTA/SM，但 255-register 上限仍造成少量
  stack spill；下一步应缩短 score/output live range，或拆分计算阶段减少 spill；
- non-causal/causal 仍由运行时 `bool` 选择；拆成编译期 specialization 可以继续减少热循环
  predicate 和分支；
- Nsight Compute 硬件计数器在当前机器上受 `ERR_NVGPUCTRPERM` 权限限制，shared bank
  conflict、warp stall 和实际 L2/DRAM throughput 尚未获得硬件计数器验证；
- causal 当前针对 square self-attention；cross-attention 的非方形 causal 语义尚未实现。
