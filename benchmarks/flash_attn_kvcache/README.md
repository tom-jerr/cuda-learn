# FlashAttention-2 KV-cache 离散访存 benchmark

本目录测试 vendored FlashAttention 2.8.3 的 `with_kvcache` 路径。测试只编译官方
FP16、head-dim 64 的 regular/split-KV specialization，避免安装完整 Python wheel；
kernel 源码没有修改。

## 对照设计

四组输入拥有完全相同的逻辑 Q/K/V、有效 KV 长度和计算量：

- `dense`：连续 `[B, Sk, Hkv, D]` cache，API 默认的 regular FA2 kernel；
- `dense_split`：仍是连续 cache，但强制 split-KV kernel；
- `paged_contiguous`：page table 指向相邻且顺序排列的页；
- `paged_random`：page table 在等大的物理 pool 内随机选页并打乱顺序。

主指标 `paged_random / paged_contiguous` 只改变物理页顺序，能够隔离离散访存；
`paged_contiguous / dense_split` 衡量 page-table indirection；`paged_random / dense`
则是用户从 dense API 切到随机 paged API 时看到的总差异。FA2 2.8.3 要求 page size
是 256 的倍数，默认值 256 也恰好等于 D64 split-KV kernel 的 N tile。

## 运行

```bash
source ~/.python/miniinfer/bin/activate
python benchmarks/flash_attn_kvcache/benchmark.py \
  --output benchmarks/flash_attn_kvcache/results/timing.json

# 大 batch 下成对交替运行顺序页/随机页，消除固定 case 顺序带来的 DVFS 偏差
python benchmarks/flash_attn_kvcache/benchmark_large_batch.py \
  --output benchmarks/flash_attn_kvcache/results/timing_large_batch.json
```

首次运行会通过 PyTorch extension cache 编译两个 vendored FA2 specialization。

Profiler：

```bash
source ~/.python/miniinfer/bin/activate
bash benchmarks/flash_attn_kvcache/profile_ncu.sh
bash benchmarks/flash_attn_kvcache/profile_nsys.sh
python benchmarks/flash_attn_kvcache/profile_torch.py
```

Nsight Compute 脚本固定 `B=8, Sq=1, Hq=Hkv=8, Sk=8192, D=64`，分别采集
连续 split、顺序 paged、随机 paged 的 DRAM/L2 吞吐、L2 hit rate、long-scoreboard
stall 和 SM throughput。若系统未授权 GPU performance counters，`ncu` 会报告
`ERR_NVGPUCTRPERM`；CUDA Event benchmark 不受影响。

`profile_torch.py` 是无需 GPU performance-counter 权限的 CUPTI fallback：它能确认
kernel 名称、调用次数和纯 GPU kernel 时间，但不能提供 L2/stall 指标。

## 本机结果（RTX 4060 Laptop, sm_89）

参数：`B=8, Sq=1, Hq=Hkv=8, D=64, FP16, page=256, fragmentation=2x`。
CUDA Event 对每种布局独立做 30 次 warmup，再做 200 次稳态采样。

| KV length | dense | dense split | paged contiguous | paged random | random / contiguous |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 242.7 us | 130.0 us | 135.2 us | 135.2 us | 1.000x |
| 4,096 | 476.2 us | 288.8 us | 297.0 us | 298.0 us | 1.003x |
| 8,192 | 925.7 us | 589.8 us | 601.1 us | 599.6 us | 0.997x |

原始数据见 [`results/timing_rtx4060.json`](results/timing_rtx4060.json)。PyTorch
Profiler（CUPTI）在 KV=8192 上确认三个对照都只运行一个同名
`flash_fwd_splitkv_kernel`；均值分别为 629.4、636.1、634.5 us，结果见
[`results/torch_profiler_rtx4060.json`](results/torch_profiler_rtx4060.json)。Profiler
本身有约 6% tracing overhead，应比较比例而不是与 Event 绝对值混用。

### 结论与原因

1. **随机页几乎没有额外代价。** FA2 D64 split-KV 的 `kBlockN=256`，恰好等于
   本测试的 page size。随机只改变每个 256-token tile 的起始地址；tile 内的 K/V
   `cp.async` 仍是相同的合并访问，读取字节数和计算量也相同。一次 decode scan
   不复用前一页，因此物理相邻页没有额外的 cache-line reuse 优势。
2. **可测到的是页表寻址，不是页顺序。** Paged 分支每推进一个 K/V tile 都读取
   两个 block-table entry，计算 page index/offset 和地址 delta；dense split 只做
   固定 stride 的指针递减。对应实现位于
   [`flash_fwd_kernel.h`](../../third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h)
   的初始寻址（约 583--594 行）和 tile 推进（约 943--974 行）。稳态 Event
   测试中该开销约 1--3%；2K 的 CUPTI 单-kernel 对照为 3.7%。
3. **不要用 `paged_random / dense` 推断离散访存成本。** FA2 2.8.3 在有
   `block_table` 时强制 split-KV kernel，而这里的 dense 默认走 regular forward。
   Sq=1 时 split-KV 的 64x256 tile 比 regular D64 的 128x128 tile 更适合 decode，
   所以 paged 反而显著快于 dense。`paged_random / paged_contiguous` 才是本问题的
   直接指标，`paged_contiguous / dense_split` 才是页表间接寻址指标。

本机 NCU 被系统权限拦截（`ERR_NVGPUCTRPERM`），无法取得 L2 hit-rate、DRAM bytes
和 long-scoreboard counter；因此不把未经采集的 counter 当作实测结论。授权后运行
`profile_ncu.sh` 即可补齐这些指标。当前源码控制流与 CUPTI kernel trace 已足以说明：
两种 paged case 的 kernel、指令路径和 tile 内内存事务形态一致，差别仅是页表值。

## 大 batch 结果

`benchmark_large_batch.py` 对顺序 paged 和随机 paged 做成对采样，并在每次采样时
交替 launch 顺序，避免后运行的 case 固定受 GPU boost/温度影响。其余参数与上面相同。

| BS | KV length | paged contiguous | paged random | random / contiguous |
|---:|---:|---:|---:|---:|
| 8 | 4,096 | 324.6 us | 324.6 us | 1.0000x |
| 8 | 8,192 | 573.4 us | 573.4 us | 1.0000x |
| 16 | 4,096 | 593.9 us | 593.9 us | 1.0000x |
| 16 | 8,192 | 1,204.2 us | 1,204.2 us | 1.0000x |
| 32 | 4,096 | 1,157.1 us | 1,155.1 us | 0.9982x |
| 32 | 8,192 | 2,260.0 us | 2,257.9 us | 0.9991x |
| 64 | 4,096 | 2,221.1 us | 2,214.9 us | 0.9972x |
| 64 | 8,192 | 4,398.1 us | 4,399.6 us | 1.0003x |
| 128 | 4,096 | 4,645.9 us | 4,645.9 us | 1.0000x |

原始数据见
[`results/timing_large_batch_rtx4060.json`](results/timing_large_batch_rtx4060.json)。
BS128/KV4K 单独结果见
[`results/timing_bs128_rtx4060.json`](results/timing_bs128_rtx4060.json)；BS128/KV8K
超过本机 8 GiB 显存可安全容纳的四组对照容量，未运行。BS64/KV8K 按实际 K+V
数据量计算，两种布局都达到约 244 GB/s useful bandwidth。
额外把 BS32/KV4K 的 page pool 扩大到 `8x`：每个 K 或 V pool 跨度达到 1 GiB，
500 次成对采样仍为 `1157.1 / 1157.1 us = 1.0000x`，见
[`results/timing_bs32_frag8_paired_rtx4060.json`](results/timing_bs32_frag8_paired_rtx4060.json)。

### 为什么离散页没有明显损失

- **这里是粗粒度 paged access，不是逐 token gather。** FA2 2.8.3 的最小 page 是
  256 token；D64 split-KV 恰好每次也处理 256 token。只有 tile 与 tile 之间跳一次
  地址，页内仍是长而连续的合并加载。若 page size 是 16 或逐 token 随机，结论不能
  外推。
- **FA2 的“离散页”仍来自一个连续 page-pool tensor。** `block_table` 存的是
  `k_cache/v_cache` 第一维的 slot id，API 不接受一组独立 `cudaMalloc` 指针。本测试
  随机访问的是同一块 1 GiB allocation 内相距很远的 slot，这就是 FA2 能表达的真实
  paged-KV 语义；它不等价于跨 allocation 的物理碎片或 Unified Memory page fault。
- **GPU coalescing 发生在一次 warp load 内。** 它要求同一条 load 指令的 lanes
  访问相邻 cache line，并不要求上一个 tile 与下一个 tile 的虚拟地址相邻。随机
  block id 因而不会增加每个 tile 的 memory transaction 数量。
- **KV decode 本来就是 streaming workload。** 每次 attention 要扫描每个有效 K/V
  tile，跨 page 没有可利用的数据复用；把下一页放在相邻虚拟地址不会像 CPU 顺序读取
  那样自动带来显著 prefetch 收益。虚拟地址连续也不等价于落在同一个 GDDR row，GPU
  memory controller 会把地址交错映射到多个 channel/bank。
- **页表很小而且所有线程访问一致。** 最大测试的 block table 只有
  `BS64 * 32 pages * 4 B = 8 KiB`，很容易驻留 cache；随机的是 entry 的值，不是
  读取 page table 本身的地址。额外成本主要是每 tile 的少量整数地址计算。
- **并发隐藏了页边界延迟。** BS 从 8 增至 64 后有 512 个 head/sequence CTA；当某个
  CTA 等待一个新页的 load/TLB translation 时，其它 CTA 仍可执行。结果中没有看到
  随 BS 增长而扩大的惩罚。

因此更准确的结论是：**FA2 当前支持的 256-token coarse-grained paged KV，在充分
并发、一次完整 scan 的 decode 场景下，随机 page order 没有可测的额外代价**；这不
代表细粒度随机 gather、跨 allocation 指针数组、低并发或频繁 page fault 也免费。
