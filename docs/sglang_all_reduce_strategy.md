# SGLang 1-shot / 2-shot all-reduce 选择策略

本文中的 “shot” 指跨 rank 数据交换与同步阶段，不等于 host 侧 kernel launch 数。
SGLang 的 `2shot_pull` 把 reduce-scatter 与 all-gather 写在一个 CUDA kernel 中，但
中间仍有第二次跨 GPU 同步，所以算法上依然是 two-shot。

以下记号统一为：

- `P`：world size；
- `M`：每个 rank 的输入字节数；
- all-reduce 操作为 sum，每个 rank 最终得到完整的 `M` 字节结果；
- “远端流量”只统计穿过 NVLink/PCIe/NVSwitch 的 payload，不含本地 HBM/L2 访问、
  flag、alignment padding 和输出 copy。

## 1. 三条实现路径

| 路径 | 第一个通信阶段 | 第二个通信阶段 | 同步 | 结果位置 |
|---|---|---|---|---|
| `1shot_push` | 每个 source 把完整输入 push 到每个 destination | 无 | payload marker | 每个 rank 的普通 output |
| `1shot_pull` | 每个 rank 从所有 source 拉取完整输入并归约 | 无 | reduce 前/后 barrier | 每个 rank 的普通 output |
| `2shot_pull` | rank `r` 只归约约 `1/P` 的结果分片 | 把该分片 fan-out 到所有 rank | reduce 前 + reduce/all-gather 间 barrier | 每个 symmetric workspace 都得到完整结果 |

源码入口分别是
[`all_reduce_1shot_push_kernel`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh#L136-L200)、
[`all_reduce_1shot_pull_kernel`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh#L202-L229) 和
[`all_reduce_2shot_pull_kernel`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh#L231-L262)。

## 2. 1-shot 为什么延迟低、流量高

### 2.1 Push

对 source rank `s`：

```text
input[s] ──► workspace[0].slot[s]
         ├─► workspace[1].slot[s]
         └─► ... workspace[P-1].slot[s]
```

其中一次是写自己的 workspace，另外 `P-1` 次是远端写，所以：

```text
每 rank 远端流量 = (P - 1) M
全局远端流量    = P (P - 1) M
```

destination 随后从自己的本地 workspace 读取 `P` 份输入并相加。这会产生约 `P*M`
的本地 HBM/L2 load，但不再穿过互连。

Push 的优势不是少传数据，而是协议短：store 的 payload 同时承担 arrival signal，
无需先 staging、再发独立 semaphore。小 tensor 时，省下的一次 copy、barrier latency
和 flag round trip 往往比多出的带宽更重要。

缺点是：

- source 要向 `P` 个地址 fan-out，`P` 增长时 store 指令和远端流量都线性增长；
- 每个 destination 重复完成同一份完整归约，计算量全局为 `P²` 量级；
- Lamport slot/epoch 是有状态的，同一 communicator 不能在无顺序的多个 stream 上并发；
- workspace 要为每个 source 保留两个 epoch，容量约 `2 * P * max_message_size`/rank。

### 2.2 Pull

每个 rank 从所有 source 的同一 offset 读数据：

```text
input[0] ─┐
input[1] ─┼──► rank r reduce ──► output[r]
...       │
input[P-1]┘
```

一次是 local read，`P-1` 次是 remote read，所以远端流量公式与 push 相同：

```text
每 rank 远端流量 = (P - 1) M
全局远端流量    = P (P - 1) M
```

Pull 与 push 的差别因此主要是方向和准备成本，而不是理论互连字节数：

- eager pull 先把普通 input copy 到 symmetric workspace；
- CUDA Graph pull 可以从 capture 后登记的 peer input pointers 直接读，省掉 staging；
- pull 使用显式 block-level barrier，协议比 marker polling 更通用；
- 在某些拓扑上 remote read 与 remote write 的有效带宽、L2 行为不同，最终必须实测。

## 3. 2-shot 为什么适合带宽主导区间

### 3.1 Reduce-scatter

rank `r` 只负责结果的约 `M/P` 分片。为了归约这段，它从每个 source 读取 `M/P`：

```text
每 rank 远端 read = (P - 1) M / P
全局远端 read     = (P - 1) M
```

### 3.2 All-gather

rank `r` 再把自己已经归约的 `M/P` 分片写到其余 `P-1` 个 rank：

```text
每 rank 远端 write = (P - 1) M / P
全局远端 write     = (P - 1) M
```

合计：

```text
每 rank 远端流量 = 2 (P - 1) M / P
全局远端流量    = 2 (P - 1) M
```

因此 1-shot 与 2-shot 的理论远端流量比为：

```text
P (P - 1) M / [2 (P - 1) M] = P / 2
```

这给出几个直观结论：

- `P=2`：理论远端流量相同；2-shot 多一道阶段和同步，通常没有优势；
- `P=4`：2-shot 约为 1-shot 的一半远端流量；
- `P=8`：2-shot 约为 1-shot 的四分之一；
- `P=16`：2-shot 约为 1-shot 的八分之一。

代价是更长的 dependency chain：所有 rank 必须先完成 reduce-scatter 并通过带
release/acquire 语义的 barrier，才能读取其他 rank 的 reduced shard。小消息时第二道
同步、分片边界和额外地址计算会盖过节省的带宽。

## 4. Push、普通 Pull 与 NVLS Pull

2-shot 的 `LoadStoreImpl` 用普通 peer pointers 执行 `P` 次 load 和 `P` 次 store。
如果存在 multicast VA，`MultiCastImpl` 改用：

- `multimem.ld_reduce`：从 multicast team 所有物理 replica load 并在 fabric 中归约；
- `multimem.st`：一次写入所有 replica。

普通 P2P 2-shot 已经减少算法流量；NVLS 又减少 GPU 指令、地址 fan-out 和部分 switch
数据移动开销。它只在支持 multicast objects、NVLink SHARP/NVSwitch 的系统上成立，
不能因为 GPU 是 SM90 就假定一定可用。程序必须查询
`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`。官方要求见
[CUDA VMM Multicast Memory Sharing](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#multicast-memory-sharing)。

## 5. 实际选择顺序

当前 V2 的 dispatch 顺序是：

```text
nbytes <= push_threshold
        │ yes
        └────► 1shot_push
        │ no
nbytes <= pull_threshold
        │ yes
        └────► 1shot_pull
        │ no
multicast 可用且 nbytes 在 mc range
        │ yes
        └────► NVLS 2shot_pull
        │ no
nbytes <= two_shot_threshold
        │ yes
        └────► 普通 2shot_pull
        │ no
        └────► NCCL / 其他 backend
```

对应源码为
[`CustomAllReduceV2._pick_config`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py#L335-L347)。

### 5.1 可以先用的判断规则

| 条件 | 首选 | 原因 |
|---|---|---|
| `P=2`，小到中等消息 | 1-shot | 2-shot 不省理论远端流量，却多一次阶段 |
| `P>=4`，很小的 decode TP 消息 | `1shot_push` | latency/launch/barrier 主导，payload marker 协议最短 |
| Graph 中输入已登记，消息仍较小 | `1shot_pull` 或 push | pull 可直接读 graph peer pointer，免 staging；具体由实测阈值决定 |
| `P>=4`，消息进入 bandwidth-bound | `2shot_pull` | 远端流量从 `P(P-1)M` 降至 `2(P-1)M` |
| NVLS multicast 可用 | multicast `2shot_pull` | fabric reduction/fan-out，减少显式 peer 指令 |
| PCIe-only、多节点非 NVLink clique、消息过大 | NCCL | custom spin/barrier 和全连接 P2P 不再有稳定优势或根本不安全 |
| communicator 会被多个无序 stream 并发 | NCCL 或每 stream 独立 communicator | push epoch、pull semaphore 都是有状态协议 |

### 5.2 为什么不能给一个固定 crossover

crossover 同时依赖：

- `P` 与拓扑：直连 NVLink、NVSwitch、PCIe、MNNVL；
- graph/eager：eager pull 有 staging copy，graph pointer table 可以 zero-copy；
- dtype 与对齐：FP16/BF16 的 arithmetic throughput、16B packing；
- GPU 架构和 SM 数：决定可驻留 block 数以及 spin kernel 是否阻塞其他工作；
- 当前模型的消息分布：decode 的大量小 tensor 与 prefill/大 batch 的较大 tensor；
- 是否有 NVLS multicast，以及 Fabric Manager/IMEX 等系统组件是否正常；
- NCCL 版本和它在当前平台上的协议选择。

所以 SGLang 把 graph/eager、architecture、world size、block count 与 multicast range
写成 benchmark 得到的表，而不是用上述流量公式直接算阈值。流量公式只能判断趋势，
不能替代 benchmark。

## 6. 当前配置实例

以 SGLang 当前 SM90/H200 表为例：

| World | Context | `1shot_push` 上限 | `1shot_pull` 上限 | 2-shot / multicast |
|---:|---|---:|---:|---|
| 2 | graph | 16 MB | 128 MB | pull 已覆盖到 128 MB，通常不进入 2-shot |
| 2 | eager | 32 MB | 128 MB | 同上 |
| 4 | graph | 384 KB | 384 KB | 之后普通 2-shot，最大 128 MB |
| 4 | eager | 896 KB | 896 KB | 之后优先 multicast，最大 32 MB |
| 8 | graph | 128 KB | 128 KB | 128–512 KB 普通 2-shot，之后 multicast |
| 8 | eager | 128 KB | 128 KB | 之后优先 multicast，最大 128 MB |

这张表恰好体现趋势：world size 越大，one-shot crossover 越早；`P=2` 则愿意让
one-shot 覆盖很大的范围。数值来自当前 commit 的调优结果，只适合解释，不应复制成
另一台机器的固定配置。完整表见
[`configs/custom_all_reduce_v2.py`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py#L233-L315)。

## 7. 推荐调优方法

1. 先按模型真实 TP tensor shape 收集消息大小直方图，不要只扫连续的 1 KB–128 MB。
2. 分开 benchmark eager 与 CUDA Graph，二者的 staging/launch 成本不同。
3. 每个 `(GPU arch, topology, P)` 分别测 push、pull、2-shot、NCCL。
4. 除平均 latency 外观察 P99、SM occupancy、NVLink TX/RX、L2 hit rate 和功耗。
5. 在连续多轮和 CUDA Graph replay 下测试，单轮不会暴露 epoch/semaphore 复用问题。
6. 测试与模型 compute 并发的情形；spin-wait collective 单独最快，不代表端到端最快。
7. 在 crossover 附近留 hysteresis/保守区间，避免小的 shape 波动频繁切换 backend。

最终选择原则可以压缩成一句话：**小消息为同步付费，选 1-shot；大消息为带宽付费，
选 2-shot；拓扑或协议能力不满足时，选 NCCL。**
