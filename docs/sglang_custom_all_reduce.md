# SGLang custom all-reduce：实现思路与纯 CUDA 复刻

本文基于 2026-08-28 的 SGLang `main`，commit
[`d5670645`](https://github.com/sgl-project/sglang/tree/d56706459c8e52ec3ab1c41dae778e4fe03e0da3)。
当前 CUDA 默认路径是 JIT 编译的 `CustomAllReduceV2`；环境变量
`SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0` 才会退回 legacy 实现。入口选择见
[`custom_all_reduce.py`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py#L344-L369)。

先说结论：它不是用 CUDA core 算加法比 NCCL 更快，而是针对单机、小到中等
tensor-parallel 消息，省掉通用通信库的调度层、channel/protocol 状态机和额外
kernel launch。代价是只覆盖受控的拓扑、dtype、对齐和调用顺序；超过实测阈值
仍回退 NCCL。

## 1. 整体分层

V2 把“内存如何被所有 rank 看见”和“如何归约”拆开：

1. Python 建立 symmetric memory。每个 rank 的 slab 都能拿到所有 peer 的映射，
   并切成 `push slots | pull buffer | semaphores`。push 部分有
   `2 * world_size` 个槽，两个 epoch 各为每个 source rank 留一个槽；pull 部分有
   一个 tensor 大小的 staging buffer；每个 kernel block 另有 128B 对齐的
   semaphore。布局见
   [`CustomAllReduceV2._init_workspace`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py#L217-L299)。
2. `PushPlane`、`PullPlane` 只保存 peer pointer table、counter、semaphore 和可选
   multicast VA，`Communicator` 再把 plane 与调优后的 block 数组合起来。
3. JIT kernel 只接收 trivially-copyable 参数，不负责分配、IPC 或进程组通信。
   这样同一组 communication planes 还能被融合算子复用。CUDA 端的三个算法入口见
   [`custom_all_reduce.cuh`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh#L1-L20)。

legacy 实现更接近早期 vLLM：Python 用 `cudaMalloc` 建 buffer，交换
`cudaIpcMemHandle_t` 后在各进程打开 peer pointer；CUDA 类内部同时管理 pointer
注册、CUDA Graph 地址和 kernel launch。它有 1-stage 与 2-stage 两种算法，核心见
[`legacy custom_all_reduce.cuh`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh#L324-L518)。

## 2. V2 的三个算法

令每个 rank 的消息为 `N` bytes，rank 数为 `P`。

### 2.1 `1shot_push`：小消息优先

每个 rank 在同一个 kernel 中完成：

```text
local input
   │  16B vector store，分别 push 到所有 rank 的 workspace
   ├──────────────► dst 0: slot[src]
   ├──────────────► dst 1: slot[src]
   └──────────────► ...

本 rank workspace: slot[0], slot[1], ... slot[P-1]
                         │ 本地轮询、相加
                         ▼
                       output
```

它没有独立的全局 barrier。workspace 初始全是正零 `+0.0`；producer 把 payload
中的正零改写成负零 `-0.0`，两者数值相同但 bit pattern 不同。consumer 对一个
16B vector 反复 load，只要任一 32-bit atom 仍为全零就说明该 word 尚未到达；
全部非零才做归约。完成后把槽写回正零。这个 Lamport-style 数据即信号协议见
[`LamportTrait`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh#L98-L156)，
完整 push/poll kernel 见
[`all_reduce_1shot_push_kernel`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh#L136-L200)。

必须有两个 epoch：consumer 可能仍在清空第 `k` 轮槽位时，快的 producer 已进入
第 `k+1` 轮；若复用同一槽，会把新数据清掉。V2 用每 block counter 在两半
workspace 间翻转，并要求 grid 大小保持不变。`PushEpoch` 的地址计算见
[`communicator.cuh`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh#L204-L239)。

这一算法每个 rank 写 `P*N`、读 `P*N`，总流量随 `P²` 增长，所以适合 latency
主导的小消息，不适合大消息。它还隐含一个严格约束：同一 communicator 上各 rank
必须以同样顺序发 collective；不能从两个互不排序的 stream 并发复用同一套 epoch
和 slot。

### 2.2 `1shot_pull`：每个 rank 拉取所有输入

eager 模式先把本 rank 输入拷入 symmetric pull buffer，所有 rank 到达 block-level
barrier 后，每个 rank 从 `P` 个 peer buffer 读取同一位置、相加并写自己的 output。
CUDA Graph 模式可以直接从事先登记的 graph input pointer table 读，跳过 staging
copy。计算循环见
[`all_reduce_1shot_pull_kernel`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh#L202-L229)。

它同样是每 rank 读取 `P*N`，但省去 push 的 `P*N` fan-out store；对某些大小与
拓扑更合适。

### 2.3 `2shot_pull`：大消息的 reduce-scatter + all-gather

第 1 shot 中 rank `r` 只负责输出的约 `1/P` 分片：从所有 rank 拉取该分片并求和。
第 2 shot 把已归约分片写进所有 rank 的 workspace，相当于 fused all-gather。
每 rank 的核心网络流量从 one-shot 的 `O(P*N)` 降到 `O(N)`，因此 rank 或消息变大
后更划算。实现把 `vec_offset` 预加到每个 peer base pointer 上，使 peer pointer
留在 uniform register，避免为每个线程重复做 64-bit 地址加法；源码注释记录这项
优化约省 12 registers/thread、提升约 10%。见
[`LoadStoreImpl` 与 2shot kernel](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh#L64-L109)。

在 H100/H200/Blackwell 等支持 NVLink multicast/NVLS 的系统上，pull plane 还可能
有 multicast VA。此时 `multimem.ld_reduce` 一条指令从 multicast team 的所有物理
副本 load 并求和，`multimem.st` 一次 fan-out 到所有副本；这是 NVSwitch fabric
in-network reduction 路径，不是普通 P2P load 的语法糖。SGLang PTX 封装见
[`ptx.cuh`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/include/sgl_kernel/distributed/ptx.cuh#L91-L159)，
CUDA 的 multicast/VMM 关系见 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#multicast-memory-sharing)。

## 3. 算法选择与 fallback

`_pick_config` 的顺序是：`1shot_push` → `1shot_pull` → multicast `2shot_pull` →
普通 `2shot_pull` → fallback。阈值分别按 CUDA Graph/eager、GPU 架构、SM 数和
world size 离线调优，不是一个普适常量；逻辑见
[`_pick_config`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py#L324-L361)，
表见
[`custom_all_reduce_v2.py config`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py)。

运行条件包括：

- tensor contiguous/weak-contiguous、总 byte 数是 16 的倍数；kernel 按 16B 向量化；
- world size 在当前架构 config 中；同节点必须构成完整 NVLink clique；
- 多节点还要求同一 NVLink clique 和 VMM-backed allocator；
- 消息大于调优阈值或 workspace 容量时回退 NCCL。

所以它是 inference TP fast path，不是 NCCL 的通用替代品。

## 4. 用到的 CUDA 特性

### P2P、UVA 与 IPC/VMM

peer access 让 GPU kernel 直接解引用另一张 GPU 的 device pointer。UVA 使这些
pointer 位于统一虚拟地址空间；是否可达仍须检查拓扑并启用 P2P。CUDA 官方说明见
[Multi-Device P2P Access](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-device-peer-to-peer-transfers-and-memory-access)。

多进程下裸 pointer 不能直接交换。legacy 使用 `cudaIpcGetMemHandle` /
`cudaIpcOpenMemHandle`；V2 由 PyTorch symmetric memory rendezvous 建 plane。CUDA
Graph 的普通 allocator 输入在 capture 后交换 IPC handle，expandable-segments
输入走 VMM remap，然后把每次 collective 的 peer pointers 写入 GPU pointer table。
这让被 capture 的 kernel 参数保持稳定。相关代码见
[`_register_graph_inputs`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py#L393-L458)，
IPC 机制见 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)。

### CUDA memory model 与 system scope

跨 GPU 可寻址不等于有正确的可见性。SGLang 的 peer payload 用
`ld/st.relaxed.sys.global.v4.b32`，barrier flag 用
`red.release.sys.global.add.u32` 与 `ld.acquire.sys.global.u32`。`.sys` 覆盖不同 GPU；
`.gpu` 只覆盖同一 device。release/acquire barrier 保证 barrier 前写入对 barrier 后
读可见；push 协议则把到达标记编码进 payload 本身，所以 relaxed access 足够。
封装见
[`ptx.cuh`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/include/sgl_kernel/distributed/ptx.cuh#L63-L89)，
线程 scope 语义见 [CUDA C++ Memory Model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html#thread-scopes)。

### 16B vector access、模板展开和 `__grid_constant__`

FP32 每线程一次处理 4 个元素，FP16/BF16 一次处理 8 个元素。16B 对齐减少指令数，
world size 是模板参数，peer loop 可完全展开。参数结构使用 `__grid_constant__`，
避免每个线程生成私有参数副本，并利于把相同 peer bases 放在 uniform/constant
datapath。这也是输入 byte 数必须是 16 的倍数的直接原因。

### block-level 跨 GPU barrier

pull 算法为每个 block 建独立 semaphore，前 `P` 个线程各通知一个 destination，
本 rank 对应线程轮询本地累计值。semaphore pad 到 128B，避免相邻 block 的 flag
共享 cache line。开始 barrier 只需 relaxed；2shot 在发布归约分片后使用
release/acquire barrier，再开始 all-gather。实现见
[`Barrier`](https://github.com/sgl-project/sglang/blob/d56706459c8e52ec3ab1c41dae778e4fe03e0da3/python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh#L158-L202)。

### Programmatic Dependent Launch（SM90+）

V2 在支持的架构上让 staging memcpy kernel 与后继 all-reduce 通过 PDL 链接，使用
`cudaGridDependencySynchronize`/trigger 语义减少同 stream kernel 间的完整串行空隙。
这是 opportunistic overlap，不能把它当成必然并发。官方语义见
[Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

## 5. 本仓库的非 CUTLASS examples

[`examples/custom_all_reduce/simple.cu`](../examples/custom_all_reduce/simple.cu) 是可独立
编译的 FP32 `1shot_push`：只使用 CUDA Runtime、普通 C++ 与两条 inline PTX
16B system-scope load/store，没有 CUTLASS、NCCL、PyTorch 或 SGLang 依赖。

它保留了关键机制：

- 全连接 P2P 检查与 `cudaDeviceEnablePeerAccess`；
- `[2 epochs][world ranks][vectors]` workspace；
- `+0.0 → -0.0` payload 编码、local polling、清零复用；
- 16B vector transaction、compile-time world size、每 block epoch counter；
- 多次连续 enqueue，用于覆盖双缓冲复用。

其余生产机制拆成独立文件，避免掩盖主 kernel：

| 文件 | 聚焦机制 | 本机要求 |
|---|---|---|
| [`fused_residual_rmsnorm.cu`](../examples/custom_all_reduce/fused_residual_rmsnorm.cu) | FP32 1-shot push 后直接融合 residual add、行内 RMSNorm 与 affine weight；带拆分基线和 CPU reference | 1 张 GPU 可做双逻辑 rank 验证；2 张 P2P GPU 走真实路径 |
| [`ipc_1shot_push.cu`](../examples/custom_all_reduce/ipc_1shot_push.cu) | `fork` 前建 socket、两进程分别绑定 GPU、交换 `cudaIpcMemHandle_t`、打开/关闭 peer mapping 与 exporter 生命周期 | Linux、2 张 P2P GPU、SM70+ |
| [`cuda_graph_pointer_table.cu`](../examples/custom_all_reduce/cuda_graph_pointer_table.cu) | capture 时只固定 device pointer table；capture 后登记地址；同一 graph replay 两组不同 pointer rows | 1 张 CUDA GPU |
| [`pdl_staged_pull.cu`](../examples/custom_all_reduce/pdl_staged_pull.cu) | eager pull 的 staging kernel → reduction kernel PDL 依赖；secondary 在 wait 前预取独立 peer 数据 | SM90+ |
| [`vmm_multicast_reduce.cu`](../examples/custom_all_reduce/vmm_multicast_reduce.cu) | `cuMulticastCreate/AddDevice/BindMem`、UC/MC 双 VA、`multimem.ld_reduce`、proxy fence 和完整释放顺序 | 2 张支持 multicast object 的 SM90+ NVSwitch GPU |

这些仍是教学拆解：IPC 文件只实现 FP32 push；Graph 与 PDL 文件用单 GPU 上的两个
buffer 隔离讲解地址登记/依赖语义；multicast 文件是单进程两 GPU，故不再重复
POSIX FD/FABRIC multicast-handle 交换。生产代码还需要进程组、错误传播、timeout、
自动阈值和 NCCL fallback。

1-shot/2-shot 的逐阶段远端流量公式、SM90 配置实例和选择矩阵另见
[`sglang_all_reduce_strategy.md`](sglang_all_reduce_strategy.md)。

编译与运行：

```bash
make -C examples custom_all_reduce/simple
make -C examples custom_all_reduce/fused_residual_rmsnorm
make -C examples custom_all_reduce/ipc_1shot_push
make -C examples custom_all_reduce/cuda_graph_pointer_table

# SM90+ 的两个独立目标
make -C examples custom_all_reduce/pdl_staged_pull
make -C examples custom_all_reduce/vmm_multicast_reduce

# simple 参数依次为 float 元素数、使用的 GPU 数、迭代次数
./examples/custom_all_reduce/simple 1048576 2 100

# AR -> residual -> RMSNorm 的拆分/融合数值对照
./examples/custom_all_reduce/fused_residual_rmsnorm

# IPC 参数依次为 float 元素数、迭代次数；固定 fork 两个 rank
./examples/custom_all_reduce/ipc_1shot_push 1048576 100
```

`simple` 和 IPC 示例要求至少两张互相可 P2P 的 SM70+ GPU。融合示例在单 GPU
环境用同一 grid 的两个 block 模拟逻辑 rank；双 GPU 时自动使用真实 P2P。程序会
在每个 rank 上和 CPU reference 比较，最后输出 `PASS` 或 `FAIL`。

## 6. 正确性陷阱

- 不能把普通 `volatile` 当成跨 GPU memory ordering；scope 和 acquire/release 必须
  与通信协议匹配。
- spin-wait kernel 必须控制 grid/occupancy。若同一 GPU 上等待的 grid 占满所有
  执行资源，而负责推进协议的 kernel 还未运行，可能永久死锁。
- communicator 是有状态的：epoch、slot、semaphore 都依赖所有 rank 的 collective
  顺序一致。不要从多个未排序 stream 并发复用。
- P2P “可用”不代表“值得用”。PCIe-only 多卡下 custom path 可能不如 NCCL，SGLang
  因而执行 NVLink clique 与消息大小检查。
- 正零 marker 方案只适合浮点 payload，且 producer 必须转换每个正零 atom；漏掉一个
  合法的 `+0.0` 会让 consumer 永久等待。
- two-shot 的第二阶段必须在归约分片对所有 peer 可见后才开始，单纯 `__syncthreads()`
  只同步一张 GPU 上的一个 block，不能替代 system-scope barrier。
