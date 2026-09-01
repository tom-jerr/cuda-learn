# CUDA Stream-Ordered Allocator 与 Memory Pool IPC

本文结合 CUDA Programming Guide 的 Stream-Ordered Memory Allocator 章节和本仓库的
[`examples/cuda_allocator/import_export_pool.cu`](../examples/cuda_allocator/import_export_pool.cu)，
说明 stream allocator、memory pool、exporter/importer 的关系，以及跨进程共享 allocation
时必须满足的同步与生命周期规则。

示例实现的是一个最小的 Linux 单 GPU 流程：parent 是 exporter，child 是 importer；
exporter 创建并导出 pool 和 allocation，importer 在同一块 GPU 上写入共享 allocation，
最后由 exporter 读回并验证结果。

官方参考：

- [CUDA Programming Guide：Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
- [CUDA Runtime API：Stream Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html)

## 1. 三层概念

这套 IPC 机制不是简单地把一个 device pointer 发给另一个进程，而是由三层机制共同完成：

| 层级 | 主要 API | 解决的问题 |
|---|---|---|
| Stream ordering | `cudaMallocAsync`、`cudaFreeAsync`、event | allocation 何时生效、何时失效 |
| Memory pool | `cudaMemPoolCreate`、`cudaMallocFromPoolAsync` | allocation 来自哪里、如何缓存和复用 |
| Memory pool IPC | pool handle、pointer export data、IPC event | 谁可以导入、导入哪次 allocation、何时可以访问 |

IPC 又分为两次共享：

1. 先共享 pool，建立访问能力和安全域；
2. 再共享 pool 中的一次具体 allocation。

其中 pool handle 和 allocation export data 不是同一种东西：

| 数据 | 含义 | 传输方式 |
|---|---|---|
| POSIX FD / Win32 handle / Fabric handle | pool 的 OS 或 fabric shareable handle | 使用对应平台的 handle 传输机制 |
| `cudaMemPoolPtrExportData` | 某次 allocation 的 opaque identity | socket、pipe、共享内存等任意 IPC |
| `cudaIpcEventHandle_t` | 跨进程 CUDA event 的 opaque handle | socket、pipe、共享内存等任意 IPC |
| `void *ptr` | 当前进程 CUDA 地址空间里的 device pointer | 不能直接发送给其他进程使用 |

共享 pool 只表示 importer 有能力导入属于该 pool 的 allocation，并不会自动把所有
allocation 映射到 importer。反过来，`cudaMemPoolPtrExportData` 只有和对应的 imported
pool 一起使用才有意义。

## 2. Stream-ordered allocator

传统 `cudaMalloc` / `cudaFree` 可能引入跨 stream 的同步。Stream-ordered allocator 把
分配和释放也作为 stream 中有顺序的操作：

```cpp
cudaMallocAsync(&ptr, size, stream);
kernel<<<grid, block, 0, stream>>>(ptr);
cudaFreeAsync(ptr, stream);
```

同一 stream 内天然满足：

```text
allocation 生效 → kernel 使用 → free 生效
```

但 `cudaMallocAsync` 或 `cudaMallocFromPoolAsync` 返回 pointer 时，分配操作可能仍在 stream
中等待执行。pointer 可以立即作为 opaque value 传给 CUDA API，但不能在 allocation
操作完成前读写。合法访问区间是：

```text
allocation 操作完成
        ↓
所有 kernel / memcpy 访问
        ↓
free 操作开始生效
```

在其他 stream 中使用时，必须用 event 或同步 API 建立依赖：

```cpp
cudaMallocAsync(&ptr, size, alloc_stream);
cudaEventRecord(ready, alloc_stream);

cudaStreamWaitEvent(use_stream, ready);
kernel<<<grid, block, 0, use_stream>>>(ptr);
cudaEventRecord(used, use_stream);

cudaStreamWaitEvent(free_stream, used);
cudaFreeAsync(ptr, free_stream);
```

否则可能发生 use-before-allocation 或 use-after-free，行为未定义。跨进程共享不会改变
这条基本规则，只是普通 event 需要换成 IPC event。

### 2.1 Memory pool 的作用

所有 `cudaMallocAsync` allocation 都来自 memory pool：

- `cudaMallocAsync` 使用 stream 所在设备的 current pool；
- `cudaMallocFromPoolAsync` 明确指定 pool，不需要修改设备的 current pool；
- 每个设备都有 default/implicit pool；
- `cudaMemPoolCreate` 创建 explicit pool，可以指定 location、IPC handle 类型、最大尺寸等属性。

普通 pool 会优先复用已经通过 `cudaFreeAsync` 释放的内存，再向 OS/driver 申请更多物理内存。
复用行为由以下属性控制：

- `cudaMemPoolReuseFollowEventDependencies`：根据 event 依赖复用其他 stream 释放的内存；
- `cudaMemPoolReuseAllowOpportunistic`：复用已经实际完成、但没有显式依赖关系的 free；
- `cudaMemPoolReuseAllowInternalDependencies`：允许 allocator 插入内部 stream 依赖以完成复用。

这些属性默认开启。普通非 IPC pool 还可以用 `cudaMemPoolAttrReleaseThreshold` 控制缓存
多少物理内存，用 `cudaMemPoolTrimTo` 主动缩减 pool，并查询 reserved/used 的当前值和峰值。

默认 pool 不支持 IPC，因此需要共享 allocation 时必须创建 IPC-capable explicit pool。

## 3. 完整的 exporter/importer 时序

示例的核心 happens-before 关系如下：

```text
Exporter stream                         Importer stream
────────────────────────────────────────────────────────────
MallocFromPoolAsync
        │
Record READY
        │
        └──── IPC READY ───────────────► Wait READY
                                         │
                                         write_kernel
                                         │
                                         Free imported ptr
                                         │
                                     Record DONE
        ◄──── IPC DONE ──────────────────┘
Wait DONE
        │
Memcpy D2H
        │
Free exporter ptr
```

展开后是：

```text
Exporter allocation
    happens-before
Importer kernel
    happens-before
Importer free
    happens-before
Exporter final use
    happens-before
Exporter free
```

其中 READY 防止 importer 过早访问，DONE 防止 exporter 过早释放。pool handle 和 pointer
export data 只描述共享对象，不会自动建立上述 GPU 执行依赖。

## 4. 示例逐步说明

### 4.1 在 CUDA 初始化之前 fork

程序先创建 Unix `socketpair`，然后 `fork`：

```cpp
int sv[2];
socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
pid_t pid = fork();
```

这发生在 parent 或 child 调用任何 CUDA API 之前，使两个进程分别初始化自己的 CUDA
runtime/context。socket 同时负责传递 pool FD、allocation export data、IPC event handle
和最后的 ACK。

### 4.2 Exporter 创建 IPC-capable pool

Exporter 创建 explicit pool：

```cpp
cudaMemPoolProps props{};
props.allocType = cudaMemAllocationTypePinned;
props.location.type = cudaMemLocationTypeDevice;
props.location.id = 0;
props.handleTypes = cudaMemHandleTypePosixFileDescriptor;

cudaMemPoolCreate(&pool, &props);
```

这些属性的含义是：

- allocation 位于 GPU 0；
- `cudaMemAllocationTypePinned` 结合 `cudaMemLocationTypeDevice` 表示驻留在该 device 的
  allocation，不是 `cudaMallocHost` 创建的 host pinned buffer；
- `handleTypes` 非零使 pool 具备 IPC 能力；
- handle 类型必须和后续 export/import API 使用的类型一致。

Exporter 随后导出 POSIX FD：

```cpp
int pool_fd;
cudaMemPoolExportToShareableHandle(
    &pool_fd,
    pool,
    cudaMemHandleTypePosixFileDescriptor,
    0);
```

`cudaMemPoolExportToShareableHandle` 的第一个参数是用于接收 FD 的地址。对应的 import API
则把收到的 FD 值转换为 `void *` 传入，这是 POSIX FD handle 类型规定的调用形式。

### 4.3 POSIX FD 必须使用 `SCM_RIGHTS`

FD 数值只在本进程的 descriptor table 中有意义。直接执行下面的操作是错误的：

```cpp
write(sock, &pool_fd, sizeof(pool_fd));
```

即便 importer 收到相同的整数，也可能指向完全不同的文件对象。示例正确使用
`sendmsg`、`recvmsg` 和 `SCM_RIGHTS`：

```cpp
cmsg->cmsg_level = SOL_SOCKET;
cmsg->cmsg_type = SCM_RIGHTS;
memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
```

发送成功后 exporter 可以关闭自己的导出 FD；通过 `SCM_RIGHTS` 传递给 importer 的是一个
新的 FD reference。长期运行的 importer 在成功导入 pool 后，也应按照平台 handle 的所有权
规则及时关闭收到的 FD，避免 descriptor 泄漏。本最小示例中的 child 很快退出，所以未关闭
的 FD 最终仍会由 OS 回收。

Windows 应使用 Win32 handle 及对应的跨进程 handle 复制机制；不能套用 POSIX FD 的传输方法。

### 4.4 Importer 导入 pool

Child 收到 FD 后创建 imported pool：

```cpp
cudaMemPoolImportFromShareableHandle(
    &pool,
    (void *)(intptr_t)pool_fd,
    cudaMemHandleTypePosixFileDescriptor,
    0);
```

Imported pool 是一个导入视图，不是完整 allocator。它不能：

```cpp
cudaDeviceSetMemPool(device, imported_pool);               // 不允许
cudaMallocFromPoolAsync(&ptr, size, imported_pool, stream); // 不允许
```

Importer 只能导入 exporter 已从原始 pool 创建并明确导出的 allocation。因此 pool 的内存
复用策略对于 imported pool 也没有实际意义。

如果 importer 使用的 GPU 与 pool resident device 不同，还要在 importer 进程中执行：

1. `cudaDeviceCanAccessPeer` 检查访问拓扑；
2. `cudaMemPoolSetAccess` 给访问 GPU 设置读写权限。

Imported pool 不继承 exporter 进程中设置的额外设备访问权限。当前示例的两个进程都在
GPU 0 上，resident device 默认已经可访问，所以不需要 `cudaMemPoolSetAccess`。

### 4.5 Exporter 创建并导出 allocation

Exporter 明确从该 pool 异步分配：

```cpp
float *ptr = nullptr;
cudaMallocFromPoolAsync(
    reinterpret_cast<void **>(&ptr),
    N * sizeof(float),
    pool,
    stream);
```

然后导出这次 allocation 的身份：

```cpp
cudaMemPoolPtrExportData data{};
cudaMemPoolExportPointer(&data, ptr);
```

`cudaMemPoolPtrExportData` 不是 OS handle，也不是 pointer。本机 CUDA 头文件中它当前是一个
opaque 结构，应用不得解析其内容。它可以作为普通字节通过 socket、pipe 或 shared memory
传输，但只能在对应 allocation 的有效生命周期中使用。

### 4.6 Importer 导入 allocation

Importer 用 imported pool 和 export data 创建本进程中的 pointer：

```cpp
float *ptr = nullptr;
cudaMemPoolImportPointer(
    reinterpret_cast<void **>(&ptr),
    imported_pool,
    &data);
```

该 pointer 映射到和 exporter 相同的底层 allocation，但属于 importer 的 CUDA 地址空间，
数值不保证与 exporter 的 pointer 相同。因此日志中两个 `%p` 是否相等没有语义，不能用来
判断是否共享成功。

`cudaMemPoolImportPointer` 不会等待 exporter stream 中的 allocation 操作完成。以下代码仍然
可能发生 use-before-allocation：

```cpp
cudaMemPoolImportPointer(...);
kernel<<<grid, block, 0, stream>>>(ptr); // 缺少 READY 依赖
```

### 4.7 READY：保证 importer 不会过早访问

Exporter 创建可跨进程使用、且不记录 timing 的 event：

```cpp
cudaEventCreateWithFlags(
    &ready,
    cudaEventInterprocess | cudaEventDisableTiming);
cudaEventRecord(ready, exporter_stream);
```

因为 `READY` 记录在 allocation 后面的同一 stream：

```text
MallocFromPoolAsync → READY
```

Exporter 用 `cudaIpcGetEventHandle` 得到 handle 并发给 importer。Importer 打开并等待：

```cpp
cudaIpcOpenEventHandle(&ready, ready_handle);
cudaStreamWaitEvent(importer_stream, ready, 0);
write_kernel<<<1, 32, 0, importer_stream>>>(ptr, N);
```

最终形成：

```text
Exporter allocation → READY → Importer wait → Importer kernel
```

注意 exporter 调用 `cudaEventRecord` 只是把 event 加入 stream，然后就可以立刻发送 event
handle。socket 传输不需要等 event 完成；importer 的 `cudaStreamWaitEvent` 会在 GPU 侧等待
event 真正完成。

### 4.8 Importer 必须先释放

Importer 在同一 stream 中使用和释放 pointer：

```cpp
write_kernel<<<1, 32, 0, stream>>>(ptr, N);
cudaFreeAsync(ptr, stream);
cudaEventRecord(done, stream);
```

顺序为：

```text
Importer kernel → Importer free → DONE
```

这里 importer 的 `cudaFreeAsync` 释放本进程对 allocation 的导入映射/引用，不会让 exporter
持有的 pointer 立即失效。Importer 在这个 free 之后不能再访问自己的 pointer，但 exporter
仍然可以继续访问原 allocation。

如果使用同步 `cudaFree`，应用仍需先显式保证该进程中所有异步访问已经结束；不能依赖它
替代正确的 stream/event 排序。

### 4.9 DONE：保证 exporter 最后释放

Exporter 打开 importer 发来的 DONE event，并把等待插入自己的 stream：

```cpp
cudaIpcOpenEventHandle(&done, done_handle);
cudaStreamWaitEvent(exporter_stream, done, 0);

cudaMemcpyAsync(host, ptr, sizeof(host), cudaMemcpyDeviceToHost,
                exporter_stream);
cudaFreeAsync(ptr, exporter_stream);
```

Exporter 在等待 DONE 后仍然读取了 allocation，验证 importer 写入的值，然后才执行最终
free。这说明 importer free 和 exporter free 的含义不同：前者释放导入引用，后者结束原始
allocation 的生命周期。

CUDA 要求所有 importer 都在 exporter 之前释放：

```text
Importer 0 free ─┐
Importer 1 free ─┼─► all importers done ─► Exporter free
Importer N free ─┘
```

多 importer 场景需要每个 importer 的 DONE event、host barrier 或可靠的引用计数协议。
只等待其中一个 importer 不够。

Exporter 可以在 host 侧较早调用 `cudaFreeAsync`，但该 free 所在 stream 必须依赖所有
importer 的 free。也就是说，host API 的调用时间可以早，GPU stream order 不能早。

### 4.10 清理对象

本例在 exporter 完成 D2H copy、最终 free 和 stream synchronization 后向 child 发送 ACK。
Child 收到 ACK 后才销毁 IPC event、imported pool 和 stream，保证这些 CUDA 对象覆盖对方的
使用期。

推荐的总体清理顺序是：

```text
Importer: last use → free imported pointer → free completed → destroy imported pool
Exporter: wait all importers → last use → free original pointer → destroy export pool
```

`cudaMemPoolDestroy` 可以在仍有未完成 allocation/free 时返回，资源会延迟到 outstanding
allocation 消失后回收；但工程代码仍应显式同步并按照清晰的所有权顺序销毁，避免把延迟
清理行为当作正常同步机制。

## 5. 关键不变量和常见错误

### 5.1 必须始终成立的规则

1. Allocation 必须来自 IPC-capable explicit pool；default pool 不支持 IPC。
2. Pool handle 类型必须在创建 pool 时写入 `cudaMemPoolProps::handleTypes`。
3. OS handle 必须使用该平台规定的 IPC 方式传递。
4. 不得把 exporter 的原始 device pointer 直接发送给 importer 使用。
5. Import pointer 成功不代表 exporter 的异步 allocation 已经 ready。
6. Importer 第一次访问必须 happens-after exporter allocation。
7. Importer 的 free 必须 happens-after importer 的所有访问。
8. 所有 importer 的 free 必须 happens-before exporter 的 free。
9. Pool、allocation export data、IPC event 和 OS handle 都要有明确的所有权与生命周期。
10. 多 GPU access 必须在 importer 中单独配置，不能假设继承 exporter 设置。

### 5.2 两类同步不能混淆

示例同时存在 host IPC 同步和 GPU stream 同步：

| 同步 | 示例机制 | 保证什么 |
|---|---|---|
| Host/control plane | socket 的 `read/write/sendmsg/recvmsg` | metadata 或 handle 已经送达 |
| GPU/data plane | IPC event + `cudaStreamWaitEvent` | GPU allocation、kernel、free 的执行顺序 |

收到 READY event handle 不表示 READY 已经完成；收到 DONE handle也不表示 importer free 已经
完成。只有把 event wait 加入目标 stream，才能建立 GPU 侧的 happens-before 关系。

### 5.3 常见错误

- 直接发送 `pool_fd` 的整数值，而不是用 `SCM_RIGHTS`；
- 发送 exporter 的 `ptr`，而不是 `cudaMemPoolPtrExportData`；
- `cudaMemPoolImportPointer` 后立即 launch kernel，没有等待 READY；
- exporter 看到 host 消息后立即 free，没有等待 importer 的 GPU free；
- importer free 后继续从 importer pointer 读写；
- 把 imported pool 设置为 current pool或尝试从中分配；
- 多 GPU 时只在 exporter 中调用 `cudaMemPoolSetAccess`；
- 多 importer 时只等待一个 DONE；
- 长期缓存已经失效 allocation 对应的 `cudaMemPoolPtrExportData`；
- 在长生命周期进程中忘记关闭收到的 OS handle。

## 6. 适用范围

### 6.1 适合的场景

- 同一台机器上的多进程 GPU pipeline，例如 decode、inference、postprocess 分进程运行；
- 一个中心 exporter/allocator 管理显存，多个 worker/importer 消费 buffer；
- 本地 MPI rank 或其他多进程计算任务共享临时工作区；
- allocation 创建和释放频繁，需要 pool 复用来减少同步和 driver/OS 分配开销；
- pool 只共享一次，之后持续共享多个 allocation；
- 同机多 GPU 共享，但拓扑支持 peer access，并正确配置 `cudaMemPoolSetAccess`；
- 希望通过 OS handle 控制哪些进程有资格导入 pool，而不是向任意进程暴露所有 allocation。

Pool-level handle 建立一次后，每个新 allocation 只需发送较小的
`cudaMemPoolPtrExportData` 和同步信息，适合长期运行的 producer/consumer 模型。

### 6.2 不太适合的场景

| 需求 | 更合适的选择或原因 |
|---|---|
| 偶尔共享一块长期存在的普通 `cudaMalloc` 内存 | 传统 `cudaIpcGetMemHandle` 可能更简单 |
| Importer 需要自主从共享 pool 分配 | Imported pool 不允许 allocation |
| 通过普通网络跨机器发送 POSIX FD | FD 只在同一 OS/节点的 handle 传递机制中有效 |
| CUDA 与 Vulkan/OpenGL/D3D 共享资源 | CUDA External Memory/Semaphore |
| 精确控制 VA、物理页、映射位置和子区间权限 | CUDA Virtual Memory Management API |
| 数据本来就需要跨节点通信/规约 | NCCL、GPUDirect RDMA 或显式网络传输 |
| 显存峰值后必须立即归还系统 | 当前 IPC pool 的 trim/release 存在限制 |

现代 CUDA 还提供 Fabric handle，可支持特定的跨节点共享配置，但它依赖驱动、硬件和管理员
配置的 IMEX channel。它不是把 POSIX FD 通过 TCP 发送到另一台机器，也不是对普通集群环境
透明可用的通用网络内存机制。

## 7. 当前 IPC pool 的限制

CUDA Programming Guide 当前列出的限制包括：

- IPC export pool 不能把 physical blocks 释放回 OS；
- `cudaMemPoolTrimTo` 对 IPC pool 没有效果；
- `cudaMemPoolAttrReleaseThreshold` 对 IPC pool 实际被忽略；
- imported pool 不能创建新 allocation、不能成为 current pool；
- imported pool 的 allocation reuse 属性没有实际含义；
- imported pool 的资源统计只反映该进程已导入的 allocation 和相关物理内存。

不能释放 physical blocks 是 driver 实现限制，官方说明它可能随未来驱动变化。因此长期运行、
负载峰谷差异很大的 IPC 服务需要特别关注显存高水位，不能照搬普通 pool 的 trim 策略。

## 8. 支持性与部署检查

正式代码应先查询设备和驱动能力：

```cpp
int pools_supported = 0;
cudaDeviceGetAttribute(
    &pools_supported,
    cudaDevAttrMemoryPoolsSupported,
    device);

int handle_types = 0;
cudaDeviceGetAttribute(
    &handle_types,
    cudaDevAttrMemoryPoolSupportedHandleTypes,
    device);

if (!pools_supported ||
    !(handle_types & cudaMemHandleTypePosixFileDescriptor)) {
  // 当前设备/驱动不支持示例所需的 memory pool IPC。
}
```

`cudaDevAttrMemoryPoolSupportedHandleTypes` 从 CUDA 11.3 开始提供；兼容更老驱动的程序应先
检查 driver version，避免属性查询本身返回 `cudaErrorInvalidValue`。

部署时还要检查：

- exporter/importer 是否能看到预期的 CUDA device；
- 容器是否正确暴露 GPU device、Unix socket 和相关权限；
- 多 GPU topology 是否支持所需的 peer access；
- handle 类型是否受当前 OS、driver 和 device 支持；
- exporter 异常退出、importer 超时或 importer crash 时，控制面如何回收/重建共享状态；
- 是否有可靠协议确保 exporter 不会在尚有 importer 时复用或释放 allocation。

## 9. 示例的消息协议

当前最小示例通过一个全双工 Unix socket 按以下顺序传输：

```text
Exporter → Importer: pool POSIX FD       (SCM_RIGHTS)
Exporter → Importer: allocation data     (plain bytes)
Exporter → Importer: READY event handle  (plain bytes)

Importer → Exporter: DONE event handle   (plain bytes)
Exporter → Importer: ACK                 (plain byte)
```

这个协议适合演示，但生产环境通常还应增加：

- protocol version、message type 和 payload length；
- allocation ID、size、dtype/layout 等业务 metadata；
- 多 importer reference count 或 barrier；
- timeout、peer crash 检测和错误传播；
- allocation generation，避免把过期 export data 误用于已经复用的地址；
- handle 和 event 的异常路径清理。

## 10. 总结

可以用一句话区分三类对象：

> Pool handle 解决“允许谁导入”，pointer export data 解决“导入哪一次分配”，READY/DONE
> event 解决“什么时候可以访问和释放”。

完整生命周期必须满足：

```text
Create exportable pool
  → Share/import pool
  → Allocate in exporter
  → Share/import allocation
  → READY dependency
  → Importer use
  → Importer free
  → DONE dependency
  → Exporter final use/free
  → Destroy pools and IPC objects
```

只完成 pool/pointer 的 export/import 还不够；真正决定程序是否正确的是跨 stream、跨进程的
allocation/use/free 顺序，以及 importer-first、exporter-last 的生命周期协议。
