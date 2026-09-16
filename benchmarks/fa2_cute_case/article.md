---
title: "用 CuTe 手写 FlashAttention-2：从线程布局、Online Softmax 到流水线与性能对照"
created: 2026-09-10
updated: 2026-09-10
tags:
  - CUDA
  - LLMInference
description: 以一份 BM=BN=D=64 的 CuTe FlashAttention-2 前向 Kernel 为案例，沿 global、shared memory、register 追踪线程所有权，推导 online softmax 和多 stage 流水，并与官方 FlashAttention 2.8.3 做可复现性能对照。
katex: true
---

# 用 CuTe 手写 FlashAttention-2：从线程布局、Online Softmax 到流水线与性能对照

FlashAttention-2 的公式不复杂，真正难的是把一块数据从 HBM 搬到 shared memory，再交给 `ldmatrix` 和 `mma.sync` 的寄存器 ABI；score 在寄存器里完成 softmax 后，还要原地变成下一次 MMA 的 A operand。最后的 O 明明已经在寄存器里，却又写一次 shared memory，才能高效落回 HBM。这些步骤表面上都是 `Tensor` 和 `Layout`，背后却对应 CTA、warp、lane 在不同时刻的所有权变化。

本文不从生产版模板展开，而是分析一份固定配置的教学实现：FP16 输入、FP32 累加，`BM=BN=D=64`，128 threads，`SM80_16x8x16_F32F16F16F32_TN`，支持 causal/non-causal 和 2/3-stage K/V 流水。固定配置让我们能把下面几个问题推到具体的线程和寄存器：

1. 一个 CTA、一个 warp、一个 4-lane group 和一个 thread 分别算哪一块 tile？
2. PTX 的四个 C 寄存器怎样变成 CuTe 打印出的 `trS: ((_2,_2),_1,_8)`？
3. 4-lane subgroup 为什么恰好拥有一整行 score，softmax 又怎样沿这个 layout 归约？
4. 第一次 GEMM 的 C fragment 怎样零搬运地成为第二次 GEMM 的 A operand？
5. 为什么 O 要从 register 写进 shared memory，再以 128 bit transaction 写回 HBM？
6. `partition_S/D`、`partition_fragment_A/B/C`、`retile_D`、`compose` 到底移动了数据，还是只建立 view？

先给出边界：这个 Kernel 使用 Ampere 引入的 `cp.async`、`ldmatrix` 和 `mma.sync.m16n8k16` 数据通路，但本次可用设备是 **RTX 4060 Laptop，compute capability 8.9（Ada）**。所以后文数字只能称为“SM89 实测”，不能冒充 A100 或 RTX 30 系列的 Ampere 实测。文末脚本提供 `--require-ampere`，可以在 SM80/86/87 上原样复跑并拒绝错误硬件。

## 1. 先走一遍 FA2：一个 query tile 如何流过所有 KV tiles

对一个 query tile，注意力前向为：

$$
S=QK^T,\qquad P=\operatorname{softmax}(S/\sqrt d),\qquad O=PV.
$$

若序列长为 $N$，显式保存 $S$ 或 $P$ 会产生 $O(N^2)$ 的 HBM 中间数据。FlashAttention 的 IO-aware 核心是把 K/V 沿序列维切成 tile：对一个 `64×64` query tile，CTA 每次只把当前 `64×64` K/V tile 搬进 shared memory，在寄存器里短暂保留当前 score fragment，并为每个 query row 维护三个跨 tile 状态：

$$
m_i=\max_{j\in\text{processed}} s_{ij},
$$

$$
\ell_i=\sum_{j\in\text{processed}}e^{s_{ij}-m_i},
$$

$$
o_i=\sum_{j\in\text{processed}}e^{s_{ij}-m_i}v_j.
$$

其中 $o_i$ 还是未归一化 numerator。所有 KV tile 完成后才计算 $O_i=o_i/\ell_i$。因此主循环要守住的算法不变量不是“当前 tile 做完了一次 softmax”，而是：`running_max`、`running_sum` 和 `trO` 始终使用同一个指数基准。

一块 CTA 固定一个 `(batch, head, query_block)`，因此只负责最终 O 的一个 `[64,64]` tile。grid 为：

```cpp
dim3 grid(seqlen / 64, heads, batch);
```

non-causal 时它遍历全部 KV tiles；causal 时 query block `q_block` 只遍历 `0..q_block`，并在最后一个 tile 内屏蔽上三角。把源码压缩成一轮可以得到：

```cpp
// CTA owns Q[q_block:q_block+64, 0:64] and O of the same shape.
load Q gmem -> smem -> trQ;
prefetch K/V tiles into a staged smem ring;

for (int t = 0; t < kv_tiles; ++t) {
  load K_t smem -> trK;
  trS = trQ * trK;                   // QK^T, FP32 C fragment
  mask(trS);                         // only diagonal tile for causal
  online_softmax(trS, m, l, trO);    // trS is overwritten by P_t
  trP_as_a = C_layout_to_A_view(trS);
  load V_t smem -> trV;
  trO += trP_as_a * trV;             // unnormalized numerator
}

trO /= l;
store trO registers -> O smem -> O gmem;
```

这段流程有两个不同层次的“不落 HBM”。score/P 从未物化为 $N\times N$ 的 global-memory 矩阵；O 的最终 epilogue 仍会经过 shared memory，但那只是 CTA 内的一次 ownership permutation，不是 $O(N^2)$ 中间结果。

## 2. 整体数据流：CuTe 编排的是坐标交接

```text
Q gmem ──G2SRow/cp.async──> Q smem ──ldmatrix.N──> trQ(A)
                                                        \
K gmem ──G2SRow/cp.async──> K smem ──ldmatrix.N──> trK(B) ── QKᵀ ──> trS(C)
                                                                           │
                                                        causal mask + online softmax
                                                                           │
                                                           C-layout ──view──> trP(A)
                                                                            \
V gmem ──G2SCol/cp.async──> V smem ──ldmatrix.T──> trV(B) ──────────────────── PV ──> trO(C)

trO(C, FP32) ──normalize/convert──> trO_half ──scatter──> O smem ──128-bit copy──> O gmem
```

这里最容易误读的是“同一个 Tensor 为什么有这么多 view”。CuTe 的 `Tensor` 是 `Engine + Layout`：Engine 决定数据位于 pointer 还是寄存器，Layout 是“逻辑坐标到 index”的函数。`partition` 和 `retile` 多数只创建 view，不发出内存指令；真正的数据移动发生在 `copy()`，真正的矩阵乘发生在 `gemm()`。

## 3. CTA、warp、thread：先写清楚谁拥有哪块 tile

本文的计算配置是：

```cpp
using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMma = decltype(make_tiled_mma(
    MmaAtom{},
    Layout<Shape<_4, _1, _1>>{},   // 4 warps along M
    Tile<_64, _64, _16>{}));       // CTA-level M,N,K tile
```

`MmaAtom` 描述一个 warp 执行的 `m16n8k16` 指令；`Layout<Shape<_4,_1,_1>>` 把 4 个 warp 全铺在 M 方向；`Tile<_64,_64,_16>` 再声明 CTA 层的逻辑 MMA tile。三层 ownership 如下：

![CTA、warp 与 lane 对 score tile 的所有权](img/cute-fa2-case/fa2_cta_warp_thread.svg)

| 层级 | 数量 | 在一个 `64×64` score tile 中的工作 |
| --- | ---: | --- |
| CTA | 1 | 固定一个 `(batch, head, q_block)`，产生 O 的 `64×64` tile |
| warp | 4 | warp $w$ 独占 query rows $16w\ldots16w+15$，N 方向覆盖全部 64 列 |
| 4-lane group | 每 warp 8 组 | group $g$ 共同拥有 row $16w+g$ 与 $16w+g+8$ |
| lane | 每组 4 个 | lane 内 $q=lane\bmod4$，每行持有 16 个离散列 |

这正是 FA2 论文所说的 Q-split warp partition：多个 warp 各算不同的 Q row slice，同时读取同一个 K/V tile。FA1 的 split-K 方案让 warp 产生同一 output tile 的部分和，需要经 shared memory 汇总；FA2 让每个 warp 在主循环中独占自己的 O rows，所以 QK→softmax→PV 期间没有跨 warp 的 partial-O reduction。

这里的“没有跨 warp 通信”只描述主循环。它不等于 kernel 从此不使用 shared memory：G2S/S2R 仍把 shared memory 当作 global transaction 与 MMA fragment 之间的交换点，最终输出也需要一次 shared-memory ownership permutation。

## 4. O 为什么先写 shared memory，再写 global HBM

先看一个容易混淆的事实：`trO` 已经包含最终数值，而且每个元素都有唯一 owner，直接按标量地址写 global memory 在正确性上没有问题。问题在于 owner 排列来自 MMA C fragment，而 HBM 希望看到连续、对齐且合并的 row-major transaction。

对 warp $w$、lane 内 $g=\lfloor lane/4\rfloor$、$q=lane\bmod4$，`trO` 的每个值满足：

$$
m=16w+g+8r,\qquad d=8j+2q+c,
$$

其中 $r,c\in\{0,1\}$，$j\in[0,7]$。固定一行和一个 8-half 区间后，这 8 个连续 half 分散在 $q=0,1,2,3$ 四个 lane：每个 lane 只拿相邻的两个。`compose` 只能让**同一 thread**用另一组坐标访问自己的寄存器，不能把四个 lane 的值聚到一个 thread 手里。因此它无法凭空得到 `uint128_t` 所需的连续 8 half。

这份实现复用已经结束生命周期的 Q shared buffer，做两次真实 copy：

```cpp
Tensor sO = make_tensor(make_smem_ptr(q_smem), SmemLayoutO{});
Tensor tOsO = thr_mma.partition_C(sO);
copy(trO_half, tOsO);                 // C owners scatter to logical sO(m,d)
__syncthreads();                       // all lanes have finished the scatter

auto s2g_thr = S2GRow{}.get_slice(threadIdx.x);
Tensor tOsO_vec = s2g_thr.partition_S(sO);
Tensor tOgO_vec = s2g_thr.partition_D(gO);
copy(S2GRow{}, tOsO_vec, tOgO_vec);   // one 16-byte vector per copy atom
```

第一段按 MMA C ownership 把值散到逻辑 `sO(m,d)`；barrier 后，第二套 `S2GRow` ownership 再让每个 thread 读取连续 8 half，并以 `UniversalCopy<uint128_t>` 写回。shared memory 在这里相当于一个显式、可同步的 lane-to-lane transpose。它不做数值归约，也没有增加峰值 shared-memory 容量，因为 `sO` 复用了 Q 的 8 KiB 区域。

论文解释了两件上层事实：FlashAttention 用 tiling 避免 score/P 的 $N^2$ HBM materialization；FA2 的 Q-split 避免主循环中的 inter-warp partial-output reduction。论文没有把 O epilogue 的每条 copy 展开成一个独立优化。更直接的证据来自官方 v2.8.3 源码：它同样先用 `make_tiled_copy_C` 将 `acc_o` 写入 `sO`，同步后再由 gmem tiled copy 从 `sO` 写 `gO`。因此，“主循环不做跨 warp 归约”和“epilogue 通过 smem 重排 store ownership”并不矛盾。

## 5. 读 CuTe API：先区分 view、owning fragment 与指令

下面这张表可以作为后文的索引。判断一行代码的第一问应当是：它只建立坐标函数，还是会分配/移动数据？

| API | 本例中的作用 | 是否移动数据 |
| --- | --- | --- |
| `Shape` / `Stride` / `Layout` | 定义坐标域以及 `coord → offset` | 否 |
| `make_tensor(engine, layout)` | 将 pointer、shared pointer 或寄存器存储与 Layout 绑定 | 否；但 register engine 的具体构造可能拥有存储 |
| `local_tile(tensor, tile, coord)` | 从全局 tensor 取 CTA 逻辑 tile view | 否 |
| `composition` / `Swizzle` / `tile_to_shape` | 组合地址函数，并把 atom 平铺到目标 shape | 否 |
| `Copy_Atom` | 描述一条 `cp.async`、`ldmatrix` 或 vector copy 的 value ABI | 否 |
| `make_tiled_copy` | 用 thread/value layout 把 Copy Atom 平铺成 cooperative copy | 否 |
| `get_slice(tid)` | 选择当前 thread 在 tiled operation 中的那一份 | 否 |
| `partition_S` / `partition_D` | 为当前 thread 创建 source/destination view | 否 |
| `copy(tiled_copy, src, dst)` | 按 atom 发出实际 copy 指令 | **是** |
| `MMA_Atom` / `make_tiled_mma` | 描述 PTX MMA ABI，并将 warp atom 平铺成 CTA tile | 否 |
| `partition_fragment_A/B/C` | 创建当前 thread 对应的 owning register fragment | 分配寄存器语义，不读原 tensor |
| `make_tiled_copy_A/B/C` | 从 MMA operand/accumulator ownership 派生兼容 copy | 否 |
| `retile_D(fragment)` | 将已有寄存器按 copy destination value layout 建 view | 否 |
| `make_tensor_like<T>(x)` | 新建与 `x` 同 shape/layout 的 owning register tensor | **新存储**；元素仍需显式写入 |
| `left_inverse` / `compose` | 建立坐标变换或重解释已有 engine | 否 |
| `gemm(tiled_mma, A, B, C)` | 发出 `mma.sync` 并更新 C fragment | **是** |

CuTe 文档对 `Tensor` 的核心定义是 `data()[layout()(coord)]`。这句话很短，却解释了为什么同一 pointer 可以先被看作 `V[N,D]`，再被看作 `Vᵀ[D,N]`；也解释了为什么 `partition_fragment_A(gQ)` 不会读取 `gQ`：这里需要的是 `gQ` 的 shape 来推导 fragment repeat，返回值的 engine 是线程私有寄存器。

## 6. Global memory：V 的转置只是换坐标解释

Q、K、O 的物理布局都是连续 `[N,D]`：

```cpp
make_shape(seqlen, HeadDim{}),
make_stride(HeadDim{}, _1{})
```

对 CTA 内坐标，地址分别为：

$$
Q(m,d)=Q_{base+(64q_{block}+m)64+d},
$$

$$
K(n,d,t)=K_{base+(64t+n)64+d}.
$$

第二次 GEMM 需要 $P[M,N]V[N,D]$。代码把同一份物理 V 创建为逻辑 `Vᵀ[D,N]`：

```cpp
Tensor mVt = make_tensor(
    make_gmem_ptr(v + bh_base),
    make_shape(HeadDim{}, seqlen),
    make_stride(_1{}, HeadDim{}));
```

其地址函数为：

$$
L_{V^T}(d,n)=d+64n=L_V(n,d).
$$

这里没有 transpose kernel，也没有额外 HBM 流量。发生变化的是坐标语义，物理地址完全相同。

## 7. G2S：128 个线程怎样搬完一个 64×64 tile

Q/K 使用的 `G2SRow` 为：

```cpp
using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>;
using G2SRow = decltype(make_tiled_copy(
    G2SAtom{},
    Layout<Shape<_32, _4>, Stride<_4, _1>>{},
    Layout<Shape<_1, _8>>{}));
```

一条 copy 搬 16 B，即 8 个 FP16。令：

$$
r=\lfloor tid/4\rfloor,\qquad c=tid\bmod4.
$$

基础 copy tile 是 `32×32`，完整 `64×64` 在行列方向各重复两次。线程 `tid` 搬运：

$$
row=r+32\rho_r,
$$

$$
col=8c+v+32\rho_c,
$$

其中 $v\in[0,7]$，$\rho_r,\rho_c\in\{0,1\}$。因此每线程搬 32 个 half，全 CTA 正好覆盖 `128×32=4096` 个元素。

```cpp
auto copy_thread = g2s_row.get_slice(threadIdx.x);
Tensor tQgQ = copy_thread.partition_S(gQ);
Tensor tQsQ = copy_thread.partition_D(sQ);
copy(g2s_row, tQgQ, tQsQ);
```

这四行可以逐句理解：

- `get_slice(tid)` 从 TiledCopy 的 `(thread,value)` 布局取出当前线程的方案；
- `partition_S(gQ)` 创建当前线程的 source view；
- `partition_D(sQ)` 用相同的逻辑坐标创建 destination view；
- `copy()` 才真正发出 `cp.async`。

source 是 row-major，destination 是 XOR-swizzled shared layout，但它们通过相同的 `(row,col)` 坐标相遇。producer thread 与后续 S2R consumer thread 也无需相同，shared memory 就是两套 ownership 的交换点。

V 使用 `G2SCol`，把连续的 8-value vector 放在逻辑 D 维：

$$
d=8c+v+32\rho_d,\qquad n=r+32\rho_n.
$$

因为 $V^T(d,n)$ 的 stride 是 `(1,64)`，这仍是连续的 16-byte global transaction。

## 8. Shared memory：Swizzle 同时服务 producer 与 consumer

Q/K 的 shared-memory atom 是：

```cpp
composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<_8, _64>, Stride<_64, _1>>{})
```

对逻辑 `(row,d)`，当前固定配置可写成：

$$
offset(row,d)=64row+\left(d\oplus((row\bmod8)\ll3)\right).
$$

最低 3 bit 不变，所以 8-half 的 `cp.async` 仍落在连续 16 B；row bits 被 XOR 到 column chunk bits 后，`ldmatrix` 读取多个 row 时又能避开朴素 row-major 的 bank 聚集。完整的 bank/word 推导可参考站内文章 [从 Bank Conflict 到 CUTLASS Swizzle](从%20Bank%20Conflict%20到%20CUTLASS%20Swizzle：推导%20ldmatrix%20的访存布局.md)。

shared-memory 容量随 stage 数变化：

$$
bytes=(1+2\times kStages)\times64\times64\times2.
$$

Q 在整个 KV 循环中不变，只有 1 份；K/V 是环形 buffer，各有 `kStages` 份：

| 配置 | Q | K | V | dynamic shared memory |
| --- | ---: | ---: | ---: | ---: |
| 2-stage | 8 KiB | 16 KiB | 16 KiB | 40 KiB |
| 3-stage | 8 KiB | 24 KiB | 24 KiB | 56 KiB |

这张表已经提示了一个性能边界：stage 增加的不是免费并行度，而是用 shared-memory residency 换取更深的在途 copy。

## 9. S2R：MMA 先定义寄存器 ABI，ldmatrix 再负责填入

TiledMMA 为：

```cpp
using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMma = decltype(make_tiled_mma(
    MmaAtom{},
    Layout<Shape<_4, _1, _1>>{},
    Tile<_64, _64, _16>{}));
```

4 个 warp 全部铺在 M 方向：warp 0–3 分别负责 16 行。MMA A operand 的寄存器位置是硬件 ABI，必须先由 `partition_fragment_A` 创建：

```cpp
auto mma_thread = tiled_mma.get_slice(threadIdx.x);
Tensor trQ = mma_thread.partition_fragment_A(gQ);
```

这里 `gQ` 只提供 shape、元素类型和 repeat 次数，没有读取 global memory。`trQ` 是 owning register Tensor，概念 shape 为 `(8,1,4)`，即每线程 32 个 FP16。

接着从 MMA A ownership 派生兼容的 `ldmatrix` copy：

```cpp
auto q_s2r = make_tiled_copy_A(S2RAtomN{}, tiled_mma);
auto q_loader = q_s2r.get_slice(threadIdx.x);
Tensor sQ_for_load = q_loader.partition_S(sQ);
Tensor rQ_for_load = q_loader.retile_D(trQ);
copy(q_s2r, sQ_for_load, rQ_for_load);
```

`retile_D(trQ)` 没有再分配寄存器，也没有 copy。`rQ_for_load` 与 `trQ` 指向同一批寄存器，只是前者按 `ldmatrix` destination value layout 看，后者按 MMA A fragment layout 看：

```text
ldmatrix 写 rQ_for_load
mma.sync 读 trQ
```

因此这段依赖关系应读成：MMA 决定消费 ABI，S2R copy 决定怎样从 swizzled shared memory 把数据直接装进该 ABI。

## 10. 从 PTX 的四个 C 寄存器推导 `trS: ((_2,_2),_1,_8)`

不要从 CuTe 打印结果猜语义，先回到 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 的 PTX lane ABI。对一个 warp，令 $g=\lfloor lane/4\rfloor$、$q=lane\bmod4$。PTX ISA 规定每个 lane 的 accumulator 是四个 FP32 寄存器：

| register | atom row | atom column |
| --- | ---: | ---: |
| `c0` | $g$ | $2q$ |
| `c1` | $g$ | $2q+1$ |
| `c2` | $g+8$ | $2q$ |
| `c3` | $g+8$ | $2q+1$ |

两个二元坐标就能描述这四个值：`c` 选择相邻的两个 N columns，`r` 选择相隔 8 的两个 M rows。若 atom 内寄存器次序为 `c+2r`，四项依次是 `c0,c1,c2,c3`。

本例的 warp tile 是 `16×64`。M 方向只需要一个 `m16` atom，N 方向需要 $64/8=8$ 次重复，所以 `partition_fragment_C` 打印为：

```text
shape  = ((_2,_2),_1,_8)
stride = ((_1,_2),_0,_4)
          │    │    │
          │    │    └─ N repeat 每前进一次，跨 4 个 FP32 slots
          │    └────── M repeat 只有 1，stride 退化为 0
          └─────────── atom 内 slot = c + 2r
```

第一维本身还是嵌套坐标 `(c,r)`。源码中的：

```cpp
trS(make_coord(col_item, row_item), 0, nj)
```

对应 thread-local register slot：

$$
slot_C(c,r,j)=c+2r+4j.
$$

![PTX C fragment、CuTe trS 与 P-as-A 的坐标关系](img/cute-fa2-case/fa2_fragment_layout.svg)

令：

$$
w=\lfloor tid/32\rfloor,\quad lane=tid\bmod32,
$$

$$
g=\lfloor lane/4\rfloor,\quad q=lane\bmod4.
$$

score C fragment 中，当前线程拥有的坐标为：

$$
m=16w+g+8r,
$$

$$
n=8j+2q+c,
$$

其中 $r,c\in\{0,1\}$，$j\in[0,7]$。它和上面的 slot 公式共同给出 `fragment coordinate → register slot → logical matrix coordinate` 的完整映射。每 thread 持有 $2×2×8=32$ 个 FP32 score：两个 query rows，每行 16 个列元素。

固定 `w,g,r` 后，连续 4 个 lane 的 `q=0..3` 在每个 $j$ 上覆盖 8 个相邻列；遍历 8 个 $j$ 后正好覆盖整行 64 列：

$$
4\ lanes\times16\ scores/lane=64\ scores.
$$

这也说明 `trS` 的 layout 不是结果生成后附加的元数据，而是 Tensor Core 把 accumulator 交给 lane 时已经确定的 ABI。

causal mask 同样直接使用这套坐标：

```cpp
const int query_row = row_item == 0 ? row0 : row1;
const int key_col = nj * 8 + lane_col * 2 + col_item;
if (key_col > query_row) score = -INFINITY;
```

这里只需要 mask 对角 query tile。更早的 KV tiles 全在因果范围内，更晚的 KV tiles 根本不会进入循环。

## 11. 沿 `trS` 做 Online Softmax：为什么归约宽度是 4

softmax 首先在每个 thread 内沿 `c` 和 `j` 扫描。对固定的 `r`，一个 lane 得到自己 16 个数的局部 max：

```cpp
for (int j = 0; j < 8; ++j)
  for (int c = 0; c < 2; ++c)
    tile_max[r] = max(tile_max[r], trS(make_coord(c, r), 0, j));
```

固定 `w,g,r` 后，`q=0..3` 的 4 个 lane 合起来覆盖同一 row 的全部 64 列，所以第二步只需在宽度为 4 的 subgroup 内做 XOR shuffle：

```cpp
for (int delta = 1; delta < 4; delta <<= 1)
  value = op(value, __shfl_xor_sync(0xffffffffu, value, delta, 4));
```

`width=4` 将 warp 切成 8 个独立 subgroup，恰好对应 `g=0..7`。每个 subgroup 同时负责两个 rows，因此 `tile_max[2]`、`tile_sum[2]`、`running_max[2]` 和 `running_sum[2]` 都是长度 2。归约宽度来自 PTX C ownership；改变 MMA atom、warp layout 或 tile N 后，必须重新推导。

当前 tile 得到 row max $m_t$ 后，与历史状态合并：

$$
m_{new}=\max(m_{old},m_t),
$$

$$
\alpha=e^{m_{old}-m_{new}}.
$$

当前 tile 概率分子为：

$$
p_j=e^{s_j-m_{new}},\qquad \ell_t=\sum_jp_j.
$$

于是新的分母是：

$$
\ell_{new}=\alpha\ell_{old}+\ell_t.
$$

旧 numerator 原先以 $m_{old}$ 为指数基准，也必须换到 $m_{new}$：

$$
o_{new}=\alpha o_{old}+P_tV_t.
$$

这解释了代码中常被漏看的 rescale：

```cpp
trO(... row0 ...) *= alpha0;
trO(... row1 ...) *= alpha1;
gemm(tiled_mma, trP_as_a, trV, trO);
```

若只更新 `running_sum` 而不缩放旧 `trO`，每个 KV tile 实际使用了不同的指数基准，结果会随 tile 边界改变。

本实现先对 raw QK score 求 max，再乘 `kScale=1/8`。因为 scale 为正数，`max(scale*x)=scale*max(x)`，这样少做了 score fragment 上的大量乘法。官方 FA2 进一步用 `exp2f(x*log2(e))`，让缩放和减 max 更容易合成 FMA；教学版使用 `__expf`，语义清楚，但少了一层生产优化。

## 12. P 如何从 C layout 变成下一次 MMA 的 A layout

第一次 GEMM 后，`trS` 先被逐元素覆盖为 FP32 概率，再显式转换成一个新的 FP16 owning fragment：

```cpp
Tensor trP_as_c = make_tensor_like<Element>(trS);
for (int i = 0; i < size(trS); ++i)
  trP_as_c(i) = Element(trS(i));
```

这里确实发生了 FP32→FP16 的寄存器写入。接下来第二次 GEMM 是 $P[M,N]V[N,D]$，P 必须满足 MMA A operand 的 fragment ABI。关键问题不是两个 fragment 都有 32 个元素，而是每个 A 所需元素是否仍由同一个 thread 持有。

当前 warp 只沿 M 分布，一个 warp 已拥有完整 `16×64` score strip。C fragment 的寄存器编号可写成：

$$
slot_C(c,r,j)=c+2r+4j.
$$

对 `m16n8k16` 的 A operand，一个 lane 在一个 atom 内持有 8 个 half。把 atom value 写成 `(c,r,h)`：`c,r,h` 都取 0/1，其中 `h` 选择 K16 内的 `0/+8` 半区。N=64 作为第二次 GEMM 的 K 维，还要重复 $64/16=4$ 次，记为 $\kappa$。概念 shape 为：

```text
(((2,2),2),1,4)
   c r  h  M K-repeat
```

其 thread-local slot 与逻辑矩阵坐标是：

$$
slot_A(c,r,h,\kappa)=c+2r+4h+8\kappa,
$$

$$
m=16w+g+8r,\qquad k=16\kappa+2q+c+8h.
$$

第二次 GEMM 的 A-k 就是第一次 score 的 n。令 $j=2\kappa+h$，便有：

$$
8j+2q+c=16\kappa+8h+2q+c,
$$

$$
slot_C(c,r,j)=c+2r+4(2\kappa+h)=slot_A(c,r,h,\kappa).
$$

所以 A 需要的每个元素不仅留在同一 thread，甚至已经位于相同的线性 register slot，只需改变嵌套坐标。代码没有手写 $j=2\kappa+h$，而是让 Layout algebra 从同一个 compact logical element space 自动求出映射：

```cpp
auto layout_as_c = mma_thread.partition_C(compact_score).layout();
auto layout_as_a = mma_thread.partition_A(compact_score).layout();
auto a_to_c = left_inverse(layout_as_c).compose(layout_as_a);
auto trP_as_a = trP_as_c.compose(a_to_c);
```

`layout_as_c` 将 C fragment 坐标映射到逻辑 P 元素编号，`layout_as_a` 对 A fragment 做同样的事；`left_inverse(layout_as_c).compose(layout_as_a)` 得到“A坐标 → C坐标”。最后的 `compose` 只创建 view，`trP_as_a.data()` 与 `trP_as_c.data()` 相同，没有 copy、shuffle 或 shared-memory round trip。

这个零搬运转换成立，是因为 `Layout<Shape<_4,_1,_1>>` 让两次 GEMM 都按 M 切 warp，P 的 thread ownership 没变。若第二次 MMA 换了 warp partition，layout algebra 最多能描述目标坐标，不能把别的 thread 的寄存器搬过来；那时仍需要 shuffle 或 shared memory。官方 FA2 的 `convert_layout_acc_Aregs` 完成的正是同类 C-accumulator→A-register view 转换。

## 13. Epilogue：从 FP32 `trO` 到合并写回

主循环结束时：

```text
trO          : FP32 numerator accumulator
running_sum  : 每个 query row 的 softmax denominator
```

当前线程负责两个 query rows，每行 16 个 D 元素，因此先做每行归一化并转换为 FP16：

```cpp
trO_half((col, 0), 0, j) = half(trO((col, 0), 0, j) / running_sum[0]);
trO_half((col, 1), 0, j) = half(trO((col, 1), 0, j) / running_sum[1]);
```

MMA C layout 适合 `mma.sync`，却不保证跨线程 global store 合并。Kernel 复用 Q 的 shared buffer 完成一次 ownership 转换：

```text
register C fragment
  → mma_thread.partition_C(sO) scatter 到 swizzled shared memory
  → CTA barrier
  → S2GRow 以每线程 16 B 的连续向量写 global memory
```

P 的 C→A 转换能零搬运，是因为只涉及同一 thread 内的寄存器坐标；output store 关系到四个 lane 共同拥有的一段连续地址，必须真正重排数据。第 4 节已经给出官方实现的同构 epilogue；这里的 `partition_C(sO)` 与 `S2GRow.partition_S(sO)` 正是 shared memory 两侧的两套 ownership。

## 14. 2/3-stage 流水：模板参数会穿过容量、索引和等待协议

![2-stage K/V 环形流水](img/cute-fa2-case/fa2_two_stage_pipeline.svg)

对 `kStages`，prologue 先填入：

$$
\min(kv\_tiles,kStages-1)
$$

个 K/V tiles。steady state 的索引为：

```cpp
read_stage    = tile % kStages;
prefetch_tile = tile + kStages - 1;
write_stage   = prefetch_tile % kStages;
```

计算 tile $t$ 时，producer 把 tile $t+kStages-1$ 写入环形 buffer。下一轮消费前：

```cpp
cp_async_wait<kStages - 2>();
__syncthreads();
```

这两步不能合并理解：

- `cp_async_wait<N>` 约束当前线程发出的 async-copy group，确保未决 group 数降到阈值；
- `__syncthreads()` 让 CTA threads 在 ownership 交接点会合，防止 consumer 在其他 producer threads 尚未完成或旧 stage 尚未读完时继续。

进入 tail 后不再提交新 group，代码改用 `cp_async_wait<0>()` 完全 drain。若仍使用 `wait<kStages-2>`，它可以在最后一个预取 group 尚未完成时合法返回。

3-stage 还会把 dynamic shared memory 从 40 KiB 提到 56 KiB，因此 launch helper 必须通过 `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` 申请超过传统 48 KiB 的容量。只把 layout 的 stage extent 从 2 改成 3，既不构成正确的环形流水，也可能直接 launch 失败。

## 15. 与官方 FlashAttention 2.8.3 比的究竟是什么

对照固定在 upstream tag v2.8.3 的 D64 FP16 forward specialization。薄绑定直接调用：

```cpp
flash::run_mha_fwd_<cutlass::half_t, 64, causal>(params, stream);
```

计时路径没有 Python transpose/copy，也不包含 output allocation。双方读取同一份物理 `[B,H,N,64]` 数据，softmax scale 都是 `1/8`。

这仍不是功能等价的产品对比。教学版只覆盖固定 D=64、N 为 64 的倍数、无 dropout、定长 MHA；官方 Kernel 还保留通用参数协议并写出 `softmax_lse`。更关键的实现差异是：

| 维度 | CuTe 教学版 | 官方 FA2 v2.8.3 D64 路径 |
| --- | --- | --- |
| tile | `64×64` | `128×128` |
| threads | 128（4 warps） | 128（4 warps） |
| shared memory | 40 KiB（2-stage）/56 KiB（3-stage） | 由 `128×128` traits 决定的 Q/K/V workspace |
| 输出 | O | O + softmax LSE |
| shape 支持 | D=64、N%64=0 | 生产级多 shape/边界分派的一条特化路径 |
| softmax exponent | `__expf` | `exp2f` + `log2(e)` scale |

所以小 shape 的结果尤其不能解读为“教学实现全面击败官方”。`64×64` tile 对 N=64/128 的工作粒度更贴合，且输出契约更窄；官方 `128×128` 路径在这些输入上承担了更多固定成本。

## 16. 正确性与计时方法

测试在 PowerShell 启动的 Ubuntu-22.04 WSL 中完成，环境如下：

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，24 SM，compute capability 8.9 |
| CUDA / nvcc | 12.8 / V12.8.93 |
| PyTorch | 2.10.0+cu128 |
| CUTLASS | v4.6.1 对应 commit `e05f953...` |
| FlashAttention | v2.8.3 对应 commit `060c918...` |
| 编译 | `-O3 -std=c++17 --use_fast_math -arch=sm_89` |
| dtype / D | FP16 / 64 |

同一份 2-stage 与 3-stage 源码还额外使用 `-arch=sm_80` 做了 compile-only 检查，两者均编译通过，`ptxas` 报告 0 stack frame、0 spill store、0 spill load。这个检查证明 Ampere target 能生成代码，不代表取得了 Ampere 硬件性能数据。

正确性不是只抽查几行：FP32 参考实现关闭 TF32，按 128 个 query rows 分块计算完整 attention；测试覆盖 26 组 `(B,H,N,causal)`，N 从 64 到 4096，并额外加入 N=320 的 uniform-score 和放大 Q/K 的 concentrated-softmax 输入。所有输出满足 `atol=2e-3, rtol=2e-3`，观测到的最大绝对误差为 `1.26e-3`。

计时使用预分配 output、同地址 warmup、CUDA Graph 内重复 100 次 forward、CUDA Event 计时，共 9 rounds，并轮转三个实现的测量顺序。图中点为 median，淡色带是观测到的 min–max。笔记本测试结束时 GPU 达到 84°C，短 kernel 的样本范围也较宽，因此原始 JSON 保留了每轮数据，不能把小于几个百分点的差异当作稳定排序。

> [!NOTE]
> `compute-sanitizer --tool racecheck` 在当前 WSL/WDDM 设备上报告“不支持调试接口”，因此本文不声称完成 racecheck。同步正确性的证据来自布局 ownership 推导与上述数值边界测试；若在 Linux 原生 Ampere 环境复跑，应补上 racecheck。

## 17. 性能结果：小 shape 赢在粒度，高并发回到资源效率

先看单 head：

![CuTe FA2 与官方 FA2 延迟对比](img/cute-fa2-case/fa2_latency.svg)

代表性中位延迟如下，单位为 μs：

| B×H | N | mode | CuTe 2-stage | CuTe 3-stage | 官方 FA2 | 2-stage / 官方 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 64 | non-causal | 3.154 | 3.163 | 8.509 | 2.70× faster |
| 1 | 512 | non-causal | 10.536 | 10.660 | 18.862 | 1.79× faster |
| 1 | 2048 | non-causal | 66.058 | 75.592 | 69.376 | 1.05× faster |
| 1 | 4096 | non-causal | 231.178 | 261.089 | 285.706 | 1.24× faster |
| 1 | 512 | causal | 11.026 | 11.243 | 18.452 | 1.67× faster |
| 1 | 4096 | causal | 187.023 | 184.945 | 193.800 | 1.04× faster |
| 8 | 4096 | non-causal | 1592.410 | 1918.032 | 1562.163 | 1.02× slower |
| 32 | 2048 | non-causal | 1612.339 | 1958.623 | 1566.177 | 1.03× slower |
| 32 | 2048 | causal | 919.859 | 1101.025 | 863.498 | 1.07× slower |

`B×H=1` 时，总 CTA 数等于 query tile 数。教学版 BM=64 产生的 CTAs 是官方 BM=128 的两倍，在只有 24 个 SM 的设备上更容易填出 wave；小 N 时 `64×64` 也减少 tile 浪费和固定工作。这解释了 N=64–1024 的明显优势。

当 `B×H` 增加到 8 或 32，CTA 数不再稀缺，官方更大的 `128×128` tile 能摊薄调度、softmax 与 epilogue 成本。到 `B=4,H=8,N=2048`，官方 non-causal 快约 3%，causal 快约 7%。瓶颈从“有没有足够 block”迁移为“每个 block 的有效工作与资源效率”。

这里还有一个反常点：单 head、N=4096 的教学版仍更快，但 `B×H=8/32` 时优势消失。这说明不能从一个长序列 shape 推出 Kernel 的普遍吞吐量；head/batch 并行度会改变 CTA wave 和 tile 固定成本的占比。

## 18. 3-stage 为什么多数更慢

![2-stage 与 3-stage 延迟比例](img/cute-fa2-case/fa2_stage_ratio.svg)

在选取的多 head/batch workload 中，3-stage 比 2-stage 慢 10%–66%。编译结果没有 spill：2-stage non-causal 使用 159 registers/thread，3-stage 使用 160；差异主要来自 shared memory 从 40 KiB 增至 56 KiB，以及更深流水没有创造足够的独立计算来抵消 residency 下降。

这揭示了 stage 调优的共同约束：

$$
\text{useful overlap}=f(\text{async depth},\text{compute window},\text{smem},\text{registers},\text{resident CTAs}).
$$

K/V tile 每轮只有 8 KiB+8 KiB，QK、softmax、PV 已提供一段计算窗口。2-stage 能在消费当前 tile 时预取下一 tile；增加第三份 buffer 并不自动增加 global-memory bandwidth，反而可能让一个 SM 同时驻留的 CTA 更少。stage 数必须与 tile、寄存器预算和目标 SKU 一起选择。

## 19. 这个案例里真正可迁移的东西

这份固定 Kernel 的 `64×64`、4-lane subgroup、`Swizzle<3,3,3>` 和精确寄存器公式都不是通用答案。值得迁移的是推导顺序：

1. 先用 TiledMMA 写出 `(warp,lane,value)→(m,n,k)`，再决定 softmax reduction group；
2. 把 G2S 与 S2R 当成两套独立 ownership，通过 shared memory 的逻辑坐标交接；
3. 先让 MMA 定义 register ABI，再从它派生 `ldmatrix` copy；
4. 判断 fragment 是否能零搬运重解释时，检查的是 thread ownership，而不是 shape 看起来是否相同；
5. 模板化 stage 时同时修改 layout 容量、环形索引、prologue、steady-state wait、tail drain 和 launch shared-memory 属性；
6. 性能结论必须跨序列长度与 `B×H` 检查，否则会把 occupancy 问题误判成单 CTA 效率。

生产版还必须处理非整 tile、变长序列、GQA/MQA、dropout、local window、不同 head dimension 和架构 dispatch。教学 Kernel 的价值是把坐标、数据移动、数值状态与流水协议放在同一个可验证案例里，而不是代替官方实现。

## 20. 如何复现

完整快照、官方 D64 薄绑定、结果 JSON/CSV 和作图脚本位于本文随附的 `code/fa2_cute_case/`；开发仓库中的可执行版本位于 `benchmarks/fa2_cute_case/`。在当前 WSL 环境运行：

```bash
cd /home/lzy/cuda_learn
CUDA_HOME=/usr/local/cuda-12.8 \
  /home/lzy/.python/miniinfer/bin/python \
  benchmarks/fa2_cute_case/bench.py \
  --output benchmarks/fa2_cute_case/results_sm89_clean
```

在 A100/A10/RTX 30 等 Ampere 设备上复跑时增加硬件守卫：

```bash
python benchmarks/fa2_cute_case/bench.py \
  --require-ampere \
  --output benchmarks/fa2_cute_case/results_ampere
```

脚本会按当前 compute capability 编译 2/3-stage 教学 Kernel 与官方 FA2 D64 specialization，若设备不是 SM80/86/87 则直接退出。A100 与 GA10x 的 SM 数、shared-memory 容量、时钟和功耗差别很大，结果应分别记录，不能合并成一个“Ampere 数字”。

## Reference

- Tri Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), 2023.
- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Dao-AILab FlashAttention v2.8.3, [`flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/flash_fwd_kernel.h) 与 [`flash_fwd_launch_template.h`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/flash_fwd_launch_template.h).
- NVIDIA CUTLASS v4.6.1, [CuTe Tensor 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html)、[CuTe GEMM tutorial](https://github.com/NVIDIA/cutlass/blob/v4.6.1/media/docs/cpp/cute/0x_gemm_tutorial.md)、[`mma_atom.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.6.1/include/cute/atom/mma_atom.hpp) 与 [`copy_atom.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.6.1/include/cute/atom/copy_atom.hpp).
- NVIDIA, [PTX ISA 11.0：`mma.m16n8k16` fragment layout](https://docs.nvidia.com/cuda/archive/11.0/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type)、[`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async) 与 [`ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix).
- 站内：[CuTe 初探：以 FlashAttention-2 拆解 Layout、TiledCopy 与 TiledMMA](CuTe%20初探：以%20FlashAttention-2%20拆解%20Layout、TiledCopy%20与%20TiledMMA.md)。
