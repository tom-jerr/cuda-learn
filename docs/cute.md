# CuTe C++ 入门：从 Layout 到 TiledCopy、TiledMMA 和流水线 GEMM

对应代码：[`examples/cute_gemm/simple.cu`](../examples/cute_gemm/simple.cu)。它依据所给
Ampere HGEMM 实现缩小而来，保留最能体现 CuTe 抽象层次的主循环，暂时把复杂的 shared
memory epilogue 换成直接写回，以便先看清 CuTe 如何描述数据和线程。

## 1. 引入与构建

CuTe C++ 是 NVIDIA CUTLASS 仓库中的 header-only 组件。本项目将 CUTLASS 固定为
`third_party/cutlass` Git submodule，核心入口是：

```cpp
#include <cute/tensor.hpp>
```

这不等于只需要 CUDA Toolkit；还必须初始化 CUTLASS submodule。本示例用 CUTLASS
v4.6.1 和 CUDA 13.0 在 sm_89 上验证过：

```bash
git submodule update --init --recursive

# 方式一：只编 standalone 示例
make -C examples cute_gemm/simple
./examples/cute_gemm/simple 256 256 256

# 方式二：使用项目 CMake target
cmake -B build-cute \
  -DPython_EXECUTABLE="$HOME/.python/miniinfer/bin/python"
cmake --build build-cute --target cute_hgemm -j
./build-cute/examples/cute_gemm/cute_hgemm 256 256 256
```

CuTe target 默认进入 CMake 配置，因此 `compile_commands.json` 会包含正确的 CUTLASS
include path，clangd 可以解析模板；它使用 `EXCLUDE_FROM_ALL`，普通
`cmake --build build` 不会额外编译这个教学程序。若只想配置主库，可传
`-DCUDA_LEARN_ENABLE_CUTE_EXAMPLE=OFF`。

## 2. 先建立总图：CuTe 的层级

CuTe 的核心思想不是“提供一个 GEMM 函数”，而是用 layout algebra 描述：逻辑坐标如何
映射到存储地址、线程编号、线程持有的 value 编号，以及硬件指令的寄存器槽位。

```text
标量层       Int<N>, Shape, Stride, Coord
                   │
映射层             Layout : Coord -> linear index
                   │
数据层             Tensor = Engine + Layout
                   │
分块层             Tile / local_tile / partition
                  /                              \
搬运层   CopyOperation -> CopyAtom -> TiledCopy   计算层   MMAOperation -> MMAAtom -> TiledMMA
                              │                                      │
                         ThrCopy + fragment                     ThrMMA + fragment
                              \                                      /
                               copy() -> pipeline -> gemm() -> epilogue
```

读一个 CuTe kernel 时，建议始终按下面四个问题追踪：

1. 当前 Tensor 的逻辑 shape 是什么？
2. 它的 layout 把坐标映射到哪个地址空间、什么 offset？
3. 哪个 TiledCopy/TiledMMA 把 tile 分给了哪些线程和 value？
4. `partition_*` 后新增的每个 mode 表示什么？

## 3. 静态整数、Shape、Stride 与 Layout

### 3.1 `Int<N>` 和普通 `int`

```cpp
auto M = int(m);       // runtime value，类型是 int
auto BM = Int<128>{};  // compile-time value，类型携带 128
```

CuTe 大量优化依赖编译期 shape。`Int<128>` 让编译器知道循环次数、fragment 大小和 layout
变换结果，因此能展开循环并做静态合法性检查。问题的 M/N/K 可以动态，但 CTA/MMA tile
通常应静态。

常见下划线别名 `_1`、`_8`、`_16` 等就是静态整数类型。占位符 `_` 则不是 `_1`；它表示
“保留这个 mode，不在本次切片中固定它”。

### 3.2 Layout 是函数，不是内存

```cpp
auto row_major = make_layout(
    make_shape(Int<4>{}, Int<8>{}),
    make_stride(Int<8>{}, Int<1>{}));
```

该 layout 可以写成：

```text
shape  = (4,8)
stride = (8,1)
L(i,j) = i*8 + j
```

所以：

- `shape(layout)`：坐标域；
- `stride(layout)`：每个 mode 对地址的贡献；
- `layout(coord)`：坐标到线性 offset 的映射；
- `size(layout)`：逻辑元素数，shape 各 mode 的乘积；
- `cosize(layout)`：覆盖该映射所需的存储容量，swizzle 或带洞 stride 时不一定等于 size。

两者区别很重要：分配 shared memory 应使用 `cosize`，而不是凭逻辑元素数猜容量。

### 3.3 Shape 可以是层次化的

CuTe 的 mode 可以嵌套：

```text
((_4,_8),(_2,_2))
```

外层 rank 是 2，但第 0、1 个 mode 内部又各有子 mode。这让一个 layout 同时表达
lane/value、warp/MMA repeat 等层次，不必把所有东西压平成一个整数。`size<0>(x)` 访问
外层第 0 mode；继续访问可写成 `size<0,1>(x)`。

### 3.4 `composition` 和 Swizzle

示例的 shared layout：

```cpp
using SmemLayoutAtom = decltype(composition(
    Swizzle<3,3,3>{},
    make_layout(make_shape(Int<8>{}, Int<32>{}),
                make_stride(Int<32>{}, Int<1>{}))));
```

可以把它读成函数复合：

```text
(row,k) --普通 row-major layout--> linear offset --XOR swizzle--> smem offset
```

Swizzle 改变物理 offset，却不改变逻辑 `(row,k)` 坐标。目的在于让 `ldmatrix` 所需的
线程访问更均匀地落到 32 个 shared-memory bank，同时仍能通过同一个逻辑坐标访问。
这正是 layout algebra 的价值：算法代码仍在使用 `(m,k)`，bank 映射封装在 layout 中。

`tile_to_shape(atom, (BM,BK,Stages))` 再把小 layout atom 重复铺满整个 shared tensor。

## 4. Tensor：Engine + Layout

```cpp
Tensor mA = make_tensor(
    make_gmem_ptr(a),
    make_shape(m, k),
    make_stride(k, Int<1>{}));
```

Tensor 不是 owning container；它近似一个零开销 view：

```text
Tensor = Engine + Layout
Engine = pointer / register array / counting iterator + address-space information
Layout = logical coordinate -> Engine offset
```

因此 `mA(i,j)` 最终是 `a[i*k+j]`，而 `make_gmem_ptr` 告诉 CuTe 这是 global-memory
地址。类似地，`make_smem_ptr` 标记 shared memory。寄存器 fragment 则通常由
`make_fragment_*` 或 `partition_fragment_*` 创建，Engine 是线程私有的 owning storage。

示例的数据语义是：

```text
A: (M,K), stride (K,1), row-major
B: (N,K), stride (K,1), row-major
D: (M,N), stride (N,1), row-major
D = A * B^T
```

这就是名字中的 TN。参考实现虽然 PyTorch binding 检查的是 `b[K,N]`，kernel 却把传入
指针构造成 `(N,K)` row-major；只有当调用侧实际传入物理转置且 contiguous 的 B 时才与
kernel 一致。直接把普通 contiguous `[K,N]` 交进去，在多数非方阵 shape 上语义会错。

## 5. Tile 与 `local_tile`

```cpp
Tensor gA = local_tile(mA, make_tile(BM{}, BK{}),
                       make_coord(blockIdx.y, _));
```

`local_tile` 做两件事：

1. 用 `(BM,BK)` 把完整 `(M,K)` tensor 分块；
2. 固定 M tile 为 `blockIdx.y`，用 `_` 保留 K tile mode。

结果可按 `(BM, BK, k_tile)` 理解。`gA(_,_,tile)` 是当前 CTA 在第 `tile` 次 mainloop
需要的 A tile。B 同理，D 的 M/N tile 都已由 blockIdx 固定，因此没有 K tile mode。

这里的 `_` 也用于 Tensor slicing：

```cpp
gA(_, _, 0)       // 固定第 2 mode，保留前两个 mode
sA(_, _, stage)   // 取某一个 pipeline stage
```

切片通常只生成新的 Tensor view，不搬数据。

## 6. Atom、TiledCopy 与线程分区

### 6.1 CopyOperation 和 CopyAtom

```cpp
using G2SCopyAtom =
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>;
```

- `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`：具体硬件 copy operation，一条指令搬 128 bit；
- `Copy_Atom<..., half_t>`：在该指令上附加 source/destination value layout 和元素类型；
- 一个 CopyAtom 描述最小、不可再分的协作搬运单元。

S2R 使用：

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, half_t>
```

它对应 non-transpose 的 `ldmatrix.x4` 型 shared-to-register 搬运。名称带 SM75 是因为
`ldmatrix` 从 Turing 引入，不表示它不能用于 SM80/SM89。

### 6.2 `make_tiled_copy`

```cpp
using G2SCopy = decltype(make_tiled_copy(
    G2SCopyAtom{},
    make_layout(make_shape(Int<32>{}, Int<4>{}),
                make_stride(Int<4>{}, Int<1>{})), // ThrLayout: 128 threads
    make_layout(make_shape(Int<1>{}, Int<8>{}))   // ValLayout: 8 half/thread
));
```

TiledCopy = CopyAtom + ThrLayout + ValLayout + 推导出的 Tiler。

- `ThrLayout`：tile 坐标如何映射为 thread id；这里 `32*4=128` 个线程；
- `ValLayout`：同一线程的一条 copy 指令拥有哪些 value；这里是连续 8 个 half，即 16 B；
- `Tiler`：这组 thread/value 一轮共同覆盖的逻辑 tile；程序打印为 `(32,32)`。

“每线程 8 个 half”不代表每个完整 A tile 只搬 8 个。TiledCopy 的基础 coverage 是
`32x32`，`partition_*` 会把它继续重复到 `128x32` 的 CTA A tile。

### 6.3 TiledCopy → ThrCopy → thread tensor

```cpp
auto thr_copy = g2s_copy.get_slice(threadIdx.x);
Tensor tAgA = thr_copy.partition_S(gA);
Tensor tAsA = thr_copy.partition_D(sA);
```

这三层分别是：

```text
TiledCopy：整个 CTA 的搬运方案
ThrCopy：其中一个 thread 的搬运方案
tAgA/tAsA：该 thread 在具体 source/destination Tensor 上看到的 fragment view
```

`partition_S` 和 `partition_D` 必须来自同一个 ThrCopy。二者 shape 兼容，于是：

```cpp
copy(g2s_copy, tAgA(_,_,_,tile), tAsA(_,_,_,stage));
```

能发出该线程对应的 `cp.async`。所有线程执行同一行代码，合起来完成整个 CTA tile。

常见 fragment mode 注释：

```text
tAgA: (CPY, CPY_M, CPY_K, k_tile)
tAsA: (CPY, CPY_M, CPY_K, stage)
```

第 0 个 `CPY` mode 是单个 CopyAtom 内的 value mode；中间 mode 是 atom 在 CTA tile 上
重复的次数；最后一个 mode 来自原 tensor 的 K tile 或 pipeline stage。

## 7. MMAAtom、TiledMMA 与寄存器 fragment

### 7.1 MMAAtom 封装一条 warp 指令

```cpp
using MmaAtom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
```

该 atom 表示 32 个 lane 协作执行一次：

```text
D[16,8] = A[16,16] * B[8,16]^T + C[16,8]
input/accumulator/output: FP16
```

它不仅包含 shape，还包含 lane id 和 lane 内寄存器 value 到 A/B/C 逻辑坐标的 TV
(thread-value) layout。手写 PTX 时最难维护的 fragment 寄存器映射，就在这里被类型化了。

### 7.2 TiledMMA 把 atom 铺到更多 warp/value

```cpp
using TiledMma = decltype(make_tiled_mma(
    MmaAtom{},
    make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{})),
    Tile<Int<32>, Int<32>, Int<16>>{}));
```

三个参数分别是：

1. `MmaAtom`：一条 warp MMA 指令的线程/value 映射；
2. atom thread layout `(2,2,1)`：M/N 方向各放 2 个 warp，共 4 warp = 128 threads；
3. permutation/atom tiler `(32,32,16)`：规定该 tiled MMA 的基础逻辑 value tile。

不要把第三个参数直接当成 CTA tile。这里 TiledMMA 的基础 tile 是 32x32x16，但它可以在
更大的 `gD[128,128]`、`gA[128,32]` 上重复 partition；重复次数会成为 fragment 的
`MMA_M/MMA_N/MMA_K` mode。

### 7.3 TiledMMA → ThrMMA → fragment

```cpp
auto thr_mma = tiled_mma.get_slice(threadIdx.x);
auto tDgD = thr_mma.partition_C(gD);
auto tDrA = thr_mma.partition_fragment_A(gA(_,_,0));
auto tDrB = thr_mma.partition_fragment_B(gB(_,_,0));
auto tDrD = thr_mma.make_fragment_C(tDgD);
clear(tDrD);
```

- `partition_A/B/C`：按 MMA 的 TV layout 给当前线程投影 source/destination view；
- `partition_fragment_A/B`：同时分区并分配符合 MMA operand 类型的寄存器；
- `make_fragment_C`：根据 C 投影 shape 分配 accumulator 寄存器；
- `clear`： accumulator 清零。

由 TiledMMA 创建 fragment 的重要意义是：不要手算 lane 0 的 `a_frag[0]` 对应哪几个
矩阵元素。只要后续 copy 与 TiledMMA 兼容，CuTe 会让寄存器 layout 对齐。

## 8. 为什么 S2R TiledCopy 要从 TiledMMA 派生

```cpp
auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtom{}, tiled_mma);
auto s2r_thr_a = s2r_copy_a.get_slice(threadIdx.x);
auto tAsA_mma = s2r_thr_a.partition_S(sA);
auto tArA = s2r_thr_a.retile_D(tDrA);
```

`make_tiled_copy_A/B(atom, tiled_mma)` 让 copy 的 thread/value 分布与 MMA operand 的
thread/value 分布兼容。然后：

- `partition_S(sA)`：用 copy layout 看 shared source；
- `retile_D(tDrA)`：不重新分配寄存器，只给已有 MMA fragment 建立一个 copy-compatible
  destination view。

`partition` 是“把一个大 tensor 分给线程”，`retile` 是“同一线程已经拥有这些值，现在
换一种逻辑分组看它们”。这是 `retile_D/retile_S` 最实用的理解。

于是下面这行完成 shared → MMA operand registers，并落在正确的寄存器槽位：

```cpp
copy(s2r_copy_a, tAsA_mma(_,_,k_block,stage), tArA(_,_,k_block));
```

## 9. `cute::copy` 与 `cute::gemm` 是分派入口

```cpp
copy(g2s_copy, src, dst);    // 选中 cp.async atom
copy(s2r_copy_a, src, dst);  // 选中 ldmatrix atom
gemm(tiled_mma, rA, rB, rD); // 选中 mma.sync atom
```

这些调用表面相似，但不是运行时检查类型再决定指令。copy/MMA atom、layout 和 fragment
类型都在编译期确定，模板展开后生成相应硬件指令。Layout 自身通常也不产生额外运行时
对象或查表。

## 10. 两级流水线

参考实现同时流水化两个距离不同的数据通路：

```text
global tile i+1 --cp.async--> shared stage write
shared tile i   --ldmatrix--> register fragment next
register fragment current --mma.sync--> accumulator
```

示例使用两个 shared stages：

1. prologue 先提交 stage 0 的 A/B；
2. `cp_async_fence()` 提交一个 async group；
3. `cp_async_wait<0>()` 等到所需 group 完成；
4. `__syncthreads()` 保证数据对整个 CTA 可见；
5. 在当前 register fragment 做 MMA 时，提交下一个 global tile；
6. `read_stage/write_stage` 环形递增。

`cp_async_wait` 只处理异步 copy group 的完成状态，不是 CTA barrier；因此后面的
`__syncthreads()` 不能省。反过来，只调用 `__syncthreads()` 也不能替代 async wait。

寄存器又做了一层 ping-pong：计算 `k_block` 前，先加载 `next_k_block`。本例 BK=32、
MMA_K=16，所以每个 shared stage 含两个 `k_block`。

## 11. Epilogue：本示例与所给实现的差别

本示例最后执行：

```cpp
copy(tDrD, tDgD);
```

`tDrD` 与 `tDgD` 都由同一个 ThrMMA 的 C partition 得到，逻辑 ownership 相同，所以
写回正确。它的优点是短且容易验证，缺点是 accumulator fragment 的 lane 分布未必对应
最理想的连续 global store。

所给实现使用更完整的 epilogue：

```text
MMA accumulator registers
  │ make_tiled_copy_C + retile_S
  ▼
shared C scratchpad（重新排布）
  │ 自定义 S2G TiledCopy，每线程 uint128_t
  ▼
row-major global D（连续 16-byte stores）
```

其中：

- `make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma)` 把 C fragment 的寄存器分布转换成
  shared-memory copy 视图；
- `retile_S(tCrD)` 是同一 accumulator 的 source 重解释；
- `partition_D(sC)` 决定每线程写 shared 的位置；
- 第二个 `S2GCopyC` 再独立决定 shared → global 的合并写分布；
- `group_modes<1,3>` 把多个层次 mode 压成一个遍历维，便于按 batch/pipe 循环；
- 中间 `make_tensor_like<T>` 处理 accumulator 类型和最终 output 类型不同的情况。

所以 epilogue 的 shared memory 不是为了跨 K 重用，而是 layout conversion scratchpad：把
Tensor Core 喜欢的寄存器布局转成 global memory 喜欢的连续布局。

## 12. 如何读程序打印的类型

运行示例会打印类似：

```text
TiledMMA:
  ThrLayoutVMNK:  (_32,_2,_2,_1):(_1,_32,_64,_0)
  PermutationMNK: (_32,_32,_16)
MMA_Atom
  Shape_MNK:      (_16,_8,_16)

G2SCopy:
  Tiler_MN:       (_32,_32)
  ValueType:      16b
```

冒号左侧是 shape，右侧是 stride。`ThrLayoutVMNK` 中 `_32` 是 atom 内 32 lanes，后面的
`_2,_2,_1` 是 atom 在 M/N/K 上的 warp repeat；stride `_1,_32,_64,_0` 表示 lane 连续，
两个 M warp 相差 32、两个 N warp 相差 64，K 方向只有一个位置所以 stride 为 0。

`G2SCopy Tiler_MN=(32,32)` 表示一轮 TiledCopy 的逻辑 coverage，不是说 CTA tensor 只能
是 32x32。打印是理解复杂类型最快的办法；需要图形时也可在 host 侧用 CuTe 的
`print_latex(layout)` 生成 LaTeX 映射图。

## 13. 所给实现逐段映射到层级

| 所给代码 | CuTe 层级 | 解决的问题 |
|---|---|---|
| `make_shape/make_stride` | 基础代数 | 定义坐标域和坐标到 offset 的函数 |
| `make_tensor` | 数据 view | 将 pointer/address space 与 layout 组合 |
| `local_tile` | CTA 分块 | 从完整矩阵得到当前 CTA tile |
| `Swizzle + composition` | shared layout | 改变 bank 映射而保持逻辑坐标 |
| `Copy_Atom<cp.async>` | 搬运 atom | 选择 16-byte global→shared 指令 |
| `make_tiled_copy` | CTA copy | 定义 128 threads 与每线程 values 的 ownership |
| `partition_S/D` | thread copy | 把实际 source/destination tensor 投影到一个线程 |
| `MMA_Atom` | 计算 atom | 描述一条 warp Tensor Core 指令及 TV layout |
| `make_tiled_mma` | CTA MMA | 将 MMA atom 铺到多个 warp 和 value tile |
| `partition_fragment_A/B` | MMA registers | 创建符合 operand 寄存器 layout 的 fragment |
| `make_tiled_copy_A/B` | copy/MMA 桥接 | 让 ldmatrix 结果匹配 MMA 输入槽位 |
| `retile_D/S` | thread 内重解释 | 同一批寄存器换成 copy-compatible view |
| `cute::gemm` | 指令分派 | 在 fragment 上生成 `mma.sync` |
| `make_tiled_copy_C` | epilogue 桥接 | 从 accumulator TV layout 转到 shared layout |
| `group_modes` | layout 变换 | 合并层次化 mode 以便线性遍历 |

## 14. 这个教学实现刻意保留的边界

- 仅适用于 compute capability 8.0+ 的 FP16 Tensor Core / `cp.async` 路径；仓库默认
  sm_89 合法；
- 要求 `M%128==0、N%128==0、K%32==0`，且 K 至少 32；没有 predication/zero fill；
- B 的物理 layout 是 contiguous `[N,K]`，计算 `A @ B.T`；
- accumulator 是 FP16，与所给 `F16F16F16F16` atom 一致。生产 GEMM 通常更偏好
  FP32 accumulator，例如相应的 F16/F16/F32 atom；
- 两级 pipeline 展示结构，但没有自动 stage 选择、persistent scheduling、split-K 等；
- direct epilogue 以清晰为目标，不代表最佳带宽。

所给实现的 tile 是 `BM=128、BN=256、BK=32`，因此它自己的无 predication fast path
要求 `M%128==0、N%256==0、K%32==0`。它的 prologue 还会无条件预取 `Stages-1` 个 K
tile，所以还要满足 `K/BK >= Stages-1`；例如 Stages=4 时 K 至少为 96。kernel 开头只
判断 tile 左上角是否越界，不能保护 tile 内的尾部元素。

继续优化时，合理顺序是：先加边界 predication，再把 accumulator 换成 FP32，然后移植
所给 R2S/S2G epilogue，最后才比较 Stages=2/3/4 和 block swizzle。每一步都应保留单独的
正确性测试，因为 CuTe 类型能保证 layout 兼容性，却不会替你证明业务层的矩阵方向和
边界索引正确。

## 参考资料

- [NVIDIA CUTLASS：CuTe dense GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)
- [NVIDIA CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- CUTLASS checkout 中的 `examples/cute/tutorial/sgemm_sm80.cu`
