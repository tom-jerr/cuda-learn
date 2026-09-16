# 从 simple CUDA GEMM 理解分块、优化与评估

对应可运行代码：`examples/gemm/simple.cu`。它计算 row-major 的单精度矩阵乘：

\[
C_{M\times N}=A_{M\times K}B_{K\times N},\qquad
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}.
\]

## 1. 先写对：naive GEMM

naive 版让一个线程计算一个 `C[row,col]`：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
float acc = 0.0f;
for (int k = 0; k < K; ++k)
  acc = fmaf(A[row * K + k], B[k * N + col], acc);
C[row * N + col] = acc;
```

它是最重要的正确性基线，但数据复用很差。计算一个输出要读 `K` 个 A 和 `K` 个 B；相邻
线程虽然能合并读取 B，同一行 A 也可能命中 cache，但算法没有显式保证复用。GEMM 共执行
约 `2MNK` FLOPs，而 naive 的 A/B 读取上界接近 `8MNK` bytes，算术强度很低。

## 2. simple 优化版的数据流

代码采用：

```text
BM=32, BN=32, BK=16       CTA 输出 tile 和 K tile
TM=4,  TN=4               每线程输出 tile
block=(8,8)               64 threads = 2 warps

global A[32,16] ─┐
                  ├─> shared memory ─> registers ─> 4x4 outer product
global B[16,32] ─┘
```

一个 CTA 负责 `C` 的 `32×32` 区域。K 方向每次前进 16：先由整个 CTA 把 A 的
`32×16` tile 和 B 的 `16×32` tile 搬入 shared memory，再由每个线程计算自己的
`4×4` 输出。

### shared-memory tiling 为什么快

在一个 K stage 中：

- 计算量：`2×BM×BN×BK = 32768` FLOPs；
- A/B 搬运：`4×(BM×BK + BK×BN) = 4096` bytes；
- 忽略最终 C store，CTA 内算术强度约 `8 FLOP/byte`。

同一个 A 元素被 BN 个输出复用，同一个 B 元素被 BM 个输出复用。代价是每个 K stage
需要两次 `__syncthreads()`：第一次保证 tile 已写完，第二次保证 tile 已读完才能覆盖。

### register tiling 为什么继续快

如果每线程只算一个 C，读一次 shared A/B 只能做一次 FMA。现在每线程保存 16 个
accumulator；每个 `kk` 只读 4 个 A 和 4 个 B，就能做 `4×4=16` 次 FMA。这是寄存器级
outer product reuse。

tile 也不能无限增大：`TM×TN` 越大，寄存器越多，可能产生 register spilling，并降低每个
SM 可同时驻留的 CTA 数；shared tile 越大也会压低 occupancy。因此 BM/BN/BK/TM/TN 是
需要按 GPU 和 shape 调优的参数，而不是越大越好。

### 边界为什么要分别使用 M/N/K

三种矩阵的合法坐标分别是：

```text
A: row < M, col < K
B: row < K, col < N
C: row < M, col < N
```

只用 `128×128×128` 方阵测试会掩盖把 N、K 或 block 行列写混的 bug。示例默认使用
`M=513,N=509,K=515`，故意覆盖 M/N/K 都不是 tile 整数倍的路径，越界 load 以 0 填充。

## 3. 从本例继续优化

建议一次只改变一个因素，并始终对照 naive 和库基线：

1. **合并和向量化访存**：让 warp 搬运连续地址；对齐后用 `float4`/128-bit load。边界 tile
   保留 scalar predicated path。
2. **shared-memory layout**：用 Nsight Compute 检查 bank conflict；必要时 padding 或
   XOR swizzle。不要看到二维 shared 数组就盲目 padding，应由真实访问映射决定。
3. **增大 CTA/warp tile**：例如 128×128 CTA、每 warp 固定负责子 tile，提高数据复用和
   每次同步对应的计算量；同时观察寄存器、shared memory 和 occupancy。
4. **双缓冲**：准备两份 shared tile，在计算 tile `i` 时预取 tile `i+1`。Ampere 及以后可用
   `cp.async` 隐藏 global-to-shared 延迟。仅换成异步指令后立刻 wait，并没有形成流水线。
5. **Tensor Core**：FP16/BF16/TF32 可用 WMMA 或 `mma.sync`，FP32 accumulator；性能跃迁
   同时来自低精度、Tensor Core 和更低带宽，评估时必须和相同 dtype 的 cuBLAS 比较。
6. **CTA 调度与 L2 locality**：大矩阵可改变 `(blockIdx.x, blockIdx.y)` 到 tile 的映射，缩短
   A 或 B 的 L2 reuse distance；收益依赖 shape 和 GPU，适合 autotune。
7. **epilogue fusion**：把 bias、activation、scale 等融合进 C 写回，减少一次完整的读写。

生产场景优先使用 cuBLAS/cuBLASLt 或 CUTLASS；手写 kernel 的意义是理解数据搬运、针对固定
shape/dtype/fusion 做专门化，而不是期待一个 tile 配置赢过所有形状。

## 4. 如何评估 GEMM

### 4.1 正确性

至少测试：

- 方阵、长条矩阵、`M/N/K` 互不相等；
- tile 整数倍和非整数倍；
- 小尺寸、`K` 很大、含正负数和不同数值尺度；
- NaN/Inf 检查，以及 `max_abs`、`max_rel`；
- 按 dtype 设容差，不能把 FP32 容差照搬给 FP16/BF16。

本例先比较 tiled 和 naive 的全部输出，再抽样用 CPU double accumulation 检查，避免两个 GPU
kernel 恰好共享同一种索引错误。完整数值测试还应使用高精度 CPU/cuBLAS reference。

### 4.2 延迟和吞吐量

用同一 CUDA stream 上的 CUDA Event 计 kernel 时间，先 warmup，再记录多次结果。不要把
首次 context 初始化、JIT、malloc 或 H2D copy 混进 kernel 时间；如果产品关心端到端延迟，
再单独测端到端。

```text
FLOPs       = 2*M*N*K
GFLOP/s     = FLOPs / time_ms / 1e6
TFLOP/s     = FLOPs / time_ms / 1e9
speedup     = baseline_time / kernel_time
peak效率    = measured_TFLOP/s / 对应dtype理论峰值
库效率      = cublas_time / kernel_time
```

报告 min、median、mean 或 P50/P90，而不是只挑最好的一次。本例以 median 计算吞吐量，并同时
打印 min/median/mean。

### 4.3 带宽、算术强度与 Roofline

如果理想化地认为 A/B 各读一次、C 写一次，则最低逻辑字节数为：

\[
Q_{min}=sizeof(float)(MK+KN+MN),\quad AI_{ideal}=\frac{2MNK}{Q_{min}}.
\]

`Q_min/time` 只能叫 logical/effective GB/s，不能声称是实际 HBM 带宽，因为 cache、CTA 间
重复 load、write allocate 都会改变 DRAM traffic。实际 bytes、L2 hit rate 和 DRAM 吞吐应由
Nsight Compute 硬件计数器测量。

Roofline 上界为：

\[
P \le \min(P_{peak},\ BW_{peak}\times AI).
\]

点落在斜线区通常偏 memory-bound，应提高 reuse/合并访存；落在平台区则偏 compute-bound，
应关注指令吞吐、Tensor Core 和流水线。

### 4.4 profiler 指标

常用观察项包括：

- SM 与 DRAM 吞吐占峰值百分比；
- global load/store 合并程度、实际 DRAM bytes、L2 hit rate；
- shared-memory bank conflict；
- eligible/active warps、stall reason；
- registers/thread、shared memory/CTA、active CTA/SM 和 occupancy；
- local-memory load/store——它常意味着 accumulator 或临时数组 spill。

occupancy 不是最终目标。低 occupancy 但 ILP 和数据复用高的 GEMM，可能比高 occupancy kernel
更快；最终仍看正确性、稳定延迟、GFLOP/s 和相同 dtype/shape 下相对 cuBLAS 的比例。

## 5. 编译与运行

```bash
cd examples
make gemm/simple NVCCFLAGS='-std=c++17 -O3 -arch=sm_89'
./gemm/simple                    # 默认 513x509x515，50 次
./gemm/simple 1024 1024 1024 100
```

`-arch` 应换成目标 GPU 的 compute capability。benchmark 时保持 GPU 空闲，并关注温度、功耗
和动态频率；需要严谨对比时交替执行候选 kernel，避免固定执行顺序偏向某一个版本。
