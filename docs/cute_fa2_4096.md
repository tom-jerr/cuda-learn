# 面向 SM89 的 CuTe FlashAttention-2：固定 4096×4096 Score 的布局设计与实现

## 摘要

本文给出一个从布局约束出发设计的 FlashAttention-2 forward 教学实现。目标设备是
Ada/SM89，固定输入为 `Q,K,V ∈ FP16[4096,64]`，注意力 score 矩阵为
`S ∈ FP32[4096,4096]`，输出为 `O ∈ FP16[4096,64]`。实现使用 64 个 CTA、
`mma.sync.m16n8k16`、XOR-swizzled shared memory、16-byte `cp.async`、
`ldmatrix.x4`、双 stage KV 流水、寄存器 online softmax，以及从 QK accumulator 到 PV
A operand 的 `left_inverse + compose` 布局变换。

设计参考 HPC-Ops prefill attention 的计算结构，但不复制其硬件后端：HPC-Ops 当前路径
面向 SM90、BF16、D=128，采用 TMA 和 WGMMA；本文针对仓库实际使用的 RTX 4060
Laptop/SM89，将相同的数据流原则重新落到 `cp.async + ldmatrix + mma.sync`。最终 kernel
使用 40 KiB shared memory、161 registers/thread，无 register spill；在本机固定输入上 50
次平均约 0.253 ms，对应约 17.0 TFLOP/s，覆盖每个 CTA/warp 的 256 行参考检查与 GPU
结果逐元素一致。

对应源码为 [`examples/cute_fa2/fa2_4096.cu`](../examples/cute_fa2/fa2_4096.cu)。

## 1. 问题定义和边界

本文把“4096×4096 方阵”解释为注意力 score，而不是 head dimension：

\[
Q,K,V\in\mathbb{R}^{4096\times64},\qquad
S=\frac{QK^T}{\sqrt{64}}\in\mathbb{R}^{4096\times4096},
\]

\[
P=\operatorname{softmax}_{\rm row}(S),\qquad
O=PV\in\mathbb{R}^{4096\times64}.
\]

当前 specialization 有意限定为：

- batch 和 head 均为 1；
- sequence length 固定 4096，head dimension 固定 64；
- FP16 输入/输出，两个 Tensor Core GEMM 都使用 FP32 accumulator；
- non-causal forward，无 dropout、mask、反向和变长边界；
- 输入均为连续 row-major，地址来自 `cudaMalloc`，满足 16-byte 对齐。

固定边界使 layout、循环次数和谓词全部成为编译期常量，便于集中分析 MMA/copy/smem
之间的兼容性。它不是通用 FlashAttention API。

## 2. 参考结构及架构迁移

HPC-Ops 的 multi-stage prefill kernel 采用以下结构：

1. Q tile 在一个 CTA 生命周期内保持不变；
2. K/V 按 sequence tile 流式进入 shared memory；
3. 每个 tile 依次执行 QK、online softmax 和 PV；
4. 输出 numerator 与 row max/sum 常驻寄存器；
5. 使用 `left_inverse(layout_asC).compose(layout_asA)` 将 score accumulator 作为 PV 的
   A operand；
6. 最后统一除以 online-softmax denominator，再通过 shared-memory epilogue 写回。

这些结构可见于 HPC-Ops 固定提交
[`f39028d`](https://github.com/Tencent/hpc-ops/blob/f39028d9f5ab77f71906fbf929d1b611859ab6b7/src/attention/prefill/sm90/kernels.cuh#L3157-L3434)，
其 D128 配置见
[`multi_stage_dim128.cu`](https://github.com/Tencent/hpc-ops/blob/f39028d9f5ab77f71906fbf929d1b611859ab6b7/src/attention/prefill/sm90/multi_stage_dim128.cu#L41-L66)。

本文保留上述算法结构，但硬件映射如下：

| 设计层 | HPC-Ops SM90 | 本文 SM89 |
|---|---|---|
| Global→shared | TMA | 128 个线程执行 16B `cp.async` |
| Shared→register | WGMMA descriptor | `ldmatrix.x4` / `ldmatrix.x4.trans` |
| MMA | warpgroup WGMMA | warp `m16n8k16` |
| 并行分工 | warpgroup | 4 warps/CTA，全部沿 M 排列 |
| KV 流水 | barrier 管理的 stage | 2-stage `cp.async` ring buffer |
| 数据类型 | BF16，D=128 | FP16，D=64 |

## 3. CTA 分解

选择：

\[
B_M=64,\qquad B_N=64,\qquad D=64.
\]

一个 CTA 负责 64 个 query 行，并遍历全部 4096 个 key/value 行：

\[
N_{CTA}=4096/64=64,
\]

\[
N_{KV\ tile}=4096/64=64.
\]

因此 grid 为 `dim3(64)`。第 `b` 个 CTA 和第 `t` 个 KV tile 计算：

\[
Q_b=Q[64b:64b+64,:],
\]

\[
K_t=K[64t:64t+64,:],\qquad
V_t=V[64t:64t+64,:].
\]

每个 CTA 最终只写：

\[
O_b=O[64b:64b+64,:].
\]

不同 CTA 没有通信。K/V 会被不同 CTA 重读，但在 4096×64 的固定问题上完整 K 或 V
仅占 512 KiB，L2 能提供显著复用。

## 4. Online softmax 不变量

令当前处理到第 `t` 个 KV tile，每一行维护三个 FP32 状态：

- `m_t`：目前所有 score 的最大值；
- `l_t`：相对于 `m_t` 的指数和；
- `o_t`：尚未除以 `l_t` 的输出 numerator。

当前 tile 的 scaled score 为：

\[
S_t=Q_bK_t^T/\sqrt{64}.
\]

更新公式为：

\[
m_t=\max(m_{t-1},\operatorname{rowmax}(S_t)),
\]

\[
\alpha_t=\exp(m_{t-1}-m_t),
\]

\[
\widetilde P_t=\exp(S_t-m_t),
\]

\[
l_t=\alpha_tl_{t-1}+\operatorname{rowsum}(\widetilde P_t),
\]

\[
o_t=\alpha_to_{t-1}+\operatorname{FP16}(\widetilde P_t)V_t.
\]

循环结束后：

\[
O_b=o_{63}/l_{63}.
\]

`m/l/o` 均为 FP32；只有送入第二个 Tensor Core GEMM 的
`\widetilde P_t` 转成 FP16。首个 tile 通过 `l_{-1}=0` 特判令 `α_0=0`，避免计算
`exp(-∞-m_0)`。

## 5. MMA layout 的推导

### 5.1 Atom 的硬件约束

MMA atom 为：

```cpp
MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>
```

一条指令执行：

\[
C_{16\times8}\mathrel{+}=A_{16\times16}B_{8\times16}^{T}.
\]

令 lane 分解为：

\[
l_0=lane\bmod4,\qquad l_1=\lfloor lane/4\rfloor.
\]

Atom A layout 为：

```text
((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
```

若 A value 坐标为 `(a0,a1,a2)`，则逻辑线性位置：

\[
p_A=32l_0+l_1+16a_0+8a_1+128a_2=m+16k,
\]

所以：

\[
m=l_1+8a_1,\qquad k=2l_0+a_0+8a_2.
\]

每 lane 有 `2×2×2=8` 个 A values。

Atom B layout 为：

```text
((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
```

对应：

\[
p_B=16l_0+l_1+8b_0+64b_1=n+8k,
\]

\[
n=l_1,\qquad k=2l_0+b_0+8b_1.
\]

每 lane 有 4 个 B values。

Atom C layout 为：

```text
((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
```

对应：

\[
p_C=32l_0+l_1+16c_0+8c_1=m+16n,
\]

\[
m=l_1+8c_1,\qquad n=2l_0+c_0.
\]

因此相邻 4 lanes 共同覆盖两条完整的 8-column score 行片段；这也是 online softmax
使用 4-lane XOR reduction 的原因。

### 5.2 为什么四个 warp 只沿 M 排列

线程 layout 选择：

```cpp
Layout<Shape<_4,_1,_1>>{}
```

实际展开为：

```text
ThrLayoutVMNK = (_32,_4,_1,_1):(_1,_32,_0,_0)
```

即：

\[
threadIdx.x=lane+32\cdot warp_M.
\]

四个 warp 分别拥有 16 个 query 行，共覆盖 `4×16=64` 行。没有 warp 沿 N 排列，这是
QK→PV 寄存器复用的必要设计：一个 warp 必须持有其 16 行对应的完整 64-column score，
才能直接把它解释成 PV 的 `A[16,64]`。如果采用 GEMM 常见的 `(2,2,1)` warp layout，
score 的 N 方向会分散到两个 warp；PV 的每个 warp 却需要完整 P 行，从而需要 shared
memory 或 shuffle 做跨 warp 交换，破坏本文要验证的 C→A 零拷贝变换。

### 5.3 为什么 permutation 是 64×64×16

四个 warp 的自然 coverage 为：

\[
(4\cdot16,1\cdot8,1\cdot16)=(64,8,16).
\]

指定：

```cpp
Tile<_64,_64,_16>{}
```

将 N footprint 从 8 扩展到 64。线程数不增加；每个 warp 在 N 上获得 8 组 atom
values。于是基础 TiledMMA footprint 正好为一个 CTA 的 score/PV output tile：

\[
64\times64\times16.
\]

对于 QK：

```text
trQ = (V8, MMA_M=1, MMA_K=4)
trK = (V4, MMA_N=8, MMA_K=4)
trS = (V4, MMA_M=1, MMA_N=8)
```

对于 PV：

```text
trP = (V8, MMA_M=1, MMA_K=4)
trV = (V4, MMA_N=8, MMA_K=4)
trO = (V4, MMA_M=1, MMA_N=8)
```

每个 warp 每个 QK/PV 分别执行：

\[
8\;N\text{-blocks}\times4\;K\text{-blocks}=32
\]

条 MMA。四 warp、两个 GEMM 合计每个 KV tile 256 条；64 个 KV tile、64 个 CTA 的
总指令数为：

\[
256\times64\times64=1,048,576.
\]

每条 MMA 计 `2×16×8×16=4096` FLOPs，总计算量：

\[
1,048,576\times4096=4,294,967,296\ \text{FLOPs},
\]

等于两次矩阵乘法的：

\[
4\cdot4096^2\cdot64.
\]

## 6. Global→shared copy layout

每条 `cp.async` 搬运：

\[
128\ bits=16\ bytes=8\ FP16.
\]

### 6.1 Q/K 的 row copy

线程和值 layout：

```cpp
ThrLayout = (32,4):(4,1)
ValLayout = (1,8)
```

对于 `tid∈[0,128)`：

\[
t_r=\lfloor tid/4\rfloor,\qquad t_c=tid\bmod4.
\]

基础 `32×32` copy tile 中第 `v` 个 value 坐标为：

\[
row=t_r,\qquad col=8t_c+v,\quad 0\le v<8.
\]

扩展到 `64×64` 后，每线程坐标为：

\[
row=t_r+32r_r,
\]

\[
col=8t_c+v+32r_c,
\]

其中 `r_r,r_c∈{0,1}`。所以每线程每个 tile 发出 4 条 `cp.async`，CTA 共搬：

\[
128\times4\times8=4096\ FP16=64\times64.
\]

### 6.2 V 的 column-logical copy

PV 的 B operand 逻辑坐标为 `(output_dim, reduction_row)`，但 V 的物理存储是
`V[reduction_row,output_dim]`，因此 V tensor 使用：

```text
shape  = (D,N)
stride = (1,D)
```

线程和值 layout 交换两个逻辑 mode：

```cpp
ThrLayout = (4,32):(1,4)
ValLayout = (8,1)
```

令：

\[
t_d=tid\bmod4,\qquad t_n=\lfloor tid/4\rfloor.
\]

基础 copy 坐标：

\[
d=8t_d+v,\qquad n=t_n.
\]

物理地址仍为：

\[
n\cdot64+d,
\]

所以每条指令仍读取 V row 中连续、对齐的 8 个 FP16，而 shared tensor 已经具有 PV
B operand 所需的 `(D,N)` 逻辑方向。

## 7. Shared-memory layout 与 swizzle

### 7.1 Row-major atom

Q/K 使用：

```cpp
composition(
    Swizzle<3,3,3>{},
    Layout<Shape<_8,_64>,Stride<_64,_1>>{})
```

普通 element offset：

\[
x=64r+c.
\]

`Swizzle<3,3,3>` 执行：

\[
x'=x\oplus\left(((x\gg6)\mathbin{\&}7)\ll3\right).
\]

在 `8×64` atom 内可化为：

\[
x'=64r+\left(c\oplus((r\mathbin{\&}7)\ll3)\right).
\]

令 16-byte segment 编号 `j=⌊c/8⌋`，则：

\[
j'=j\oplus(r\mathbin{\&}7).
\]

这意味着：

- 最低 3 个 element bits 不变，8 个 FP16 始终连续且保持 16B 对齐；
- 每行的 8 个 16B segments 按行号做 XOR permutation；
- shared bank 为 `bank=(x'>>1) mod 32`，segment 起始 bank 为 `4j' mod 32`，不同
  `ldmatrix` 行被分散到不同的 bank quartet。

因此同一个 layout 同时适配 16B `cp.async` 和 `ldmatrix`。

### 7.2 V 的 column-logical atom

V 使用：

```cpp
composition(
    Swizzle<3,3,3>{},
    Layout<Shape<_64,_8>,Stride<_1,_64>>{})
```

对逻辑坐标 `(d,n)`：

\[
x=d+64n,
\]

\[
x'=64n+\left(d\oplus((n\mathbin{\&}7)\ll3)\right).
\]

物理上它与 row-major V 完全一致，只是逻辑 mode 顺序变成 `(D,N)`，从而可直接配合
`ldmatrix.x4.trans` 填充 MMA B fragment。

### 7.3 容量

Q 无需多 stage：

\[
64\times64\times2\ bytes=8\ KiB.
\]

K/V 各双 stage：

\[
64\times64\times2\ stages\times2\ bytes=16\ KiB.
\]

总 shared memory：

\[
8+16+16=40\ KiB.
\]

Stage stride 为 4096 elements。它位于 swizzle 涉及的 bit 3–8 之外，因此两个 stage
不会互相扰动。

## 8. Shared→register copy layout

Q/K 使用：

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, half_t>
```

一条 `ldmatrix.x4` 由一个 warp 读取四个 `8×8` FP16 matrix：

\[
4\times8\times8=256\ FP16=8\ FP16/lane.
\]

`make_tiled_copy_A/B(atom,tiled_mma)` 不再手写 lane→register mapping，而是从
TiledMMA 的 A/B TV layout 推导 destination，使 ldmatrix 输出直接落入 MMA ABI 要求的
寄存器槽位。

V 的物理行列方向与 MMA B 逻辑方向相反，因此使用：

```cpp
Copy_Atom<SM75_U16x8_LDSM_T, half_t>
```

该名字中的 U16x8 表示每 lane 得到 8 个 16-bit values，对应 PTX
`ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16`。它不是额外的显式 transpose kernel；
转置发生在 shared→register 指令内部。

对 K/V，每 lane 的完整 B fragment 含：

\[
4\;atom\ values\times8\;N\text{-blocks}\times4\;K\text{-blocks}
=128\ FP16.
\]

每条 x4 产生 8 FP16，因此每 warp 每个 K/V tile 发出 16 条 ldmatrix.x4。Q fragment
每 lane 有 `8×4=32` FP16，对应每 warp 4 条 x4。

## 9. QK accumulator 到 PV A operand

QK score fragment 的紧凑寄存器 layout 为：

```text
trP_as_C:
((_2,_2),_1,_8):((_1,_2),_0,_4)
```

PV 所需 A fragment 为：

```text
trP_as_A:
((_2,_2,_2),_1,_4):((_1,_2,_4),_0,_8)
```

两者均有 32 FP16 values/thread。令：

- `L_C : coord_C -> matrix_offset` 为 score 的 C partition；
- `L_A : coord_A -> matrix_offset` 为同一个 `64×64` 逻辑矩阵的 A partition；
- `R_C : coord_C -> register_offset` 为 FP16 P 的物理寄存器 layout。

因为当前 warp layout 没有沿 N 拆分，`L_C` 对当前线程持有的 score 是连续的一一映射，
可以求左逆：

\[
L_C^{-1}:matrix\_offset\rightarrow coord_C.
\]

先得到 A 坐标到 C 坐标的映射：

\[
T_{A\rightarrow C}=L_C^{-1}\circ L_A.
\]

再与实际 register layout 复合：

\[
R_A=R_C\circ T_{A\rightarrow C}
=R_C\circ L_C^{-1}\circ L_A.
\]

对应 CuTe：

```cpp
auto a_to_c = left_inverse(layout_as_c).compose(layout_as_a);
auto trP_as_a = trP_as_c.compose(a_to_c);
```

整个过程只构造编译期 layout view，不产生数据搬运、shared-memory round trip 或
register shuffle。

## 10. 双 stage `cp.async` 流水

Prologue 将 Q、K0、V0 提交为一个 async group，随后 `wait<0> + __syncthreads()`；Q
只从 shared 加载一次并常驻寄存器。稳定循环采用：

| 时间 | write stage | read stage | 计算 |
|---|---:|---:|---|
| tile `t` 开始 | 提交 K/V `[t+1]` | K/V `[t]` | — |
| QK | async copy 继续 | ldmatrix K `[t]` | `QKᵀ` |
| softmax | async copy 继续 | — | 更新 `m/l`，缩放旧 `o` |
| PV | async copy 继续 | ldmatrix V `[t]` | `P_tV_t` |
| tile `t` 结束 | `wait<0>` | — | CTA barrier 后交换 stage |

Stage 索引：

\[
write=read\oplus1.
\]

`cp_async_wait` 只保证 async group 完成，不能代替 CTA barrier；`__syncthreads()` 保证
所有 warp 已停止读取旧 stage，并且新 stage 对整个 CTA 可见。写、读 stage 物理分离，
所以 global→shared copy 可覆盖 QK、softmax 和 PV 的大部分执行时间。

## 11. Output epilogue

MMA C fragment 的 global ownership 是离散的，直接写回会产生大量小 store。循环结束后：

1. FP32 `trO/l` 转成 FP16；
2. 按 `thr_mma.partition_C(sO)` 写入复用的 Q shared-memory 区域；
3. CTA barrier；
4. 使用与 G2S row copy 相同的 `(32,4)` thread layout，每线程四条 16B store 写回
   row-major global O。

shared swizzle 保持每个 8-half vector 连续，所以输出 exchange 同时完成 MMA C layout 到
global vector layout 的转换。

## 12. 实验方法与结果

环境：

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, SM89 |
| CUDA | 13.0.88 |
| CUTLASS | 4.6.1，提交 `e05f953` |
| 编译 | `nvcc -std=c++17 -O2 -arch=sm_89` |

PTXAS 资源报告：

```text
Used 161 registers
0 bytes spill stores
0 bytes spill loads
shared memory: 40960 bytes/CTA
occupancy limit: 2 CTA/SM = 8 resident warps/SM
```

固定输入、3 次 warmup、50 次 CUDA-event 计时的代表结果：

```text
latency: 约 0.253 ms
mathematical throughput: 约 17.0 TFLOP/s
sampled-row max abs error: 约 4e-7
all output finite: yes
```

在使用逐 tile online-softmax CPU 参考的最终测试中，打印误差为 `0.0`；上面的 `4e-7`
是与一次性 full-row softmax 参考比较时观察到的差异上界，来自不同 FP16 P 舍入位置。

对最终 cubin 执行 `cuobjdump --dump-sass` 还能确认抽象确实落到了目标指令：

```text
LDGSTS.E...128       # cp.async 16B
LDSM.16.M88.4        # ldmatrix.x4
LDSM.16.MT88.4       # ldmatrix.x4.trans
HMMA.16816.F32       # m16n8k16, FP32 accumulator
STG.E.128            # vectorized output store
```

正确性参考在 CPU 上对每个 CTA 的每个 warp 所拥有的首行完整执行
`QKᵀ-online-softmax-PV`，共检查 256 行，覆盖全部 64 个 CTA 和全部 4 个 warp；参考按相同
64-column tile 顺序更新 `m/l/o`，并在每个 tile 的 P 进入 PV 前舍入为 FP16，以匹配
Tensor Core 输入。程序另对全部 `4096×64` 输出做 finite scan。

当前 Windows/WSL 驱动环境不能初始化 Compute Sanitizer debugger interface，因此
sanitizer 不是本报告的通过项。

## 13. 构建与复现

Makefile：

```bash
make -C examples cute_fa2/fa2_4096
./examples/cute_fa2/fa2_4096 50
```

CMake：

```bash
cmake -S . -B build-cute \
  -DPython_EXECUTABLE=/home/lzy/.python/miniinfer/bin/python
cmake --build build-cute --target cute_fa2_4096 -j
./build-cute/examples/cute_fa2/cute_fa2_4096 50
```

命令行参数是 benchmark iteration 数，不改变固定 shape。

## 14. 结论与限制

本设计的关键不是单独某个 swizzle 或 copy atom，而是以下约束同时成立：

\[
\text{warp layout沿M}
\Rightarrow C\rightarrow A\text{无需跨warp交换},
\]

\[
8\ FP16/cp.async
\Rightarrow 16B\ vector
\Rightarrow Swizzle<3,3,3>\text{保留低3位},
\]

\[
64\times64\text{ tile}
\Rightarrow 4\ warps\times8\ N\text{-atoms},
\]

\[
2\ stages\times(K,V)
\Rightarrow 40\ KiB\ smem
\Rightarrow copy/compute overlap.
\]

当前结果证明了固定 4096×4096 score 下布局代数与流水结构的正确性，但离通用、生产级
FA2 仍有明确距离：尚未支持 causal/变长/batch/head、D128、边界 predication、dropout、
反向、持久化调度和架构自动选择；性能数据也仅代表本机固定输入，不能外推到其他 GPU。

## 参考资料

1. Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*,
   [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)。
2. Tencent HPC-Ops, [SM90 prefill multi-stage kernel](https://github.com/Tencent/hpc-ops/blob/f39028d9f5ab77f71906fbf929d1b611859ab6b7/src/attention/prefill/sm90/kernels.cuh#L3157-L3434)。
3. NVIDIA CUTLASS, [CuTe dense GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)。
4. 本仓库 CUTLASS 的 [`mma_traits_sm80.hpp`](../third_party/cutlass/include/cute/atom/mma_traits_sm80.hpp)
   与 [`copy_traits_sm75.hpp`](../third_party/cutlass/include/cute/atom/copy_traits_sm75.hpp)。
