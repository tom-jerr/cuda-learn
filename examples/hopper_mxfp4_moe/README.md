# Hopper MXFP4 × FP8 MoE：layout 推导与 RS WGMMA

这个 example 实现 MoE 第一层的 grouped GEMM：

```text
Y[e, token, channel] = X_fp8[e, token, k] · W_mxfp4[e, channel, k]^T
```

为了让量化权重位于 WGMMA 的寄存器 A 侧，kernel 实际计算转置后的矩阵：

```text
Y^T[N, M] = W[N, K] · X^T[K, M]
             A (RF)      B (SMEM)
```

权重从 packed E2M1 直接加载到寄存器，用 `prmt.b32` LUT 展开成已经带 K-group scale 的 E4M3，随后直接传给 `SM90_64x128x32_F32E4M3E4M3_RS_TN`。激活以 E4M3 保存在共享内存。这里的“融合”是一个 kernel 内的转换指令和 WGMMA 指令流水重叠，不是硬件存在 FP4 WGMMA 指令。

## 1. 一般 MoE 尺寸

默认参数对应常见的 gated FFN 第一层：

| 维度 | 默认值 | 含义 |
|---|---:|---|
| `E` | 8 | 单 GPU local experts |
| `M_capacity` | 128 | 每个 expert 路由并 padding 后的 token 容量 |
| `K` | 4096 | hidden size |
| `intermediate` | 14336 | FFN intermediate size |
| `N` | 28672 | gate 与 up 两个 projection 合并，即 `2 × intermediate` |

每个 expert 计算 `Y^T[28672,128] = W[28672,4096] × X^T[4096,128]`。真实 token 数由 `tokens_per_expert[e]` 指定，小于 capacity 的行补零且不写输出。

默认输入显存约为：packed 权重 448 MiB、权重 scale 28 MiB、FP8 激活 4 MiB、FP32 输出 112 MiB。没有 Hopper 时可只运行 CPU layout 检查；`--smoke` 将几何缩为 `E=1,M=128,N=128,K=4096`。

## 2. CTA tile 为什么是 `(N,M,K)=(64,128,32)`

Hopper 的 RS atom 固定覆盖：

```text
A: 64 × 32 E4M3，来自 128 个线程的寄存器
B: 128 × 32 E4M3，来自共享内存 descriptor
C: 64 × 128 FP32 accumulator
```

因此一个 CTA/warpgroup 负责 64 个输出 channel、128 个 routed token；K 每轮前进 32，正好等于 MXFP4 的 scale group。默认网格为：

```text
grid = (N / 64, M_capacity / 128, E) = (448, 1, 8)
block = 128 threads = 4 warps = 1 warpgroup
K loop = 4096 / 32 = 128 iterations
```

全局 tensor 的物理布局保持简单的 row-major：

```text
X bits     [E, M_capacity, K]       uint8/E4M3
W packed   [E, N, K/2]              uint8，偶数 k 在低 nibble
W offset   [E, N, K/32]             uint8，值域 [1,12]
tokenScale [E, M_capacity]          float
residual   [E]                      float
Y          [E, M_capacity, N]       float
```

## 3. A 寄存器 layout

CuTe 给该 atom 的 `ALayout_64x32` 为：

```text
LayoutA_TV:
  ((_4,_8,_4), (_4,_2,_2))
  ((_256,_1,_16), (_64,_8,_1024))
```

第一组是 128 个 thread，第二组是每线程 16 个 value。比直接解读扁平 stride 更清楚的等价坐标公式如下。令：

```text
warp = tid / 32
lane = tid % 32
i    = 0..15
v0   = i % 4
v1   = (i / 4) % 2
v2   = i / 8
```

那么本线程第 `i` 个 A fragment 值对应：

```text
row = 16 * warp + lane / 4 + 8 * v1
k   =  4 * (lane % 4) + v0 + 16 * v2
```

以 lane 0 为例：

```text
i=0..3    -> row 0, k 0..3
i=4..7    -> row 8, k 0..3
i=8..11   -> row 0, k 16..19
i=12..15  -> row 8, k 16..19
```

这解释了转换器为什么接收两个 `fp4x8` 寄存器以及两个 scale：两个寄存器各自的低 4 个值属于 `lo_row`，高 4 个值属于 `hi_row`。对所有 128 个线程枚举可得到 `128 × 16 = 64 × 32`，每个 `(row,k)` 恰好一个 owner；程序的 `--layout` 模式会实际验证这个双射和 scale 分组假设。

注意 `partition_A(identity)` 的 layout 含有 coordinate-valued stride，只负责返回逻辑坐标；拥有实际寄存器存储的 tensor 必须由 `partition_shape_A(...)` 创建。把前者直接拿来分配 fragment 会得到 tuple offset，而不是合法的线性 RF offset。

## 4. packed FP4 到 E4M3 的 layout

全局地址由上面的 `(row,k)` 直接计算：

```text
byte = W[((e * N + row) * K/2) + k/2]
code = (byte >> (4 * (k & 1))) & 0xf
```

每线程 16 个 code 按 fragment value 次序打包成两个 `uint32`。每个 `uint32` 是 8 个 nibble；转换后对应一个 `uint64`，即 8 个 E4M3 byte。LUT 的无符号部分为：

```cpp
lut_0_to_3 = offset * 0x08080800 + 0x0c080000;
lut_4_to_7 = offset * 0x08080808 + 0x1c181410;
```

E4M3 有 3 个 mantissa bit，指数加一会让编码增加 `0x08`。乘法器中的 `0x00` 让 FP4 零值始终保持零。符号位从每个 nibble 的 bit 3 取出，经 `prmt` 排成 E4M3 byte 的 bit 7。于是 offset 为 `t` 时，寄存器中实际数值是：

```text
W_tilde = q_e2m1 * 2^(t - 6)
```

该转换不产生中间 float，也不把展开后的权重写回 shared/global memory。

## 5. E8M0 scale 的预处理契约

kernel 输入的 `W offset` 不是原始 E8M0 byte。对每个 expert，设原始 scale 为 `2^(e-127)`，预处理应计算：

```text
b  = max(e_min, e_max - 11)
e' = max(e, b)
t  = e' - b + 1              # 传给 kernel，范围 [1,12]
r  = 2^(b-127) / 2
residual = 64 * r = 2^(b-122)
```

mainloop 得到 `q' * 2^(t-6)`，epilogue 再乘 `residual`：

```text
q' * 2^(e'-b-5) * 2^(b-122) = q' * 2^(e'-127)
```

如果 expert 内原始指数跨度超过 11，小 scale 被抬到 `b` 时必须同时重写/重新量化对应的 FP4 payload。这个 example 的随机数据已经直接生成合法的 `t` 和 residual，重点放在 GPU mainloop；接真实 checkpoint 时，权重预处理必须遵守上述契约。

激活使用 per-token scale，因此可以在完整 K reduction 后与 expert residual 一起应用：

```text
Y = accumulator * token_scale[e, token] * residual[e]
```

若激活 scale 也随 K block 变化，则不能这样移到最终 epilogue，必须对对应 K 部分和先缩放再累加。

## 6. B 共享内存 layout

每个 stage 保存 `128 × 32 = 4096` 个 E4M3 byte。K 只有 32 byte，所以使用 `GMMA::Layout_K_SW32_Atom<FP8>`；SW64/SW128 atom 的 K 基本块分别为 64/128，不能整除本例的 K32 tile。

CuTe 打印结果为：

```text
Sw<1,4,3> o ((_8,_16),(_32,_1)):((_32,_256),(_1,_0))
```

忽略 swizzle 时，`token = t0 + 8*t1`，基础 offset 为：

```text
p0 = 32*t0 + 256*t1 + k = 32*token + k
```

`Swizzle<1,4,3>` 用 bit 7 异或 bit 4：

```text
p = p0 ^ ((p0 & 0x80) >> 3)
  = 32*token + (k ^ (16 if (token & 4) else 0))
```

即每 8 个 token row 中，后 4 行交换两个 16-byte half，以改变同时访问时的 shared-bank 映射。`make_fragment_B(partition_B(s_x))` 不加载 B 寄存器，而是从该 layout 构造 Major-K GMMA descriptor；这正是 RS 中的 S 操作数。

## 7. C accumulator layout

每 CTA 产生 `64 × 128 = 8192` 个 FP32，平均每线程 64 个。`partition_C(identity)` 返回每个 accumulator 的 `(channel_in_tile, token_in_tile)`，所以无需猜测寄存器编号即可直接写回 row-major `Y[token,channel]`。程序枚举 128 个线程并确认 `128 × 64 = 8192` 个坐标也是严格双射。

其逐 lane 公式与 A 很相似。令 `i=0..63`，并分解为：

```text
v0 = i % 2
v1 = (i / 2) % 2
v2 = i / 4
```

则：

```text
channel = 16 * warp + lane / 4 + 8 * v1
token   =  2 * (lane % 4) + v0 + 8 * v2
```

例如 lane 0 的前八个坐标是 `(0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9)`，与 `--layout` 的实测输出一致。

生产 kernel 通常会先用 STSM 将 C 写入一个 swizzled epilogue buffer，再用 TMA store 获得更合并的全局写；这里使用 coordinate-driven direct store，让 A/B/C 三种 layout 的对应关系更容易观察。

## 8. 双缓冲与异步生命周期

kernel 保留两份 activation SMEM stage 和两份展开后的 A RF slot：

```text
iteration k:   convert W[k] -> A_slot[k%2]
               WGMMA(A_slot[k%2], B_stage[k%2]) + commit
               wait_group<1>
               load X[k+1] -> B_stage[(k+1)%2]
iteration k+1: convert W[k+1] while WGMMA(k) may still be pending
```

`wait_group<1>` 允许最新一组继续执行，但保证更早一组完成。因此两轮后复用同一个 A slot/B stage 时，旧 WGMMA 已经不再读取它。最后用 `wait_group<0>` 等待 accumulator 完整可见。普通 shared store 之后还需要 async-proxy fence 和 `__syncthreads()`，才能安全地由 GMMA descriptor 读取。

## 9. 构建和运行

直接使用 examples Makefile：

```bash
make -C examples hopper_mxfp4_moe/main
./examples/hopper_mxfp4_moe/main --layout
./examples/hopper_mxfp4_moe/main --smoke
./examples/hopper_mxfp4_moe/main
```

或使用 CMake：

```bash
cmake -S . -B build-hopper-moe \
  -DCUDA_LEARN_ENABLE_HOPPER_MXFP4_MOE=ON
cmake --build build-hopper-moe --target hopper_mxfp4_moe -j
./build-hopper-moe/examples/hopper_mxfp4_moe/hopper_mxfp4_moe --layout
```

必须生成 `sm_90a`，不能只生成 forward-compatible `sm_90`。`--layout` 是 CPU-only，可在非 Hopper 机器上验证 CuTe 映射；GPU 路径在非 SM90 机器上会打印 `SKIP`。运行时可用 `--experts/--tokens/--channels/--k` 覆盖尺寸，但四个矩阵维度必须满足 tile 对齐要求。

可用下面命令确认 cubin 中确实同时存在 LUT permutation 和寄存器-共享内存 WGMMA：

```bash
cuobjdump --dump-sass ./examples/hopper_mxfp4_moe/main | \
  grep -E 'PRMT|QGMMA|WARPGROUP.DEPBAR'
```

CUDA 13 在 Hopper 上反汇编为类似：

```text
PRMT ...
QGMMA.64x128x32.F32.E4M3.E4M3 R..., R..., gdesc[...], ...
WARPGROUP.DEPBAR.LE gsb0, 0x1
```

其中第二个 `R...` 是寄存器 A operand，`gdesc[...]` 是共享内存 B operand，证明走的是 RS 路径。
