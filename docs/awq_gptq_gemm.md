# AWQ / GPTQ W4A16 GEMM：权重重排与 CUDA 数据通路

## 先区分量化算法与推理 kernel

AWQ 和 GPTQ 主要决定“FP16 权重如何变成更准确的 INT4 权重”：

- AWQ 用校准 activation 找 salient channels，并通过等价缩放减少重要权重的量化误差；
- GPTQ 使用近似 Hessian，逐列量化并把当前列的误差补偿到后续列；`desc_act`
  还会引入 activation-order / `g_idx`；
- 模型保存后，推理 kernel 不再运行上述搜索，只消费 `qweight + scales +
  zero-points/g_idx`。所以 AWQ/GPTQ 的高性能 GEMM 核心可以共用，主要差别是解码语义和
  checkpoint layout。

本仓库的 `awq_gptq_gemm.cu` 复刻推理数据通路；为了自包含，它用普通 min/max 产生测试
INT4，不声称复现 AWQ calibration 或 GPTQ Hessian optimizer。
示例覆盖常见的 asymmetric AWQ 和 symmetric GPTQ `uint4b8`，GPTQ 教学路径关闭
`desc_act`；生产框架对非对称 checkpoint、`g_idx` 和 tensor parallel metadata 另有分支。

## 三层 layout

设逻辑权重 `W[K,N]`，4 bit、group size `G`，每个 `uint32` 放 8 个 nibble。

| 层次 | qweight 形状/语义 | 目的 |
|---|---|---|
| AWQ checkpoint | `[K,N/8]`，沿 N pack，nibble interleave | 兼容 AWQ checkpoint/dequantizer |
| GPTQ checkpoint | `[K/8,N]`，沿 K pack；可带 `g_idx[K]` | 兼容 GPTQ checkpoint |
| 教学 GEMM layout | `[K/8,N]`，同一 K-pack 下 N 连续 | 一个 warp/线程组读取连续输出列 |
| Marlin layout | tile-major packed weights + permuted scales/zp | 直接匹配 warp MMA 消费顺序 |

AWQ 4-bit pack 的 interleave 是 `[0,2,4,6,1,3,5,7]`。加载阶段需先撤销这个
interleave，再把 pack 轴从 N 改成 K。GPTQ 标准 qweight 已沿 K pack，但 Marlin 仍会
进行第二次 tile-major repack。

本例的统一 runtime 公式：

```text
AWQ : W_hat[k,n] = scale[k/G,n] * (uint4 - zero_point[k/G,n])
GPTQ: W_hat[k,n] = scale[k/G,n] * (uint4 - 8)
C = A_fp16 @ W_hat
```

## vLLM 当前加载路径

vLLM 的 `auto_awq.py` 先把 AWQ `[K,N/8]`、特殊 bit order 转成标准 GPTQ-like
`[K/8,N]`，qzeros 也转成通用格式，然后交给统一 mixed-precision kernel selector。
Marlin backend 在 `process_weights_after_loading` 中：

1. 必要时对 N/K 做 tile padding；
2. `g_idx` 存在时排序并产生 `g_idx_sort_indices`；
3. `gptq_marlin_repack` 把 qweight 改成 MMA tile-major；
4. `marlin_permute_scales` 重排 scale；AWQ runtime zero-point 先 unpack，再按同样 tile
   逻辑重排与 repack；
5. forward 直接调用 `marlin_gemm`，不再转换 checkpoint layout。

当前 selector 不只可能选择 Marlin，还可能根据平台/shape 选择其他 WNA16 backend；因此
“GPTQ/AWQ”描述量化格式，“Marlin/其他 backend”描述执行 kernel，二者不要混为一谈。

### Tensor parallel 与 shape 适配

- column-parallel 切 N，checkpoint loader 必须按 packed output dim 正确切分，且本地 N 要
  满足 pack/tile 对齐；
- row-parallel 切 K，本地 K 必须保持 group 完整，否则一个 scale group 会跨 rank；
- GPTQ `desc_act` 的 `g_idx` 关联全局 K 顺序。row-parallel 时部分路径需要在各 rank 重复
  scale/g_idx metadata，而不能像普通 group quant 一样直接切片；
- 当前 vLLM 对无 `g_idx` 的部分 tile-misaligned shape 可在加载期补零，forward 后裁掉
  padded N；有 act-order 时映射更复杂，通常执行严格 shape check。

## SGLang 当前加载路径

SGLang 同样在 `process_weights_after_loading` 做一次性重排，但 AWQ 与 GPTQ 保留了两条
更显式的入口：

- AWQ：`awq_marlin_repack`，然后分别执行 `marlin_permute_scales` 和
  `awq_to_marlin_zero_points`；
- GPTQ：`gptq_marlin_repack(..., perm=g_idx_sort_indices)`；有 act-order 时先
  `argsort(g_idx)`，再重排 qweight；scale 使用 `marlin_permute_scales`；
- forward 最终都进入 Marlin linear helper，并传 workspace、重排后的 scale/zp、
  `g_idx` 和 sort indices。

## 高性能 CUDA kernel 为什么快

1. **压缩流量**：decode 阶段 small-M GEMM 常受显存带宽限制，INT4 weight 理论流量约为
   FP16 的 1/4；scale/zp 是额外但较小的 group metadata。
2. **离线重排**：global memory 中的 nibble 顺序就是 warp 即将送入 MMA 的顺序，避免
   forward 做大范围 transpose。
3. **向量化解码**：128-bit load 后用 shift/mask、`lop3`/`prmt` 等指令并行展开多个
   INT4，并尽快形成 FP16/BF16 fragments。
4. **Tensor Core tile**：warp 负责固定 M×N tile，K 维循环执行 `mma.sync`；activation、
   packed weight 和 scale 的访问都围绕该 tile 设计。
5. **流水与驻留**：global→shared 双缓冲，解码与 MMA overlap；控制寄存器和 shared
   memory，使每个 SM 同时驻留足够 CTA。
6. **small-M / split-K**：decode 常只有几个 token，需要沿 N/K 提供并行度；K 很大时
   split-K，最后用 workspace reduction 或适用场景下 atomic add。

本例只保留最容易看懂的数据通路：K-packed 合并读取、每个 32-bit word 一次加载后展开
8 个 INT4、shared-memory tiled GEMM。它用 CUDA cores 做 FP32 accumulate，没有实现
Marlin 的 128-bit load、Tensor Core tile、复杂 swizzle、异步流水或 split-K，所以
适合验证格式，不用于和生产 Marlin 比性能。

## 运行

```bash
make -C examples quant/awq_gptq_gemm
./examples/quant/awq_gptq_gemm
```

## 官方源码入口

- vLLM [`auto_awq.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/auto_awq.py) /
  [`auto_gptq.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/auto_gptq.py)
- vLLM [`mixed_precision/marlin.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/mixed_precision/marlin.py)
  与 [`marlin_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py)
- SGLang [`AWQMarlinLinearKernel`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py)
  与 [`GPTQMarlinLinearKernel`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py)
- IST-DASLab [`marlin_cuda_kernel.cu`](https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu)
