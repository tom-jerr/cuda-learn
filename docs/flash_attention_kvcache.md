# FA2 KV cache 与 `flash_attn_multistage_kvcache`

本文对应：

- 官方 Python API：
  [`flash_attn_interface.py`](../third_party/flash-attention/flash_attn/flash_attn_interface.py)；
- 官方 C++ 参数与 dispatch：
  [`flash_api.cpp`](../third_party/flash-attention/csrc/flash_attn/flash_api.cpp)；
- 官方 split-KV / append kernel：
  [`flash_fwd_kernel.h`](../third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h)；
- 本仓库实现：
  [`flash_attn_multistage_kvcache.cu`](../src/flash_attn_multistage_kvcache.cu)；
- Python API：[`ops.py`](../python/cuda_learn/ops.py)；
- 针对性测试：
  [`test_flash_attn_kvcache.py`](../python/cuda_learn/tests/test_flash_attn_kvcache.py)。

## 1. 官方 `flash_attn_with_kvcache` 的语义

官方物理布局是：

```text
q:        [B, Q, Hq, D]
k_cache:  [Bcache, capacity, Hkv, D]
v_cache:  [Bcache, capacity, Hkv, D]
knew:     [B, Qnew, Hkv, D]       # optional
vnew:     [B, Qnew, Hkv, D]       # optional
lengths:  [B] int32                # append 前的有效 cache 长度
```

不传 `knew/vnew` 时，kernel 只读取每个 batch 的 `cache[:lengths[b]]`。传入新 K/V 时，
它先执行：

```text
k_cache[b, lengths[b] + i, hk, :] = rotary(knew[b, i, hk, :])
v_cache[b, lengths[b] + i, hk, :] = vnew[b, i, hk, :]
actual_k_len[b] = lengths[b] + Qnew
```

然后直接让 Q attend 更新后的 cache。cache 被原地修改，但 `cache_seqlens` 不会自动增加；
调用方必须在下一 decode step 更新长度。官方也要求预分配容量足够。

官方还支持：

- `Hq % Hkv == 0` 的 MQA/GQA；
- `cache_batch_idx` 做 beam/cache slot 重映射；
- `block_table` paged KV cache；
- append 时融合 RoPE；
- left padding、local window、ALiBi、softcap；
- KV 维 split，以及 partial-result combine。

### 1.1 Bottom-right causal mask

KV-cache 的 Q 通常对应整个 key 序列的最后 `Q` 个位置，所以 causal 不是 PyTorch
上三角从左上角开始的简单 `row >= col`。对 query 局部行 `m` 和 key 列 `n`：

```text
query_absolute_position = actual_k_len - q_len + m
keep(n) <=> n <= query_absolute_position
```

例如 `Q=2, K=5`：

```text
1 1 1 1 0
1 1 1 1 1
```

当 `Q=1` 时，唯一 query 就是最后一个位置，可以看到全部 cache。因此官方在无 ALiBi
时直接把 `seqlen_q == 1` 的 causal 当 non-causal specialization 运行。

## 2. 为什么官方 KV-cache 走 split-KV

普通 FA2 以 query row block 提供 CTA 并行度。decode 时 `Q=1`，即使 cache 很长，每个
batch-head 也通常只有一个 CTA，GPU 很容易吃不满。官方为此沿 KV block 再切 `S` 份：

```text
                         KV cache
              +------------+------------+------------+
same Q  ----> | split 0 CTA| split 1 CTA| split 2 CTA|
              +------------+------------+------------+
                    |             |             |
               (m0,l0,O0)    (m1,l1,O1)    (m2,l2,O2)
                    +-------------+-------------+
                                  |
                         online-softmax combine
                                  |
                                  O
```

每个 split 输出 FP32 partial `O` 和 LSE；combine kernel 用各 split 的 LSE 重新缩放后求和。
官方启发式遍历候选 split 数，估算：

```text
waves      = batch_heads_mblocks * splits / effective_sms
efficiency = waves / ceil(waves)
```

过滤不会改变每份 KV block 数的冗余候选后，选择第一个达到最高效率 85% 的 split 数。
FA2 代码把 `effective_sms` 取成 `2 * num_sms`，因为 split kernel 是 128 threads/CTA。

只要出现 append、`cache_batch_idx` 或 paged cache，C++ dispatch 就强制进入 split-KV
kernel；否则 `num_splits==1` 时仍可使用普通 forward kernel。

## 3. 官方融合 append 如何避免额外 kernel

split-KV kernel 先构造指向 cache 和 `knew/vnew` 的两个 tensor view。它从最后一个 KV
block 反向循环到包含 `cache_seqlen` 的 block，只复制满足下面区间的元素：

```text
cache_seqlen <= key_position < cache_seqlen + seqlen_knew
```

K 可以在寄存器中应用 interleaved 或 NeoX-style RoPE，再写入 cache。写完后执行 CTA
barrier，保证本 CTA 后续重新读取 K/V 时能看到刚写入的值。官方注释指出：GQA 下多个
Q-head CTA 可能向同一个 KV-head 写相同内容，形式上存在 race，但写入值完全相同；这样
每个 CTA 都无需等待另一个 CTA 就能继续。

之后 kernel 才加载 Q，并从 cache 的最后一个有效 KV block 反向执行 QK、online
softmax、P×V。反向遍历让只有最后一个、通常不完整的 block 需要边界 mask，也可以少保留
一个循环边界寄存器。

## 4. 我们的 multi-stage KV-cache 版本

仓库沿用 head-major 布局：

```text
q/out:     [B, Hq, Q, 64]
k/v cache: [B, Hkv, capacity, 64]
lengths:   [B] int32
knew/vnew: [B, Hkv, Q, 64]
```

当前约束是 FP16、`D=64`、`1 <= Q <= 64`、连续 cache。每个 `(batch,q_head)` 启动一个
128-thread CTA：

1. 读取并 clamp `cache_seqlens[b]`；
2. append specialization 用 128-bit global load/store 原地写入新 K/V；
3. CTA barrier 后发出 Q、K0、V0 的 `cp.async`；
4. K/V 使用两个 64×64 XOR-swizzled stage；计算 tile `t` 时预取 `t+1`；
5. 最后不足 64 的 cache tile通过 `cp.async ... src-size=0` zero-fill；
6. K 使用 `ldmatrix.x4`，V 使用 `ldmatrix.x4.trans`；
7. score 中同时应用 `key < actual_len` 和 bottom-right causal mask；
8. 两个相邻 N=8 score accumulator 直接 pack 成 P 的 MMA A operand；
9. accumulator 经 swizzled Q shared tile转换后 128-bit 写回 output。

cache-only 路径支持 GQA：

```text
kv_head = q_head / (Hq / Hkv)
```

融合 append 当前要求 `Hq == Hkv`，避免教育实现引入官方所接受的跨 CTA 同址写 race。

### 4.1 Python 使用

```python
import torch
from cuda_learn import flash_attn_multistage_kvcache

B, H, Q, capacity = 2, 8, 1, 2048
q = torch.randn(B, H, Q, 64, device="cuda", dtype=torch.float16)
k_cache = torch.empty(B, H, capacity, 64,
                      device="cuda", dtype=torch.float16)
v_cache = torch.empty_like(k_cache)
cache_seqlens = torch.tensor([511, 700], device="cuda", dtype=torch.int32)
knew = torch.randn(B, H, Q, 64, device="cuda", dtype=torch.float16)
vnew = torch.randn_like(knew)

out = flash_attn_multistage_kvcache(
    q, k_cache, v_cache, cache_seqlens,
    k=knew, v=vnew, causal=True)

# API 与官方一致：本次调用不修改 cache_seqlens。
cache_seqlens += Q
```

cache 会被原地写入，因此 Python binding 对非连续 cache 直接报错，不能悄悄调用
`.contiguous()` 后更新一个临时副本。

## 5. 正确性与资源

定向测试覆盖：

- `Q=1/4/5/17/64`；
- cache length `0/1/3/7/64/65/70/80/129/130`；
- capacity 非 64 倍数；
- MHA、cache-only GQA；
- causal/non-causal；
- append 跨 64/128 tile 边界，且旧 cache prefix 保持不变。

与 FP32 PyTorch reference 的最大绝对误差不超过 `9.77e-4`。四个编译期
`AppendKV × Causal` specialization 均为 40,960 B static shared、0 spill；寄存器范围
是 135--147/thread。

## 6. Decode 性能与当前最重要的缺口

RTX 4060 Laptop、`B=2,H=8,Q=1,D=64`，交替执行次序后取 11--15 轮 paired median：

| cache capacity | 官方自动 splits | multi-stage | 官方 FA2 KV-cache | ours / FA2 |
|---:|---:|---:|---:|---:|
| 128 | 1 | 0.02177 ms | 0.02148 ms | 1.013x |
| 256 | 1 | 0.02236 ms | 0.02135 ms | 1.047x |
| 512 | 2 | **0.02147 ms** | 0.02494 ms | 0.861x |
| 1024 | 2 | 0.03000 ms | **0.02571 ms** | 1.167x |
| 2048 | 3 | 0.05746 ms | **0.03421 ms** | 1.679x |
| 4096 | 3 | 0.11145 ms | **0.05776 ms** | 1.929x |

在 `capacity=1024` 的另一组 15×200 paired 测量中，我们是 `0.03001 ms`，官方 auto
split 是 `0.02633 ms`；官方强制 `split=1` 则是 `0.03688 ms`。也就是说：我们的单
CTA/head kernel 本身并不差，甚至比官方单 split 快约 19%；长 cache 的主要差距来自官方
把 KV 并行拆成多个 CTA，而不是 `ldmatrix` 或 P-layout conversion。

下一步若以 decode 性能为目标，应优先增加：

1. `Split > 1` 的 partial `(m,l,O)` workspace；
2. 单独 combine kernel；
3. 根据 `B*H`、SM 数、KV block 数选择最小高效 split；
4. 再扩展 paged cache、GQA fused append 和 RoPE。

直接继续微调当前单 CTA 的 shared layout，对 2K--4K cache 不可能追回由 CTA 级并行度
造成的 1.7--1.9 倍差距。

## 7. 复现

```bash
source ~/.python/miniinfer/bin/activate
source scripts/env.sh
cmake --build build -j4

pytest -q python/cuda_learn/tests/test_flash_attn_kvcache.py

MAX_JOBS=1 python -m cuda_learn.bench \
  --warmup 5000 --iters 1000 test_flash_attn_multistage_kvcache_decode
```

官方 benchmark binding 只新增 D64 FP16 split-KV translation unit，不修改官方 kernel
源码。首次运行会在 torch extension cache 中 JIT 编译。
