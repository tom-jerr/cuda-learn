# RMSNorm：与 LayerNorm 的差异及 small hidden size CUDA 优化

本文说明 RMSNorm 相比 LayerNorm 省掉了什么、这种简化的收益与边界，以及本仓库
`D <= 128` 专用 CUDA kernel 的设计和实测结果。对应代码：

- CUDA kernel 与 TVM FFI 注册：[src/rmsnorm.cu](../src/rmsnorm.cu)；
- Python 接口：[python/cuda_learn/ops.py](../python/cuda_learn/ops.py)；
- 正确性与 benchmark：[python/cuda_learn/tests/test_ops.py](../python/cuda_learn/tests/test_ops.py)；
- 参数校验测试：[python/cuda_learn/tests/test_rmsnorm_validation.py](../python/cuda_learn/tests/test_rmsnorm_validation.py)。

当前 `ops.rmsnorm(x)` 接受连续或非连续的二维 FP32 CUDA tensor，沿最后一维归一化。
它遵循原始 `rmsnorm.cu` 的语义，不包含可学习的 `weight` 和 `bias`：

```python
y = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
```

标准模型中的 RMSNorm 通常还会乘可学习参数 `weight`。本仓库已有的
`rmsnorm_and_add` 则实现了 residual add、RMSNorm 和 weight 的融合版本。

## 1. RMSNorm 与 LayerNorm

设一行输入为 `x = [x_0, ..., x_{D-1}]`。LayerNorm 先计算均值和方差：

```text
mean = (1 / D) * sum_i(x_i)
var  = (1 / D) * sum_i((x_i - mean)^2)
y_i  = (x_i - mean) * rsqrt(var + eps) * gamma_i + beta_i
```

RMSNorm 不做均值中心化，只计算均方根：

```text
mean_square = (1 / D) * sum_i(x_i^2)
y_i         = x_i * rsqrt(mean_square + eps) * gamma_i
```

两者的核心区别不是 `sqrt` 或 `rsqrt`，而是 RMSNorm 删除了 `mean` 统计量和
`x_i - mean`。原始 RMSNorm 论文将其描述为：保留 re-scaling invariance，但放弃
LayerNorm 的 re-centering invariance。论文实验显示它在所测试模型上取得相近效果，
但这不是“任何模型都可无损替换”的保证。背景与论文结果见
[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。

### 1.1 RMSNorm 的工程优势

| 项目 | LayerNorm | RMSNorm |
|---|---|---|
| 行统计量 | `sum(x)` 与方差 | `sum(x^2)` |
| 中心化 | 需要 `x - mean` | 不需要 |
| 可学习参数 | 通常有 `gamma`、`beta` | 通常只有 `gamma` |
| reduction 状态 | 至少维护均值与方差 | 只维护平方和 |
| CUDA 实现 | reduction 和 epilogue 更复杂 | 更容易做单一平方和 reduction |
| 算子融合 | 要同时携带 mean/variance | 更容易与 residual、weight 融合 |

在 CUDA 上，LayerNorm 并不一定真的读取输入更多次：高质量实现可以在一次扫描中同时
累计 `sum(x)` 和 `sum(x^2)`，或者使用 Welford reduction。但它仍需要维护两个统计量、
执行中心化，并在并行 reduction 中传播更多状态。RMSNorm 的主要优势是计算图和 reduction
状态更简单，而不是笼统地声称“一定少一次 HBM 读取”。

### 1.2 什么时候不能直接替换

RMSNorm 对缩放保持不变，但不消除整行共同的偏移。如果：

- 模型依赖 LayerNorm 的零均值输出；
- 要加载已有 LayerNorm checkpoint；
- 替换后不允许重新训练或验证模型质量；

就不能仅凭 kernel 更快而直接替换。RMSNorm 与 LayerNorm 是不同算子，选择通常应在模型
设计或训练阶段确定。本仓库的精度测试只证明 CUDA 实现符合 RMSNorm 公式，不证明它与
LayerNorm 输出相等。

## 2. 通用 kernel：一行一个 block

通用路径 `rmsnorm_kernel` 让一个 CUDA block 处理一行：

1. 每个线程跨步读取若干元素，累计 `x_i^2`；
2. partial sum 写入 shared memory；
3. 用二次幂树形 reduction 得到整行平方和；
4. 计算 `inverse_rms = rsqrt(mean_square + eps)`；
5. 再次读取输入并写出归一化结果。

线程数根据 hidden size 选择：

| hidden size | threads/row |
|---:|---:|
| `D <= 128` 的非专用情况 | 32 |
| `128 < D <= 256` | 64 |
| `256 < D <= 512` | 128 |
| `D > 512` | 256 |

所有线程数都是 2 的幂，因此 shared-memory reduction 对非 2 次幂的 `D` 仍然正确；越界
元素根本不进入局部平方和。`D=17、1003、4097` 都由测试覆盖。

### 2.1 宽行为什么主要受访存限制

不计标量统计量，通用 kernel 每个元素大约执行：

```text
第一次读取 x：4 B
第二次读取 x：4 B
写出 y：      4 B
总流量：     12 B/element
```

对应的有效浮点工作只有平方、累加、除法摊销和输出乘法，约 4 FLOP/element，算术强度约为
`0.33 FLOP/byte`。因此宽行更容易受显存/L2 带宽限制，而不是受 FP32 算力限制。

本次在 `D=4096` 上试过三种替代映射：通用 `float4`、把整行分片留在寄存器、一个 warp
独立处理一行。它们均未超过原始 256-thread/block 版本：第二遍输入读取通常能得到缓存
帮助，而更长的单线程累加链、额外寄存器或较低的有效并行度抵消了收益。因此最终宽行路径
保留简单实现，而不是保留 benchmark 更慢的“优化”。

## 3. Small hidden size 专用优化

当 `D <= 128` 且 `D % 4 == 0` 时，一行最多只有 32 个 `float4`。这恰好可以映射给一个
warp 的 32 个 lane，因此使用独立的 `rmsnorm_small_kernel`：

```text
一个 128-thread block
├── warp 0 -> row 0
├── warp 1 -> row 1
├── warp 2 -> row 2
└── warp 3 -> row 3
```

每个 lane 的数据流为：

```text
HBM --float4 load--> 4 个 FP32 register
                         │
                         ├── x² 局部和
                         │      │
                         │   warp shuffle reduction
                         │      │
                         └── × inverse_rms --float4 store--> HBM
```

### 3.1 为什么它比通用路径快

专用路径同时做了四件事：

1. **一 warp 一行**：平方和只需 5 轮 `__shfl_down_sync`，不经过 shared memory；
2. **四行一 block**：避免一个 32-thread block 只占一个 warp，却消耗一个 block slot；
3. **寄存器保留 `float4`**：reduction 后直接复用原值，输入只读一次；
4. **向量化读写**：每个活跃 lane 使用一次 128-bit load 和一次 128-bit store。

因为 `D <= 128`，每个 lane 最多持有一个 `float4`，不会像宽行寄存器缓存那样快速增加
寄存器压力。`cuobjdump --dump-resource-usage` 显示该 kernel 使用 19 registers/thread，
没有 local-memory spill，也不使用 shared memory。

### 3.2 对齐与尾部条件

只有 `D % 4 == 0` 才进入 `float4` 路径。PyTorch CUDA allocation 的基地址满足 16-byte
对齐，且每行包含整数个 `float4`，所以每行起点仍然对齐。对于 `D < 128`，超出
`D / 4` 的 lane 以零参与 reduction，并跳过 load/store。不是 4 的倍数时回退到标量通用
路径，不做不安全的越界向量读取。

## 4. 正确性测试

测试参考实现为：

```python
def rmsnorm_ref(x, eps=1e-5):
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
```

使用 `rtol=1e-5, atol=1e-6`，覆盖：

| 类别 | 用例 |
|---|---|
| 宽行 benchmark | `(4096, 4096)` |
| small-size benchmark | `(8192, 128)` |
| 非 2 次幂/标量尾部 | `D=17、1003、4097` |
| 常见宽度 | `D=128、256、1024、4096` |
| 数值范围 | 大幅值输入、全零输入 |
| Python API | 非连续输入、空维度、非二维、错误 dtype/device/epsilon |

运行方法：

```bash
source scripts/env.sh
source ~/.python/miniinfer/bin/activate
cmake --build build -j 8

python -m cuda_learn.bench --check-only \
  test_rmsnorm test_rmsnorm_small_hidden test_rmsnorm_edge_cases
python -m unittest cuda_learn.tests.test_rmsnorm_validation -v
```

## 5. 性能与 profiler 结论

测试设备为仓库当前目标设备 RTX 4060 Laptop GPU（sm_89）。CUDA Event 端到端结果包含
Python wrapper、输出分配、FFI 和 kernel launch：

| 输入 | 当前实现 median | 256-thread baseline median | speedup |
|---|---:|---:|---:|
| `(4096, 4096)` | `0.674 ms` | `0.677 ms` | `1.00x` |
| `(8192, 128)` | `0.058 ms` | `0.094 ms` | `1.62x` |

PyTorch CUDA profiler 单独统计 `(8192,128)` 的 kernel 时间：

| kernel | 平均 GPU 时间 | calls |
|---|---:|---:|
| `rmsnorm_small_kernel` | `8.992 us` | 100 |
| `rmsnorm_kernel`，固定 256 threads | `93.090 us` | 100 |

kernel-only 约为 `10.35x`，而端到端只有 `1.62x`。差距说明 small tensor 场景已经开始受
输出 tensor 分配、FFI 与 launch 固定开销影响；继续只压缩 kernel 指令不会等比例改善
Python 端到端时间。

Nsight Compute 在当前机器上因 `ERR_NVGPUCTRPERM` 无法读取硬件 performance counters，
所以本轮没有给出 DRAM/L2 throughput 百分比。瓶颈判断来自算术强度、CUDA kernel timeline、
编译资源占用以及多组 A/B 微基准；若系统开放计数器，应继续验证：

- `dram__throughput` 与 L2 hit rate；
- shared-memory reduction 的 barrier stall；
- active warps、block limit 和 achieved occupancy；
- global load/store sector utilization。

## 6. 下一步值得做的优化

按预期收益排序：

1. **融合 weight、residual 与后继逐元素算子**：宽行受访存限制，减少 HBM 往返通常比
   替换 reduction 指令更有效；
2. **支持复用输出或 CUDA Graph**：降低 small-size 路径已经占主导的 allocation/launch
   固定开销；
3. **扩展 `D=256/512` 专用路径**：每 lane 缓存 2/4 个 `float4`，但必须实测寄存器压力；
4. **增加 FP16/BF16 输入、FP32 accumulation**：更贴近 LLM 推理实际数据类型；
5. **加入 LayerNorm CUDA baseline**：在相同 shape、dtype 和融合边界下比较，而不是用
   不同框架调用的总时间推断 RMSNorm 的收益。

核心经验是：small hidden size 的瓶颈主要是过度并行、同步和固定开销；宽 hidden size 的
瓶颈主要是数据移动。两者需要不同 kernel，不能用同一种“向量化”策略覆盖所有尺寸。
