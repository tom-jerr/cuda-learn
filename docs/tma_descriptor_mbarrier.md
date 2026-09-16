# TMA descriptor 与 mbarrier：从一块8×32 tile开始

本教程参考 [reed-lau/cute-gemm 的 mbarrier 实验](https://github.com/reed-lau/cute-gemm/tree/37f3a01eeef7ac7f404343b2cc67e8f0c21018f2/mbarrier) 与 [tma-desc 实验](https://github.com/reed-lau/cute-gemm/tree/37f3a01eeef7ac7f404343b2cc67e8f0c21018f2/tma-desc)，固定参考commit为 `37f3a01eeef7ac7f404343b2cc67e8f0c21018f2`。

代码使用CUDA Driver API与少量inline PTX，没有CUTLASS依赖。先直接看清TMA/同步协议，再与 [FA3/FlashMLA](hopper_fp8_attention_layouts.md) 中的CuTe封装对应。

## 1. 三个程序的分工和运行方式

| 文件 | 做什么 | 运行要求 |
|---|---|---|
| [`examples/mbarrier/main.cu`](../examples/mbarrier/main.cu) | CPU计数模型、128线程arrival实验、Hopper事务计数实验 | `--model`无需GPU；arrival需要SM80+；事务实验需要SM90+ |
| [`examples/mbarrier/primitives.cuh`](../examples/mbarrier/primitives.cuh) | init/arrive/wait/expect_tx等PTX包装 | 被其他示例包含 |
| [`examples/tma_desc/main.cu`](../examples/tma_desc/main.cu) | 打印descriptor参数、核对layout、可选调用官方编码API | 默认/`--layout`无需GPU；`--encode`需要支持的驱动环境 |
| [`examples/tma_desc/config.cuh`](../examples/tma_desc/config.cuh) | 固定矩阵规格、SW128地址与descriptor编码 | 两个TMA程序共用 |
| [`examples/tma_desc/tma_copy.cu`](../examples/tma_desc/tma_copy.cu) | TMA load→shared坐标计算→TMA store，复用buffer十二轮 | GPU执行需要SM90+；`--layout`无需GPU |

从仓库根目录运行：

```bash
make -C examples -j3 hopper-basics

# 不执行GPU指令：先把计数和地址算清楚。
./examples/mbarrier/main --model
./examples/tma_desc/main --layout

# GPU基础barrier实验；SM89上会明确跳过事务计数部分。
./examples/mbarrier/main

# 官方descriptor编码；与运行TMA是两个独立检查。
./examples/tma_desc/main --encode

# 在Hopper上验证两种shared layout和十二轮buffer复用。
./examples/tma_desc/tma_copy
```

这些目标独立于默认 `make -C examples all`，不会把默认SM89示例全部改成SM90。`hopper-basics`负责构建，不自动启动GPU实验。

`mbarrier/main`包含SM80 cubin、SM90 cubin和compute90 PTX。SM89运行基础arrival分支；SM90运行完整实验。`tma_copy`默认编译为SM90，可通过 `HOPPER_NVCCFLAGS`覆盖；本例只使用TMA/mbarrier，不要求WGMMA所使用的SM90a构建设置。

退出码：正常完成所选检查返回0；独立的encode/TMA执行不受支持返回77，并打印SKIP；真实错误或结果不一致返回1。基础arrival程序在SM89通过后返回0，同时明确打印事务部分SKIP，不能将这个0解释为事务实验也通过。

## 2. 参考代码有哪些地方需要区分实验与正式接口

原 `mbarrier/main.cu` 用位域解释64-bit对象，并在打印前调用 `mbarrier.inval`，注释称其为禁用barrier cache。这里必须更正：**inval会让barrier失效；失效后继续arrive/wait而不重新init，是未定义行为。** 它不是刷新缓存或读取计数器的API。

原 `tma-desc` 还包含逆向得到的descriptor编码与位域打印。本教程使用官方 `cuTensorMapEncodeTiled`，打印其输入参数与逻辑地址关系。`CUtensorMap` 的128 bytes与mbarrier的8 bytes都按opaque对象使用。

因此程序中的CPU `Model` 只是解释“还差多少arrival/bytes”的概念计数器，不是硬件位域解析器；GPU实验用合法的wait结果和数据校验观察行为。关于对象失效后的限制，见 [PTX mbarrier.inval](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval)。

## 3. 从我们需要搬的tile确定descriptor参数

固定逻辑矩阵：32行、96列、float。为了让stride错误暴露出来，每行实际分配112个float，其中后16个是padding。

```text
global，一行共112个float：
┌────────────32───────────┬────────────32───────────┬────────────32───────────┬──16──┐
│ 第0个列tile             │ 第1个列tile             │ 第2个列tile             │ pad  │
└────────────────────────┴────────────────────────┴────────────────────────┴──────┘
逻辑列数96                                                         分配行距112

一个CTA搬运小块：8行 × 32列 float = 256个float = 1024 bytes
整个矩阵：沿行4块 × 沿列3块 = 12个tile
shared：只分配一个8×32槽，复用12次
```

用前面CuTe的写法，global是 `(32,96):(112,1)`，stride单位为float元素。TMA API按内层连续维优先描述，因此参数顺序为 `(col,row)`：

| API参数 | 本例的值 | 从tile角度理解 |
|---|---|---|
| `tensorDataType` | FLOAT32 | 单个元素4 bytes |
| `tensorRank` | 2 | col、row两个维度 |
| `globalAddress` | cudaMalloc返回的矩阵基址 | 描述整张矩阵，不是每个tile单独换基址 |
| `globalDim` | `{96,32}` | 整张矩阵各维的逻辑元素数 |
| `globalStrides` | `{448}` | 从一行走到下一行：112×4 bytes |
| `boxDim` | `{32,8}` | 一次搬32列、8行 |
| `elementStrides` | `{1,1}` | 逐元素取样；不是每行字节数 |
| `interleave` | NONE | 不使用交错通道格式 |
| `swizzle` | NONE或128B | shared目的排列；不修改global逻辑布局 |
| `l2Promotion` | NONE | 此教学例不额外设置L2 promotion |
| `oobFill` | FLOAT_OOB_FILL_NONE | 不请求特殊NaN填充；本例所有tile都在边界内 |

API的dim0 stride隐含为元素大小4 bytes，所以二维tensor只传**一个**global stride。把 `{4,448}` 全部当成二维 `globalStrides` 传入会把参数含义弄错。

`elementStrides` 是取样步长；本例固定1。`interleave=NONE`时dim0的取样步长不支持、该项被忽略，不能用它给最内维制造任意字节间隔。[官方tensor map参数约束](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)。

## 4. 跨tile时global/shared各增加多少

先看NONE shared layout：`(8,32):(32,1)`。

| 当前坐标的移动 | global变化 | shared变化 |
|---|---:|---:|
| 向右1列 | +1 float / +4 B | +1 float / +4 B |
| 向下1行 | +112 float / +448 B | +32 float / +128 B |
| 向右一个32列tile | +32 float / +128 B | 下一轮复用同一槽的基址 |
| 向下一个8行tile | +896 float / +3584 B | 下一轮复用同一槽的基址 |

跨tile的global步长是**下一次TMA的源起点变化**。它不是mbarrier要等待的字节数。

尽管global八行之间有pitch，TMA真正传输的有效tile仍只有 `8×32×4=1024 B`。它不会把每行padding和另外两个列tile也一起计入当前传输。

本例tile枚举为先向右再向下：

```text
            col 0..31   col 32..63   col 64..95
row  0..7      0            1            2
row  8..15     3            4            5
row 16..23     6            7            8
row 24..31     9           10           11
```

kernel传给TMA的坐标依次为 `(0,0),(32,0),(64,0),(0,8),...`，顺序是 `(col,row)`，单位是元素坐标。这种扫描顺序由kernel循环决定；descriptor本身不负责按某个顺序扫描全部tile。

## 5. SW128怎样改变shared layout

本例一行32个float，恰好128 bytes。把一行切成8个16-byte包，每包4个float。SW128重新排列这些包，包内4个float仍然连续：

```text
逻辑包编号       0 1 2 3 4 5 6 7
row0物理位置     0 1 2 3 4 5 6 7
row1物理位置     1 0 3 2 5 4 7 6
row2物理位置     2 3 0 1 6 7 4 5
row3物理位置     3 2 1 0 7 6 5 4
row4物理位置     4 5 6 7 0 1 2 3
```

例如逻辑 `tile(1,0..3)`：NONE放在物理float槽32..35；SW128放在36..39。TMA已经按swizzle写入，普通CUDA Core读写shared时必须按同一映射访问。

代码为本例固定布局使用：

```cpp
row * 32 + (col ^ (4 * (row % 8)))
```

这是float元素偏移。**它依赖32-float行宽，以及shared基址1024-byte对齐。** 代码显式声明 `__align__(1024)`，使tile起点位于完整swizzle周期边界。不要把这个公式直接套到任意宽度、dtype或shared子视图基址。

CPU检查枚举全部256个坐标，确认恰好覆盖全部256个物理槽。真正的GPU检查还会给每个逻辑坐标加上不同的值：

```text
tile(row,col) += 1 + 3*row + col
```

如果只是给每个物理元素都加1，即使逻辑swizzle索引用错，完整遍历所有槽后也可能通过检查；带坐标的修改能检出这种错误。

## 6. descriptor究竟是什么，在哪里创建和传递

主机先分配真实global数据，再编码：

```cpp
alignas(64) CUtensorMap input{};
cuTensorMapEncodeTiled(
    &input, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
    input_device,
    dims, strides_in_bytes, box, element_steps,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

实际程序检查API返回值。失败不会记为ENCODE PASS；驱动返回801“不支持”时单独打印SKIP。

descriptor同时包含矩阵基址、维度、stride、搬运box及模式。它没有把矩阵数据装进128 bytes；GPU发起TMA时仍从descriptor指定的global地址读数据。

本例分别为输入和输出创建descriptor，通过kernel参数传入：

```cpp
__global__ void copy_tiles(
    __grid_constant__ const CUtensorMap input,
    __grid_constant__ const CUtensorMap output);
```

`__grid_constant__`允许所有线程使用kernel参数对象的公共地址，避免取参数地址时引入每线程局部副本。本例的主机descriptor在launch时按值复制到kernel参数；被descriptor引用的device数据要保持有效直到kernel完成。

descriptor动态修改属于另一个问题，需要专门的tensor-map更新及proxy同步协议。本例encode后只读，循环只修改TMA坐标，不修改descriptor。

不要把下面三个对象混在一起：

| 对象 | 大小/表示 | 作用 |
|---|---|---|
| TMA `CUtensorMap` | 128-byte opaque对象；编码API要求至少64-byte对齐 | 描述global tensor与TMA搬运规则 |
| WGMMA shared descriptor | 64-bit指令操作数 | 描述WGMMA从shared读取矩阵的地址/步长/swizzle |
| shared address | 示例中通过`__cvta_generic_to_shared`得到32-bit地址 | 给PTX提供shared目标或barrier位置 |

两种descriptor之间没有直接按位转换关系。

## 7. mbarrier只需要先理解两笔账

mbarrier维护阶段完成条件，可以先理解为：

```text
本阶段还差的arrival数 == 0
并且
本阶段还差的transaction数 == 0
→ 阶段完成
```

TMA的 `complete_tx::bytes` 协议中transaction按**字节**计数。mbarrier本身不是一个通用“所有异步指令完成”的开关，只有关联到它的操作才会更新它。

| 指令 | arrival账 | transaction账 | 是否等待 |
|---|---|---|---|
| `init(count)` | 初始化本轮/后续轮次期望数 | 初始化为0 | 否 |
| `arrive()` | 完成1次arrival | 不变 | 否 |
| `expect_tx(bytes)` | 不变 | 增加待完成字节 | 否 |
| `arrive.expect_tx(bytes)` | 完成1次arrival | 增加待完成字节 | 否 |
| `complete_tx(bytes)` | 不变 | 扣除已完成字节 | 否；该指令本身不搬数据 |
| `test_wait/try_wait` | 不变 | 不变 | 检查或尝试等待指定阶段完成 |
| `inval` | 对象失效 | 对象失效 | 只能在所有使用结束后执行 |

`init(128)`并不自动表示“CTA有128线程”；它表示期望128个arrival。可以让128线程各arrive一次，也可以在合法计数规则下由其他分工贡献这些arrival。**wait的人数和arrive次数是两个不同的数量。**

## 8. 程序一：基础arrival与数据可见性

`arrival_visibility<<<1,128>>>`分六轮执行：

```text
初始化 ready，期望128个arrival
CTA同步发布初始化

每线程写 values[tid]
每线程 arrive(ready)
每线程 wait(ready,当前phase)
线程0读取全部128个values并计算sum
CTA同步，确认线程0读完
下一轮，phase翻转
```

arrive的默认release语义与成功wait的默认acquire语义，使完成阶段的写入对等待者可见。host核对六轮的sum，既观察计数，也检查数据可见性。

为什么wait后仍有 `__syncthreads()`？ready完成只表示各线程完成了本轮写入/arrival。线程0随后还要读取；其他线程不能立即覆盖下一轮values。后面的CTA barrier保护**读者的使用期**。

这个模式对应之前GEMM的两类同步：数据已经到齐，以及数据已经用完。它们不能合并成同一个含糊的“同步完成”。

## 9. 程序一的Hopper分支：arrival齐了也可能不能通过

`transaction_accounting<<<1,1>>>`故意只用一个线程，先init为3，再调用三次arrive。这样可以按顺序观察计数，不把warp调度混进来。

| 操作后 | 待arrival | 待bytes | 当前活动phase | 等phase0能否完成 |
|---|---:|---:|---:|---|
| init(3) | 3 | 0 | 0 | 否 |
| expect_tx(1024) | 3 | 1024 | 0 | 否 |
| arrive一次 | 2 | 1024 | 0 | 否 |
| complete_tx(256) | 2 | 768 | 0 | 否 |
| 再arrive两次 | 0 | 768 | 0 | 否 |
| complete_tx(512) | 0 | 256 | 0 | 否 |
| complete_tx(256) | 自动恢复3 | 0 | 1 | 是，phase0已完成 |

随后再做一轮 `arrive_expect_tx(512)`、两次arrive、`complete_tx(512)`，phase1完成，活动phase回到0。

这里的手工 `complete_tx` **只是模拟事务完成的记账效果**，没有执行异步copy。真实TMA load会在完成时自动提交1024字节的完成量，不能又由线程手工complete一次。

表中的pending/tx是教学推导，并非程序读取的硬件字段。GPU记录的是完成/未完成的观察结果；已完成的检查使用wait确保指令观察到完成。

## 10. phase翻转：等待的是旧阶段完成

barrier初始化时活动phase为0：

```text
活动phase0：等待本轮arrival/bytes
       ↓ 本轮完成
活动phase1：下一轮计数开始
       ↓ 下一轮完成
活动phase0：再下一轮计数开始
```

因此 `wait(bar,0)`表示等phase0完成，而不是等“barrier的当前phase变成0”。消费者初始保存0，成功等待后才把自己的phase变量异或1。

parity只有一位，不能辨认任意久以前的轮次。PTX等待协议针对当前或紧邻前一阶段，程序必须约束producer不能无限超前绕回。本例每轮的buffer使用与CTA同步使所有参与者保持相同轮次。

没有必要每轮 `inval + init`。正常完成会自动开始下一阶段，并恢复expected arrival计数；每一轮需要按实际新搬运重新登记transaction bytes。

## 11. 程序三：把descriptor与mbarrier接起来

`tma_copy.cu`使用128线程、一个shared tile、一个ready barrier。核心结构如下，省略错误检查和host初始化：

```cpp
if (tid == 0) {
  init(&ready, 1);
  async_shared_fence();
}
__syncthreads();

unsigned phase = 0;
for (int tile_id = 0; tile_id < 12; ++tile_id) {
  if (tid == 0) {
    arrive_expect_tx(&ready, 1024);
    load_2d(shared_tile, &input_map, col, row, &ready);
  }
  wait(&ready, phase);
  phase ^= 1;

  // 128线程各处理两个元素，SW128分支使用对应shared地址。
  transform_shared_tile();
  async_shared_fence();
  __syncthreads();

  if (tid == 0) {
    tma_store(...);
    bulk_commit_group();
    bulk_wait_group_0();
  }
  __syncthreads();
}
```

### 11.1 为什么init是1，不是128

只有thread0调用 `arrive_expect_tx`，所以expected arrival为1。128线程都可以wait同一个barrier，但wait不会减少arrival计数。

如果init设为128却只有thread0 arrive，本阶段永远缺127次arrival；如果128线程都执行 `arrive_expect_tx(1024)`，又只发一条1024-byte TMA，就会登记过多字节。

### 11.2 为什么先登记1024，再issue load

本例让thread0先完成arrival并登记1024字节：arrival归零时tx仍不为零，phase不会提前完成。随后TMA发起并自动扣除完成字节，两个条件同时满足后phase翻转。

这个顺序把依赖写得直观。若要改成别的登记/发起顺序，必须重新验证计数与异步完成之间的竞态，不能因为某次运行“足够慢”就认为顺序正确。

load PTX为：

```text
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
```

操作数包括shared目标地址、tensor-map地址、`{col,row}`坐标和mbarrier地址。后缀明确指出使用这个barrier按bytes报告完成。本例不需要两CTA cluster launch；`shared::cluster`是该Hopper指令的shared目标地址空间形式，并不自动创建cluster或multicast。

### 11.3 wait保证load完成，但不代表计算完成

wait成功后128个线程才能读取shared tile。每线程处理两个逻辑值：第一个是`tid`，第二个是`tid+128`，再按NONE/SW128映射到物理shared地址。

计算结束后每个writer执行async proxy fence，再做CTA同步，thread0才issue TMA store。这把CUDA线程的generic shared写入正确发布给异步TMA读取。单独让thread0 fence不能替代其他writer的发布协议。

### 11.4 TMA store使用bulk group，不使用这个ready barrier

store PTX为：

```text
cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
cp.async.bulk.commit_group
cp.async.bulk.wait_group 0
```

load的ready barrier只跟踪G2S，S2G走发起线程自己的bulk async group。`wait_group 0`在本例等待完整store完成；随后CTA barrier使其他线程也知道shared可以复用。

本例选择完整wait，便于教学。更精细的实现可以区分源shared已读完与global写入已完成，例如研究 `.read` 等待形式；但只等待load的mbarrier绝对不足以保护store的源buffer。

## 12. 初始化发布、数据发布、buffer复用分别保证什么

```text
init ready
   │  初始化发布：async proxy fence + CTA barrier
   ▼
登记bytes、发起TMA load
   │  ready wait：输入数据完成并可见
   ▼
读取/修改shared tile
   │  writer fence + CTA barrier：所有计算结果可供TMA读取
   ▼
发起TMA store
   │  bulk wait + CTA barrier：store结束，shared可以被下一轮覆盖
   └──────────────────────────────→ 下一轮load
```

这个循环只用一个buffer，刻意将生命周期展示完整；它没有声称实现多级copy/compute重叠。下一步加入double buffering时，可以将每个槽拆成ready和free两种状态，让producer等待所有consumer释放，而不是在每轮阻塞整个CTA。

两buffer的phase必须按**同一个槽复用了几次**计算：

| tile_id | buffer槽 | 本次应等待的ready phase |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 1 |
| 4 | 0 | 0 |

一buffer时phase是`tile_id%2`；两buffer时是`(tile_id/2)%2`。`buffer index`与`phase`不同，不能把槽编号直接当成phase。ready/free barrier首次使用的启动条件也要分别设计，不能机械复制单buffer循环。

## 13. 对应到CuTe/FA3中的API

| 这里显式写的内容 | 在CuTe/CUTLASS中对应的职责 |
|---|---|
| `cuTensorMapEncodeTiled`与dims/strides/box | `make_tma_copy`根据global tensor与shared layout构造tensor map |
| `load_2d` | TMA copy atom，如`SM90_TMA_LOAD` |
| descriptor地址与TMA坐标 | `get_tma_tensor`、`partition_S/D`得到的TMA视图 |
| `init / arrive_expect_tx / wait` | transaction barrier与pipeline acquire/commit/wait |
| 等consumer用完buffer | pipeline consumer release / producer acquire |
| generic→async shared发布 | `fence_view_async_shared`等指令封装 |

CuTe的 `partition` 是把索引分工应用到tensor；descriptor是TMA实际执行所需的硬件描述；mbarrier负责传输阶段完成。这三个API层次各自处理不同的问题。

在FA3的SS QK中，TMA ready之后WGMMA可以通过shared descriptor读Q/K，不一定经过ldmatrix。到了FlashMLA稀疏FP8 decode，输入需要先gather/dequant，这部分通常是线程级load和shared store，再按其发布与barrier协议交给WGMMA。不能只看到Hopper就把所有G2S改成同一条TMA指令。

## 14. 验证范围与排查顺序

本次环境为CUDA13、RTX4060 Laptop SM89：

- 三个程序均可构建，Hopper分支通过PTX汇编。
- CPU layout覆盖与SW128双射检查、CPU计数模型通过。
- GPU基础arrival实验通过：128线程，6轮phase，读取全部shared值验证可见性。
- 当前驱动的官方descriptor编码返回801，不标为编码成功。
- transaction accounting和完整TMA执行需要SM90+，本机未运行；程序明确打印SKIP。

在Hopper运行时，TMA测试对NONE与SW128各处理12个tile，检查3072个有效float，以及512个padding元素保持sentinel。坐标计算、行pitch、swizzle、barrier重复使用都参与这个检查；它是正确性示例，不做性能benchmark。

遇到卡住或错误时，沿数据生命周期检查：先看descriptor的维度顺序与**字节**stride，再看expected arrival数与tx bytes是否与真实发起量一致，然后看等待的是哪一轮phase，最后检查shared的最后一个读者是否已经完成。无需读取opaque对象的内部位域来定位这些问题。

官方参考：[Tensor Memory API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)、[异步barrier](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)、[异步数据搬运](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。
