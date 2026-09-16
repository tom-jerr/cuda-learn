# mbarrier实验

完整解释见 [TMA descriptor与mbarrier教程](../../docs/tma_descriptor_mbarrier.md)。

```bash
# 仓库根目录执行
make -C examples mbarrier/main
./examples/mbarrier/main --model   # CPU概念计数，不读取硬件位域
./examples/mbarrier/main           # GPU arrival + SM90 transaction分支
```

SM89能运行128线程、6阶段的arrival/数据可见性实验；expect_tx/complete_tx需要SM90。
程序明确报告每一部分PASS或SKIP。`primitives.cuh`也被TMA完整搬运示例使用。
