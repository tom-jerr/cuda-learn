# TMA descriptor与完整搬运

完整解释见 [TMA descriptor与mbarrier教程](../../docs/tma_descriptor_mbarrier.md)。

```bash
# 仓库根目录执行
make -C examples tma_desc/main tma_desc/tma_copy
./examples/tma_desc/main --layout  # 无需GPU，检查32×96矩阵、112-float行pitch
./examples/tma_desc/main --encode  # 调用官方cuTensorMapEncodeTiled
./examples/tma_desc/tma_copy       # SM90+：NONE/SW128，12轮G2S→计算→S2G
```

每次搬8×32 float，transaction为1024 bytes。一个shared缓冲槽和ready barrier复用12轮。
kernel检查坐标相关计算与padding，防止错误的swizzle/stride被单纯拷贝掩盖。
不支持实际编码或TMA执行时返回77；默认layout检查不声称执行了TMA。
