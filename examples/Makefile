# 一键重编 standalone 演示（原始学习材料）。
# flash_attn/simple.cu 只有 kernel 没有 main，不在此列；
# gemm/mma.cu 为空文件（留作后续迁移的占位）。
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_89

BINS := vector_add/simple \
        transpose/simple \
        dotproduct/simple \
        gemm/simple \
        fusion/silu_and_mul \
        softmax/simple \
        cuda_allocator/minimal_allocator \
        cuda_allocator/import_export_pool \
        cuda_graph/simple \
        cuda_graph/fork_join

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean
