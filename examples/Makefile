# 一键重编 standalone 演示（原始学习材料）。
# flash_attn/simple.cu 只有 kernel 没有 main，不在此列；
# gemm/mma.cu 为空文件（留作后续迁移的占位）。
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_89
CLUSTER_NVCCFLAGS ?= -std=c++17 -O2 -arch=sm_90

BINS := vector_add/simple \
        embedding/simple \
        kv_cache/simple \
        transpose/simple \
        dotproduct/simple \
        gemm/simple \
        fusion/silu_and_mul \
        softmax/simple \
        sgemv_k32/simple \
        topk/simple \
        prefix_sum/simple \
        quant/awq_gptq_gemm \
        cuda_allocator/minimal_allocator \
        cuda_allocator/import_export_pool \
        cuda_graph/simple \
        cuda_graph/fork_join \
        custom_all_reduce/simple \
        custom_all_reduce/fused_residual_rmsnorm \
        custom_all_reduce/ipc_1shot_push \
        custom_all_reduce/cuda_graph_pointer_table

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Thread block clusters require compute capability 9.0 or newer, so keep this
# target separate from `all` (the repository's default GPU is sm_89).
histogram/cluster: histogram/cluster.cu
	$(NVCC) $(CLUSTER_NVCCFLAGS) $< -o $@

# Make the default-stream behavior in this teaching example explicit.
cuda_stream/non_blocking: cuda_stream/non_blocking.cu
	$(NVCC) $(NVCCFLAGS) --default-stream legacy $< -o $@

custom_all_reduce/pdl_staged_pull: custom_all_reduce/pdl_staged_pull.cu
	$(NVCC) $(CLUSTER_NVCCFLAGS) $< -o $@

custom_all_reduce/vmm_multicast_reduce: custom_all_reduce/vmm_multicast_reduce.cu
	$(NVCC) $(CLUSTER_NVCCFLAGS) $< -lcuda -o $@

clean:
	rm -f $(BINS) histogram/cluster cuda_stream/non_blocking \
		custom_all_reduce/pdl_staged_pull \
		custom_all_reduce/vmm_multicast_reduce

.PHONY: all clean
