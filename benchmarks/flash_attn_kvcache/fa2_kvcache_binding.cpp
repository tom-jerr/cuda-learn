#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cmath>

#include "cutlass/half.h"
#include "flash.h"

namespace {

void check_common(const torch::Tensor &q, const torch::Tensor &k,
                  const torch::Tensor &v, const torch::Tensor &out,
                  const torch::Tensor &softmax_lse,
                  const torch::Tensor &cache_seqlens) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(),
              "q, k, v and out must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 &&
                  k.scalar_type() == torch::kFloat16 &&
                  v.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "this focused benchmark supports FP16 only");
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 64,
              "q must have shape [batch, seqlen_q, heads_q, 64]");
  TORCH_CHECK(k.dim() == 4 && v.sizes() == k.sizes(),
              "k and v must be identically shaped rank-4 tensors");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  out.is_contiguous(),
              "q, k, v and out must be contiguous");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");
  TORCH_CHECK(k.size(2) > 0 && q.size(2) % k.size(2) == 0,
              "KV heads must divide Q heads");
  TORCH_CHECK(cache_seqlens.is_cuda() && cache_seqlens.is_contiguous() &&
                  cache_seqlens.scalar_type() == torch::kInt32 &&
                  cache_seqlens.numel() == q.size(0),
              "cache_seqlens must be contiguous CUDA int32 [batch]");
  TORCH_CHECK(softmax_lse.is_cuda() &&
                  softmax_lse.scalar_type() == torch::kFloat32 &&
                  softmax_lse.sizes() == torch::IntArrayRef(
                      {q.size(0), q.size(2), q.size(1)}),
              "softmax_lse must be CUDA float32 [batch, heads_q, seqlen_q]");
}

void set_base_params(flash::Flash_fwd_params &params, const torch::Tensor &q,
                     const torch::Tensor &k, const torch::Tensor &v,
                     const torch::Tensor &out,
                     const torch::Tensor &softmax_lse,
                     const torch::Tensor &cache_seqlens, int seqlen_k) {
  params = {};
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();

  // Logical layouts are BSND for Q/O and either BSND or paged PSND for K/V.
  // All FlashAttention strides are expressed in elements.
  params.q_batch_stride = q.stride(0);
  params.q_row_stride = q.stride(1);
  params.q_head_stride = q.stride(2);
  params.k_batch_stride = k.stride(0);
  params.k_row_stride = k.stride(1);
  params.k_head_stride = k.stride(2);
  params.v_batch_stride = v.stride(0);
  params.v_row_stride = v.stride(1);
  params.v_head_stride = v.stride(2);
  params.o_batch_stride = out.stride(0);
  params.o_row_stride = out.stride(1);
  params.o_head_stride = out.stride(2);

  params.softmax_lse_ptr = softmax_lse.data_ptr();
  params.b = static_cast<int>(q.size(0));
  params.h = static_cast<int>(q.size(2));
  params.h_k = static_cast<int>(k.size(2));
  params.h_h_k_ratio = params.h / params.h_k;
  params.seqlen_q = static_cast<int>(q.size(1));
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = ((params.seqlen_q + 127) / 128) * 128;
  params.seqlen_k_rounded = ((seqlen_k + 127) / 128) * 128;
  params.d = params.d_rounded = 64;

  params.scale_softmax = 1.0f / std::sqrt(64.0f);
  params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;
  params.p_dropout = 1.0f;  // FA2 stores keep probability here.
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = params.scale_softmax;
  params.window_size_left = -1;
  params.window_size_right = -1;
  params.is_bf16 = false;
  params.is_causal = false;  // q_len=1 decode is non-causal in FA2 as well.
  params.cu_seqlens_k = cache_seqlens.data_ptr<int>();
  params.is_seqlens_k_cumulative = false;
  params.num_splits = 1;  // One kernel, no accumulator/combine allocation.
}

}  // namespace

// mode:
//   0 = dense cache, regular FA2 forward path
//   1 = dense cache, forced split-KV path (same kernel family as paged modes)
//   2 = paged cache, split-KV path
void fa2_kvcache_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor out, torch::Tensor softmax_lse,
                         torch::Tensor cache_seqlens,
                         c10::optional<torch::Tensor> block_table, int64_t mode) {
  check_common(q, k, v, out, softmax_lse, cache_seqlens);
  TORCH_CHECK(mode >= 0 && mode <= 2, "mode must be 0, 1, or 2");

  const bool paged = mode == 2;
  int seqlen_k = 0;
  if (paged) {
    TORCH_CHECK(block_table.has_value(), "paged mode requires block_table");
    const auto &table = *block_table;
    TORCH_CHECK(table.is_cuda() && table.is_contiguous() &&
                    table.scalar_type() == torch::kInt32 && table.dim() == 2 &&
                    table.size(0) == q.size(0),
                "block_table must be contiguous CUDA int32 [batch, pages]");
    TORCH_CHECK(k.size(1) % 256 == 0,
                "FA2 page size must be a multiple of 256");
    seqlen_k = static_cast<int>(table.size(1) * k.size(1));
  } else {
    TORCH_CHECK(!block_table.has_value(), "dense mode does not take block_table");
    TORCH_CHECK(k.size(0) == q.size(0),
                "dense K/V batch must equal Q batch");
    seqlen_k = static_cast<int>(k.size(1));
  }

  flash::Flash_fwd_params params;
  set_base_params(params, q, k, v, out, softmax_lse, cache_seqlens,
                  seqlen_k);
  if (paged) {
    params.block_table = block_table->data_ptr<int>();
    params.block_table_batch_stride = block_table->stride(0);
    params.page_block_size = static_cast<int>(k.size(1));
  } else {
    params.page_block_size = 1;
  }

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (mode == 0) {
    flash::run_mha_fwd_<cutlass::half_t, 64, false>(params, stream);
  } else {
    flash::run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64, false>(params,
                                                                    stream);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &fa2_kvcache_forward,
             "Focused FlashAttention-2 FP16/D64 KV-cache forward");
}
