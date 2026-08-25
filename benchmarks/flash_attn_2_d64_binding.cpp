#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cmath>

#include "cutlass/half.h"
#include "flash.h"

// A minimal benchmark-only binding around flash-attention 2.8.4's official
// FP16, head-dim-64 forward specializations. Keeping only causal/non-causal D64
// avoids compiling the library's roughly 100 unrelated translation units,
// without changing either kernel being measured.
void flash_attn_2_fwd_d64(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor out, torch::Tensor softmax_lse,
                          bool causal) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(),
              "q, k, v and out must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 &&
                  k.scalar_type() == torch::kFloat16 &&
                  v.scalar_type() == torch::kFloat16 &&
                  out.scalar_type() == torch::kFloat16,
              "q, k, v and out must be float16");
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 64,
              "expected q shaped [B,H,N,64]");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() &&
                  out.sizes() == q.sizes(),
              "q, k, v and out must have identical shapes");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  out.is_contiguous(),
              "q, k, v and out must be contiguous");

  const int batch = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int seqlen = static_cast<int>(q.size(2));
  TORCH_CHECK(softmax_lse.is_cuda() &&
                  softmax_lse.scalar_type() == torch::kFloat32 &&
                  softmax_lse.sizes() == torch::IntArrayRef({batch, heads, seqlen}),
              "softmax_lse must be float32 CUDA [B,H,N]");

  flash::Flash_fwd_params params = {};
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();

  // Inputs use cuda_learn's [B,H,N,D] physical layout. FlashAttention accepts
  // arbitrary row/head strides, so present the same storage as logical BNHD
  // without a transpose or copy in the timed path.
  const int64_t batch_stride = static_cast<int64_t>(heads) * seqlen * 64;
  const int64_t head_stride = static_cast<int64_t>(seqlen) * 64;
  params.q_batch_stride = params.k_batch_stride = params.v_batch_stride =
      params.o_batch_stride = batch_stride;
  params.q_row_stride = params.k_row_stride = params.v_row_stride =
      params.o_row_stride = 64;
  params.q_head_stride = params.k_head_stride = params.v_head_stride =
      params.o_head_stride = head_stride;

  params.softmax_lse_ptr = softmax_lse.data_ptr();
  params.b = batch;
  params.h = params.h_k = heads;
  params.h_h_k_ratio = 1;
  params.seqlen_q = params.seqlen_k = seqlen;
  params.seqlen_q_rounded = params.seqlen_k_rounded =
      ((seqlen + 127) / 128) * 128;
  params.d = params.d_rounded = 64;
  params.scale_softmax = 1.0f / 8.0f;
  params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;
  params.p_dropout = 1.0f; // flash-attention stores keep probability here.
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = params.scale_softmax;
  params.window_size_left = -1;
  params.window_size_right = causal ? 0 : -1;
  params.is_bf16 = false;
  params.is_causal = causal;
  params.is_seqlens_k_cumulative = true;
  params.num_splits = 0;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (causal) {
    flash::run_mha_fwd_<cutlass::half_t, 64, true>(params, stream);
  } else {
    flash::run_mha_fwd_<cutlass::half_t, 64, false>(params, stream);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &flash_attn_2_fwd_d64,
             "flash-attention 2 FP16 D=64 forward");
}
