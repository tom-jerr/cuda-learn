"""Focused correctness tests for the continuous multi-stage KV cache."""

import torch

from cuda_learn import ops


def _reference(q, k_cache, v_cache, cache_seqlens, causal):
    batch, heads_q, q_len, _ = q.shape
    heads_kv = k_cache.shape[1]
    group = heads_q // heads_kv
    out = torch.zeros_like(q)
    for b in range(batch):
        key_len = int(cache_seqlens[b])
        if key_len == 0:
            continue
        for h in range(heads_q):
            scores = (q[b, h].float()
                      @ k_cache[b, h // group, :key_len].float().T) / 8.0
            if causal:
                q_row = torch.arange(q_len, device=q.device)[:, None]
                k_col = torch.arange(key_len, device=q.device)[None, :]
                scores.masked_fill_(k_col > key_len - q_len + q_row,
                                    -torch.inf)
            probability = torch.nan_to_num(torch.softmax(scores, dim=-1))
            out[b, h] = (probability
                         @ v_cache[b, h // group, :key_len].float()).half()
    return out


def test_flash_attn_multistage_kvcache_cache_only():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(2)
    for q_len, lengths, causal in (
            (1, [1, 65], True),
            (5, [3, 70], True),
            (17, [64, 129], False),
            (64, [80, 130], True),
            (4, [0, 7], True)):
        q = torch.randn(2, 4, q_len, 64, device="cuda", dtype=torch.float16)
        k_cache = torch.randn(
            2, 2, 141, 64, device="cuda", dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)
        cache_seqlens = torch.tensor(lengths, device="cuda",
                                     dtype=torch.int32)
        actual = ops.flash_attn_multistage_kvcache(
            q, k_cache, v_cache, cache_seqlens, causal=causal)
        expected = _reference(
            q, k_cache, v_cache, cache_seqlens, causal=causal)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_flash_attn_multistage_kvcache_fused_append():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(3)
    batch, heads, q_len, capacity = 2, 3, 7, 137
    q = torch.randn(batch, heads, q_len, 64,
                    device="cuda", dtype=torch.float16)
    k_cache = torch.randn(batch, heads, capacity, 64,
                          device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    k_before, v_before = k_cache.clone(), v_cache.clone()
    knew = torch.randn(batch, heads, q_len, 64,
                       device="cuda", dtype=torch.float16)
    vnew = torch.randn_like(knew)
    cache_seqlens = torch.tensor([63, 129], device="cuda",
                                 dtype=torch.int32)

    actual = ops.flash_attn_multistage_kvcache(
        q, k_cache, v_cache, cache_seqlens, knew, vnew, causal=True)
    for b, old_len in enumerate((63, 129)):
        assert torch.equal(k_cache[b, :, old_len:old_len + q_len], knew[b])
        assert torch.equal(v_cache[b, :, old_len:old_len + q_len], vnew[b])
        assert torch.equal(k_cache[b, :, :old_len], k_before[b, :, :old_len])
        assert torch.equal(v_cache[b, :, :old_len], v_before[b, :, :old_len])

    expected = _reference(
        q, k_cache, v_cache, cache_seqlens + q_len, causal=True)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
