import unittest

import torch

from cuda_learn import ops


def rope_reference(x, cos_cache, sin_cache, position_ids):
    rotary_half = cos_cache.shape[1]
    cos = cos_cache[position_ids.long()].float().unsqueeze(1)
    sin = sin_cache[position_ids.long()].float().unsqueeze(1)
    out = x.clone()
    x0 = x[..., :rotary_half].float()
    x1 = x[..., rotary_half:2 * rotary_half].float()
    out[..., :rotary_half] = (x0 * cos - x1 * sin).to(x.dtype)
    out[..., rotary_half:2 * rotary_half] = (x0 * sin + x1 * cos).to(x.dtype)
    return out


class RoPENeoxValidationTest(unittest.TestCase):
    def make_inputs(self, head_dim=12, rotary_dim=8):
        tokens, q_heads, kv_heads, max_position = 5, 4, 2, 11
        q = torch.randn(tokens, q_heads, head_dim, device="cuda",
                        dtype=torch.float16)
        k = torch.randn(tokens, kv_heads, head_dim, device="cuda",
                        dtype=torch.float16)
        positions = torch.tensor([7, 0, 4, 4, 2], device="cuda",
                                 dtype=torch.int32)
        inv_freq = 1.0 / (10000 ** (torch.arange(
            0, rotary_dim, 2, device="cuda", dtype=torch.float32) /
            rotary_dim))
        angles = (torch.arange(max_position, device="cuda").float().unsqueeze(1)
                  * inv_freq.unsqueeze(0))
        return q, k, angles.cos().half(), angles.sin().half(), positions

    def test_gqa_partial_rotary_and_repeated_positions(self):
        q, k, cos, sin, positions = self.make_inputs()
        expected_q = rope_reference(q, cos, sin, positions)
        expected_k = rope_reference(k, cos, sin, positions)
        actual_q, actual_k = ops.rope_neox(q, k, cos, sin, positions)
        torch.testing.assert_close(actual_q, expected_q, rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(actual_k, expected_k, rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(actual_q[..., 8:], expected_q[..., 8:])
        torch.testing.assert_close(actual_k[..., 8:], expected_k[..., 8:])

    def test_noncontiguous_inputs_return_rotated_contiguous_copies(self):
        q, k, cos, sin, positions = self.make_inputs(head_dim=16, rotary_dim=8)
        q = q[..., ::2]
        k = k[..., ::2]
        self.assertFalse(q.is_contiguous())
        self.assertFalse(k.is_contiguous())
        expected_q = rope_reference(q, cos, sin, positions)
        expected_k = rope_reference(k, cos, sin, positions)
        actual_q, actual_k = ops.rope_neox(q, k, cos, sin, positions)
        self.assertTrue(actual_q.is_contiguous())
        self.assertTrue(actual_k.is_contiguous())
        torch.testing.assert_close(actual_q, expected_q, rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(actual_k, expected_k, rtol=3e-3, atol=3e-3)

    def test_rejects_bad_shapes(self):
        q, k, cos, sin, positions = self.make_inputs()
        cases = [
            (q[0], k, cos, sin, positions),
            (q, k[:-1], cos, sin, positions),
            (q, k, cos, sin[:, :-1], positions),
            (q, k, cos, sin, positions[:-1]),
            (q, k, cos[:, :3], sin[:, :3], positions),
            (q, k, torch.empty(11, 8, device="cuda", dtype=torch.float16),
             torch.empty(11, 8, device="cuda", dtype=torch.float16), positions),
        ]
        for args in cases:
            with self.subTest(shapes=[tuple(x.shape) for x in args]), \
                    self.assertRaises(ValueError):
                ops.rope_neox(*args)

    def test_rejects_wrong_dtype_device_and_alias(self):
        q, k, cos, sin, positions = self.make_inputs()
        with self.assertRaises(TypeError):
            ops.rope_neox(q.float(), k, cos, sin, positions)
        with self.assertRaises(TypeError):
            ops.rope_neox(q, k, cos, sin, positions.long())
        with self.assertRaises(TypeError):
            ops.rope_neox(q.cpu(), k, cos, sin, positions)
        with self.assertRaises(ValueError):
            ops.rope_neox(q, q, cos, sin, positions)


if __name__ == "__main__":
    unittest.main()
