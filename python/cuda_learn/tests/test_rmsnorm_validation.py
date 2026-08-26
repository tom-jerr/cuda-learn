import unittest

import torch

from cuda_learn import ops


class RMSNormValidationTest(unittest.TestCase):
    def test_noncontiguous_input(self):
        x = torch.randn(17, 7, device="cuda").t()
        self.assertFalse(x.is_contiguous())
        expected = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5)
        torch.testing.assert_close(
            ops.rmsnorm(x), expected, rtol=1e-5, atol=1e-6)

    def test_rejects_non_2d_and_empty_shapes(self):
        with self.assertRaises(ValueError):
            ops.rmsnorm(torch.randn(8, device="cuda"))
        with self.assertRaises(ValueError):
            ops.rmsnorm(torch.empty(0, 8, device="cuda"))
        with self.assertRaises(ValueError):
            ops.rmsnorm(torch.empty(8, 0, device="cuda"))

    def test_rejects_invalid_epsilon(self):
        x = torch.randn(2, 8, device="cuda")
        for epsilon in (-1.0, float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                ops.rmsnorm(x, epsilon)
        with self.assertRaises(TypeError):
            ops.rmsnorm(x, "1e-5")

    def test_rejects_cpu_and_non_float32(self):
        with self.assertRaises(TypeError):
            ops.rmsnorm(torch.randn(2, 8))
        with self.assertRaises(TypeError):
            ops.rmsnorm(torch.randn(2, 8, device="cuda", dtype=torch.float16))


if __name__ == "__main__":
    unittest.main()
