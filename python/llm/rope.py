import torch


# θ_i = base^(−2i/d)
def build_rope(max_len: int, head_dim: int, rope_theta: int):
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2))) / head_dim
    pos = torch.arange(0, max_len)

    freq = torch.outer(pos, inv_freq)  # [S, D]
    return freq


def apply_rope(x: torch.Tensor, freq: torch.Tensor):
    # x: [B, S, H, D/2]
    x_even = x[:, :, :, ::2]
    x_odd = x[:, :, :, 1::2]

    cos = freq.cos()[None, :, None, :]
    sin = freq.sin()[None, :, None, :]

    x_even = x_even * cos - x_odd * sin
    x_odd = x_even * sin + x_odd * cos
    return torch.stack(x_even, x_odd, dim=-1).flatten(2)  # [B, S, D]
