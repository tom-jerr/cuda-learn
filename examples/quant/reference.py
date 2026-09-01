"""Compare INT8 per-tensor, per-token and per-channel quantization errors.

Tensor convention in this example: x.shape == [batch, tokens, channels].

- per-tensor:  one qparam for the complete tensor;
- per-token:   one qparam for every x[b, t, :];
- per-channel: one qparam for every x[..., c].
"""

from typing import Literal

import torch

Granularity = Literal["per_tensor", "per_token", "per_channel"]


def _reduce_dims(x: torch.Tensor, granularity: Granularity) -> tuple[int, ...]:
    if granularity == "per_tensor":
        return tuple(range(x.ndim))
    if granularity == "per_token":
        if x.ndim < 1:
            raise ValueError("per-token quantization expects at least 1 dimension")
        return (x.ndim - 1,)
    if granularity == "per_channel":
        if x.ndim < 2:
            raise ValueError("per-channel quantization expects at least 2 dimensions")
        return tuple(range(x.ndim - 1))
    raise ValueError(f"unsupported granularity: {granularity}")


def _reduce(
    x: torch.Tensor, granularity: Granularity, reduction: str
) -> torch.Tensor:
    dims = _reduce_dims(x, granularity)
    keepdim = granularity != "per_tensor"
    if reduction == "min":
        return x.amin(dim=dims, keepdim=keepdim)
    if reduction == "max":
        return x.amax(dim=dims, keepdim=keepdim)
    raise ValueError(f"unsupported reduction: {reduction}")


def symm_int8_quant(
    x: torch.Tensor, granularity: Granularity = "per_tensor"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantization using [-127, 127]."""
    qmin, qmax = -127, 127
    max_abs = _reduce(x.abs(), granularity, "max")
    scales = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    values = torch.clamp(torch.round(x / scales), qmin, qmax).to(torch.int8)
    return scales, values


def symm_int8_dequant(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return x.float() * scales


def asymm_int8_quant(
    x: torch.Tensor, granularity: Granularity = "per_tensor"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric signed INT8 quantization using [-128, 127]."""
    qmin, qmax = -128, 127
    xmin = _reduce(x, granularity, "min")
    xmax = _reduce(x, granularity, "max")

    # Include real zero so zero_point remains representable for one-sided data.
    xmin = torch.minimum(xmin, torch.zeros_like(xmin))
    xmax = torch.maximum(xmax, torch.zeros_like(xmax))
    value_range = xmax - xmin
    scales = torch.where(
        value_range > 0,
        value_range / (qmax - qmin),
        torch.ones_like(value_range),
    )
    computed_zp = torch.clamp(torch.round(qmin - xmin / scales), qmin, qmax)
    # For an all-zero range, scale=1 and zero_point=0 is the clearest convention.
    zp = torch.where(value_range > 0, computed_zp, torch.zeros_like(computed_zp))
    zp = zp.to(torch.int32)
    values = torch.clamp(torch.round(x / scales) + zp, qmin, qmax).to(torch.int8)
    return values, scales, zp


def asymm_int8_dequant(
    x: torch.Tensor, scales: torch.Tensor, zp: torch.Tensor
) -> torch.Tensor:
    return (x.float() - zp.float()) * scales


def error_metrics(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, float]:
    error = x - x_hat
    mse = error.square().mean()
    signal_power = x.square().mean()
    return {
        "mae": error.abs().mean().item(),
        "rmse": mse.sqrt().item(),
        "max_error": error.abs().max().item(),
        "sqnr_db": (10 * torch.log10(signal_power / mse)).item(),
    }


def make_test_tensor() -> torch.Tensor:
    """Create [B,T,C] data with channel/token scale variation and outliers."""
    torch.manual_seed(2026)
    batch, tokens, channels = 4, 128, 256
    x = torch.randn(batch, tokens, channels)
    token_scales = torch.linspace(0.25, 2.0, tokens).view(1, tokens, 1)
    channel_scales = torch.logspace(-1.0, 0.6, channels).view(1, 1, channels)
    x = x * token_scales * channel_scales
    x[0, 0, 0] = 30.0
    x[1, 63, 127] = -24.0
    return x


def compare_granularities(x: torch.Tensor) -> None:
    print(f"tensor shape: {tuple(x.shape)}, dtype: {x.dtype}")
    print(
        f"{'scheme':<11} {'granularity':<12} {'qparams':>7} "
        f"{'MAE':>10} {'RMSE':>10} {'MAX':>10} {'SQNR(dB)':>10}"
    )
    print("-" * 79)

    granularities: tuple[Granularity, ...] = (
        "per_tensor",
        "per_token",
        "per_channel",
    )
    for scheme in ("symmetric", "asymmetric"):
        for granularity in granularities:
            if scheme == "symmetric":
                scales, q = symm_int8_quant(x, granularity)
                x_hat = symm_int8_dequant(q, scales)
            else:
                q, scales, zp = asymm_int8_quant(x, granularity)
                x_hat = asymm_int8_dequant(q, scales, zp)

            metrics = error_metrics(x, x_hat)
            print(
                f"{scheme:<11} {granularity:<12} {scales.numel():>7d} "
                f"{metrics['mae']:>10.6f} {metrics['rmse']:>10.6f} "
                f"{metrics['max_error']:>10.6f} {metrics['sqnr_db']:>10.2f}"
            )


if __name__ == "__main__":
    compare_granularities(make_test_tensor())
