import torch


def symm_int8_quant(x):
    qmin, qmax = -127, 127
    xmax = x.abs().max(dim=-1)

    scale = (xmax / qmax).clamp(1e-5)
    val = torch.clamp(torch.round(x / scale), qmin, qmax).to(torch.int8)

    return scale, val


def symm_int8_dequant(val, scale):
    return val.float() * scale


def asymm_int8_quant(x):
    qmin, qmax = -128, 127
    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin) / (qmax - qmin)
    zp = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
    x = torch.clamp(torch.round(zp + x / scale), qmin, qmax).to(torch.int8)
    return x, scale, zp


def asymm_int8_dequant(x, scale, zp):
    return (x.float() - zp) * scale
