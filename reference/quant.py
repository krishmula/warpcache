"""Symmetric INT8 quantization with one scale per (token, KV head) vector."""

import torch

QMAX = 127


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize along the last dimension.

    x: [..., D] floating point. Returns (q int8 [..., D], scale fp32 [...]).

    Rounding is round-half-to-even (torch.round), matching CUDA __float2int_rn.
    Vectors whose scale would be zero get scale 1.0, so they quantize to zeros
    and dequantize to exact zeros. Non-finite inputs are rejected.
    """
    if not torch.isfinite(x).all():
        raise ValueError("quantize: non-finite input is unsupported")
    x = x.float()
    scale = x.abs().amax(dim=-1) / QMAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(x / scale.unsqueeze(-1)).clamp(-QMAX, QMAX).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.unsqueeze(-1)
