"""Python entry to the warp sum and max warm-up kernel in warp_reduce.cu."""

from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


@cache
def _ext():
    return load_inline(
        name="warp_reduce",
        cpp_sources="std::tuple<torch::Tensor, torch::Tensor> warp_sum_max(torch::Tensor x);",
        cuda_sources=(Path(__file__).parent / "warp_reduce.cu").read_text(),
        functions=["warp_sum_max"],
    )


def warp_sum_max(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row sums and row maxes of x [N, 128], fp32 on the GPU. Compiles on first call."""
    return _ext().warp_sum_max(x)
