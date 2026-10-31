"""Python entry to the decode attention kernel in decode.cu."""

from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


@cache
def _ext():
    return load_inline(
        name="decode",
        cpp_sources=(
            "torch::Tensor decode(torch::Tensor q, torch::Tensor k_cache,"
            " torch::Tensor v_cache, std::optional<torch::Tensor> k_scale,"
            " std::optional<torch::Tensor> v_scale,"
            " torch::Tensor block_table, torch::Tensor seq_lens);"
        ),
        cuda_sources=(Path(__file__).parent / "decode.cu").read_text(),
        functions=["decode"],
    )


def decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_chunks: int = 1,
) -> torch.Tensor:
    """Decode attention with the contract in docs/scope.md. Returns fp16 [B, 16, 128]."""
    if num_chunks != 1:
        raise NotImplementedError("Path B (num_chunks > 1) is day 3")
    return _ext().decode(q, k_cache, v_cache, k_scale, v_scale, block_table, seq_lens)
