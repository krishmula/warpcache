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
            "torch::Tensor decode_fp16(torch::Tensor q, torch::Tensor k_cache,"
            " torch::Tensor v_cache, torch::Tensor block_table, torch::Tensor seq_lens);"
        ),
        cuda_sources=(Path(__file__).parent / "decode.cu").read_text(),
        functions=["decode_fp16"],
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
    if k_cache.dtype == torch.int8:
        raise NotImplementedError("INT8 cache comes after the FP16 kernel")
    if num_chunks != 1:
        raise NotImplementedError("Path B (num_chunks > 1) is day 3")
    return _ext().decode_fp16(q, k_cache, v_cache, block_table, seq_lens)
