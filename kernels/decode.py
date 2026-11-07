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
            "std::vector<torch::Tensor> decode(torch::Tensor q, torch::Tensor k_cache,"
            " torch::Tensor v_cache, std::optional<torch::Tensor> k_scale,"
            " std::optional<torch::Tensor> v_scale,"
            " torch::Tensor block_table, torch::Tensor seq_lens, int num_chunks);"
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
    result = _ext().decode(
        q, k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, num_chunks
    )
    if len(result) == 1:
        return result[0]
    return _merge(*result)


def _merge(m: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Merge the partial states over the chunk dim: m, l [B, HQ, C], u [B, HQ, C, D].

    Unlike reference.attention.merge_states, this never reads a value back to
    the CPU, so the CPU can queue the next decode step while the GPU works.
    Chunk 0 is never empty because every history has at least 1 token, so the
    max over chunks is finite.
    """
    w = torch.exp(m - m.amax(dim=2, keepdim=True))
    l_all = torch.einsum("bhc,bhc->bh", w, l)
    u_all = torch.einsum("bhc,bhcd->bhd", w, u)
    return (u_all / l_all.unsqueeze(-1)).half()
