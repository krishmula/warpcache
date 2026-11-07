"""Python entry to the cache_write kernel in cache_write.cu."""

from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


@cache
def _ext():
    return load_inline(
        name="cache_write",
        cpp_sources=(
            "void cache_write(torch::Tensor k, torch::Tensor v, torch::Tensor slots,"
            " torch::Tensor k_cache, torch::Tensor v_cache,"
            " std::optional<torch::Tensor> k_scale,"
            " std::optional<torch::Tensor> v_scale);"
        ),
        cuda_sources=(Path(__file__).parent / "cache_write.cu").read_text(),
        functions=["cache_write"],
    )


def cache_write(
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    """Store new tokens in place, matching PagedKVCache.write.

    k, v: fp16 [N, 8, 128]. slots: int64 [N] slot numbers, unique within one call.
    Every value must be finite. The kernel does not check this, because the
    check would make the CPU wait for the GPU on every decode step.
    """
    _ext().cache_write(k, v, slots, k_cache, v_cache, k_scale, v_scale)
