"""Paged KV cache matching the standalone contract in docs/scope.md.

K and V: [num_pages, page_size, Hkv, D], int8 or floating point.
K and V scales (int8 only): [num_pages, page_size, Hkv], fp32.
A slot is page_id * page_size + offset.
"""

from dataclasses import dataclass

import torch

from reference.attention import decode_attention, split_decode_attention
from reference.quant import dequantize, quantize


@dataclass
class PagedKVCache:
    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor | None
    v_scale: torch.Tensor | None

    @classmethod
    def allocate(
        cls,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> "PagedKVCache":
        shape = (num_pages, page_size, num_kv_heads, head_dim)
        if dtype != torch.int8:
            return cls(torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype), None, None)
        return cls(
            torch.zeros(shape, dtype=torch.int8),
            torch.zeros(shape, dtype=torch.int8),
            torch.ones(shape[:-1]),
            torch.ones(shape[:-1]),
        )

    @property
    def page_size(self) -> int:
        return self.k.shape[1]

    @property
    def quantized(self) -> bool:
        return self.k_scale is not None

    def write(self, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
        """Store new tokens. k, v: [N, Hkv, D] floating point. slots: [N] int64."""
        page, offset = slots // self.page_size, slots % self.page_size
        if not self.quantized:
            self.k[page, offset] = k.to(self.k.dtype)
            self.v[page, offset] = v.to(self.v.dtype)
            return
        self.k[page, offset], self.k_scale[page, offset] = quantize(k)
        self.v[page, offset], self.v_scale[page, offset] = quantize(v)

    def gather(self, block_table: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct one request's history as fp32 [seq_len, Hkv, D]."""
        pos = torch.arange(seq_len)
        page, offset = block_table[pos // self.page_size], pos % self.page_size
        k, v = self.k[page, offset], self.v[page, offset]
        if not self.quantized:
            return k.float(), v.float()
        return (
            dequantize(k, self.k_scale[page, offset]),
            dequantize(v, self.v_scale[page, offset]),
        )

    def nbytes(self) -> int:
        tensors = [self.k, self.v, self.k_scale, self.v_scale]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def slots_for(block_table: torch.Tensor, positions: torch.Tensor, page_size: int) -> torch.Tensor:
    return block_table[positions // page_size] * page_size + positions % page_size


def paged_decode_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    num_splits: int = 1,
) -> torch.Tensor:
    """q [B, Hq, D], block_tables [B, max_pages], seq_lens [B]. Returns fp32 [B, Hq, D]."""
    out = []
    for b in range(q.shape[0]):
        k, v = cache.gather(block_tables[b], int(seq_lens[b]))
        if num_splits == 1:
            out.append(decode_attention(q[b], k, v))
        else:
            out.append(split_decode_attention(q[b], k, v, num_splits))
    return torch.stack(out)
