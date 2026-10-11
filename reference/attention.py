"""Single-query decode attention: direct form and split-history form with merge.

Shapes for one request: q [Hq, D], k and v [L, Hkv, D]. Query head h reads
KV head h // (Hq // Hkv). All math runs in fp32.
"""

import math

import torch


def _expand_kv(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    hq, hkv = q.shape[0], kv.shape[1]
    if hq % hkv:
        raise ValueError(f"query heads {hq} not divisible by KV heads {hkv}")
    return kv.float().repeat_interleave(hq // hkv, dim=1)


def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """softmax(q k^T / sqrt(D)) v for one request. Returns [Hq, D]."""
    if k.shape[0] == 0:
        raise ValueError("decode attention needs at least one cached token")
    k, v = _expand_kv(q, k), _expand_kv(q, v)
    scores = torch.einsum("hd,lhd->hl", q.float(), k) / math.sqrt(q.shape[-1])
    return torch.einsum("hl,lhd->hd", torch.softmax(scores, dim=-1), v)


def partial_state(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partial attention state (m, l, u) over one chunk of history.

    m [Hq]: max score. l [Hq]: sum exp(score - m). u [Hq, D]: sum exp(score - m) * v.
    An empty chunk returns m = -inf, l = 0, u = 0, which contributes no mass.
    """
    hq, d = q.shape
    if k.shape[0] == 0:
        return (
            torch.full((hq,), -math.inf),
            torch.zeros(hq),
            torch.zeros(hq, d),
        )
    k, v = _expand_kv(q, k), _expand_kv(q, v)
    scores = torch.einsum("hd,lhd->hl", q.float(), k) / math.sqrt(d)
    m = scores.amax(dim=-1)
    p = torch.exp(scores - m.unsqueeze(-1))
    return m, p.sum(dim=-1), torch.einsum("hl,lhd->hd", p, v)


def merge_states(m: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Merge S partial states stacked on dim 0: m [S, Hq], l [S, Hq], u [S, Hq, D]."""
    m_all = m.amax(dim=0)
    if torch.isinf(m_all).any():
        raise ValueError("merge needs at least one non-empty chunk per head")
    w = torch.exp(m - m_all)
    l_all = (w * l).sum(dim=0)
    u_all = (w.unsqueeze(-1) * u).sum(dim=0)
    return u_all / l_all.unsqueeze(-1)


def split_decode_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_splits: int
) -> torch.Tensor:
    """Split history into num_splits contiguous chunks, then merge. Returns [Hq, D]."""
    chunk = math.ceil(k.shape[0] / num_splits)
    states = [
        partial_state(q, k[i * chunk : (i + 1) * chunk], v[i * chunk : (i + 1) * chunk])
        for i in range(num_splits)
    ]
    m, l, u = (torch.stack(s) for s in zip(*states))
    return merge_states(m, l, u)
