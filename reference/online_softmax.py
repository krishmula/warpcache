"""Online softmax over one token range, in the shape of one CUDA thread block.

One call covers one query head and tokens [start, end) of its history, reading
each token once. Path A is one call over the whole history. Path B is one call
per chunk, then merge_states in reference/attention.py.
"""

import math

import torch


def token_range_state(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, end: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partial state (m, l, u) over tokens [start, end), one token at a time.

    q [D], k [L, D], v [L, D], all fp32. Score of token i is q . k[i] / sqrt(D).
    Returns m and l as 0-d tensors and u as [D], with the meaning documented in
    reference.attention.partial_state. An empty range returns m = -inf, l = 0,
    u = 0.
    """

    d = q.shape[0]

    m = torch.tensor(-math.inf)
    l = torch.tensor(0.0)
    u = torch.zeros(d)

    for i in range(start, end):
        z = torch.dot(q, k[i]) / math.sqrt(d)
        new_m = torch.maximum(m, z)
        shrink = torch.exp(m - new_m)
        w = torch.exp(z - new_m)
        l = l * shrink + w
        u = u * shrink + w * v[i]
        m = new_m

    return m, l, u
