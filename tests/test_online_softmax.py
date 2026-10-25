import math

import pytest
import torch

from reference.attention import decode_attention, merge_states
from reference.online_softmax import token_range_state

D = 128


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def online_decode(q, k, v, num_chunks):
    """Split [0, L) the way split_decode_attention does, then merge."""
    length = k.shape[0]
    chunk = math.ceil(length / num_chunks)
    states = [
        token_range_state(q, k, v, min(i * chunk, length), min((i + 1) * chunk, length))
        for i in range(num_chunks)
    ]
    m = torch.stack([s[0] for s in states]).unsqueeze(1)
    l = torch.stack([s[1] for s in states]).unsqueeze(1)
    u = torch.stack([s[2] for s in states]).unsqueeze(1)
    return merge_states(m, l, u)[0]


def direct(q, k, v):
    return decode_attention(q[None], k[:, None], v[:, None])[0]


def test_hand_worked_example():
    q = torch.tensor([1.0])
    k = torch.tensor([[2.0], [1.0], [3.0]])
    v = torch.tensor([[10.0], [20.0], [30.0]])
    m, l, u = token_range_state(q, k, v, 0, 3)
    assert m.item() == pytest.approx(3.0)
    assert l.item() == pytest.approx(1.5032147244080551)
    assert u.tolist() == pytest.approx([36.385500076446675])
    assert (u / l).tolist() == pytest.approx([24.20512484720024])


def test_empty_range_contributes_nothing():
    q, k, v = torch.randn(D), torch.randn(5, D), torch.randn(5, D)
    m, l, u = token_range_state(q, k, v, 3, 3)
    assert m.item() == -math.inf
    assert l.item() == 0.0
    assert torch.equal(u, torch.zeros(D))


@pytest.mark.parametrize("length", [1, 7, 300])
@pytest.mark.parametrize("num_chunks", [1, 3, 16])
def test_matches_reference(length, num_chunks):
    q, k, v = torch.randn(D), torch.randn(length, D), torch.randn(length, D)
    out = online_decode(q, k, v, num_chunks)
    torch.testing.assert_close(out, direct(q, k, v), rtol=1e-5, atol=1e-5)


def test_large_scores_stay_finite():
    q, k, v = 40 * torch.randn(D), torch.randn(300, D), torch.randn(300, D)
    out = online_decode(q, k, v, 1)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, direct(q, k, v), rtol=1e-5, atol=1e-5)


def test_token_order_does_not_change_the_output():
    q, k, v = torch.randn(D), torch.randn(50, D), torch.randn(50, D)
    perm = torch.randperm(50)
    torch.testing.assert_close(
        online_decode(q, k[perm], v[perm], 1), online_decode(q, k, v, 1), rtol=1e-5, atol=1e-5
    )
