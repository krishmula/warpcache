import math

import pytest
import torch
import torch.nn.functional as F

from reference.attention import (
    decode_attention,
    merge_states,
    split_decode_attention,
)
from reference.paged import PagedKVCache, paged_decode_attention, slots_for
from reference.quant import dequantize, quantize

HQ, HKV, D, PAGE = 12, 2, 128, 16


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def build_cache(lengths, dtype, num_pages=64):
    """Write random fp16 K/V for each request into shuffled physical pages."""
    max_pages = math.ceil(max(lengths) / PAGE)
    perm = torch.randperm(num_pages)
    block_tables = torch.zeros(len(lengths), max_pages, dtype=torch.long)
    cache = PagedKVCache.allocate(num_pages, PAGE, HKV, D, dtype)
    history, used = [], 0
    for b, n in enumerate(lengths):
        pages = math.ceil(n / PAGE)
        block_tables[b, :pages] = perm[used : used + pages]
        used += pages
        k = torch.randn(n, HKV, D).half()
        v = torch.randn(n, HKV, D).half()
        cache.write(k, v, slots_for(block_tables[b], torch.arange(n), PAGE))
        history.append((k, v))
    return cache, block_tables, torch.tensor(lengths), history


# Reference attention against a trusted implementation.


@pytest.mark.parametrize("length", [1, 7, 300])
def test_decode_matches_sdpa(length):
    q = torch.randn(HQ, D)
    k, v = torch.randn(length, HKV, D), torch.randn(length, HKV, D)
    expected = F.scaled_dot_product_attention(
        q[None, :, None, :],
        k.permute(1, 0, 2)[None],
        v.permute(1, 0, 2)[None],
        enable_gqa=True,
    )[0, :, 0, :]
    torch.testing.assert_close(decode_attention(q, k, v), expected)


# Quantization.


def test_quant_error_within_half_step():
    x = torch.randn(100, HKV, D) * 5
    q, s = quantize(x)
    assert q.dtype == torch.int8 and s.shape == (100, HKV)
    assert q.min() >= -127
    err = (dequantize(q, s) - x).abs()
    assert (err <= s.unsqueeze(-1) / 2 * (1 + 1e-5)).all()


def test_quant_zero_vector():
    q, s = quantize(torch.zeros(3, D))
    assert (q == 0).all() and (s == 1).all()
    assert (dequantize(q, s) == 0).all()


@pytest.mark.parametrize("big", [65504.0, 1e30])
def test_quant_extreme_finite(big):
    x = torch.zeros(D)
    x[0], x[1] = big, -big
    q, s = quantize(x)
    assert q[0] == 127 and q[1] == -127
    assert torch.isfinite(dequantize(q, s)).all()


@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_quant_rejects_nonfinite(bad):
    x = torch.zeros(D)
    x[3] = bad
    with pytest.raises(ValueError):
        quantize(x)


# Paged storage.

LENGTHS = [1, 15, 16, 17, 50]


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_gather_shuffled_pages_and_partial_tail(dtype):
    cache, tables, lens, history = build_cache(LENGTHS, dtype)
    for b, (k, v) in enumerate(history):
        got_k, got_v = cache.gather(tables[b], int(lens[b]))
        if dtype == torch.int8:
            k, v = dequantize(*quantize(k)), dequantize(*quantize(v))
        assert torch.equal(got_k, k.float()) and torch.equal(got_v, v.float())


def test_append_across_page_boundary_matches_bulk_write():
    n = PAGE + 3
    table = torch.tensor([5, 2])
    k, v = torch.randn(n, HKV, D), torch.randn(n, HKV, D)
    bulk = PagedKVCache.allocate(8, PAGE, HKV, D, torch.int8)
    bulk.write(k, v, slots_for(table, torch.arange(n), PAGE))
    step = PagedKVCache.allocate(8, PAGE, HKV, D, torch.int8)
    for t in range(n):
        before = step.k_scale.clone()
        step.write(k[t : t + 1], v[t : t + 1], slots_for(table, torch.tensor([t]), PAGE))
        changed = (step.k_scale != before).nonzero()
        assert (changed[:, :2] == torch.tensor([table[t // PAGE], t % PAGE])).all()
    for a, b in zip(step.gather(table, n), bulk.gather(table, n)):
        assert torch.equal(a, b)


def test_int8_storage_bytes():
    # Per token and KV head: 2*128 int8 bytes + 2 fp32 scales vs 2*128 fp16 values.
    fp16 = PagedKVCache.allocate(4, PAGE, HKV, D, torch.float16).nbytes()
    int8 = PagedKVCache.allocate(4, PAGE, HKV, D, torch.int8).nbytes()
    assert round(1 - int8 / fp16, 4) == 0.4844


# Correctness hierarchy for the paged decode path.


def test_int8_paged_matches_reference_over_reconstructed_cache():
    cache, tables, lens, history = build_cache(LENGTHS, torch.int8)
    q = torch.randn(len(LENGTHS), HQ, D)
    got = paged_decode_attention(q, cache, tables, lens)
    for b, (k, v) in enumerate(history):
        k, v = dequantize(*quantize(k)), dequantize(*quantize(v))
        torch.testing.assert_close(got[b], decode_attention(q[b], k, v))


def test_int8_quantization_error_vs_full_precision():
    cache, tables, lens, history = build_cache(LENGTHS, torch.int8)
    q = torch.randn(len(LENGTHS), HQ, D)
    got = paged_decode_attention(q, cache, tables, lens)
    want = torch.stack([decode_attention(q[b], k, v) for b, (k, v) in enumerate(history)])
    err = (got - want).abs().max().item()
    # Synthetic N(0, 1) data only. Real tolerances come from model-derived captures.
    assert 0 < err < 0.05, err


# Split history and merge.


def test_merge_weights_by_softmax_mass():
    # Chunk outputs 2 and 10 with masses 9 and 1 merge to 2.8, not their mean 6.
    m = torch.zeros(2, 1)
    l = torch.tensor([[9.0], [1.0]])
    u = torch.tensor([[[18.0]], [[10.0]]])
    assert merge_states(m, l, u).item() == pytest.approx(2.8)


@pytest.mark.parametrize("length", [1, 17, 300])
@pytest.mark.parametrize("splits", [2, 4, 32])
def test_split_matches_direct_including_empty_chunks(length, splits):
    q = torch.randn(HQ, D)
    k, v = torch.randn(length, HKV, D), torch.randn(length, HKV, D)
    got = split_decode_attention(q, k, v, splits)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, decode_attention(q, k, v))


def test_split_paged_matches_direct_paged():
    cache, tables, lens, _ = build_cache(LENGTHS, torch.int8)
    q = torch.randn(len(LENGTHS), HQ, D)
    torch.testing.assert_close(
        paged_decode_attention(q, cache, tables, lens, num_splits=4),
        paged_decode_attention(q, cache, tables, lens),
    )
