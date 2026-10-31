import math

import pytest
import torch
from test_reference import HKV, HQ, PAGE, build_cache

from kernels.decode import decode
from reference.paged import paged_decode_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def run(lengths, q_scale=1.0):
    num_pages = sum(math.ceil(n / PAGE) for n in lengths)
    cache, tables, lens, _ = build_cache(lengths, torch.float16, num_pages)
    q = (torch.randn(len(lengths), HQ, 128) * q_scale).half()
    got = decode(
        q.cuda(),
        cache.k.cuda(),
        cache.v.cuda(),
        None,
        None,
        tables.int().cuda(),
        lens.int().cuda(),
    )
    want = paged_decode_attention(q.float(), cache, tables, lens)
    return got, want


def test_one_token_returns_its_v_row():
    cache, tables, lens, history = build_cache([1], torch.float16)
    q = torch.randn(1, HQ, 128).half()
    got = decode(
        q.cuda(), cache.k.cuda(), cache.v.cuda(), None, None, tables.int().cuda(), lens.int().cuda()
    )
    _, v = history[0]
    assert torch.equal(got[0].cpu(), v[0].repeat_interleave(HQ // HKV, dim=0))


@pytest.mark.parametrize("length", [2, PAGE - 1, PAGE, PAGE + 1, 300, 2048])
def test_matches_reference(length):
    got, want = run([length])
    torch.testing.assert_close(got.float().cpu(), want, rtol=1e-3, atol=1e-3)


def test_batch_with_shuffled_pages():
    got, want = run([1, 15, 16, 17, 50])
    torch.testing.assert_close(got.float().cpu(), want, rtol=1e-3, atol=1e-3)


def test_large_scores_stay_finite():
    got, want = run([300], q_scale=30.0)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.float().cpu(), want, rtol=1e-3, atol=1e-3)


def test_rejects_int64_block_table():
    cache, tables, lens, _ = build_cache([5], torch.float16)
    q = torch.randn(1, HQ, 128).half().cuda()
    with pytest.raises(RuntimeError, match="block_table must be int32"):
        decode(q, cache.k.cuda(), cache.v.cuda(), None, None, tables.cuda(), lens.int().cuda())
