import math

import pytest
import torch
from test_reference import HKV, HQ, PAGE, build_cache

from kernels.decode import decode
from reference.paged import paged_decode_attention
from reference.quant import dequantize, quantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")

DTYPES = [torch.float16, torch.int8]


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def call(q, cache, tables, lens):
    scales = (None, None)
    if cache.quantized:
        scales = (cache.k_scale.cuda(), cache.v_scale.cuda())
    return decode(
        q.cuda(), cache.k.cuda(), cache.v.cuda(), *scales, tables.int().cuda(), lens.int().cuda()
    )


def run(lengths, dtype, q_scale=1.0):
    num_pages = sum(math.ceil(n / PAGE) for n in lengths)
    cache, tables, lens, _ = build_cache(lengths, dtype, num_pages)
    q = (torch.randn(len(lengths), HQ, 128) * q_scale).half()
    return call(q, cache, tables, lens), paged_decode_attention(q.float(), cache, tables, lens)


@pytest.mark.parametrize("dtype", DTYPES)
def test_one_token_returns_its_v_row(dtype):
    cache, tables, lens, history = build_cache([1], dtype)
    q = torch.randn(1, HQ, 128).half()
    got = call(q, cache, tables, lens)
    v = history[0][1][0]
    if dtype == torch.int8:
        v = dequantize(*quantize(v)).half()
    assert torch.equal(got[0].cpu(), v.repeat_interleave(HQ // HKV, dim=0))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", [2, PAGE - 1, PAGE, PAGE + 1, 300, 2048])
def test_matches_reference(length, dtype):
    got, want = run([length], dtype)
    torch.testing.assert_close(got.float().cpu(), want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_with_shuffled_pages(dtype):
    got, want = run([1, 15, 16, 17, 50], dtype)
    torch.testing.assert_close(got.float().cpu(), want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_large_scores_stay_finite(dtype):
    got, want = run([300], dtype, q_scale=30.0)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.float().cpu(), want, rtol=1e-3, atol=1e-3)


def test_rejects_int64_block_table():
    cache, tables, lens, _ = build_cache([5], torch.float16)
    q = torch.randn(1, HQ, 128).half().cuda()
    with pytest.raises(RuntimeError, match="block_table must be int32"):
        decode(q, cache.k.cuda(), cache.v.cuda(), None, None, tables.cuda(), lens.int().cuda())


def test_rejects_int8_cache_without_scales():
    cache, tables, lens, _ = build_cache([5], torch.int8)
    q = torch.randn(1, HQ, 128).half().cuda()
    with pytest.raises(RuntimeError, match="k_scale and v_scale are required for int8"):
        decode(q, cache.k.cuda(), cache.v.cuda(), None, None, tables.int().cuda(), lens.int().cuda())
