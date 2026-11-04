import math

import pytest
import torch
from test_reference import HKV, HQ, PAGE, build_cache

from kernels.decode import _ext, decode
from reference.attention import partial_state
from reference.paged import paged_decode_attention
from reference.quant import dequantize, quantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")

DTYPES = [torch.float16, torch.int8]


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def gpu_args(q, cache, tables, lens):
    scales = (None, None)
    if cache.quantized:
        scales = (cache.k_scale.cuda(), cache.v_scale.cuda())
    return q.cuda(), cache.k.cuda(), cache.v.cuda(), *scales, tables.int().cuda(), lens.int().cuda()


def call(q, cache, tables, lens, num_chunks=1):
    return decode(*gpu_args(q, cache, tables, lens), num_chunks=num_chunks)


def run(lengths, dtype, q_scale=1.0, num_chunks=1):
    num_pages = sum(math.ceil(n / PAGE) for n in lengths)
    cache, tables, lens, _ = build_cache(lengths, dtype, num_pages)
    q = (torch.randn(len(lengths), HQ, 128) * q_scale).half()
    got = call(q, cache, tables, lens, num_chunks)
    return got, paged_decode_attention(q.float(), cache, tables, lens)


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
@pytest.mark.parametrize("num_chunks", [1, 4])
def test_large_scores_stay_finite(num_chunks, dtype):
    got, want = run([300], dtype, q_scale=30.0, num_chunks=num_chunks)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.float().cpu(), want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "lengths, num_chunks", [([300], 4), ([2048], 4), ([1, 15, 16, 17, 50], 4), ([5], 8)]
)
def test_each_chunk_writes_its_partial_state(lengths, num_chunks, dtype):
    num_pages = sum(math.ceil(n / PAGE) for n in lengths)
    cache, tables, lens, _ = build_cache(lengths, dtype, num_pages)
    q = torch.randn(len(lengths), HQ, 128).half()
    states = [t.cpu() for t in _ext().decode(*gpu_args(q, cache, tables, lens), num_chunks)]
    for b, n in enumerate(lengths):
        k, v = cache.gather(tables[b], n)
        size = math.ceil(n / num_chunks)
        for c in range(num_chunks):
            chunk = slice(c * size, (c + 1) * size)
            want = partial_state(q[b].float(), k[chunk], v[chunk])
            got = tuple(s[b, :, c] for s in states)
            torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_chunks", [2, 4, 8])
@pytest.mark.parametrize("lengths", [[300], [2048], [1, 15, 16, 17, 50]])
def test_merged_chunks_match_reference(lengths, num_chunks, dtype):
    got, want = run(lengths, dtype, num_chunks=num_chunks)
    assert got.dtype == torch.float16 and got.shape == (len(lengths), HQ, 128)
    torch.testing.assert_close(got.float().cpu(), want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_more_chunks_than_tokens(dtype):
    got, want = run([5], dtype, num_chunks=8)
    torch.testing.assert_close(got.float().cpu(), want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_chunks", [1, 4])
def test_empty_batch_returns_empty_output(num_chunks):
    cache, tables, lens, _ = build_cache([5], torch.float16)
    q = torch.randn(0, HQ, 128).half()
    got = call(q, cache, tables[:0], lens[:0], num_chunks)
    assert got.shape == (0, HQ, 128)


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
