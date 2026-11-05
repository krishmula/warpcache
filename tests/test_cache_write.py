import pytest
import torch
from test_reference import HKV, PAGE

from kernels.cache_write import cache_write
from reference.paged import PagedKVCache, slots_for

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")

D = 128
DTYPES = [torch.float16, torch.int8]
NUM_PAGES = 64


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def filled_cache(dtype):
    """A cache full of nonzero values, so a write that does nothing fails."""
    cache = PagedKVCache.allocate(NUM_PAGES, PAGE, HKV, D, dtype)
    if cache.quantized:
        cache.k.random_(-127, 128)
        cache.v.random_(-127, 128)
        cache.k_scale.uniform_(0.5, 2.0)
        cache.v_scale.uniform_(0.5, 2.0)
    else:
        cache.k.normal_()
        cache.v.normal_()
    return cache


def to_gpu(cache):
    scales = (None, None)
    if cache.quantized:
        scales = (cache.k_scale.cuda(), cache.v_scale.cuda())
    return PagedKVCache(cache.k.cuda(), cache.v.cuda(), *scales)


def clone(cache):
    scales = (None, None)
    if cache.quantized:
        scales = (cache.k_scale.clone(), cache.v_scale.clone())
    return PagedKVCache(cache.k.clone(), cache.v.clone(), *scales)


def gpu_write(cache, k, v, slots):
    cache_write(
        k.cuda(), v.cuda(), slots.cuda(), cache.k, cache.v, cache.k_scale, cache.v_scale
    )


def write_both(k, v, slots, dtype):
    """Write the same tokens with the kernel and the reference. Returns (gpu, ref)."""
    start = filled_cache(dtype)
    gpu, ref = to_gpu(start), clone(start)
    gpu_write(gpu, k, v, slots)
    ref.write(k, v, slots)
    return gpu, ref


def assert_same_cache(gpu, ref):
    assert torch.equal(gpu.k.cpu(), ref.k), "K cache differs"
    assert torch.equal(gpu.v.cpu(), ref.v), "V cache differs"
    if ref.quantized:
        assert torch.equal(gpu.k_scale.cpu(), ref.k_scale), "K scale differs"
        assert torch.equal(gpu.v_scale.cpu(), ref.v_scale), "V scale differs"


def shuffled_slots(lengths):
    """Slot numbers for each request's history, on shuffled pages."""
    perm = torch.randperm(NUM_PAGES)
    out, used = [], 0
    for n in lengths:
        pages = -(-n // PAGE)
        table = perm[used : used + pages]
        used += pages
        out.append(slots_for(table, torch.arange(n), PAGE))
    return torch.cat(out)


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_rows(dtype):
    k = v = torch.zeros(3, HKV, D, dtype=torch.half)
    slots = torch.tensor([0, 17, 40])
    gpu, ref = write_both(k, v, slots, dtype)
    assert_same_cache(gpu, ref)
    if dtype == torch.int8:
        page, page_slot = slots // PAGE, slots % PAGE
        assert (gpu.k.cpu()[page, page_slot] == 0).all()
        assert (gpu.k_scale.cpu()[page, page_slot] == 1.0).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_fp16_max(dtype):
    k = torch.randn(2, HKV, D).half()
    k[:, :, 0], k[:, :, 1] = 65504.0, -65504.0
    v = -k
    slots = torch.tensor([5, 33])
    gpu, ref = write_both(k, v, slots, dtype)
    assert_same_cache(gpu, ref)
    if dtype == torch.int8:
        rows = gpu.k.cpu()[slots // PAGE, slots % PAGE]
        assert (rows[..., 0] == 127).all() and (rows[..., 1] == -127).all()


def test_rounds_half_to_even():
    k = torch.zeros(1, HKV, D, dtype=torch.half)
    k[0, :, 0] = 127.0
    k[0, :, 1:5] = torch.tensor([0.5, 1.5, 2.5, -2.5], dtype=torch.half)
    v = k.clone()
    slots = torch.tensor([3])
    gpu, ref = write_both(k, v, slots, torch.int8)
    assert_same_cache(gpu, ref)
    assert (gpu.k_scale.cpu()[0, 3] == 1.0).all()
    assert gpu.k.cpu()[0, 3, :, 1:5].tolist() == [[0, 2, 2, -2]] * HKV


@pytest.mark.parametrize("dtype", DTYPES)
def test_shuffled_pages_and_partial_tail(dtype):
    slots = shuffled_slots([1, 15, 16, 17, 50])
    k = (torch.randn(len(slots), HKV, D) * 5).half()
    v = (torch.randn(len(slots), HKV, D) * 5).half()
    assert_same_cache(*write_both(k, v, slots, dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_rows_of_very_different_sizes(dtype):
    n = 256
    slots = torch.randperm(NUM_PAGES * PAGE)[:n]
    size = 10.0 ** torch.empty(n, HKV, 1).uniform_(-3, 3)
    k = (torch.randn(n, HKV, D) * size).half()
    v = (torch.randn(n, HKV, D) * size).half()
    assert_same_cache(*write_both(k, v, slots, dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_one_token_at_a_time_across_page_boundary(dtype):
    n = PAGE + 3
    table = torch.tensor([5, 2])
    k = torch.randn(n, HKV, D).half()
    v = torch.randn(n, HKV, D).half()
    start = filled_cache(dtype)
    gpu, ref = to_gpu(start), clone(start)
    for t in range(n):
        gpu_write(gpu, k[t : t + 1], v[t : t + 1], slots_for(table, torch.tensor([t]), PAGE))
    ref.write(k, v, slots_for(table, torch.arange(n), PAGE))
    assert_same_cache(gpu, ref)


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_tokens_changes_nothing(dtype):
    k = v = torch.zeros(0, HKV, D, dtype=torch.half)
    gpu, ref = write_both(k, v, torch.zeros(0, dtype=torch.long), dtype)
    assert_same_cache(gpu, ref)


def test_rejects_fp32_input():
    gpu = to_gpu(filled_cache(torch.int8))
    k = torch.randn(1, HKV, D)
    with pytest.raises(RuntimeError, match="fp16"):
        gpu_write(gpu, k, k, torch.tensor([0]))
