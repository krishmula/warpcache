import pytest
import torch

from kernels.warp_reduce import warp_sum_max

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def check(x):
    s, m = warp_sum_max(x)
    torch.testing.assert_close(s, x.sum(dim=1), rtol=1e-5, atol=1e-4)
    assert torch.equal(m, x.amax(dim=1))


@pytest.mark.parametrize("rows", [1, 5, 1000])
def test_random_rows(rows):
    check(torch.randn(rows, 128, device="cuda"))


def test_every_column_counts():
    check(torch.eye(128, device="cuda"))


def test_all_negative_rows():
    check(-1 - torch.rand(3, 128, device="cuda"))


def test_max_in_first_and_last_column():
    x = torch.randn(2, 128, device="cuda")
    x[0, 0] = 100
    x[1, 127] = 100
    check(x)
