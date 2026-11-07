import pytest

from kernels.decode import select_chunks


@pytest.mark.parametrize(
    "max_history, chunks",
    [(1, 1), (15, 1), (16, 1), (32, 2), (1000, 62), (2048, 128), (4096, 128), (16384, 128)],
)
def test_select_chunks(max_history, chunks):
    assert select_chunks(max_history) == chunks
