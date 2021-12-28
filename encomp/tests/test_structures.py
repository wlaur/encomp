from hypothesis import given
from hypothesis.strategies import lists, integers

from encomp.structures import flatten, divide_chunks


def test_flatten():

    nested = [(1, 2, 3), 3, [6, 7]]

    flat = list(flatten(nested))

    assert len(flat) == 6


    nested = ([1, 2, 3], 3, [6, 7])

    flat = list(flatten(nested))

    assert len(flat) == 6


@given(
    lst=lists(integers(), min_size=1, max_size=100),
)
def test_divide_chunks(lst):

    m = len(lst)

    for N in range(1, m + 1):
        chunked = list(divide_chunks(lst, N))

        k = len(chunked)

        if m % N == 0:
            assert k == m // N
        else:
            assert k == m // N + 1
