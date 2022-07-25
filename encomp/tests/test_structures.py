import pytest

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

    deep_nested = [[[[[[[[1]]]]]]]]
    assert next(flatten(deep_nested)) == 1

    x = list(flatten(deep_nested, max_depth=5))
    assert isinstance(x, list)
    assert len(x) == 1

    deep_nested = [[[[[[[[1]]]]]]]]
    assert list(flatten(deep_nested)) == [1]

    deep_nested = [[[[[[[[1]]]]]]]]
    list(flatten(deep_nested, max_depth=5)) == [[[[1]]]]

    deep_nested = [[[[[[[[1]]], [2]]]]]]
    list(flatten(deep_nested, max_depth=2)) == [[[[[[[1]]], [2]]]]]

    recursive = [None]
    recursive[0] = recursive

    with pytest.raises(RecursionError):

        next(flatten(recursive))

    y = next(flatten(recursive, max_depth=100))

    assert isinstance(y, list)
    assert y == recursive
    assert y == recursive[0]
    assert y == recursive[0][0]


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


def test_divide_chunks_errors():

    with pytest.raises(TypeError):
        divide_chunks([])

    with pytest.raises(ValueError):
        next(divide_chunks([1, 2, 3], 0))

    with pytest.raises(ValueError):
        next(divide_chunks([1, 2, 3], -5))

    with pytest.raises(ValueError):
        next(divide_chunks([], 2))
