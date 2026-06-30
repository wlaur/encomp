from typing import assert_type

import polars as pl
from pytest import raises

from ..units import Quantity as Q
from ..utypes import Mass, Numpy1DArray


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_abs() -> None:
    assert_type(abs(Q(2, "kg")), Q[Mass, float])  # pyrefly: ignore[assert-type]
    assert_type(abs(Q([1, 2], "kg")), Q[Mass, Numpy1DArray])  # pyrefly: ignore[assert-type]
    assert_type(abs(Q(pl.Series([1, 2]), "kg")), Q[Mass, pl.Series])  # pyrefly: ignore[assert-type]
    assert_type(abs(Q(pl.lit(-5), "kg")), Q[Mass, pl.Expr])  # pyrefly: ignore[assert-type]


def test_min_max() -> None:
    assert_type(min(Q([25], "kg")), Q[Mass, float])
    assert_type(max(Q([25], "kg")), Q[Mass, float])

    # min/max iterate, and a pl.Series magnitude (the polars world) is not
    # iterable as a Quantity -- the reduction belongs on the magnitude (.m)
    for fn in (min, max):
        with raises(TypeError, match="is not iterable"):
            fn(Q(pl.Series([25]), "kg"))


def test_iter() -> None:
    for q in Q([1, 2, 3], "kg"):
        assert_type(q, Q[Mass, float])

    # a pl.Series magnitude routes data access through .m like pl.Expr, so the
    # Quantity itself is not iterable; iterate via .m (q.to(<unit>).m)
    with raises(TypeError, match="is not iterable"):
        next(iter(Q(pl.Series([1, 2, 3]), "kg")))
