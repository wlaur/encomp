from typing import assert_type

import polars as pl

from ..units import Quantity as Q
from ..utypes import Mass, Numpy1DArray


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_abs() -> None:
    assert_type(abs(Q(2, "kg")), Q[Mass, float])
    assert_type(abs(Q([1, 2], "kg")), Q[Mass, Numpy1DArray])
    assert_type(abs(Q(pl.Series([1, 2]), "kg")), Q[Mass, pl.Series])
    assert_type(abs(Q(pl.lit(-5), "kg")), Q[Mass, pl.Expr])


def test_min_max() -> None:
    assert_type(min(Q([25], "kg")), Q[Mass, float])
    assert_type(max(Q([25], "kg")), Q[Mass, float])

    assert_type(min(Q(pl.Series([25]), "kg")), Q[Mass, float])
    assert_type(max(Q(pl.Series([25]), "kg")), Q[Mass, float])


def test_iter() -> None:
    for q in Q([1, 2, 3], "kg"):
        assert_type(q, Q[Mass, float])

    for q in Q(pl.Series([1, 2, 3]), "kg"):
        assert_type(q, Q[Mass, float])
