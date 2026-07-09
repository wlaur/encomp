from typing import assert_type

import polars as pl

from ..units import Quantity as Q
from ..utypes import Numpy1DBoolArray


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_comparisons() -> None:
    assert_type(Q(1, "kg") > Q(25, "g"), bool)
    assert_type(Q(1, "kg") >= Q(25, "g"), bool)
    assert_type(Q(1, "kg") <= Q(25, "g"), bool)
    assert_type(Q(1, "kg") < Q(25, "g"), bool)
    assert_type(Q(1, "kg") == Q(25, "g"), bool)
    assert_type(Q(1, "kg") != Q(25, "g"), bool)

    assert_type(Q([1], "kg") > Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") >= Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") <= Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") < Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") == Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") != Q(25, "g"), Numpy1DBoolArray)

    assert_type(Q(1, "kg") > Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") >= Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") <= Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") < Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") == Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") != Q([25], "g"), Numpy1DBoolArray)

    assert_type(Q(1, "kg") > Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") >= Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") <= Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") < Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") == Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") != Q(pl.Series([25]), "g"), pl.Series)

    assert_type(Q(pl.Series([1]), "kg") > Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") >= Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") <= Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") < Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") == Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") != Q(25, "g"), pl.Series)

    assert_type(Q(pl.Series([1]), "kg") > Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") >= Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") <= Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") < Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") == Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") != Q(pl.Series([25]), "g"), pl.Series)

    assert (Q(pl.Series([0.1 + 0.2]), "m") == Q(pl.Series([0.3]), "m")).to_list() == [True]
    assert (Q(pl.Series([0.1 + 0.2]), "m") != Q(pl.Series([0.3]), "m")).to_list() == [False]
    assert (Q(pl.Series([0.1 + 0.2]), "m") == Q(0.3, "m")).to_list() == [True]
    assert (Q(0.3, "m") == Q(pl.Series([0.1 + 0.2]), "m")).to_list() == [True]

    assert_type(Q(1, "kg") > Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") >= Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") <= Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") < Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") == Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") != Q(pl.col.asd, "g"), pl.Expr)

    assert_type(Q(pl.col.asd, "kg") > Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") >= Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") <= Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") < Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") == Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") != Q(25, "g"), pl.Expr)

    assert_type(Q(pl.col.asd, "kg") > Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") >= Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") <= Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") < Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") == Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") != Q(pl.col.asd, "g"), pl.Expr)

    # operands that are equal within (rtol, atol) must satisfy every relation consistently:
    # equal, not strictly ordered either way, and non-strictly ordered both ways
    near_high = Q(1.0 + 5e-10, "m")
    near_low = Q(1.0, "m")
    assert near_high == near_low
    assert not near_high > near_low
    assert not near_high < near_low
    assert near_high <= near_low
    assert near_high >= near_low
    assert near_low <= near_high
    assert near_low >= near_high

    # a difference larger than the tolerance still orders strictly
    high = Q(2.0, "m")
    assert high > near_low
    assert not high <= near_low
    assert near_low < high
    assert not near_low >= high
