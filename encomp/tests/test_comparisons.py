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
