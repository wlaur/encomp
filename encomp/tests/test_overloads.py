from typing import assert_type, reveal_type

import numpy as np
import polars as pl

from encomp.units import Quantity as Q
from encomp.utypes import Numpy1DBoolArray


def test_reveal_type() -> None:
    reveal_type(Q(1.0, "m"))
    reveal_type(Q(1, "m"))

    reveal_type(Q(1, str("m")))  # noqa: UP018

    reveal_type(Q([1, 2, 3], "kg"))

    reveal_type(Q(pl.col.asd, "kg"))
    reveal_type(Q(pl.DataFrame({"test": []})["test"], "kg"))

    reveal_type(Q(pl.col.asd, "kg") / Q(25, "min"))
    reveal_type(Q(pl.col.asd, "kg") / Q([1, 3, 4], "day"))


def test_eq() -> None:
    r0 = Q(5, "kg") == Q(5_000, "g")
    assert_type(r0, bool)
    assert isinstance(r0, bool)
    assert r0

    r1 = Q([1, 2, 3], "kg") == Q([1, 2, 3], "kg")
    assert_type(r1, Numpy1DBoolArray)
    assert isinstance(r1, np.ndarray)
    assert r1.all()

    r2 = Q([1, 2, 3], "kg") == Q([1_000, 2_000, 3_000], "g")
    assert_type(r2, Numpy1DBoolArray)
    assert isinstance(r2, np.ndarray)
    assert r2.all()

    r3 = Q([1, 2, 3], "kg") == Q([1_001, 2_000, 3_000], "g")
    assert_type(r3, Numpy1DBoolArray)
    assert isinstance(r3, np.ndarray)
    assert not r3.all()

    r4 = Q([5, 5, 5], "kg") == Q(5_000, "g")
    assert_type(r4, Numpy1DBoolArray)
    assert isinstance(r4, np.ndarray)
    assert r4.all()

    expr = Q(pl.col.mass_kg, "kg") == Q(3_000, "g")
    assert_type(expr, pl.Expr)

    assert pl.DataFrame({"mass_kg": [1, 2, 3]}).select(expr).sum().item() == 1
