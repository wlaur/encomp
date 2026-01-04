from typing import assert_type

import numpy as np
import polars as pl

from encomp.units import Quantity as Q
from encomp.utypes import Dimensionless, Mass, MassFlow, Numpy1DArray, Numpy1DBoolArray, UnknownDimensionality


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


def test_unknown_mul_div() -> None:
    q1 = Q(2, "kilogram")
    assert_type(q1, Q[UnknownDimensionality, float])

    q2 = Q(2, "meter/kg * kg")
    assert_type(q2, Q[UnknownDimensionality, float])

    assert_type(q1 * 2, Q[UnknownDimensionality, float])
    assert_type(2 * q2, Q[UnknownDimensionality, float])

    assert_type(q1 * q2, Q[UnknownDimensionality, float])
    assert_type(q2 * q1, Q[UnknownDimensionality, float])

    assert_type(q1 / 2, Q[UnknownDimensionality, float])
    assert_type(2 / q2, Q[UnknownDimensionality, float])

    assert_type(q1 / q2, Q[UnknownDimensionality, float])
    assert_type(q2 / q1, Q[UnknownDimensionality, float])

    q3 = Q(pl.Series([1, 2, 3]), "meter/kg * kg")
    assert_type(q3 * 2, Q[UnknownDimensionality, pl.Series])
    assert_type(q3 * q1, Q[UnknownDimensionality, pl.Series])
    assert_type(q1 * q3, Q[UnknownDimensionality, pl.Series])

    assert_type(q3 / 2, Q[UnknownDimensionality, pl.Series])
    assert_type(q3 / q1, Q[UnknownDimensionality, pl.Series])
    assert_type(q1 / q3, Q[UnknownDimensionality, pl.Series])

    assert_type(Q(2, "m") / Q(25, "cm"), Q[Dimensionless, float])
    assert_type(Q(2, "kg") / Q(25, "g"), Q[Dimensionless, float])

    assert_type(Q(25, "kg") / Q([1, 2, 3], "s"), Q[MassFlow, Numpy1DArray])
    assert_type(Q(25, "kg") / Q(pl.col.test, "s"), Q[MassFlow, pl.Expr])


def test_magnitude_type_broadcasting() -> None:
    assert_type(Q(25, "kg") + Q([1, 2, 3], "g"), Q[Mass, Numpy1DArray])
    assert_type(Q(25, "kg") - Q([1, 2, 3], "g"), Q[Mass, Numpy1DArray])
    assert_type(Q([1, 2, 3], "g") - Q(25, "kg"), Q[Mass, Numpy1DArray])
    assert_type(Q([1, 2, 3], "g") + Q(25, "kg"), Q[Mass, Numpy1DArray])

    assert_type(Q(25, "kg") * Q([1, 2, 3], "m"), Q[UnknownDimensionality, Numpy1DArray])
    assert_type(Q(25, "kg") / Q([1, 2, 3], "m"), Q[UnknownDimensionality, Numpy1DArray])
    assert_type(Q([1, 2, 3], "g") * Q(25, "m"), Q[UnknownDimensionality, Numpy1DArray])
    assert_type(Q([1, 2, 3], "g") / Q(25, "m"), Q[UnknownDimensionality, Numpy1DArray])

    assert_type(Q(25, "kg") + Q(pl.col.test, "g"), Q[Mass, pl.Expr])
    assert_type(Q(25, "kg") - Q(pl.col.test, "g"), Q[Mass, pl.Expr])
    assert_type(Q(pl.col.test, "g") - Q(25, "kg"), Q[Mass, pl.Expr])
    assert_type(Q(pl.col.test, "g") + Q(25, "kg"), Q[Mass, pl.Expr])

    assert_type(Q(25, "kg") * Q(pl.col.test, "m"), Q[UnknownDimensionality, pl.Expr])
    assert_type(Q(25, "kg") / Q(pl.col.test, "m"), Q[UnknownDimensionality, pl.Expr])
    assert_type(Q(pl.col.test, "g") * Q(25, "m"), Q[UnknownDimensionality, pl.Expr])
    assert_type(Q(pl.col.test, "g") / Q(25, "m"), Q[UnknownDimensionality, pl.Expr])
