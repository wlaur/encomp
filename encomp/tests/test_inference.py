from typing import assert_type

import numpy as np
import pandas as pd
import polars as pl

from .. import utypes as ut
from ..units import Quantity as Q


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_inference() -> None:
    assert_type(Q([1]), Q[ut.Dimensionless])
    assert_type(Q(1), Q[ut.Dimensionless, float])
    assert_type(Q(np.array([1])), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q(np.array([1])), Q[ut.Dimensionless, ut.Numpy1DArray])
    assert_type(Q(pl.Series([1])), Q[ut.Dimensionless, pl.Series])
    assert_type(Q(pl.lit(1)), Q[ut.Dimensionless, pl.Expr])
    assert_type(Q(pd.Series([1])), Q[ut.Dimensionless, pd.Series])

    assert_type(Q(1) * 1, Q[ut.Dimensionless, float])
    assert_type(Q(1) * Q(1), Q[ut.Dimensionless, float])
    assert_type(Q(1), Q[ut.Dimensionless, float])
    assert_type(Q([1]) * Q(1), Q[ut.Dimensionless, np.ndarray])

    assert_type(Q(1) * Q(1, "kg"), Q[ut.Mass, float])
    assert_type(Q(1, "kg") * Q(1), Q[ut.Mass, float])
    assert_type(Q(1, "kg") / Q(1), Q[ut.Mass, float])
    assert_type(Q(1, "kg") * 1, Q[ut.Mass, float])
    assert_type(Q(1, "kg") / 1, Q[ut.Mass, float])

    assert_type(Q([1]) * Q(1, "kg"), Q[ut.Mass, np.ndarray])

    # etc: various combinations of *, /, +, -, >, <, <=, >=, ==
    # check a subset of overloads from units.py
