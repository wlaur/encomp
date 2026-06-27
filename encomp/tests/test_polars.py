from datetime import datetime

import numpy as np
import polars as pl
from pytest import raises

from ..units import Quantity as Q


def test_polars_series() -> None:
    s = pl.Series([1, 2, 3])
    assert Q(s, "kg").to("g").m[0] == 1000

    s1 = pl.Series([3, 2, 1]).alias("s1")
    assert Q(s1, "kg").to_base_units().m.name == "s1"

    s2 = pl.Series(name="s2", values=[3, 2, 1])
    assert (1 / Q(s2, "kg").to("g")).m.name == "s2"


def test_polars_expr() -> None:
    res = (Q(pl.col("test"), "kg").to("g") / Q(25, "lbs")).to("%").m
    assert isinstance(res, pl.Expr)


def test_polars_expr_data_operations_raise() -> None:
    # a pl.Expr magnitude is a deferred plan, not data. operations from the
    # numpy-data surface (aggregations, polars-native methods, numpy coercion)
    # all misfire on a plan and are refused; they belong on the magnitude (.m)
    qty = Q(pl.col("asd"), "kg")

    for name in ("mean", "sum", "min", "max", "std", "var", "rank", "alias", "over", "is_null"):
        with raises(AttributeError, match="is not supported for Quantity with pl"):
            getattr(qty, name)

    assert not hasattr(qty, "mean")

    # numpy coercion is refused rather than silently yielding a 0-d object array
    with raises(TypeError, match="cannot be converted to a numpy array"):
        np.asarray(qty)

    # the .m escape hatch is the boundary to the polars world
    assert isinstance(qty.m.mean(), pl.Expr)


def test_polars_expr_unit_algebra_works() -> None:
    # unit algebra is meaningful on a plan and keeps the unit on the expression
    qty = Q(pl.col("asd"), "kg")

    assert isinstance((qty * Q(2, "")).m, pl.Expr)
    assert isinstance(qty.to("g").m, pl.Expr)
    assert isinstance(abs(qty).m, pl.Expr)
    assert (qty.to("g")).u == Q(1, "g").u

    df = pl.DataFrame({"asd": [1.0, 2.0, 3.0]})
    assert df.select(qty.to("g").m)["asd"][0] == 1000.0


def test_polars_series_magnitude_methods_unaffected() -> None:
    # eager magnitudes (pl.Series, np.ndarray) are data, so the pint numpy
    # bridge stays enabled and reduces them in place, keeping the unit attached
    qty = Q(pl.Series("asd", [1.0, 2.0, 3.0]), "kg")
    mean = qty.mean()
    assert mean.m == 2.0
    assert mean.u == Q(1, "kg").u
    assert np.asarray(qty).tolist() == [1.0, 2.0, 3.0]


def test_polars_dataframe() -> None:
    df = pl.DataFrame({"values_kg": [1, 2, 3, 4]})

    assert df.with_columns(Q(pl.col("values_kg"), "kg").to("g").m.alias("values_g")).head(1)["values_g"][0] == 1000


def test_datetimes() -> None:
    df = pl.DataFrame(
        {
            "values_kg": [1, 2, 3, 4],
            "time": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3), datetime(2025, 1, 4)],
        }
    )

    qty = Q(df["time"])
    assert qty.m[0] == datetime(2025, 1, 1)

    with raises(TypeError):
        qty[0]
