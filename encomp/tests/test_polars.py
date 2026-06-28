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


def test_polars_series_behaves_like_expr() -> None:
    # pl.Series is the polars world too: pint's numpy bridge is flaky on it
    # (half the reductions crash, the rest lose metadata), so the numpy-data
    # surface is disabled and column ops go through .m, exactly like pl.Expr
    qty = Q(pl.Series("asd", [1.0, 2.0, 3.0]), "kg")

    for name in ("mean", "sum", "std", "var", "cumsum"):
        with raises(AttributeError, match="is not supported for Quantity with pl"):
            getattr(qty, name)

    with raises(TypeError, match="cannot be converted to a numpy array"):
        np.asarray(qty)

    # unit algebra still works; column ops + native reductions go via .m
    assert qty.to("g").m[0] == 1000.0
    assert qty.m.mean() == 2.0
    assert np.asarray(qty.m).tolist() == [1.0, 2.0, 3.0]


def test_numpy_bridge_result_metadata() -> None:
    # numpy-native magnitudes keep the numpy bridge; its results are built by
    # pint at the magnitude-agnostic subclass level (_magnitude_type is None),
    # so mt/mt_name must recover the type from the live magnitude
    reduced = Q(np.array([1.0, 4.0, 9.0]), "kg").mean()
    assert reduced.m == 14.0 / 3
    assert reduced.mt_name == "float"
    assert reduced.mt is float

    cumulative = Q(np.array([1.0, 4.0, 9.0]), "kg").cumsum()
    assert cumulative.mt_name == "ndarray"


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
