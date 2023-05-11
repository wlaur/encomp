import polars as pl

from ..units import Quantity as Q


def test_polars_series():
    s = pl.Series([1, 2, 3])
    assert Q(s, "kg").to("g").m[0] == 1000

    s1 = pl.Series([3, 2, 1]).alias("s1")
    assert Q(s1, "kg").to_base_units().m.name == "s1"

    s2 = pl.Series(name="s2", values=[3, 2, 1])
    assert (1 / Q(s2, "kg").to("g")).m.name == "s2"


def test_polars_expr():
    res = ((Q(pl.col("test"), "kg").to("g") / Q(25, "lbs"))).to("%").m
    assert str(res) == '[([([(col("test")) * (1000.0)]) / (25.0)]) * (0.220462)]'


def test_polars_dataframe():
    df = pl.DataFrame({"values_kg": [1, 2, 3, 4]})

    assert (
        df.with_columns(Q(pl.col("values_kg"), "kg").to("g").m.alias("values_g")).head(
            1
        )["values_g"][0]
        == 1000
    )
