import pandas as pd

from ..units import Quantity as Q


def test_pandas_series() -> None:
    s = pd.Series([1, 2, 3])
    assert Q(s, "kg").to("g").m[0] == 1000


def test_series_attributes() -> None:
    s = pd.Series([1, 2, 3], index=["A", "B", "C"], name="asd")

    assert Q(s, "bar").to("kPa").m.name == "asd"

    assert Q(s, "%").m.index.to_numpy()[0] == "A"

    s1 = pd.Series([1, 2, 3], name="s1")
    s2 = pd.Series([1, 2, 3], name="s2")

    q1 = Q(s1, "kg")
    q2 = Q(s2, "kg")

    assert q1.to("g").m.name == q1.m.name == "s1"
    assert q2.to("g").m.name == q2.m.name == "s2"

    assert Q(Q(Q(s1, "bar"))).m.name == "s1"

    assert (q1 / 2).m.name == "s1"
    assert (2 / q1).m.name == "s1"
    assert (q1 * 2).m.name == "s1"
    assert (2 * q1).m.name == "s1"


def test_scalar_getitem() -> None:
    index = Q(pd.DatetimeIndex(["2021-01-01", "2021-01-02"]).to_series())

    first = index[0].m
    second = index[1].m

    assert first == pd.Timestamp("2021-01-01").timestamp() * 1e9
    assert second == pd.Timestamp("2021-01-02").timestamp() * 1e9
