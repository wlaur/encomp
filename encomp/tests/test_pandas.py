import pytest
import pandas as pd

from ..units import Quantity as Q


def test_pandas_series():
    s = pd.Series([1, 2, 3])
    assert Q(s, "kg").to("g").m[0] == 1000


def test_series_attributes():
    s = pd.Series([1, 2, 3], index=["A", "B", "C"], name="asd")

    assert Q(s, "bar").to("kPa").m.name == "asd"

    assert Q(s, "%").m.index.values[0] == "A"

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


def test_datetimeindex():
    s = pd.DatetimeIndex(["2021-01-01", "2021-01-02"], name="Time")

    assert Q(s).m.name == "Time"

    assert (Q(s) + pd.Timedelta(1, "d")).m[0] == pd.Timestamp("2021-01-02")
    assert (Q(s) - pd.Timedelta(1, "d")).m[1] == pd.Timestamp("2021-01-01")

    with pytest.raises(ValueError):
        Q(s, "kg")

    with pytest.raises(TypeError):
        Q(s) * 2

    with pytest.raises(TypeError):
        2 * Q(s)


def test_scalar_getitem():
    index = Q(pd.DatetimeIndex(["2021-01-01", "2021-01-02"]))

    first = index[0].m
    second = index[1].m

    assert isinstance(first, pd.Timestamp)
    assert first == pd.Timestamp("2021-01-01")

    assert isinstance(second, pd.Timestamp)
    assert second == pd.Timestamp("2021-01-02")
