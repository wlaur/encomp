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
