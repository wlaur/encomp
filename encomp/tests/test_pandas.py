import pandas as pd

from ..units import Quantity as Q


def test_pandas_series():
    s = pd.Series([1, 2, 3])
    assert Q(s, "kg").to("g").m[0] == 1000
