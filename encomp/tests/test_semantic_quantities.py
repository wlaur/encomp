from typing import Any, assert_type, cast

import numpy as np
import polars as pl
import pytest

from ..units import DimensionalityTypeError
from ..units import Quantity as Q
from ..utypes import Numpy1DArray, SpecificEntropy, TemperatureDifference


def test_quantity_copy_preserves_semantic_dimensionality() -> None:
    difference = Q(18.0, "delta_degF").to("K")
    copied = Q(difference)
    assert_type(copied, Q[TemperatureDifference, float])
    assert copied is not difference
    assert copied.dt is TemperatureDifference
    assert copied.to("delta_degC").m == pytest.approx(10.0)

    entropy = Q(2.0, "J/kg/K").asdim(SpecificEntropy)
    assert Q(entropy).dt is SpecificEntropy
    assert Q(entropy, Q.get_unit("kJ/kg/K")).dt is SpecificEntropy

    array = difference.astype("ndarray")
    assert_type(Q(array), Q[TemperatureDifference, Numpy1DArray])
    np.testing.assert_allclose(Q(array).to("delta_degC").m, [10.0])
    series = difference.astype(pl.Series)
    assert Q(series).dt is TemperatureDifference
    assert Q(series).m.to_list() == pytest.approx([10.0])
    expression = difference.astype(pl.Expr)
    assert Q(expression).dt is TemperatureDifference
    assert pl.select(Q(expression).m).item() == pytest.approx(10.0)


def test_with_magnitude_preserves_unit_and_kind_across_containers() -> None:
    difference = Q(18.0, "delta_degF").to("K")
    scalar = difference.with_magnitude(20.0)
    array = difference.with_magnitude(np.array([20.0, 30.0]))
    series = array.with_magnitude(pl.Series("rise", [20.0, None]))
    expression = series.with_magnitude(pl.col("rise") * 2.0)
    assert_type(scalar, Q[TemperatureDifference, float])
    assert_type(array, Q[TemperatureDifference, Numpy1DArray])
    assert_type(series, Q[TemperatureDifference, pl.Series])
    assert_type(expression, Q[TemperatureDifference, pl.Expr])

    for qty in (scalar, array, series, expression):
        assert qty.dt is TemperatureDifference
        assert qty.u == difference.u
        with pytest.raises(DimensionalityTypeError):
            qty.to("degC")
    assert difference.m == pytest.approx(10.0)
    assert scalar.to("delta_degF").m == pytest.approx(36.0)
    np.testing.assert_allclose(array.to("delta_degF").m, [36.0, 54.0])
    assert series.to("delta_degF").m.to_list() == [36.0, None]
    assert pl.DataFrame({"rise": [10.0, 20.0]}).select(expression.m).to_series().to_list() == [20.0, 40.0]
    entropy = Q(2.0, "J/kg/K").asdim(SpecificEntropy)
    assert entropy.with_magnitude(pl.col("s")).dt is SpecificEntropy


def test_with_magnitude_validates_replacement() -> None:
    difference = Q(5.0, "delta_degC").to("K")
    assert difference.with_magnitude(2).m == 2.0
    assert_type(difference.with_magnitude([1.0, 2.0]), Q[TemperatureDifference, Numpy1DArray])
    with pytest.raises(TypeError):
        difference.with_magnitude(cast(Any, True))
    with pytest.raises(ValueError, match="1-dimensional"):
        difference.with_magnitude(cast(Any, np.ones((2, 2))))
