from typing import Any, assert_type, cast

import numpy as np
import polars as pl
import pytest

from ..constants import CONSTANTS
from ..thermo import specific_heat_capacity, temperature_rise
from ..units import DimensionalityTypeError, ExpectedDimensionalityError
from ..units import Quantity as Q
from ..utypes import Numpy1DArray, SpecificHeatCapacity, Temperature, TemperatureDifference


def test_conventional_heat_capacity_produces_a_temperature_rise() -> None:
    energy = Q(41.9, "kJ/kg")
    capacity = Q(4.19, "kJ/kg/K")
    rise = temperature_rise(energy, capacity)
    assert_type(rise, Q[TemperatureDifference, float])
    assert rise.dt is TemperatureDifference
    assert rise.to("delta_degC").m == pytest.approx(10.0)
    assert rise.to("delta_degF").m == pytest.approx(18.0)
    assert (Q(20.0, "degC") + rise).m == pytest.approx(30.0)
    recovered = specific_heat_capacity(energy, rise)
    assert_type(recovered, Q[SpecificHeatCapacity, float])
    assert recovered.to("kJ/kg/K").m == pytest.approx(4.19)
    assert temperature_rise(-energy, capacity).m == pytest.approx(-10.0)


def test_heat_balance_preserves_container_and_supports_scalar_broadcast() -> None:
    capacity = Q(4.0, "kJ/kg/K")
    energy = Q([40.0, 80.0], "kJ/kg")
    rise = temperature_rise(energy, capacity)
    assert_type(rise, Q[TemperatureDifference, Numpy1DArray])
    np.testing.assert_allclose(rise.to("delta_degC").m, [10.0, 20.0])
    broadcast_rise = temperature_rise(Q(40.0, "kJ/kg"), capacity.with_magnitude(np.array([4.0, 8.0])))
    np.testing.assert_allclose(broadcast_rise.m, [10.0, 5.0])
    recovered = specific_heat_capacity(energy, Q(18.0, "delta_degF"))
    assert_type(recovered, Q[SpecificHeatCapacity, Numpy1DArray])
    np.testing.assert_allclose(recovered.to("kJ/kg/K").m, [4.0, 8.0])

    series_energy = energy.with_magnitude(pl.Series("heat", [40.0, None]))
    series_rise = temperature_rise(series_energy, capacity)
    assert_type(series_rise, Q[TemperatureDifference, pl.Series])
    assert series_rise.m.to_list() == [10.0, None]
    assert specific_heat_capacity(series_energy, Q(10.0, "delta_degC")).m.to_list() == [4000.0, None]

    expr_rise = temperature_rise(energy.with_magnitude(pl.col("heat")), capacity)
    assert_type(expr_rise, Q[TemperatureDifference, pl.Expr])
    expr_capacity = specific_heat_capacity(energy.with_magnitude(pl.col("heat")), Q(10.0, "delta_degC"))
    assert_type(expr_capacity, Q[SpecificHeatCapacity, pl.Expr])
    result = pl.DataFrame({"heat": [40.0, None]}).select(expr_rise.m.alias("rise"), expr_capacity.m.alias("capacity"))
    assert result["rise"].to_list() == [10.0, None]
    assert result["capacity"].to_list() == [4000.0, None]


def test_thermal_helpers_reject_wrong_semantics() -> None:
    with pytest.raises(ExpectedDimensionalityError):
        specific_heat_capacity(Q(40.0, "kJ/kg"), cast(Any, Q(10.0, "K")))
    with pytest.raises(ExpectedDimensionalityError):
        temperature_rise(cast(Any, Q(40.0, "kJ")), Q(4.0, "kJ/kg/K"))
    with pytest.raises(ExpectedDimensionalityError):
        temperature_rise(Q(40.0, "kJ/kg"), cast(Any, Q(4.0, "kg")))


def test_general_temperature_algebra_stays_absolute() -> None:
    temperature = Q(101325.0, "Pa") * Q(0.024, "m³") / (Q(1.0, "mol") * CONSTANTS.R)
    assert temperature.dt is Temperature
    assert temperature.m_as("K") == pytest.approx(292.478313)
    specific_gas_constant = Q(287.0, "J/kg/K")
    absolute = (Q(101325.0, "Pa") / Q(1.2, "kg/m³")) / specific_gas_constant
    assert absolute.dt is Temperature
    assert absolute.m_as("K") == pytest.approx(294.207317)
    with pytest.raises(DimensionalityTypeError):
        absolute.to("delta_degC")


@pytest.mark.parametrize("unit", ["J/kg/K", "J/kg/delta_degC", "J/kg/delta_K"])
def test_heat_capacity_temperature_scale_cancels_without_offsets(unit: str) -> None:
    capacity = Q(4000.0, unit).asdim(SpecificHeatCapacity)
    assert temperature_rise(Q(40.0, "kJ/kg"), capacity).to("delta_degC").m == pytest.approx(10.0)
    capacity_fahrenheit = Q(4000.0 / 1.8, "J/kg/delta_degF").asdim(SpecificHeatCapacity)
    assert temperature_rise(Q(40.0, "kJ/kg"), capacity_fahrenheit).to("delta_degF").m == pytest.approx(18.0)
    assert specific_heat_capacity(Q(40.0, "kJ/kg"), Q(18.0, "delta_degF")).m == pytest.approx(4000.0)
