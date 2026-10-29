"""Typed sensible-heat balances for a specified constant specific heat capacity."""

from __future__ import annotations

from typing import cast, overload

import polars as pl

from .units import Quantity, require_dimensionality
from .utypes import EnergyPerMass, Numpy1DArray, SpecificHeatCapacity, TemperatureDifference

__all__ = ["specific_heat_capacity", "temperature_rise"]


@overload
def temperature_rise[MT: (float, Numpy1DArray, pl.Series, pl.Expr)](  # pyrefly: ignore[inconsistent-overload]
    energy: Quantity[EnergyPerMass, MT], heat_capacity: Quantity[SpecificHeatCapacity, MT]
) -> Quantity[TemperatureDifference, MT]: ...


@overload
def temperature_rise[MT: (float, Numpy1DArray, pl.Series, pl.Expr)](  # pyrefly: ignore[inconsistent-overload]
    energy: Quantity[EnergyPerMass, MT], heat_capacity: Quantity[SpecificHeatCapacity, float]
) -> Quantity[TemperatureDifference, MT]: ...


@overload
def temperature_rise[MT: (float, Numpy1DArray, pl.Series, pl.Expr)](  # pyrefly: ignore[inconsistent-overload]
    energy: Quantity[EnergyPerMass, float], heat_capacity: Quantity[SpecificHeatCapacity, MT]
) -> Quantity[TemperatureDifference, MT]: ...


def temperature_rise[M1: (float, Numpy1DArray, pl.Series, pl.Expr), M2: (float, Numpy1DArray, pl.Series, pl.Expr)](
    energy: Quantity[EnergyPerMass, M1], heat_capacity: Quantity[SpecificHeatCapacity, M2]
) -> Quantity[TemperatureDifference, M1] | Quantity[TemperatureDifference, M2]:
    """Return the temperature change from sensible heat per mass and heat capacity.

    ``energy`` is added heat per mass, not an absolute enthalpy or internal energy.
    ``heat_capacity`` is the specified constant (or interval-averaged) specific heat
    capacity. A negative heat input produces a negative change. This balance assumes
    no phase change, work, or reaction contribution. Units such as ``J/kg/K`` work
    without delta spellings because the function explicitly supplies the semantics.

    Matching magnitude containers combine elementwise; a scalar broadcasts onto any
    supported container. Missing values and division by zero follow the container.
    """
    validated_energy = require_dimensionality(energy, EnergyPerMass, name="energy")
    validated_capacity = require_dimensionality(heat_capacity, SpecificHeatCapacity, name="heat_capacity")
    result = (validated_energy / validated_capacity).asdim(TemperatureDifference).to("delta_K")
    return cast("Quantity[TemperatureDifference, M1] | Quantity[TemperatureDifference, M2]", result)


@overload
def specific_heat_capacity[MT: (float, Numpy1DArray, pl.Series, pl.Expr)](  # pyrefly: ignore[inconsistent-overload]
    energy: Quantity[EnergyPerMass, MT], temperature_change: Quantity[TemperatureDifference, MT]
) -> Quantity[SpecificHeatCapacity, MT]: ...


@overload
def specific_heat_capacity[MT: (float, Numpy1DArray, pl.Series, pl.Expr)](  # pyrefly: ignore[inconsistent-overload]
    energy: Quantity[EnergyPerMass, MT], temperature_change: Quantity[TemperatureDifference, float]
) -> Quantity[SpecificHeatCapacity, MT]: ...


@overload
def specific_heat_capacity[MT: (float, Numpy1DArray, pl.Series, pl.Expr)](  # pyrefly: ignore[inconsistent-overload]
    energy: Quantity[EnergyPerMass, float], temperature_change: Quantity[TemperatureDifference, MT]
) -> Quantity[SpecificHeatCapacity, MT]: ...


def specific_heat_capacity[
    M1: (float, Numpy1DArray, pl.Series, pl.Expr),
    M2: (float, Numpy1DArray, pl.Series, pl.Expr),
](
    energy: Quantity[EnergyPerMass, M1], temperature_change: Quantity[TemperatureDifference, M2]
) -> Quantity[SpecificHeatCapacity, M1] | Quantity[SpecificHeatCapacity, M2]:
    """Return interval-averaged specific heat capacity from a sensible-heat balance.

    ``energy`` is added heat per mass under the assumptions of :func:`temperature_rise`.
    ``temperature_change`` must be a difference, including when expressed in ``K``.
    Absolute temperatures are rejected. A scalar broadcasts onto any supported
    magnitude container; division by zero and missing values follow that container.
    """
    validated_energy = require_dimensionality(energy, EnergyPerMass, name="energy")
    validated_change = require_dimensionality(temperature_change, TemperatureDifference, name="temperature_change")
    result = (validated_energy / validated_change).asdim(SpecificHeatCapacity).to("J/kg/K")
    return cast("Quantity[SpecificHeatCapacity, M1] | Quantity[SpecificHeatCapacity, M2]", result)
