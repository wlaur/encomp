from typing import TYPE_CHECKING

import pytest

from ..units import Quantity as Q

if not TYPE_CHECKING:

    def reveal_type(x):
        return x


@pytest.mark.mypy_testing
def test_quantity_unspecified_type() -> None:
    return

    # some common literal units have overloaded __new__ methods
    # to infer the dimensionality
    # other units (i.e. not literals that are hard-coded to correspond
    # to a certain dimensionality), the dimensionality will be Unknown
    # the Unknown dimensionality is only used by the type checker, it will
    # never be used during runtime

    # fmt: off

    reveal_type(Q(1))  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(Q(1, ""))  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(Q(1, "%"))  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    reveal_type(Q(25, "kg"))  # R: encomp.units.Quantity[encomp.utypes.Mass]
    reveal_type(Q(25, "g"))  # R: encomp.units.Quantity[encomp.utypes.Mass]
    reveal_type(Q(25, "ton"))  # R: encomp.units.Quantity[encomp.utypes.Mass]
    reveal_type(Q(25, "tonne"))  # R: encomp.units.Quantity[encomp.utypes.Mass]
    reveal_type(Q(25, "t"))  # R: encomp.units.Quantity[encomp.utypes.Mass]
    reveal_type(Q(25, "mg"))  # R: encomp.units.Quantity[encomp.utypes.Mass]
    reveal_type(Q(25, "ug"))  # R: encomp.units.Quantity[encomp.utypes.Mass]

    reveal_type(Q(25, "L"))  # R: encomp.units.Quantity[encomp.utypes.Volume]
    reveal_type(Q(25, "l"))  # R: encomp.units.Quantity[encomp.utypes.Volume]
    reveal_type(Q(25, "liter"))  # R: encomp.units.Quantity[encomp.utypes.Volume]
    reveal_type(Q(25, "m3"))  # R: encomp.units.Quantity[encomp.utypes.Volume]
    reveal_type(Q(25, "m^3"))  # R: encomp.units.Quantity[encomp.utypes.Volume]
    reveal_type(Q(25, "m³"))  # R: encomp.units.Quantity[encomp.utypes.Volume]
    reveal_type(Q(25, "m**3"))  # R: encomp.units.Quantity[encomp.utypes.Volume]

    reveal_type(Q(25, "kg/m3"))  # R: encomp.units.Quantity[encomp.utypes.Density]
    reveal_type(Q(25, "kg/m**3"))  # R: encomp.units.Quantity[encomp.utypes.Density]
    reveal_type(Q(25, "kg/m^3"))  # R: encomp.units.Quantity[encomp.utypes.Density]
    reveal_type(Q(25, "kg/m³"))  # R: encomp.units.Quantity[encomp.utypes.Density]
    reveal_type(Q(25, "g/l"))  # R: encomp.units.Quantity[encomp.utypes.Density]
    reveal_type(Q(25, "g/L"))  # R: encomp.units.Quantity[encomp.utypes.Density]
    reveal_type(Q(25, "gram/liter"))  # R: encomp.units.Quantity[encomp.utypes.Density]

    reveal_type(Q(25, "°C"))  # R: encomp.units.Quantity[encomp.utypes.Temperature]
    reveal_type(Q(25, "degC"))  # R: encomp.units.Quantity[encomp.utypes.Temperature]
    reveal_type(Q(25, "K"))  # R: encomp.units.Quantity[encomp.utypes.Temperature]

    reveal_type(
        Q(25, "delta_°C")
    )  # R: encomp.units.Quantity[encomp.utypes.TemperatureDifference]
    reveal_type(
        Q(25, "delta_degC")
    )  # R: encomp.units.Quantity[encomp.utypes.TemperatureDifference]

    # refer to the encomp.utypes module for a list of string literals
    # that can be automatically inferred

    reveal_type(Q(25, "kg/s"))  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(Q(25, "kg"))  # R: encomp.units.Quantity[encomp.utypes.Mass]

    reveal_type(Q(25, "cm"))  # R: encomp.units.Quantity[encomp.utypes.Length]
    reveal_type(Q(25, "cm2"))  # R: encomp.units.Quantity[encomp.utypes.Area]
    reveal_type(Q(25, "cm3"))  # R: encomp.units.Quantity[encomp.utypes.Volume]

    reveal_type(Q(25, "MWh"))  # R: encomp.units.Quantity[encomp.utypes.Energy]
    reveal_type(Q(25, "kW"))  # R: encomp.units.Quantity[encomp.utypes.Power]

    reveal_type(
        Q(25, "kJ/kg/degC")
    )  # R: encomp.units.Quantity[encomp.utypes.SpecificHeatCapacity]
    reveal_type(
        Q(25, "kJ/kg/Δ°C")
    )  # R: encomp.units.Quantity[encomp.utypes.SpecificHeatCapacity]
    reveal_type(
        Q(25, "J/kg/°C")
    )  # R: encomp.units.Quantity[encomp.utypes.SpecificHeatCapacity]

    # when the unit is not a string literal, or if it's not hard-coded
    # to correspond to a specific dimensionality, Unknown will be inferred
    # by the type checker

    # TODO: mypy shows <nothing> instead of encomp.utypes.Unknown here

    reveal_type(Q(25, "kW/m^2"))  # R: encomp.units.Quantity[<nothing>]

    # string literals can be used as variables, however mypy does not handle this
    # this is correctly inferred by pyright

    unit = "kg/s"
    reveal_type(Q(25, unit))  # R: encomp.units.Quantity[<nothing>]

    unit_not_literal = str("kg/s")
    reveal_type(Q(25, unit_not_literal))  # R: encomp.units.Quantity[<nothing>]

    reveal_type(
        Q(25, "J") / Q(25, "mol") / Q(25, "delta_°C")
    )  # R: encomp.units.Quantity[encomp.utypes.MolarSpecificEntropy]
    reveal_type(
        Q(25, "J") / Q(25, "mol")
    )  # R: encomp.units.Quantity[encomp.utypes.MolarSpecificEnthalpy]
    reveal_type(
        Q(25, "J") / Q(25, "kmol")
    )  # R: encomp.units.Quantity[encomp.utypes.MolarSpecificEnthalpy]
    reveal_type(
        Q(25, "kg") / Q(25, "kmol")
    )  # R: encomp.units.Quantity[encomp.utypes.MolarMass]

    # fmt: on
