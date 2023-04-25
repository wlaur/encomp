from typing import TYPE_CHECKING

import pytest

from ..fluids import HumidAir, Water
from ..units import Quantity as Q

if not TYPE_CHECKING:

    def reveal_type(x):
        return x


@pytest.mark.mypy_testing
def test_fluids_properties_types() -> None:
    return

    w = Water(P=Q(25, "bar"), T=Q(250, "degC"))

    # fmt: off

    reveal_type(w.phase)  # R: builtins.str
    reveal_type(w.PHASE)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    reveal_type(w.P)  # R: encomp.units.Quantity[encomp.utypes.Pressure]
    reveal_type(w.T)  # R: encomp.units.Quantity[encomp.utypes.Temperature]
    reveal_type(w.H)  # R: encomp.units.Quantity[encomp.utypes.SpecificEnthalpy]
    reveal_type(w.S)  # R: encomp.units.Quantity[encomp.utypes.SpecificEntropy]
    reveal_type(w.D)  # R: encomp.units.Quantity[encomp.utypes.Density]
    reveal_type(w.V)  # R: encomp.units.Quantity[encomp.utypes.DynamicViscosity]
    reveal_type(w.Q)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(w.Z)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(w.C)  # R: encomp.units.Quantity[encomp.utypes.SpecificHeatCapacity]

    m = w.D * Q(25, "liter")
    reveal_type(m)  # R: encomp.units.Quantity[encomp.utypes.Mass]

    # only the first synonym for each property has a type hint

    reveal_type(w.SURFACE_TENSION)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    ha = HumidAir(P=Q(25, "bar"), T=Q(250, "degC"), R=Q(25, "%"))

    # the attribute names for HumidAir are different (this is based on HAPropsSI from CoolProp)

    reveal_type(ha.P)  # R: encomp.units.Quantity[encomp.utypes.Pressure]
    reveal_type(ha.T)  # R: encomp.units.Quantity[encomp.utypes.Temperature]
    reveal_type(ha.D)  # R: encomp.units.Quantity[encomp.utypes.Temperature]
    reveal_type(ha.V)  # R: encomp.units.Quantity[encomp.utypes.MixtureVolumePerDryAir]

    reveal_type(ha.Z)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # fmt: on


@pytest.mark.mypy_testing
def test_water_init_hints() -> None:
    return

    # NOTE: this is not actually type checked beyond the Quantity superclass
    # however, incorrect inputs will raise ValueError at runtime

    # fmt: off

    Water(
        T=25, P=Q(25, "bar")
    )  # E: Argument "T" to "Water" has incompatible type "int"; expected "Quantity[Any]"

    Water(
        T=(25, "degC"), P=Q(25, "bar")
    )  # E: Argument "T" to "Water" has incompatible type "Tuple[int, str]"; expected "Quantity[Any]"

    # fmt: on

    Water(P=Q(25, "bar"), T=Q(25, "degC"))
    Water(T=Q(25, "degC"), P=Q(25, "bar"))
    Water(P=Q(25, "bar"), Q=Q(50, "%"))
    Water(T=Q(25, "°C"), Q=Q(50, "%"))
    Water(Q=Q(50, "%"), P=Q(25, "kPa"))

    # TODO: is it possible to overload the __init__ method
    # to accept "P: Q[Pressure], T: Q[Temperature]", but not "P: Q[Mass]",
    # while still allowing arbitrary kwargs (for example S: Q[SpecificEntropy])
    # this could maybe be implemented by overloading *all* possible inputs to CoolProp,
    # does not seem very useful
