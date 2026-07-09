from typing import Any, assert_type, cast

import pytest

from .. import gases
from ..constants import CONSTANTS
from ..units import Quantity
from ..utypes import Pressure, Temperature


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_CONSTANTS() -> None:
    assert_type(CONSTANTS.normal_conditions_pressure, Quantity[Pressure, float])
    assert_type(CONSTANTS.normal_conditions_temperature, Quantity[Temperature, float])
    assert_type(CONSTANTS.standard_conditions_pressure, Quantity[Pressure, float])
    assert_type(CONSTANTS.standard_conditions_temperature, Quantity[Temperature, float])


def test_constant_values() -> None:
    assert Quantity(8.31446261815324, "J/mol/K") == CONSTANTS.R
    assert Quantity(5.670374419e-8, "W/m²/K⁴") == CONSTANTS.SIGMA
    assert CONSTANTS.normal_conditions_pressure == Quantity(1, "atm")
    assert CONSTANTS.normal_conditions_temperature == Quantity(0, "degC")
    assert CONSTANTS.standard_conditions_pressure == Quantity(1, "atm")
    assert CONSTANTS.standard_conditions_temperature == Quantity(15, "degC")


def test_constants_cannot_be_mutated() -> None:
    # a Quantity is mutable (.ito converts in place, the m setter replaces the magnitude), so
    # every attribute must hand out a fresh object rather than a shared one
    assert CONSTANTS.R is not CONSTANTS.R

    with pytest.raises(AttributeError):
        cast(Any, CONSTANTS).R = Quantity(1.0, "J/mol/K")

    reference = Quantity(8.31446261815324, "J/mol/K")

    CONSTANTS.R.m = 1.0
    assert reference == CONSTANTS.R

    CONSTANTS.R.ito("kJ/mol/K")
    assert reference == CONSTANTS.R
    assert CONSTANTS.R.u == Quantity.get_unit("kg*m²/K/mol/s²")

    CONSTANTS.normal_conditions_temperature.ito("degC")
    assert CONSTANTS.normal_conditions_temperature == Quantity(0, "degC")
    assert CONSTANTS.normal_conditions_temperature.u == Quantity.get_unit("K")


def test_gases_does_not_alias_the_constants() -> None:
    # gases reads CONSTANTS on every call, so mutating the gas constant cannot change a later result
    args = (Quantity(1.0, "bar"), Quantity(300.0, "K"), Quantity(29.0, "g/mol"))
    density = gases.ideal_gas_density(*args)

    CONSTANTS.R.m = 1.0
    CONSTANTS.normal_conditions_pressure.ito("kPa")

    assert gases.ideal_gas_density(*args) == density
