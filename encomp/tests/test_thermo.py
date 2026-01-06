from typing import assert_type

from ..misc import isinstance_types
from ..thermo import heat_balance, intermediate_temperatures
from ..units import Quantity as Q
from ..utypes import Energy, MassFlow, Power, Temperature, TemperatureDifference


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_heat_balance() -> None:
    assert isinstance_types(heat_balance(Q(2, "kg/s"), Q(2, "kJ/s").asdim(Power)), Q[TemperatureDifference])
    assert isinstance_types(heat_balance(Q(2, "K"), Q(2, "kJ/s").asdim(Power)), Q[MassFlow])
    assert isinstance_types(heat_balance(Q(2, "kg"), Q(2, "delta_degF")), Q[Energy])
    assert isinstance_types(heat_balance(Q(2, "kg/s"), Q(2, "delta_degF")), Q[Power])


def test_intermediate_temperatures() -> None:
    T1, T2 = intermediate_temperatures(
        Q(25, "degC"),
        Q(10, "degC"),
        Q(0.05, "W/m/K"),
        Q(10, "cm"),
        Q(1, "W/m²/K"),
        Q(2, "W/m²/K"),
        0.7,
    )

    assert_type(T1, Q[Temperature, float])
    assert_type(T2, Q[Temperature, float])

    assert T2.check(Temperature)
    assert T2.check("K")
