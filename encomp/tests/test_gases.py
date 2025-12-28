from ..gases import (
    actual_volume_to_normal_volume,
    convert_gas_volume,
    ideal_gas_density,
    mass_from_actual_volume,
    mass_to_actual_volume,
    normal_volume_to_actual_volume,
)
from ..misc import isinstance_types
from ..units import Quantity as Q
from ..utypes import Density, Mass, Volume, VolumeFlow


def test_convert_gas_volume() -> None:
    ret = convert_gas_volume(Q(1, "m3"), "N", (Q(2, "bar"), Q(25, "degC")))

    assert ret.check(Q(0, "liter"))

    ret = convert_gas_volume(Q(1, "m3/s"), "S", (Q(2, "bar"), Q(25, "degC")))

    assert isinstance_types(ret, Q[VolumeFlow])


def test_ideal_gas_density() -> None:
    assert isinstance_types(ideal_gas_density(Q(25, "degC"), Q(12, "bar"), Q(12, "g/mol")), Q[Density])


def test_gas_conversion() -> None:
    V = Q(1, "liter")
    m = Q(1, "kg")

    P = Q(1, "atm")
    T = Q(25, "degC")

    assert isinstance_types(mass_from_actual_volume(V, (P, T)), Q[Mass])

    assert isinstance_types(mass_to_actual_volume(m, (P, T)), Q[Volume])

    assert isinstance_types(actual_volume_to_normal_volume(V, (P, T)), Q[Volume])

    assert isinstance_types(normal_volume_to_actual_volume(V, (P, T)), Q[Volume])
