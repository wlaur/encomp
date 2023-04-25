from ..gases import (
    actual_volume_to_normal_volume,
    convert_gas_volume,
    ideal_gas_density,
    mass_from_actual_volume,
    mass_to_actual_volume,
    normal_volume_to_actual_volume,
)
from ..units import Quantity as Q
from ..utypes import Density, Mass, Volume, VolumeFlow


def test_convert_gas_volume():
    ret = convert_gas_volume(Q(1, "m3"), "N", (Q(2, "bar"), Q(25, "degC")))

    assert ret.check(Q(0, "liter"))

    ret = convert_gas_volume(Q(1, "m3/s"), "S", (Q(2, "bar"), Q(25, "degC")))

    assert isinstance(ret, Q[VolumeFlow])


def test_ideal_gas_density():
    assert isinstance(
        ideal_gas_density(Q(25, "degC"), Q(12, "bar"), Q(12, "g/mol")), Q[Density]
    )


def test_gas_conversion():
    V = Q(1, "liter")
    m = Q(1, "kg")

    P = Q(1, "atm")
    T = Q(25, "degC")

    assert isinstance(mass_from_actual_volume(V, (P, T)), Q[Mass])

    assert isinstance(mass_to_actual_volume(m, (P, T)), Q[Volume])

    assert isinstance(actual_volume_to_normal_volume(V, (P, T)), Q[Volume])

    assert isinstance(normal_volume_to_actual_volume(V, (P, T)), Q[Volume])
