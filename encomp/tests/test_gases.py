from typing import Any, assert_type, cast

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
from ..utypes import Density, Mass, Numpy1DArray, Volume, VolumeFlow


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_convert_gas_volume() -> None:
    ret = convert_gas_volume(Q(1, "m3"), "N", (Q(2, "bar"), Q(25, "degC")))

    assert ret.check(Q(0, "liter"))

    ret2 = convert_gas_volume(Q(1, "m3/s"), "S", (Q(2, "bar"), Q(25, "degC")))

    # TODO: inferred types are not correct
    assert isinstance_types(ret2, Q[VolumeFlow])


def test_ideal_gas_density() -> None:
    # ideal_gas_density ties T/P/M to one magnitude TypeVar (its body does arithmetic
    # across all three), so a mixed array/scalar call cannot be typed: the scalar P/M
    # args are cast (needed for both checkers), and pyrefly additionally cannot solve
    # the shared constrained TypeVar across multiple arguments at all
    assert_type(  # pyrefly: ignore[assert-type]
        ideal_gas_density(Q(25, "degC"), Q(12, "bar"), Q(12, "g/mol")), Q[Density, float]
    )

    ret = ideal_gas_density(Q([25, 26], "degC"), cast(Any, Q(12, "bar")), cast(Any, Q(12, "g/mol")))  # pyrefly: ignore[bad-argument-type, bad-specialization]
    assert_type(ret, Q[Density, Numpy1DArray])  # pyrefly: ignore[assert-type]


def test_gas_conversion() -> None:
    V = Q(1, "liter")
    m = Q(1, "kg")

    P = Q(1, "atm")
    T = Q(25, "degC")

    assert isinstance_types(mass_from_actual_volume(V, (P, T)), Q[Mass])

    assert isinstance_types(mass_to_actual_volume(m, (P, T)), Q[Volume])

    assert isinstance_types(actual_volume_to_normal_volume(V, (P, T)), Q[Volume])

    assert isinstance_types(normal_volume_to_actual_volume(V, (P, T)), Q[Volume])
