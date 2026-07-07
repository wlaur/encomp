from typing import Any, assert_type, cast

import pytest

from ..gases import (
    actual_volume_to_normal_volume,
    convert_gas_volume,
    ideal_gas_density,
    mass_from_actual_volume,
    mass_from_normal_volume,
    mass_to_actual_volume,
    mass_to_normal_volume,
    normal_volume_to_actual_volume,
)
from ..units import Quantity as Q
from ..utypes import (
    Density,
    Mass,
    MassFlow,
    NormalVolume,
    NormalVolumeFlow,
    Numpy1DArray,
    Volume,
    VolumeFlow,
)

approx = cast(Any, pytest).approx


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_convert_gas_volume() -> None:
    # the dimensionality (Volume / VolumeFlow) and units of the input are preserved
    ret = convert_gas_volume(Q(1, "m3"), "N", (Q(2, "bar"), Q(25, "degC")))
    assert_type(ret, Q[Volume, float])
    assert ret.check(Q(0, "liter"))
    assert str(ret.u) == str(Q(1, "m3").u)

    ret2 = convert_gas_volume(Q(1, "m3/s"), "S", (Q(2, "bar"), Q(25, "degC")))
    assert_type(ret2, Q[VolumeFlow, float])

    with pytest.raises(ValueError, match=r"condition_1.*'N'.*'S'"):
        convert_gas_volume(Q(1, "m3"), cast(Any, "n"), "N")

    with pytest.raises(ValueError, match=r"condition_1.*'N'.*'S'"):
        convert_gas_volume(Q(1, "m3"), cast(Any, "NS"), "N")

    with pytest.raises(TypeError, match=r"condition_2.*pressure, temperature"):
        convert_gas_volume(Q(1, "m3"), "N", cast(Any, (Q(1, "bar"),)))

    with pytest.raises(TypeError, match=r"condition_2.*pressure, temperature"):
        convert_gas_volume(Q(1, "m3"), "N", cast(Any, (Q(25, "degC"), Q(1, "bar"))))

    with pytest.raises(TypeError, match="normal_volume_to_actual_volume"):
        convert_gas_volume(cast(Any, Q(100.0, "Nm³")), "N", (Q(2, "bar"), Q(25, "degC")))

    with pytest.raises(TypeError, match="normal_volume_to_actual_volume"):
        convert_gas_volume(cast(Any, Q(100.0, "Nm³/h")), "N", (Q(2, "bar"), Q(25, "degC")))


def test_ideal_gas_density() -> None:
    # ideal_gas_density ties P/T/M to one magnitude TypeVar (its body does arithmetic
    # across all three), so a mixed array/scalar call cannot be typed: the scalar P/M
    # args are cast (needed for both checkers), and pyrefly additionally cannot solve
    # the shared constrained TypeVar across multiple arguments at all
    assert_type(  # pyrefly: ignore[assert-type]
        ideal_gas_density(Q(12, "bar"), Q(25, "degC"), Q(12, "g/mol")),  # pyrefly: ignore[bad-specialization]
        Q[Density, float],
    )

    ret = ideal_gas_density(cast(Any, Q(12, "bar")), Q([25, 26], "degC"), cast(Any, Q(12, "g/mol")))  # pyrefly: ignore[bad-argument-type, bad-specialization]
    assert_type(ret, Q[Density, Numpy1DArray])  # pyrefly: ignore[assert-type]


def test_gas_conversion() -> None:
    V = Q(1, "liter")
    m = Q(1, "kg")

    P = Q(1, "atm")
    T = Q(25, "degC")

    assert_type(mass_from_actual_volume(V, (P, T)), Q[Mass, float])
    assert_type(mass_from_actual_volume(V, "N"), Q[Mass, float])

    assert_type(mass_to_actual_volume(m, (P, T)), Q[Volume, float])
    assert_type(mass_to_actual_volume(m, "S"), Q[Volume, float])

    # the normal-volume side carries the NormalVolume dimensionality (Nm³)
    # (assert_type also verifies the runtime type via the monkeypatch above; a
    # narrowing `assert isinstance_types(...)` would poison later inference)
    nv = actual_volume_to_normal_volume(V, (P, T))
    assert_type(nv, Q[NormalVolume, Any])
    assert nv.check(Q(0, "Nm³"))
    assert_type(actual_volume_to_normal_volume(V, "N"), Q[NormalVolume, Any])

    v_actual = normal_volume_to_actual_volume(nv, (P, T))
    assert_type(v_actual, Q[Volume, Any])
    assert_type(normal_volume_to_actual_volume(nv, "S"), Q[Volume, Any])

    # round trip recovers the original value
    assert v_actual.to("liter").m == approx(V.m, rel=1e-9)


def test_mass_normal_volume_round_trip() -> None:
    m = Q(1.0, "kg")

    nv = mass_to_normal_volume(m)
    assert_type(nv, Q[NormalVolume, float])
    # air at 0 °C, 1 atm is roughly 1.276 kg/m³ -> ~0.78 Nm³ per kg
    assert nv.to("Nm³").m == approx(0.78, rel=0.05)

    m_back = mass_from_normal_volume(nv)
    assert_type(m_back, Q[Mass, float])
    assert m_back.to("kg").m == approx(m.m, rel=1e-9)

    # flows map to NormalVolumeFlow / MassFlow
    mf = Q(3600.0, "kg/h")
    nvf = mass_to_normal_volume(mf)
    assert_type(nvf, Q[NormalVolumeFlow, float])

    mf_back = mass_from_normal_volume(nvf)
    assert_type(mf_back, Q[MassFlow, float])
    assert mf_back.to("kg/h").m == approx(mf.m, rel=1e-9)


def test_normal_volume_legacy_plain_inputs() -> None:
    # plain Volume/VolumeFlow inputs (the pre-NormalVolume calling convention) are
    # still accepted at runtime and interpreted as volumes at normal conditions
    P = Q(2.0, "bar")
    T = Q(25.0, "degC")

    nv = Q(1.0, "Nm³")
    v_typed = normal_volume_to_actual_volume(nv, (P, T))

    # a legacy caller passes a plain m³ quantity where NormalVolume is expected
    # (statically a lie, hence the cast; the runtime accepts and interprets it)
    legacy = cast("Q[NormalVolume, float]", Q(1.0, "m³"))

    v_legacy = normal_volume_to_actual_volume(legacy, (P, T))
    assert v_legacy.to("m³").m == approx(v_typed.to("m³").m, rel=1e-9)

    m_typed = mass_from_normal_volume(nv)
    m_legacy = mass_from_normal_volume(legacy)
    assert m_legacy.to("kg").m == approx(m_typed.to("kg").m, rel=1e-9)
