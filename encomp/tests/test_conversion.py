from typing import Any, assert_type, cast

import numpy as np
import polars as pl
import pytest

from ..conversion import convert_volume_mass
from ..units import ExpectedDimensionalityError
from ..units import Quantity as Q
from ..utypes import Numpy1DArray, Volume, VolumeFlow


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


WATER = Q(997.0, "kg/m³")


def test_convert_volume_mass() -> None:
    mf = Q(25, "kg/s")

    assert_type(convert_volume_mass(mf, WATER), Q[VolumeFlow, float])

    mf_list = Q([25.5, 25.34], "kg/s")
    assert_type(convert_volume_mass(mf_list, WATER), Q[VolumeFlow, Numpy1DArray])

    m = Q(25, "ton")

    assert_type(convert_volume_mass(m, WATER), Q[Volume, float])

    # the density is required: there is no substance to assume
    with pytest.raises(TypeError, match="rho"):
        cast(Any, convert_volume_mass)(mf)

    # wrong dimensionality raises a proper unit error naming the argument,
    # not an internal AssertionError
    with pytest.raises(ExpectedDimensionalityError, match="rho"):
        convert_volume_mass(mf, rho=cast(Any, Q(25, "bar")))

    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(mf, rho=Q(0, "kg/m³"))

    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(mf, rho=Q(-1, "kg/m³"))

    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(mf_list, rho=Q([997.0, 0.0], "kg/m³"))

    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(mf, rho=Q(float("inf"), "kg/m³"))

    with pytest.raises(ExpectedDimensionalityError, match="inp"):
        convert_volume_mass(cast(Any, Q(25, "m/s")), WATER)


def test_missing_density_propagates_like_a_missing_input() -> None:
    # a missing density yields a missing result at that position, exactly as a missing `inp`
    # does -- NaN for float/numpy magnitudes, null for a Polars Series
    assert np.isnan(convert_volume_mass(Q(2.0, "kg"), Q(float("nan"), "kg/m³")).to("m³").m)

    volume = convert_volume_mass(Q(np.array([2.0, 2.0]), "kg"), Q(np.array([1000.0, np.nan]), "kg/m³"))
    magnitude = volume.to("m³").m
    assert magnitude[0] == pytest.approx(0.002)
    assert np.isnan(magnitude[1])

    series = convert_volume_mass(Q(pl.Series([2.0, 2.0]), "kg"), Q(pl.Series([1000.0, None]), "kg/m³"))
    assert series.to("m³").m.to_list() == [0.002, None]

    # a present value is still validated
    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(Q(np.array([2.0, 2.0]), "kg"), Q(np.array([1000.0, np.inf]), "kg/m³"))

    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(Q(np.array([2.0, 2.0]), "kg"), Q(np.array([np.nan, 0.0]), "kg/m³"))


def test_convert_volume_mass_polars_series_density() -> None:
    # a Polars-Series density is validated element-wise (finite and strictly positive),
    # mirroring the scalar/ndarray guards
    mass = Q(pl.Series([2.0, 4.0]), "kg")

    vol = convert_volume_mass(mass, rho=Q(pl.Series([1000.0, 2000.0]), "kg/m³"))
    assert_type(vol, Q[Volume, pl.Series])
    assert vol.to("m³").m.to_list() == [0.002, 0.002]

    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(mass, rho=Q(pl.Series([1000.0, -1.0]), "kg/m³"))

    # null is the polars missing sentinel, so a NaN would leak into a result that never
    # carries one
    with pytest.raises(ValueError, match="must not contain NaN"):
        convert_volume_mass(mass, rho=Q(pl.Series([1000.0, float("nan")]), "kg/m³"))

    # a null density is missing, not invalid: it propagates
    nulled = convert_volume_mass(mass, rho=Q(pl.Series([1000.0, None]), "kg/m³"))
    assert nulled.to("m³").m.to_list() == [0.002, None]


def test_nan_density_is_rejected_for_polars_inputs() -> None:
    # a polars result carries null, never NaN, and a float/ndarray density has no null
    # spelling -- so a NaN density cannot express "missing" when the input is polars
    mass_series = Q(pl.Series([2.0, 4.0]), "kg")
    mass_expr = Q(pl.col("m"), "kg")

    with pytest.raises(ValueError, match="null for a missing density"):
        convert_volume_mass(mass_series, rho=Q(float("nan"), "kg/m³"))

    with pytest.raises(ValueError, match="null for a missing density"):
        convert_volume_mass(mass_expr, rho=Q(float("nan"), "kg/m³"))

    with pytest.raises(ValueError, match="null for a missing density"):
        convert_volume_mass(mass_series, rho=Q(np.array([1000.0, np.nan]), "kg/m³"))

    # the same missing density is fine for float/numpy inputs, where NaN IS the sentinel
    assert np.isnan(convert_volume_mass(Q(2.0, "kg"), rho=Q(float("nan"), "kg/m³")).m)
