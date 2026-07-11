"""Conversions between related engineering quantity dimensionalities."""

from typing import Any, cast, overload

import numpy as np
import polars as pl

from .misc import isinstance_types
from .units import ExpectedDimensionalityError, Quantity
from .utypes import MT, Density, Mass, MassFlow, Numpy1DArray, Volume, VolumeFlow

__all__ = ["convert_volume_mass"]


@overload
def convert_volume_mass(inp: Quantity[Mass, MT], rho: Quantity[Density, Any]) -> Quantity[Volume, MT]: ...


@overload
def convert_volume_mass(inp: Quantity[MassFlow, MT], rho: Quantity[Density, Any]) -> Quantity[VolumeFlow, MT]: ...


@overload
def convert_volume_mass(inp: Quantity[Volume, MT], rho: Quantity[Density, Any]) -> Quantity[Mass, MT]: ...


@overload
def convert_volume_mass(inp: Quantity[VolumeFlow, MT], rho: Quantity[Density, Any]) -> Quantity[MassFlow, MT]: ...


@overload
def convert_volume_mass(
    inp: (Quantity[Mass, MT] | Quantity[MassFlow, MT] | Quantity[Volume, MT] | Quantity[VolumeFlow, MT]),
    rho: Quantity[Density, Any],
) -> Quantity[Mass, MT] | Quantity[MassFlow, MT] | Quantity[Volume, MT] | Quantity[VolumeFlow, MT]: ...


def convert_volume_mass(
    inp: (Quantity[Mass, Any] | Quantity[MassFlow, Any] | Quantity[Volume, Any] | Quantity[VolumeFlow, Any]),
    rho: Quantity[Density, Any],
) -> Quantity[Mass, Any] | Quantity[MassFlow, Any] | Quantity[Volume, Any] | Quantity[VolumeFlow, Any]:
    """
    Converts mass to volume or vice versa.

    Parameters
    ----------
    inp : M | V
        Input mass or volume (or flow)
    rho : Quantity[Density, Any]
        Density of the substance. A *missing* density is allowed and yields a missing
        result at that position, exactly as a missing ``inp`` does: ``NaN`` for float and
        numpy magnitudes, ``null`` for a Polars Series. A Polars result carries null
        (never NaN), so with a Polars ``inp`` a NaN density is rejected -- spell the
        missing density as a null in a ``pl.Series``. Every *present* value must be
        finite and strictly positive. Polars Expr inputs are deferred and cannot be
        data-validated here.

    Returns
    -------
    M | V
        Calculated volume or mass (or flow)
    """

    if not isinstance_types(rho, Quantity[Density, Any]):
        raise ExpectedDimensionalityError(f"rho must be a Quantity[Density], passed {rho!r} ({type(rho).__name__})")

    # a NaN density means "missing" for float/numpy data, but a polars result carries null,
    # never NaN -- and a float/ndarray density has no null spelling. A NaN there would flow
    # into the Series/Expr result as NaN, so it is rejected when the input is polars
    inp_is_polars = isinstance(inp.m, (pl.Series, pl.Expr))
    nan_rho_message = (
        "rho must not be NaN when inp has a Polars magnitude; "
        "pass rho as a pl.Series with null for a missing density, got {rho!r}"
    )

    rho_m = rho.to("kg/m³").m
    if isinstance(rho_m, float):
        if np.isnan(rho_m):
            if inp_is_polars:
                raise ValueError(nan_rho_message.format(rho=rho))
        elif not np.isfinite(rho_m) or rho_m <= 0.0:
            raise ValueError(f"rho must be finite and positive, got {rho!r}")
    elif isinstance(rho_m, np.ndarray):
        rho_arr = cast("Numpy1DArray", rho_m)
        present = ~np.isnan(rho_arr)
        if bool(np.any(present & (~np.isfinite(rho_arr) | (rho_arr <= 0.0)))):
            raise ValueError(f"rho must contain only finite positive values, got {rho!r}")
        if inp_is_polars and not bool(present.all()):
            raise ValueError(nan_rho_message.format(rho=rho))
    elif isinstance(rho_m, pl.Series):
        # null is the missing sentinel in the polars world and propagates through the
        # arithmetic below; a NaN there would leak into the result, where polars magnitudes
        # never carry one
        if rho_m.is_nan().any():
            raise ValueError(f"rho must not contain NaN, use null for a missing density, got {rho!r}")

        # Series.all() ignores nulls, which is what makes the missing values pass this guard
        valid = rho_m.is_finite() & (rho_m > 0.0)
        if not valid.all():
            raise ValueError(f"rho must contain only finite positive values, got {rho!r}")

    if isinstance_types(inp, Quantity[Mass, Any]):
        return (inp / rho).to_reduced_units().asdim(Volume)
    elif isinstance_types(inp, Quantity[MassFlow, Any]):
        return (inp / rho).to_reduced_units().asdim(VolumeFlow)
    elif isinstance_types(inp, Quantity[Volume, Any]):
        return (inp * rho).to_reduced_units().asdim(Mass)
    elif isinstance_types(inp, Quantity[VolumeFlow, Any]):
        return (inp * rho).to_reduced_units().asdim(MassFlow)
    else:
        raise ExpectedDimensionalityError(
            "inp must be a Quantity with dimensionality Mass, MassFlow, Volume or VolumeFlow, "
            f"passed {inp!r} ({type(inp).__name__})"
        )
