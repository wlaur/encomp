from typing import Any, cast, overload

import numpy as np
import polars as pl

from .misc import isinstance_types
from .units import ExpectedDimensionalityError, Quantity
from .utypes import MT, Density, Mass, MassFlow, Numpy1DArray, Volume, VolumeFlow


@overload
def convert_volume_mass(inp: Quantity[Mass, MT], rho: Quantity[Density, Any] | None = None) -> Quantity[Volume, MT]: ...


@overload
def convert_volume_mass(
    inp: Quantity[MassFlow, MT], rho: Quantity[Density, Any] | None = None
) -> Quantity[VolumeFlow, MT]: ...


@overload
def convert_volume_mass(inp: Quantity[Volume, MT], rho: Quantity[Density, Any] | None = None) -> Quantity[Mass, MT]: ...


@overload
def convert_volume_mass(
    inp: Quantity[VolumeFlow, MT], rho: Quantity[Density, Any] | None = None
) -> Quantity[MassFlow, MT]: ...


@overload
def convert_volume_mass(
    inp: (Quantity[Mass, MT] | Quantity[MassFlow, MT] | Quantity[Volume, MT] | Quantity[VolumeFlow, MT]),
    rho: Quantity[Density, Any] | None = None,
) -> Quantity[Mass, MT] | Quantity[MassFlow, MT] | Quantity[Volume, MT] | Quantity[VolumeFlow, MT]: ...


def convert_volume_mass(
    inp: (Quantity[Mass, Any] | Quantity[MassFlow, Any] | Quantity[Volume, Any] | Quantity[VolumeFlow, Any]),
    rho: Quantity[Density, Any] | None = None,
) -> Quantity[Mass, Any] | Quantity[MassFlow, Any] | Quantity[Volume, Any] | Quantity[VolumeFlow, Any]:
    """
    Converts mass to volume or vice versa.

    Parameters
    ----------
    inp : M | V
        Input mass or volume (or flow)
    rho : Quantity[Density, MT], optional
        Density, by default 997 kg/m³

    Returns
    -------
    M | V
        Calculated volume or mass (or flow)
    """

    if rho is None:
        rho = Quantity(997, "kg/m³")

    if not isinstance_types(rho, Quantity[Density, Any]):
        raise ExpectedDimensionalityError(f"rho must be a Quantity[Density], passed {rho!r} ({type(rho).__name__})")

    rho_m = rho.to("kg/m³").m
    if isinstance(rho_m, float):
        if not np.isfinite(rho_m) or rho_m <= 0.0:
            raise ValueError(f"rho must be finite and positive, got {rho!r}")
    elif isinstance(rho_m, np.ndarray):
        rho_arr = cast("Numpy1DArray", rho_m)
        if bool(np.any(~np.isfinite(rho_arr) | (rho_arr <= 0.0))):
            raise ValueError(f"rho must contain only finite positive values, got {rho!r}")
    elif isinstance(rho_m, pl.Series):
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
