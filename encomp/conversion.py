"""
Functions related to converting quantities.
"""

from typing import Any, overload

from .units import Quantity
from .utypes import Mass, MassFlow, Density, Volume, VolumeFlow


@overload
def convert_volume_mass(inp: Quantity[Mass, Any]) -> Quantity[Volume, Any]:
    ...


@overload
def convert_volume_mass(inp: Quantity[MassFlow, Any]) -> Quantity[VolumeFlow, Any]:
    ...


@overload
def convert_volume_mass(inp: Quantity[Volume, Any]) -> Quantity[Mass, Any]:
    ...


@overload
def convert_volume_mass(inp: Quantity[VolumeFlow, Any]) -> Quantity[MassFlow, Any]:
    ...


@overload
def convert_volume_mass(
    inp: Quantity,
    rho: Quantity[Density, Any] | None = None
) -> Quantity[Mass, Any] | Quantity[MassFlow, Any] | Quantity[Volume, Any] | Quantity[VolumeFlow, Any]:
    ...


def convert_volume_mass(
    inp: Quantity[Mass, Any] | Quantity[MassFlow, Any] | Quantity[Volume, Any] | Quantity[VolumeFlow, Any],
        rho: Quantity[Density, Any] | None = None
) -> Quantity[Mass, Any] | Quantity[MassFlow, Any] | Quantity[Volume, Any] | Quantity[VolumeFlow, Any]:
    """
    Converts mass to volume or vice versa.

    Parameters
    ----------
    inp : M | V
        Input mass or volume (or flow)
    rho : Quantity[Density, Any], optional
        Density, by default 997 kg/m³

    Returns
    -------
    M | V
        Calculated volume or mass (or flow)
    """

    if rho is None:
        rho = Quantity[Density, Any](997, 'kg/m³')

    if not isinstance(rho, Quantity[Density, Any]):  # type: ignore
        raise TypeError(
            f'Incorrect type for rho: {rho}'
        )

    if isinstance(inp, (Quantity[Mass, Any], Quantity[MassFlow, Any])):  # type: ignore
        return (inp / rho).to_reduced_units()

    elif isinstance(inp, (Quantity[Volume, Any], Quantity[VolumeFlow, Any])):  # type: ignore
        return (inp * rho).to_reduced_units()

    else:
        raise TypeError(
            f'Incorrect input: {inp}'
        )
