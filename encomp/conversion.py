"""
Functions related to converting quantities.
"""

from typing import overload

from .units import Quantity
from .utypes import Mass, MassFlow, Density, Volume, VolumeFlow


@overload
def convert_volume_mass(inp: Quantity[Mass],
                        rho: Quantity[Density] | None = None) -> Quantity[Volume]:
    ...


@overload
def convert_volume_mass(inp: Quantity[MassFlow],
                        rho: Quantity[Density] | None = None) -> Quantity[VolumeFlow]:
    ...


@overload
def convert_volume_mass(inp: Quantity[Volume],
                        rho: Quantity[Density] | None = None) -> Quantity[Mass]:
    ...


@overload
def convert_volume_mass(inp: Quantity[VolumeFlow],
                        rho: Quantity[Density] | None = None) -> Quantity[MassFlow]:
    ...


@overload
def convert_volume_mass(
    inp: Quantity,
    rho: Quantity[Density] | None = None
) -> Quantity[Mass] | Quantity[MassFlow] | Quantity[Volume] | Quantity[VolumeFlow]:
    ...


def convert_volume_mass(inp, rho=None):
    """
    Converts mass to volume or vice versa.

    Parameters
    ----------
    inp : M | V
        Input mass or volume (or flow)
    rho : Quantity[Density], optional
        Density, by default 997 kg/m³

    Returns
    -------
    M | V
        Calculated volume or mass (or flow)
    """

    if rho is None:
        rho = Quantity[Density](997, 'kg/m³')

    if not isinstance(rho, Quantity[Density]):  # type: ignore
        raise TypeError(
            f'Incorrect type for rho: {rho}'
        )

    if isinstance(inp, (Quantity[Mass], Quantity[MassFlow])):  # type: ignore
        return (inp / rho).to_reduced_units()

    elif isinstance(inp, (Quantity[Volume], Quantity[VolumeFlow])):  # type: ignore
        return (inp * rho).to_reduced_units()

    else:
        raise TypeError(
            f'Incorrect input: {inp}'
        )
