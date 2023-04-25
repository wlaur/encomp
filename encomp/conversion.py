"""
Functions related to converting quantities.
"""

from typing import overload

from .units import Quantity
from .utypes import MT, Density, Mass, MassFlow, Volume, VolumeFlow


@overload
def convert_volume_mass(inp: Quantity[Mass, MT]) -> Quantity[Volume, MT]:
    ...


@overload
def convert_volume_mass(inp: Quantity[MassFlow, MT]) -> Quantity[VolumeFlow, MT]:
    ...


@overload
def convert_volume_mass(inp: Quantity[Volume, MT]) -> Quantity[Mass, MT]:
    ...


@overload
def convert_volume_mass(inp: Quantity[VolumeFlow, MT]) -> Quantity[MassFlow, MT]:
    ...


@overload
def convert_volume_mass(
    inp: Quantity, rho: Quantity[Density, MT] | None = None
) -> (
    Quantity[Mass, MT]
    | Quantity[MassFlow, MT]
    | Quantity[Volume, MT]
    | Quantity[VolumeFlow, MT]
):
    ...


def convert_volume_mass(
    inp: Quantity[Mass, MT]
    | Quantity[MassFlow, MT]
    | Quantity[Volume, MT]
    | Quantity[VolumeFlow, MT],
    rho: Quantity[Density, MT] | None = None,
) -> (
    Quantity[Mass, MT]
    | Quantity[MassFlow, MT]
    | Quantity[Volume, MT]
    | Quantity[VolumeFlow, MT]
):
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
        rho = Quantity[Density, MT](997, "kg/m³")

    if not isinstance(rho, Quantity[Density]):  # type: ignore
        raise TypeError(f"Incorrect type for rho: {rho}")

    if isinstance(inp, (Quantity[Mass], Quantity[MassFlow])):  # type: ignore
        return (inp / rho).to_reduced_units()

    elif isinstance(inp, (Quantity[Volume], Quantity[VolumeFlow])):  # type: ignore
        return (inp * rho).to_reduced_units()

    else:
        raise TypeError(f"Incorrect input: {inp}")


a = convert_volume_mass(Quantity([25.2, 2.2], "kg"))
