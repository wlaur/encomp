from typing import overload

from .misc import isinstance_types
from .units import Quantity
from .utypes import MT, Density, Mass, MassFlow, Volume, VolumeFlow


@overload
def convert_volume_mass(inp: Quantity[Mass, MT]) -> Quantity[Volume, MT]: ...


@overload
def convert_volume_mass(inp: Quantity[MassFlow, MT]) -> Quantity[VolumeFlow, MT]: ...


@overload
def convert_volume_mass(inp: Quantity[Volume, MT]) -> Quantity[Mass, MT]: ...


@overload
def convert_volume_mass(inp: Quantity[VolumeFlow, MT]) -> Quantity[MassFlow, MT]: ...


@overload
def convert_volume_mass(
    inp: (Quantity[Mass, MT] | Quantity[MassFlow, MT] | Quantity[Volume, MT] | Quantity[VolumeFlow, MT]),
    rho: Quantity[Density, MT] | None = None,
) -> Quantity[Mass, MT] | Quantity[MassFlow, MT] | Quantity[Volume, MT] | Quantity[VolumeFlow, MT]: ...


def convert_volume_mass(
    inp: (Quantity[Mass, MT] | Quantity[MassFlow, MT] | Quantity[Volume, MT] | Quantity[VolumeFlow, MT]),
    rho: Quantity[Density, MT] | Quantity[Density, float] | None = None,
) -> Quantity[Mass, MT] | Quantity[MassFlow, MT] | Quantity[Volume, MT] | Quantity[VolumeFlow, MT]:
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

    if not isinstance_types(rho, Quantity[Density]):
        raise TypeError(f"Incorrect type for rho: {rho}")

    if isinstance_types(inp, Quantity[Mass]) or isinstance_types(inp, Quantity[MassFlow]):
        return (inp / rho).to_reduced_units()
    elif isinstance_types(inp, Quantity[Volume]) or isinstance_types(inp, Quantity[VolumeFlow]):
        return (inp * rho).to_reduced_units()
    else:
        raise TypeError(f"Incorrect input: {inp}")
