from typing import Any, assert_never, overload

from .misc import isinstance_types
from .units import Quantity
from .utypes import MT, Density, Mass, MassFlow, Volume, VolumeFlow


@overload
def convert_volume_mass(
    inp: Quantity[Mass, MT], rho: Quantity[Density, MT] | Quantity[Density, float] | None = None
) -> Quantity[Volume, MT]: ...


@overload
def convert_volume_mass(
    inp: Quantity[MassFlow, MT], rho: Quantity[Density, MT] | Quantity[Density, float] | None = None
) -> Quantity[VolumeFlow, MT]: ...


@overload
def convert_volume_mass(
    inp: Quantity[Volume, MT], rho: Quantity[Density, MT] | Quantity[Density, float] | None = None
) -> Quantity[Mass, MT]: ...


@overload
def convert_volume_mass(
    inp: Quantity[VolumeFlow, MT], rho: Quantity[Density, MT] | Quantity[Density, float] | None = None
) -> Quantity[MassFlow, MT]: ...


@overload
def convert_volume_mass(
    inp: (Quantity[Mass, MT] | Quantity[MassFlow, MT] | Quantity[Volume, MT] | Quantity[VolumeFlow, MT]),
    rho: Quantity[Density, MT] | Quantity[Density, float] | None = None,
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

    if not isinstance_types(rho, Quantity[Density, Any]):
        assert_never(rho)

    if isinstance_types(inp, Quantity[Mass, Any]):
        return (inp / rho).to_reduced_units().asdim(Volume)
    elif isinstance_types(inp, Quantity[MassFlow, Any]):
        return (inp / rho).to_reduced_units().asdim(VolumeFlow)
    elif isinstance_types(inp, Quantity[Volume, Any]):
        return (inp * rho).to_reduced_units().asdim(Mass)
    elif isinstance_types(inp, Quantity[VolumeFlow, Any]):
        return (inp * rho).to_reduced_units().asdim(MassFlow)
    else:
        assert_never(inp)
