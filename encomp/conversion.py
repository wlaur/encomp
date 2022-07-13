"""
Functions related to converting quantities.
"""

from typing import Optional, Union, overload

from encomp.units import Quantity
from encomp.utypes import Mass, MassFlow, Density, Volume, VolumeFlow


@overload
def convert_volume_mass(inp: Quantity[Mass],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[Volume]:
    ...


@overload
def convert_volume_mass(inp: Quantity[MassFlow],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[VolumeFlow]:
    ...


@overload
def convert_volume_mass(inp: Quantity[Volume],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[Mass]:
    ...


@overload
def convert_volume_mass(inp: Quantity[VolumeFlow],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[MassFlow]:
    ...


@overload
def convert_volume_mass(inp: Quantity,
                        rho: Optional[Quantity[Density]] = None
                        ) -> Union[Quantity[Mass],
                                   Quantity[MassFlow],
                                   Quantity[Volume],
                                   Quantity[VolumeFlow]
                                   ]:
    ...


def convert_volume_mass(inp, rho=None):
    """
    Converts mass to volume or vice versa.

    Parameters
    ----------
    inp : Union[M, V]
        Input mass or volume (or flow)
    rho : Quantity[Density], optional
        Density, by default 997 kg/m³

    Returns
    -------
    Union[V, M]
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
