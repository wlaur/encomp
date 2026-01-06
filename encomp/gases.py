"""
Functions related to gases: normal volume to mass conversion, compressibility, etc...

.. todo::
    Implement for humid air
"""

from typing import Any, Literal, cast, overload

from .constants import CONSTANTS
from .conversion import convert_volume_mass
from .fluids import Fluid
from .units import Quantity
from .utypes import (
    MT,
    Density,
    Mass,
    MassFlow,
    MolarMass,
    Pressure,
    Temperature,
    Volume,
    VolumeFlow,
)

# TODO: this module should use the NormalVolume and NormalVolumeFlow dimensionalities
# TODO: these type annotations and overloads are not correct

R = CONSTANTS.R


def ideal_gas_density(
    T: Quantity[Temperature, MT],
    P: Quantity[Pressure, MT],
    M: Quantity[MolarMass, MT],
) -> Quantity[Density, MT]:
    """
    Returns the density :math:`\\rho` of an ideal gas.

    For an ideal gas, density is calculated from the ideal gas law:

    .. math::
        \\rho = \\frac{p M}{R T}

    The gas constant :math:`R` is
    :math:`8.3144598 \\; \\frac{\\text{kg} \\,
    \\text{m}^2}{\\text{s}^2 \\, \\text{K} \\, \\text{mol}}`.

    Parameters
    ----------
    T : Quantity[Temperature, MT]
        Temperature of the gas
    P : Quantity[Pressure, MT]
        Absolute pressure of the gas
    M : Quantity[MolarMass, MT]
        Molar mass of the gas

    Returns
    -------
    Quantity[Density, MT]
        Density of the ideal gas at the specified temperature and pressure
    """

    # directly from ideal gas law
    # override the inferred type here since it's sure to be Density
    rho = (P * M) / (R * T.to("K").unknown())

    return rho.to("kg/m³").asdim(Density)


@overload
def convert_gas_volume(
    V1: Quantity[VolumeFlow, Any],
    condition_1: (tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]] | Literal["N", "S"]) = "N",
    condition_2: (tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]] | Literal["N", "S"]) = "N",
    fluid_name: str = "Air",
) -> Quantity[Volume, Any]: ...


@overload
def convert_gas_volume(
    V1: Quantity[Volume, Any],
    condition_1: (tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]] | Literal["N", "S"]) = "N",
    condition_2: (tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]] | Literal["N", "S"]) = "N",
    fluid_name: str = "Air",
) -> Quantity[VolumeFlow, Any]: ...


def convert_gas_volume(
    V1: Quantity[Volume, Any] | Quantity[VolumeFlow, Any],
    condition_1: (tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]] | Literal["N", "S"]) = "N",
    condition_2: (tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]] | Literal["N", "S"]) = "N",
    fluid_name: str = "Air",
) -> Quantity[Volume, Any] | Quantity[VolumeFlow, Any]:
    """
    Converts the volume :math:`V_1` (at :math:`T_1, P_1`) to
    :math:`V_2` (at :math:`T_1, P_1`).
    Uses compressibility factors from CoolProp.

    The values for :math:`T_i, P_i` are passed as a tuple
    using the parameter ``conditions_i``.
    Optionally, the literal 'N' or 'S' can be passed
    to indicate normal and standard conditions.

    Parameters
    ----------
    V1 : Quantity[Volume, Any] | Quantity[VolumeFlow, Any]
        Volume or volume flow :math:`V_1` at condition 1
    condition_1 : tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]] |
                  Literal['N', 'S'], optional
        Pressure and temperature at condition 1, by default 'N'
    condition_2 : tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]] |
                  Literal['N', 'S'], optional
        Pressure and temperature at condition 2, by default 'N'
    fluid_name : str, optional
        CoolProp name of the fluid, by default 'Air'

    Returns
    -------
    Quantity[Volume, Any] | Quantity[VolumeFlow, Any]
        Volume or volume flow :math:`V_2` at condition 2
    """

    n_s_conditions = {
        "N": (
            CONSTANTS.normal_conditions_pressure,
            CONSTANTS.normal_conditions_temperature,
        ),
        "S": (
            CONSTANTS.standard_conditions_pressure,
            CONSTANTS.standard_conditions_temperature,
        ),
    }

    if condition_1 in ("N", "S"):
        condition_1 = n_s_conditions[condition_1]

    if condition_2 in ("N", "S"):
        condition_2 = n_s_conditions[condition_2]

    P1, T1 = condition_1
    P2, T2 = condition_2

    Z1 = Fluid(fluid_name, T=T1, P=P1).Z
    Z2 = Fluid(fluid_name, T=T2, P=P2).Z

    # from ideal gas law PV = nRT: n and R are constant
    # also considers compressibility factor Z

    V2 = V1 * (P1 / P2) * (T2.to("K") / T1.to("K")) * (Z2 / Z1)

    # volume at P2, T2 in same units as V1
    return V2.to(V1.u)  # pyright: ignore[reportArgumentType]


@overload
def mass_to_normal_volume(mass: Quantity[Mass, MT], fluid_name: str = "Air") -> Quantity[Volume, MT]: ...


@overload
def mass_to_normal_volume(mass: Quantity[MassFlow, MT], fluid_name: str = "Air") -> Quantity[VolumeFlow, MT]: ...


def mass_to_normal_volume(
    mass: Quantity[Mass, MT] | Quantity[MassFlow, MT], fluid_name: str = "Air"
) -> Quantity[Volume, MT] | Quantity[VolumeFlow, MT]:
    """
    Convert mass to normal volume.

    Parameters
    ----------
    mass : Quantity[Mass, MT] | Quantity[MassFlow, MT]
        Input mass or mass flow
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Quantity[Volume, MT] | Quantity[VolumeFlow, MT]
        Corresponding normal volume or normal volume flow
    """

    rho = Fluid(
        fluid_name,
        P=CONSTANTS.normal_conditions_pressure,
        T=CONSTANTS.normal_conditions_temperature,
    ).D

    ret = convert_volume_mass(mass, rho=rho)

    return cast("Quantity[Volume, MT] | Quantity[VolumeFlow, MT]", ret)


@overload
def mass_to_actual_volume(
    mass: Quantity[Mass, MT],
    condition: tuple[Quantity[Pressure, MT], Quantity[Temperature, MT]],
    fluid_name: str = "Air",
) -> Quantity[Volume, MT]: ...


@overload
def mass_to_actual_volume(
    mass: Quantity[MassFlow, MT],
    condition: tuple[Quantity[Pressure, MT], Quantity[Temperature, MT]],
    fluid_name: str = "Air",
) -> Quantity[VolumeFlow, MT]: ...


def mass_to_actual_volume(
    mass: Quantity[Mass, MT] | Quantity[MassFlow, MT],
    condition: tuple[Quantity[Pressure, MT], Quantity[Temperature, MT]],
    fluid_name: str = "Air",
) -> Quantity[Volume, MT] | Quantity[VolumeFlow, MT]:
    """
    Convert mass to actual volume.

    Parameters
    ----------
    mass : Quantity[Mass, MT] | Quantity[MassFlow, MT]
        Input mass or mass flow
    condition : tuple[Quantity[Pressure, MT], Quantity[Temperature, MT]]
        Condition at which to calculate the actual volume
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Quantity[Volume, MT] | Quantity[VolumeFlow, MT]
        Corresponding actual volume or actual volume flow
    """

    rho = Fluid(fluid_name, P=condition[0], T=condition[1]).D

    ret = convert_volume_mass(mass, rho=rho)

    return cast("Quantity[Volume, MT] | Quantity[VolumeFlow, MT]", ret)


@overload
def mass_from_normal_volume(volume: Quantity[Volume, MT], fluid_name: str = "Air") -> Quantity[Mass, MT]: ...


@overload
def mass_from_normal_volume(volume: Quantity[VolumeFlow, MT], fluid_name: str = "Air") -> Quantity[MassFlow, MT]: ...


def mass_from_normal_volume(
    volume: Quantity[Volume, MT] | Quantity[VolumeFlow, MT], fluid_name: str = "Air"
) -> Quantity[Mass, MT] | Quantity[MassFlow, MT]:
    """
     Convert normal volume to mass.

     Parameters
     ----------
     volume : Quantity[Volume, MT] | Quantity[VolumeFlow, MT]
         Input normal volume or normal volume flow
     fluid_name : str, optional
         Name of the fluid, by default 'Air'

     Returns
     -------
    Quantity[Mass, MT] | Quantity[MassFlow, MT]
         Corresponding mass or mass flow
    """

    rho = Fluid(
        fluid_name,
        P=CONSTANTS.normal_conditions_pressure,
        T=CONSTANTS.normal_conditions_temperature,
    ).D

    ret = convert_volume_mass(volume, rho=rho)

    return cast("Quantity[Mass, MT] | Quantity[MassFlow, MT]", ret)


@overload
def mass_from_actual_volume(
    volume: Quantity[Volume, MT],
    condition: tuple[Quantity[Pressure, MT], Quantity[Temperature, MT]],
    fluid_name: str = "Air",
) -> Quantity[Mass, MT]: ...


@overload
def mass_from_actual_volume(
    volume: Quantity[VolumeFlow, MT],
    condition: tuple[Quantity[Pressure, MT], Quantity[Temperature, MT]],
    fluid_name: str = "Air",
) -> Quantity[MassFlow, MT]: ...


def mass_from_actual_volume(
    volume: Quantity[Volume, MT] | Quantity[VolumeFlow, MT],
    condition: tuple[Quantity[Pressure, MT], Quantity[Temperature, MT]],
    fluid_name: str = "Air",
) -> Quantity[Mass, MT] | Quantity[MassFlow, MT]:
    """
    Convert actual volume to mass.

    Parameters
    ----------
    volume : Quantity[Volume, MT] | Quantity[VolumeFlow, MT]
        Input actual volume or actual volume flow
    condition : tuple[Quantity[Pressure, MT], Quantity[Temperature, MT]]
        Condition at which to calculate the mass
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Quantity[Mass, MT] | Quantity[MassFlow, MT]
        Corresponding mass or mass flow
    """

    rho = Fluid(fluid_name, P=condition[0], T=condition[1]).D

    ret = convert_volume_mass(volume, rho=rho)

    return cast("Quantity[Mass, MT] | Quantity[MassFlow, MT]", ret)


@overload
def actual_volume_to_normal_volume(
    volume: Quantity[Volume, Any],
    condition: tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]],
    fluid_name: str = "Air",
) -> Quantity[Volume, Any]: ...


@overload
def actual_volume_to_normal_volume(
    volume: Quantity[VolumeFlow, Any],
    condition: tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]],
    fluid_name: str = "Air",
) -> Quantity[VolumeFlow, Any]: ...


def actual_volume_to_normal_volume(
    volume: Quantity[Volume, Any] | Quantity[VolumeFlow, Any],
    condition: tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]],
    fluid_name: str = "Air",
) -> Quantity[Volume, Any] | Quantity[VolumeFlow, Any]:
    """
    Convert actual volume to normal volume.

    Parameters
    ----------
    volume : Quantity[Volume, Any] | Quantity[VolumeFlow, Any]
        Input actual volume or actual volume flow
    condition : tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]]
        Condition at which to calculate the normal volume
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Quantity[Volume, Any] | Quantity[VolumeFlow, Any]
        Corresponding normal volume or normal volume flow
    """

    return convert_gas_volume(volume, condition_1=condition, condition_2="N", fluid_name=fluid_name)


@overload
def normal_volume_to_actual_volume(
    volume: Quantity[Volume, Any],
    condition: tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]],
    fluid_name: str = "Air",
) -> Quantity[Volume, Any]: ...


@overload
def normal_volume_to_actual_volume(
    volume: Quantity[VolumeFlow, Any],
    condition: tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]],
    fluid_name: str = "Air",
) -> Quantity[VolumeFlow, Any]: ...


def normal_volume_to_actual_volume(
    volume: Quantity[Volume, Any] | Quantity[VolumeFlow, Any],
    condition: tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]],
    fluid_name: str = "Air",
) -> Quantity[Volume, Any] | Quantity[VolumeFlow, Any]:
    """
    Convert normal volume to actual volume.

    Parameters
    ----------
    volume : Quantity[Volume, Any] | Quantity[VolumeFlow, Any]
        Input normal volume or normal volume flow
    condition : tuple[Quantity[Pressure, Any], Quantity[Temperature, Any]]
        Condition at which to calculate the actual volume
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Quantity[Volume, Any] | Quantity[VolumeFlow, Any]
        Corresponding actual volume or actual volume flow
    """

    return convert_gas_volume(volume, condition_1="N", condition_2=condition, fluid_name=fluid_name)
