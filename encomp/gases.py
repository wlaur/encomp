"""
Functions related to gases: normal volume to mass conversion, compressibility, etc...

.. todo::
    Implement for humid air also
"""

from typing import Union, Literal, cast

from encomp.constants import CONSTANTS
from encomp.units import Quantity
from encomp.conversion import convert_volume_mass
from encomp.utypes import (Mass,
                           MassFlow,
                           Volume,
                           VolumeFlow,
                           Temperature,
                           MolarMass,
                           Pressure,
                           Density)
from encomp.fluids import Fluid


R = CONSTANTS.R


def ideal_gas_density(T: Quantity[Temperature],
                      P: Quantity[Pressure],
                      M: Quantity[MolarMass]) -> Quantity[Density]:
    """
    Returns the density :math:`\\rho` of an ideal gas.

    For an ideal gas, density is calculated from the ideal gas law:

    .. math::
        \\rho = \\frac{p M}{R T}

    The gas constant :math:`R` is
    :math:`8.3144598 \\; \\frac{\\text{kg} \\, \\text{m}^2}{\\text{s}^2 \\, \\text{K} \\, \\text{mol}}`.

    Parameters
    ----------
    T : Quantity[Temperature]
        Temperature of the gas
    P : Quantity[Pressure]
        Absolute pressure of the gas
    M : Quantity[MolarMass]
        Molar mass of the gas

    Returns
    -------
    Quantity[Density]
        Density of the ideal gas at the specified temperature and pressure
    """

    # directly from ideal gas law
    # override the inferred type here since it's sure to be Density
    rho = cast(Quantity[Density], (P * M) / (R * T.to('K')))

    return rho.to('kg/m³')


def convert_gas_volume(
        V1: Union[Quantity[Volume], Quantity[VolumeFlow]],
        condition_1: Union[tuple[Quantity[Pressure], Quantity[Temperature]],
                           Literal['N', 'S']] = 'N',
        condition_2: Union[tuple[Quantity[Pressure], Quantity[Temperature]],
                           Literal['N', 'S']] = 'N',
        fluid_name: str = 'Air'
) -> Union[Quantity[Volume], Quantity[VolumeFlow]]:
    """
    Converts the volume :math:`V_1` (at :math:`T_1, P_1`) to
    :math:`V_2` (at :math:`T_1, P_1`).
    Uses compressibility factors from CoolProp.

    The values for :math:`T_i, P_i` are passed as a tuple using the parameter ``conditions_i``.
    Optionally, the literal 'N' or 'S' can be passed to indicate normal and standard conditions.

    Parameters
    ----------
    V1 : Union[Quantity[Volume], Quantity[VolumeFlow]]
        Volume or volume flow :math:`V_1` at condition 1
    condition_1 : Union[tuple[Quantity[Pressure], Quantity[Temperature]], Literal['N', 'S']], optional
        Pressure and temperature at condition 1, by default 'N'
    condition_2 : Union[tuple[Quantity[Pressure], Quantity[Temperature]], Literal['N', 'S']], optional
        Pressure and temperature at condition 2, by default 'N'
    fluid_name : str, optional
        CoolProp name of the fluid, by default 'Air'

    Returns
    -------
    Union[Quantity[Volume], Quantity[VolumeFlow]]
        Volume or volume flow :math:`V_2` at condition 2
    """

    n_s_conditions = {
        'N': (CONSTANTS.normal_conditions_pressure,
              CONSTANTS.normal_conditions_temperature),
        'S': (CONSTANTS.standard_conditions_pressure,
              CONSTANTS.standard_conditions_temperature)
    }

    if condition_1 == 'N' or condition_1 == 'S':
        condition_1 = n_s_conditions[condition_1]

    if condition_2 == 'N' or condition_2 == 'S':
        condition_2 = n_s_conditions[condition_2]

    if isinstance(condition_1, tuple):
        P1, T1 = condition_1
    else:
        raise ValueError(f'Incorrect value for condition 1: {condition_1}')

    if isinstance(condition_2, tuple):
        P2, T2 = condition_2
    else:
        raise ValueError(f'Incorrect value for condition 2: {condition_2}')

    Z1 = Fluid(fluid_name, T=T1, P=P1).Z
    Z2 = Fluid(fluid_name, T=T2, P=P2).Z

    # from ideal gas law PV = nRT: n and R are constant
    # also considers compressibility factor Z

    # TODO: issue with mypy for Z2 / Z1 (seems like a bug)
    V2 = V1 * (P1 / P2) * (T2.to('K') / T1.to('K')) * (Z2 / Z1)  # type: ignore

    # volume at P2, T2 in same units as V1
    return V2.to(V1.u)


def mass_to_normal_volume(mass: Union[Quantity[Mass],
                                      Quantity[MassFlow]],
                          fluid_name: str = 'Air') -> Union[Quantity[Volume],
                                                            Quantity[VolumeFlow]]:
    """
    Convert mass to normal volume.

    Parameters
    ----------
    mass : Union[Quantity[Mass], Quantity[MassFlow]]
        Input mass or mass flow
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Union[Quantity[Volume], Quantity[VolumeFlow]]
        Corresponding normal volume or normal volume flow
    """

    rho = Fluid(fluid_name,
                P=CONSTANTS.normal_conditions_pressure,
                T=CONSTANTS.normal_conditions_temperature).D

    return convert_volume_mass(mass, rho=rho)  # type: ignore


def mass_to_actual_volume(mass: Union[Quantity[Mass],
                                      Quantity[MassFlow]],
                          condition: tuple[Quantity[Pressure], Quantity[Temperature]],
                          fluid_name: str = 'Air') -> Union[Quantity[Volume],
                                                            Quantity[VolumeFlow]]:
    """
    Convert mass to actual volume.

    Parameters
    ----------
    mass : Union[Quantity[Mass], Quantity[MassFlow]]
        Input mass or mass flow
    condition : tuple[Quantity[Pressure], Quantity[Temperature]]
        Condition at which to calculate the actual volume
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Union[Quantity[Volume], Quantity[VolumeFlow]]
        Corresponding actual volume or actual volume flow
    """

    rho = Fluid(fluid_name,
                P=condition[0],
                T=condition[1]).D

    return convert_volume_mass(mass, rho=rho)  # type: ignore


def mass_from_normal_volume(volume: Union[Quantity[Volume],
                                          Quantity[VolumeFlow]],
                            fluid_name: str = 'Air') -> Union[Quantity[Mass],
                                                              Quantity[MassFlow]]:
    """
    Convert normal volume to mass.

    Parameters
    ----------
    volume : Union[Quantity[Volume], Quantity[VolumeFlow]]
        Input normal volume or normal volume flow
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Union[Quantity[Mass], Quantity[MassFlow]]
        Corresponding mass or mass flow
    """

    rho = Fluid(fluid_name,
                P=CONSTANTS.normal_conditions_pressure,
                T=CONSTANTS.normal_conditions_temperature).D

    return convert_volume_mass(volume, rho=rho)  # type: ignore


def mass_from_actual_volume(volume: Union[Quantity[Volume],
                                          Quantity[VolumeFlow]],
                            condition: tuple[Quantity[Pressure], Quantity[Temperature]],
                            fluid_name: str = 'Air') -> Union[Quantity[Mass],
                                                              Quantity[MassFlow]]:
    """
    Convert actual volume to mass.

    Parameters
    ----------
    volume : Union[Quantity[Volume], Quantity[VolumeFlow]]
        Input actual volume or actual volume flow
    condition : tuple[Quantity[Pressure], Quantity[Temperature]]
        Condition at which to calculate the mass
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Union[Quantity[Mass], Quantity[MassFlow]]
        Corresponding mass or mass flow
    """

    rho = Fluid(fluid_name,
                P=condition[0],
                T=condition[1]).D

    return convert_volume_mass(volume, rho=rho)  # type: ignore


def actual_volume_to_normal_volume(volume: Union[Quantity[Volume],
                                                 Quantity[VolumeFlow]],
                                   condition: tuple[Quantity[Pressure], Quantity[Temperature]],
                                   fluid_name: str = 'Air') -> Union[Quantity[Volume],
                                                                     Quantity[VolumeFlow]]:
    """
    Convert actual volume to normal volume.

    Parameters
    ----------
    volume : Union[Quantity[Volume], Quantity[VolumeFlow]]
        Input actual volume or actual volume flow
    condition : tuple[Quantity[Pressure], Quantity[Temperature]]
        Condition at which to calculate the normal volume
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Union[Quantity[Volume], Quantity[VolumeFlow]]
        Corresponding normal volume or normal volume flow
    """

    return convert_gas_volume(volume,
                              condition_1=condition,
                              condition_2='N',
                              fluid_name=fluid_name)


def normal_volume_to_actual_volume(volume: Union[Quantity[Volume],
                                                 Quantity[VolumeFlow]],
                                   condition: tuple[Quantity[Pressure], Quantity[Temperature]],
                                   fluid_name: str = 'Air') -> Union[Quantity[Volume],
                                                                     Quantity[VolumeFlow]]:
    """
    Convert normal volume to actual volume.

    Parameters
    ----------
    volume : Union[Quantity[Volume], Quantity[VolumeFlow]]
        Input normal volume or normal volume flow
    condition : tuple[Quantity[Pressure], Quantity[Temperature]]
        Condition at which to calculate the actual volume
    fluid_name : str, optional
        Name of the fluid, by default 'Air'

    Returns
    -------
    Union[Quantity[Volume], Quantity[VolumeFlow]]
        Corresponding actual volume or actual volume flow
    """

    return convert_gas_volume(volume,
                              condition_1='N',
                              condition_2=condition,
                              fluid_name=fluid_name)
