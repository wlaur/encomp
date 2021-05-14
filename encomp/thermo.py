"""
Functions relating to thermodynamics.
"""

from typing import Union
from scipy.optimize import fsolve

from encomp.misc import isinstance_types
from encomp.units import Quantity
from encomp.constants import CONSTANTS
from encomp.utypes import (Mass,
                           MassFlow,
                           Temperature,
                           Energy,
                           Power,
                           Length,
                           Area,
                           Temperature)


# these dimensionalities are not unique to the names here, these
# should only be defined where context makes it clear what physical
# property they describe
HeatCapacity = Energy / Mass / Temperature
ThermalConductivity = Power / Length / Temperature
HeatTransferCoefficient = Power / Area / Temperature


SIGMA = CONSTANTS.SIGMA
DEFAULT_CP = Quantity[HeatCapacity](4.18, 'kJ/kg/K')


def heat_balance(*args: Union[Quantity[Mass],
                              Quantity[MassFlow],
                              Quantity[Energy],
                              Quantity[Power],
                              Quantity[Temperature]],
                 cp: Quantity[HeatCapacity] = DEFAULT_CP) -> Union[Quantity[Mass],
                                                                   Quantity[MassFlow],
                                                                   Quantity[Energy],
                                                                   Quantity[Power],
                                                                   Quantity[Temperature]]:
    """
    Solves the heat balance equation

    .. math::
        \\dot{Q}_h = C_p \\cdot \\dot{m} \\cdot \\Delta T

    for the 3:rd unknown variable.

    Parameters
    ----------
    args : Quantity
        The two known variables in the heat balance equation:
        mass, mass flow, energy, power or temperature difference
    cp : Quantity[HeatCapacity], optional
        Heat capacity, by default 4.18 kg/kJ/K (water)

    Returns
    -------
    Quantity
        The third unknown variable
    """

    # this function might be too general to be expressed succinctly using type annotations

    if len(args) != 2:
        raise ValueError(
            'Must pass exactly two parameters out of dT, Q_h and m')

    params = {
        'm':  (Union[Quantity[Mass], Quantity[MassFlow]], ('kg', 'kg/s')),
        'dT': (Quantity[Temperature], 'delta_degC'),
        'Q_h': (Union[Quantity[Energy], Quantity[Power]], ('kJ', 'kW'))
    }

    vals = {}
    units = {a: b[1] for a, b in params.items()}

    for a in args:
        for param_name, tp in params.items():
            if isinstance_types(a, tp[0]):

                if param_name == 'dT' and not a._ok_for_muldiv():
                    raise ValueError(f'Cannot pass temperature difference using '
                                     f'degree unit {a.u}, convert to delta_deg')

                vals[param_name] = (a)

    # convert the temperature to a delta_T unit
    if 'dT' in vals:
        vals['dT'] = vals['dT'].to('delta_degC')

    # whether the calculation is per unit time or amount of mass / energy
    per_time = any(isinstance_types(
        a, Union[Quantity[MassFlow], Quantity[Power]]) for a in args)

    if per_time:
        unit_idx = 1
    else:
        unit_idx = 0

    if 'Q_h' not in vals:
        ret = cp * vals['m'] * vals['dT']
        unit = units['Q_h'][unit_idx]

    elif 'm' not in vals:
        ret = vals['Q_h'] / (cp * vals['dT'])
        unit = units['m'][unit_idx]

    elif 'dT' not in vals:
        ret = vals['Q_h'] / (cp * vals['m'])
        unit = units['dT']

        if not isinstance(ret, Quantity[Temperature]):
            raise ValueError(f'Both units must be per unit time in case one '
                             f'of them is: {vals}')

        ret.ito('delta_degC')

    else:
        raise ValueError(f'Incorrect input to heat_balance: {vals}')

    return ret.to(unit)


def intermediate_temperatures(T_b: Quantity[Temperature],
                              T_s: Quantity[Temperature],
                              k: Quantity[ThermalConductivity],
                              d: Quantity[Length],
                              h_in: Quantity[HeatTransferCoefficient],
                              h_out: Quantity[HeatTransferCoefficient],
                              epsilon: float,
                              tol: float = 1e-6) -> tuple[Quantity[Temperature],
                                                          Quantity[Temperature]]:
    """
    Solves a nonlinear system of equations to find intermediate
    temperatures of a barrier with the following modes of heat transfer:

    * inner convection
    * conduction through the barrier
    * outer convection
    * outer radiation

    Parameters
    ----------
    T_b : Quantity[Temperature]
        Bulk temperature inside the barrier
    T_s : Quantity[Temperature]
        Bulk temperature outside the barrier (surroundings)
    k : Quantity[ThermalConductivity]
        The thermal conductivity of the barrier material.
        Supply the combined value in case there are multiple layers
    d : Quantity[Length]
        Total thickness of the barrier

        .. note::
            In case ``d`` is set to 0, it will be reset to ``tol`` to
            avoid division by zero.

    h_in : Quantity[HeatTransferCoefficient]
        The convective heat transfer coefficient at the inner barrier wall
    h_out : Quantity[HeatTransferCoefficient]
        The convective heat transfer coefficient at the outer barrier wall
    epsilon : float
        The emissivity of the outside surface, used to account for radiative heat transfer
    tol : float, optional
        Numerical accuracy for the conduction layer:
        ``d`` is set to this if 0 is passed, by default 1e-6

    Returns
    -------
    tuple[Quantity[Temperature], Quantity[Temperature]]
        The intermediate temperatures :math:`T_1` and :math:`T_2`:
        the surface temperatures of the inside and outside of the barrier
    """

    # convert input to numerical values with correct unit
    T_s = T_s.to('K').m
    T_b = T_b.to('K').m
    k = k.to('W/m/K').m
    d = d.to('m').m
    h_in = h_in.to('W/m²/K').m
    h_out = h_out.to('W/m²/K').m

    if abs(d - 0) < tol:
        d = tol

    # system of coupled equations: heat transfer rate through all layers is identical
    # inner convection == conduction == (outer convection + radiation)

    def fun(x):

        T1, T2 = x

        eq1 = k / d * (T1 - T2) - h_out * (T2 - T_s) - \
            epsilon * SIGMA.m * (T2**4 - T_s**4)

        eq2 = k / d * (T1 - T2) - h_in * (T_b - T1)

        return [eq1, eq2]

    # use the boundary temperatures as initial guesses
    T1_val, T2_val = fsolve(fun, [T_b, T_s])

    return Quantity(T1_val, 'K').to('degC'), Quantity(T2_val, 'K').to('degC')
