"""
Functions relating to thermodynamics.
"""

from typing import Union

from encomp.misc import isinstance_types
from encomp.units import Quantity
from encomp.utypes import Mass, MassFlow, Temperature, Energy, Power, HeatCapacity

DEFAULT_CP = Quantity[HeatCapacity](4.18, 'kJ/kg/K')


def heat_balance(*args: Union[Quantity[Mass], Quantity[MassFlow],
                              Quantity[Energy], Quantity[Power],
                              Quantity[Temperature]],
                 cp: Quantity[HeatCapacity] = DEFAULT_CP) -> Union[Quantity[Mass], Quantity[MassFlow],
                                                                   Quantity[Energy], Quantity[Power],
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
        vals['dT'].ito('delta_degC')

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
