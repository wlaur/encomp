"""
Contains constants used elsewhere in the library.
"""

from dataclasses import dataclass

from encomp.units import Quantity as Q
from encomp.utypes import Unknown, Temperature, Pressure


@dataclass
class Constants:
    """
    Collection of constants.
    Use a single instance of this class to refer to these constants.
    """

    R: Q[Unknown] = Q(8.3144598, 'kg*m²/K/mol/s²')
    SIGMA: Q[Unknown] = Q(5.670374419e-8, 'W/m**2/K**4')

    normal_conditions_pressure = Q[Pressure](1, 'atm')
    normal_conditions_temperature = Q[Temperature](0, '°C').to('K')

    standard_conditions_pressure = Q[Pressure](1, 'atm')
    standard_conditions_temperature = Q[Temperature](15, 'degC').to('K')


CONSTANTS = Constants()
