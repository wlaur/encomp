"""
Contains constants used elsewhere in the library.
"""

from pydantic.dataclasses import dataclass

from encomp.units import Q, Quantity
from encomp.utypes import Density, Temperature, Pressure


@dataclass
class Constants:
    """
    Collection of constants.
    Use a single instance of this class to refer to these constants.
    """

    R = Q(8.3144598, 'kg*m²/K/mol/s²')
    SIGMA = Q(5.670374419e-8, 'W/m**2/K**4')

    default_density: Quantity[Density] = Q(997, 'kg/m³')

    normal_conditions_pressure: Quantity[Pressure] = Q(1, 'atm')
    normal_conditions_temperature: Quantity[Temperature] = Q(0, '°C').to('K')

    standard_conditions_pressure: Quantity[Pressure] = Q(1, 'atm')
    standard_conditions_temperature: Quantity[Temperature] = Q(15, 'degC').to('K')


CONSTANTS = Constants()
