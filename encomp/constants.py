"""
Contains constants used elsewhere in the library.
"""

from dataclasses import dataclass

from .units import Quantity as Q
from .utypes import Pressure, Temperature


@dataclass
class Constants:
    """
    Collection of constants.
    Use a single instance of this class to refer to these constants.
    """

    R = Q(8.3144598, "kg*m²/K/mol/s²")
    SIGMA = Q(5.670374419e-8, "W/m**2/K**4")

    normal_conditions_pressure = Q[Pressure, float](1, "atm")
    normal_conditions_temperature = Q[Temperature, float](0, "°C").to("K")

    standard_conditions_pressure = Q[Pressure, float](1, "atm")
    standard_conditions_temperature = Q[Temperature, float](15, "degC").to("K")


CONSTANTS = Constants()
