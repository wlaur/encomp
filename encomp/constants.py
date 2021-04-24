from pydantic import BaseModel

from encomp.units import Q

class Constants(BaseModel):
    """
    Collection of constants.
    Use a single instance of this class to refer to these constants.

    .. todo::
        Should some of these be changeable?
    """

    R: float = 8.3144598
    SIGMA: float = 5.670374419e-8

    default_density: float = 997

    normal_conditions_pressure = Q(1, 'atm')
    normal_conditions_temperature = Q(0, '°C')

    standard_conditions_pressure = Q(1, 'atm')
    standard_conditions_temperature = Q(15, 'degC')


CONSTANTS = Constants()
