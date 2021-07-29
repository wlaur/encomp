from encomp.constants import CONSTANTS
from encomp.units import Quantity
from encomp.utypes import Density, Pressure


def test_CONSTANTS():

    assert isinstance(CONSTANTS.default_density, Quantity[Density])
    assert isinstance(CONSTANTS.normal_conditions_pressure, Quantity[Pressure])
