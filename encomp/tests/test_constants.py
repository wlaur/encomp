from ..constants import CONSTANTS
from ..units import Quantity
from ..utypes import Pressure


def test_CONSTANTS():
    assert isinstance(CONSTANTS.normal_conditions_pressure, Quantity[Pressure])
