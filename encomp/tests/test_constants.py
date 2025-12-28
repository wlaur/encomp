from ..constants import CONSTANTS
from ..misc import isinstance_types
from ..units import Quantity
from ..utypes import Pressure


def test_CONSTANTS() -> None:
    assert isinstance_types(CONSTANTS.normal_conditions_pressure, Quantity[Pressure])
