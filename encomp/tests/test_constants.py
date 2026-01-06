from typing import assert_type

from ..constants import CONSTANTS
from ..units import Quantity
from ..utypes import Pressure


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_CONSTANTS() -> None:
    assert_type(CONSTANTS.normal_conditions_pressure, Quantity[Pressure, float])
