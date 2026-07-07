from typing import Any, assert_type, cast

import pytest

from ..conversion import convert_volume_mass
from ..units import ExpectedDimensionalityError
from ..units import Quantity as Q
from ..utypes import Numpy1DArray, Volume, VolumeFlow


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_convert_volume_mass() -> None:
    mf = Q(25, "kg/s")

    assert_type(convert_volume_mass(mf), Q[VolumeFlow, float])

    mf_list = Q([25.5, 25.34], "kg/s")
    assert_type(convert_volume_mass(mf_list), Q[VolumeFlow, Numpy1DArray])

    m = Q(25, "ton")

    assert_type(convert_volume_mass(m), Q[Volume, float])

    # wrong dimensionality raises a proper unit error naming the argument,
    # not an internal AssertionError
    with pytest.raises(ExpectedDimensionalityError, match="rho"):
        convert_volume_mass(mf, rho=cast(Any, Q(25, "bar")))

    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(mf, rho=Q(0, "kg/m³"))

    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(mf, rho=Q(-1, "kg/m³"))

    with pytest.raises(ValueError, match="positive"):
        convert_volume_mass(mf_list, rho=Q([997.0, 0.0], "kg/m³"))

    with pytest.raises(ExpectedDimensionalityError, match="inp"):
        convert_volume_mass(cast(Any, Q(25, "m/s")))
