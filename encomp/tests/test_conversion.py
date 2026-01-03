from typing import assert_type

import numpy as np
import pytest

from ..conversion import convert_volume_mass
from ..misc import isinstance_types
from ..units import Quantity as Q
from ..utypes import Volume, VolumeFlow


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_convert_volume_mass() -> None:
    mf = Q(25, "kg/s")

    assert isinstance_types(convert_volume_mass(mf), Q[VolumeFlow])
    assert_type(convert_volume_mass(mf), Q[VolumeFlow, float])
    assert not isinstance_types(convert_volume_mass(mf), Q[VolumeFlow, np.ndarray])

    mf_list = Q([25.5, 25.34], "kg/s")
    assert isinstance_types(convert_volume_mass(mf_list), Q[VolumeFlow])

    assert_type(convert_volume_mass(mf_list), Q[VolumeFlow, np.ndarray])

    m = Q(25, "ton")

    assert isinstance_types(convert_volume_mass(m), Q[Volume])

    with pytest.raises(AssertionError):
        convert_volume_mass(mf, rho=Q(25, "bar"))  # pyright: ignore[reportCallIssue, reportArgumentType]

    with pytest.raises(AssertionError):
        convert_volume_mass(Q(25, "m/s"))  # pyright: ignore[reportCallIssue, reportArgumentType]
