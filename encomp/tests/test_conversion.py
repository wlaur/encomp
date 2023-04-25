import numpy as np
import pytest

from ..conversion import convert_volume_mass
from ..units import Quantity as Q
from ..utypes import Volume, VolumeFlow


def test_convert_volume_mass():
    mf = Q(25, "kg/s")

    assert isinstance(convert_volume_mass(mf), Q[VolumeFlow])
    assert isinstance(convert_volume_mass(mf), Q[VolumeFlow, float])
    assert not isinstance(convert_volume_mass(mf), Q[VolumeFlow, list[float]])

    mf_list = Q([25.5, 25.34], "kg/s")
    assert isinstance(convert_volume_mass(mf_list), Q[VolumeFlow])

    # TODO: list[float] will be cast to array, this is not captured by type hints
    assert isinstance(convert_volume_mass(mf_list), Q[VolumeFlow, np.ndarray])

    m = Q(25, "ton")

    assert isinstance(convert_volume_mass(m), Q[Volume])

    with pytest.raises(TypeError):
        convert_volume_mass(mf, rho=Q(25, "bar"))

    with pytest.raises(TypeError):
        convert_volume_mass(Q(25, "m/s"))
