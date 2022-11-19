import pytest

from ..units import Quantity as Q
from ..utypes import VolumeFlow, Volume
from ..conversion import convert_volume_mass


def test_convert_volume_mass():

    mf = Q(25, 'kg/s')

    assert isinstance(convert_volume_mass(mf), Q[VolumeFlow])

    m = Q(25, 'ton')

    assert isinstance(convert_volume_mass(m), Q[Volume])

    with pytest.raises(TypeError):
        convert_volume_mass(mf, rho=Q(25, 'bar'))

    with pytest.raises(TypeError):
        convert_volume_mass(Q(25, 'm/s'))
