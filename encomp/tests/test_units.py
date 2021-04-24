import pytest

from encomp.units import Q
from encomp.utypes import *


def test_Q():

    # test that Quantity objects can be constructed
    Q(1, 'dimensionless')
    Q(1, 'kg')
    Q(1, 'bar')
    Q(1, 'h')
    Q(1, 'newton')
    Q(1, 'cSt')

    # input Quantity as unit
    Q(1, Q(2, 'bar'))

    # input Quantity as val
    Q(Q(2, 'bar'), 'kPa')

    # input Quantity as both val and unit
    Q(Q(2, 'bar'), Q(3, 'kPa'))

    # check that the dimensionality constraints work
    Q[Length](1, 'm')
    Q[Pressure](1, 'kPa')
    Q[Temperature](1, '°C')

    P = Q(1, 'bar')
    # this Quantity must have the same dimensionality as P
    Q[P](2, 'kPa')

    with pytest.raises(Exception):
        Q[Temperature](1, 'kg')
        Q[Pressure](1, 'meter')
        Q[Mass](1, P)
