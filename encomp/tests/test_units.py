import pytest
from pytest import approx
import numpy as np
import pandas as pd

from encomp.units import Quantity, Q, wraps, check
from encomp.utypes import *


def test_Q():

    # test that Quantity objects can be constructed
    Q(1, 'dimensionless')
    Q(1, 'kg')
    Q(1, 'bar')
    Q(1, 'h')
    Q(1, 'newton')
    Q(1, 'cSt')

    # make sure that the alias Q behaves identically to Quantity
    assert Q(1) == Quantity(1)
    assert type(Q(1)) is type(Quantity(1))
    assert type(Q) is type(Quantity)

    # ensure that the inputs can be nested
    Q(Q(1, 'kg'))
    mass = Q(12, 'kg')
    Q(Q(Q(Q(mass))))
    Q(Q(Q(Q(mass), 'lbs')))
    Q(Q(Q(Q(mass), 'lbs')), 'stone')

    # no unit input defaults to dimensionless
    assert Q(12).check('')
    assert Q(1) == Q(100, '%')
    Q['Dimensionless'](21)
    assert isinstance(Q(21), Q['Dimensionless'])

    # check type of "m"
    assert isinstance(Q(1, 'meter').m, int)
    assert isinstance(Q(2.3, 'meter').m, float)
    assert isinstance(Q([2, 3.4], 'meter').m, np.ndarray)
    assert isinstance(Q(np.array([2, 3.4]), 'meter').m, np.ndarray)

    # input Quantity as unit
    Q(1, Q(2, 'bar'))

    # input Quantity as val
    Q(Q(2, 'bar'), 'kPa')

    # input Quantity as both val and unit
    Q(Q(2, 'bar'), Q(3, 'kPa'))
    Q(Q(2, 'bar'), Q(3, 'mmHg'))

    # check that the dimensionality constraints work
    Q[Length](1, 'm')
    Q[Pressure](1, 'kPa')
    Q[Temperature](1, '°C')

    # the dimensionalities can also be specified as strings
    Q['Temperature'](1, '°C')

    P = Q(1, 'bar')
    # this Quantity must have the same dimensionality as P
    Q[P](2, 'kPa')

    with pytest.raises(Exception):
        Q[Temperature](1, 'kg')
        Q[Pressure](1, 'meter')
        Q[Mass](1, P)

    # in-place conversion
    # NOTE: don't use this for objects that are passed in by the user
    P3 = Q(1, 'bar')
    P3.ito('kPa')
    P3.ito(Q(123123, 'kPa'))

    assert P3.m == approx(100, rel=1e-12)

    # test conversions to np.ndarray with int/float dtypes
    a = Q([1, 2, 3], 'bar')
    a.ito('kPa')

    a = Q(np.array([1, 2, 3.0]), 'bar')
    a.ito('kPa')

    a = Q(np.array([1.0, 2.0, 3.0]), 'bar')
    a.ito('kPa')

    # conversion to new object
    P4 = Q(1, 'bar')
    P4_b = P4.to('kPa')
    P4_b = P4.to(Q(123123, 'kPa'))

    assert P4_b.m == approx(100, rel=1e-12)

    # check that nested Quantity objects can be used as input
    # only the first value is used as magnitude, the other Quantity
    # objects are only used to determine unit
    P2 = Q(Q(2, 'feet_water'), Q(321321, 'kPa')).to(Q(123123, 'feet_water'))

    # floating point math might make this off at the N:th decimal
    assert P2.m == approx(2, rel=1e-12)
    assert isinstance(P2, Q['Pressure'])

    with pytest.raises(Exception):

        # incorrect dimensionalities should raise Exception
        Q(Q(2, 'feet_water'), Q(321321, 'kg')).to(Q(123123, 'feet_water'))

    # the UnitsContainer objects can be used to construct new dimensionalities
    Q[Length * Length * Length / Temperature](1, 'm³/K')

    with pytest.raises(Exception):
        Q[Pressure / Area](1, 'bar/m')

    # percent or %
    Q(1.124124e-3, '').to('%').to('percent')
    Q(1.124124e-3).to('%').to('percent')

    # pd.Series is converted to np.ndarray
    vals = [2, 3, 4]
    s = pd.Series(vals, name='Pressure')
    arr = Q(s, 'bar').to('kPa').m
    assert isinstance(arr, np.ndarray)
    assert arr[0] == 200

    # np.ndarray magnitudes equality check
    assert (Q(s, 'bar') == Q(vals, 'bar').to('kPa')).all()

    # support a single string as input if the
    # magnitude and units are separated by one or more spaces
    assert Q('1 meter').check(Length)
    assert Q('1 meter per second').check(Velocity)
    assert (Q('1 m') ** 3).check(Volume)


def test_wraps():

    # @wraps(ret, args, strict=True|False) is a convenience
    # decorator for making the input/output of a function into Quantity
    # however, it does not enforce the return value

    @ wraps('kg', ('m', 'kg'), strict=True)
    def func(a, b):

        # this is incorrect, cannot add 1 to a dimensional Quantity
        return a * b**2 + 1

    assert isinstance(func(Q(1, 'yd'), Q(20, 'lbs')), Q['Mass'])
    assert Q(1, 'bar').check(Pressure)


def test_check():

    assert not Q(1, 'kg').check('[energy]')
    assert Q(1, 'kg').check(Mass)
    assert not Q(1, 'kg').check(Energy)

    @ check('[length]', '[mass]')
    def func(a, b):

        return a * b

    func(Q(1, 'yd'), Q(20, 'lbs'))
