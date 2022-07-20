from typing import TypedDict
from decimal import Decimal

import pytest
from pytest import approx
from typeguard import typechecked

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like as pandas_is_list_like  # type: ignore

from encomp.misc import isinstance_types
from encomp.conversion import convert_volume_mass
from encomp.units import (Quantity,
                          ureg,
                          DimensionalityTypeError,
                          ExpectedDimensionalityError)
from encomp.units import Quantity as Q
from encomp.serialize import decode
from encomp.fluids import Water
from encomp.utypes import *


def test_registry():


    from encomp.units import ureg
    from pint import _DEFAULT_REGISTRY, application_registry

    us = [ureg, _DEFAULT_REGISTRY, application_registry.get()]

    # check that all these objects are the same
    assert len(set(map(id, us))) == 1

    # check that units from all objects can be combined
    q = 1 * ureg.kg / _DEFAULT_REGISTRY.s**2 / application_registry.get().m
    assert isinstance(q, Q[Pressure])


def test_type_eq():

    q = Q(25, 'm')

    # this is the recommended way of checking type
    assert isinstance(q, Q)

    # this is overloaded to work for the Quantity base class
    # for compatibility with other libraries

    assert type(q) == Q
    assert Q == type(q)


    assert type(Q(2)) == Q
    assert Q == type(Q(25, 'bar'))

    # __eq__ is overloaded, but these are still different types
    assert not type(q) is Q

    # subclasses behave as expected

    assert type(q) == Q[Length]
    assert Q[Length] == type(q)

    assert not type(q) == Q[Dimensionless]
    assert not Q[Dimensionless] == type(q)


def test_Q():

    # test that Quantity objects can be constructed
    Q(1, 'dimensionless')
    Q(1, 'kg')
    Q(1, 'bar')
    Q(1, 'h')
    Q(1, 'newton')
    Q(1, 'cSt')

    assert Q(1, 'meter/kilometer').to_reduced_units().m == 0.001
    assert Q(1, 'km').to_base_units().m == 1000

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
    Q[Dimensionless](21)
    assert isinstance(Q(21), Q[Dimensionless])

    assert Q(1) == Q('1')
    assert Q(1) == Q('\n1\n')
    assert Q(1) == Q('1 dimensionless')

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
    Q[Temperature](1, '°C')

    P = Q(1, 'bar')
    # this Quantity must have the same dimensionality as P
    Q(2, 'kPa').check(P)

    with pytest.raises(ExpectedDimensionalityError):
        Q[Temperature](1, 'kg')

    with pytest.raises(ExpectedDimensionalityError):
        Q[Pressure](1, 'meter')

    with pytest.raises(ExpectedDimensionalityError):
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

    assert Q(1, 'bar') == Q(100, 'kPa') == Q('0.1 MPa') == Q('1e5', 'Pa')

    # check that nested Quantity objects can be used as input
    # only the first value is used as magnitude, the other Quantity
    # objects are only used to determine unit
    P2 = Q(Q(2, 'feet_water'), Q(321321, 'kPa')).to(Q(123123, 'feet_water'))

    # floating point math might make this off at the N:th decimal
    assert P2.m == approx(2, rel=1e-12)
    assert isinstance(P2, Q[Pressure])

    with pytest.raises(Exception):

        # incorrect dimensionalities should raise Exception
        Q(Q(2, 'feet_water'), Q(321321, 'kg')).to(Q(123123, 'feet_water'))

    # the UnitsContainer objects can be used to construct new dimensionalities
    class CustomDimensionality(Dimensionality):
        dimensions = Length.dimensions * Length.dimensions * \
            Length.dimensions / Temperature.dimensions

    Q[CustomDimensionality](1, 'm³/K')

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


def test_custom_units():

    # "ton" should always default to metric ton
    assert (Q(1, 'ton') == Q(1, 'Ton') == Q(1, 'TON') == Q(
        1, 'tonne') == Q(1, 'metric_ton') == Q(1000, 'kg'))

    assert Q(1, 'US_ton') == Q(907.1847400000001, 'kg')

    assert (Q(1, 'ton/hour') == Q(1, 'Ton/hour') == Q(1, 'TON/hour') ==
            Q(1, 'tonne/hour') == Q(1, 'metric_ton/hour') == Q(1000, 'kg/hour'))

    v1 = (Q(1000, 'liter') * Q(1, 'normal')).to_base_units().m
    v2 = Q(1000, 'normal liter').to_base_units().m
    v3 = Q(1, 'nm3').m
    v4 = Q(1, 'Nm3').m

    # floating point accuracy
    assert round(v1, 10) == round(v2, 10) == round(v3, 10) == round(v4, 10)

    factor = Q(12, 'Nm3 water/ (normal liter air)')
    (Q(1, 'kg water') / factor).to('pound air')

    Q[NormalVolume](2, 'nm**3')

    with pytest.raises(ExpectedDimensionalityError):
        Q[NormalVolumeFlow](2, 'm**3/hour')

    Q[NormalVolumeFlow](2, 'Nm**3/hour').to('normal liter/sec')

    class _NormalVolumeFlow(NormalVolumeFlow):
        dimensions = Normal.dimensions * VolumeFlow.dimensions

    Q[_NormalVolumeFlow](2, 'Nm**3/hour').to('normal liter/sec')

    Q(2, 'normal liter air / day')
    Q(2, '1/Nm3').to('1 / (liter normal)')


def test_wraps():

    # @ureg.wraps(ret, args, strict=True|False) is a convenience
    # decorator for making the input/output of a function into Quantity
    # however, it does not enforce the return value

    @ureg.wraps('kg', ('m', 'kg'), strict=True)
    def func(a, b):

        # this is incorrect, cannot add 1 to a dimensional Quantity
        return a * b**2 + 1

    assert isinstance(func(Q(1, 'yd'), Q(20, 'lbs')), Q[Mass])
    assert Q(1, 'bar').check(Pressure)


def test_check():

    assert not Q(1, 'kg').check('[energy]')
    assert Q(1, 'kg').check(Mass)
    assert not Q(1, 'kg').check(Energy)

    @ureg.check('[length]', '[mass]')
    def func(a, b):

        return a * b

    func(Q(1, 'yd'), Q(20, 'lbs'))


def test_typechecked():

    @typechecked
    def func_a(a: Quantity[Temperature]) -> Quantity[Pressure]:
        return Q(2, 'bar')

    assert func_a(Q(2, 'degC')) == Q(2, 'bar')

    with pytest.raises(TypeError):
        func_a(Q(2, 'meter'))

    @typechecked
    def func_b(a: Quantity) -> Quantity[Pressure]:
        return a

    assert func_b(Q(2, 'bar')) == Q(2, 'bar')
    assert func_b(Q(2, 'psi')) == Q(2, 'psi')
    assert func_b(Q(2, 'mmHg')) == Q(2, 'mmHg')

    with pytest.raises(TypeError):
        func_a(Q(2, 'meter'))


def test_dataframe_assign():

    df_multiple_rows = pd.DataFrame(
        {
            'A': [1, 2, 3],
            'B': [1, 2, 3],
        }
    )

    df_single_row = pd.DataFrame(
        {
            'A': [1],
            'B': [1],
        }
    )

    df_empty = pd.DataFrame(
        {
            'A': [],
            'B': [],
        }
    )

    for df in [df_multiple_rows, df_single_row, df_empty]:

        df['C'] = Q(df.A, 'bar') * Q(25, 'meter')

        df['Temp'] = Q(df.A, 'degC')

        with pytest.raises(AttributeError):

            density = Water(

                # this is pd.Series[float]
                T=df.Temp,
                Q=Q(0.5)

            ).D

        density = Water(

            # wrap in Quantity() to use the Water class
            T=Q(df.Temp, 'degC'),
            Q=Q(0.5)

        ).D

        # implicitly strips magnitude in whatever unit the Quantity happens to have
        df['density'] = density

        # assigns a column with specific unit
        df['density_with_unit'] = density.to('kg/m3')

        # the .m accessor is not necessary for vectors
        df['density_with_unit_magnitude'] = density.to('kg/m3').m

        # this does not work -- pandas function is_list_like(Q(4, 'bar')) -> True
        # which means that this fails internally in pandas
        # ValueError: Length of values (1) does not match length of index (3)
        # df['D'] = Q(4, 'bar')

        # i.e. the .m accessor must be used for scalar Quantity assignment

        df['E'] = Q(df.A, 'bar').m
        df['F'] = Q(4, 'bar').m


def test_generic_dimensionality():

    assert issubclass(Q[Pressure], Q)
    assert not issubclass(Q[Pressure], Q[Temperature])

    assert Q[Pressure] is Q[Pressure]
    assert Q[Pressure] == Q[Pressure]

    assert Q[Pressure] is not Q[Temperature]
    assert Q[Pressure] != Q[Temperature]

    assert isinstance_types(
        [Q, Q[Pressure], Q[Temperature]], list[type[Q]]
    )

    assert not isinstance_types(
        [Q, Q[Pressure], Q[Temperature]], list[Q]
    )

    assert isinstance_types(
        [Q[Pressure], Q[Pressure]], list[type[Q[Pressure]]]
    )

    with pytest.raises(TypeError):
        Q[1]

    with pytest.raises(TypeError):
        Q['Temperature']

    with pytest.raises(TypeError):
        Q['string']

    with pytest.raises(TypeError):
        Q[None]


def test_dynamic_dimensionalities():

    # this will create a new dimensionality type
    q1 = Q(1, 'kg^2/K^4')

    # this will reuse the previously created one
    q2 = Q(25, 'g*g/(K^2*K^2)')

    assert type(q1) is type(q2)

    # NOTE: don't use __class__ when checking this, always use type()
    assert isinstance(q1, type(q2))
    assert isinstance(q2, type(q1))
    assert isinstance(q2, Q)
    assert not isinstance(q2, Q[Pressure])

    q3 = Q(25, 'm*g*g/(K^2*K^2)')

    assert type(q3) is not type(q2)
    assert not isinstance(q3, type(q1))


def test_instance_checks():

    assert isinstance(Q(2), Q[Dimensionless])
    assert isinstance(Q(2), Q)

    class Fraction(Dimensionless):
        pass

    # Q(2) will default to Dimensionless, since that was
    # defined before Fraction
    assert not isinstance(Q(2), Q[Fraction])
    assert isinstance(Q(2), Q[Dimensionless])

    # Q[Fraction] will override the subclass
    assert isinstance(Q[Fraction](2), Q[Fraction])

    assert isinstance(Q(25, 'bar'), Q[Pressure])

    assert not isinstance(Q(25, 'C'), Q[Pressure])

    assert isinstance(Q(25, 'C'), Q[Temperature])

    assert isinstance(Q(25, 'kg'), Q[Mass])

    assert isinstance(Q(25, 'kg'), Q)

    assert isinstance_types(Q(25, 'C'), Q)
    assert isinstance_types(Q(25, 'C'), Q[Temperature])

    assert isinstance_types([Q(25, 'C')], list[Q[Temperature]])
    assert isinstance_types(
        [Q[Temperature](25, 'C'), Q(25, 'F')], list[Q[Temperature]])
    assert isinstance_types([Q(25, 'C'), Q(25, 'bar')], list[Q])

    class CustomDimensionality(Dimensionality):
        dimensions = UnitsContainer({'[mass]': 1, '[temperature]': -2})

    # Q(25, 'g/K^2') will find the correct subclass since there
    # is no other, existing dimensionality with these dimensions
    assert isinstance(Q(25, 'g/K^2'), Q[CustomDimensionality])

    assert isinstance(Q[CustomDimensionality](
        25, 'g/K^2'), Q[CustomDimensionality])

    assert isinstance(Q[CustomDimensionality](25, 'g/K^2'), Q)

    assert isinstance_types({Q(25, 'g/K^2')},
                            set[Q[CustomDimensionality]])

    assert not isinstance_types((Q(25, 'm*g/K^2'), ),
                                tuple[Q[CustomDimensionality], ...])

    assert not isinstance_types([Q(25, 'C')], list[Q[Pressure]])

    assert isinstance_types([Q[Temperature](25, 'C'), Q(25, 'F')],
                            list[Q[Temperature]])

    assert not isinstance_types([Q[Temperature](25, 'C'), Q(25, 'F/day')],
                                list[Q[Temperature]])

    assert not isinstance_types([Q(25, 'C'), Q(25, 'bar')],
                                list[Q[CustomDimensionality]])


def test_typed_dict():

    d = {
        'P': Q[Pressure](25, 'bar'),
        'T': Q[Temperature](25, 'degC'),
        'x': Q[Dimensionless](0.5)
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    # cannot use isinstance with complex types
    with pytest.raises(TypeError):
        isinstance(d, PTxDict)

    assert isinstance_types(d, PTxDict)

    d = {
        'P': Q[Pressure](25, 'bar'),
        'T': Q[Temperature](25, 'degC'),
        'x': Q[Dimensionless](0.5),
        'extra': Q
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    assert not isinstance_types(d, PTxDict)

    d = {
        'P': Q[Pressure](25, 'bar'),
        'T': Q[Temperature](25, 'degC'),
        'x': Q[Dimensionless](0.5),
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]
        missing: Q

    assert not isinstance_types(d, PTxDict)

    d = {
        'P': Q(25, 'bar'),
        'T': Q(25, 'degC'),
        'x': Q(0.5),
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    assert isinstance_types(d, PTxDict)

    d = {
        'P': Q(25, 'bar'),
        'T': Q(25, 'meter'),
        'x': Q(0.5),
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    assert not isinstance_types(d, PTxDict)


def test_convert_volume_mass():

    V = convert_volume_mass(
        Q(125, 'kg/s'),
        Q(25, 'kg/m**3')
    )

    assert V.check(VolumeFlow)

    m = convert_volume_mass(
        Q(125, 'liter/day')
    )

    assert m.check(MassFlow)


def test_compatibility():

    Q(1) + Q(2)

    Q(1, '%') - Q(2)

    Q(25, 'm') + Q(25, 'cm')

    # this subtype inherits from Dimensionless, which
    # means that it is compatible (can be added and subtracted)

    class Fraction(Dimensionless):
        pass

    q1 = Q(2)
    q2 = Q[Fraction](0.6)

    q1 + q2
    q2 + q1

    q1 - q2
    q2 - q1

    # this is a new dimensionality that happens to have
    # the same dimensions as Dimensionless
    # however, it is not compatible with Dimensionless or any of its subclasses

    class IncompatibleFraction(Dimensionality):
        dimensions = Dimensionless.dimensions

    q3 = Q[IncompatibleFraction](0.2)

    with pytest.raises(DimensionalityTypeError):
        q3 + q1

    with pytest.raises(DimensionalityTypeError):
        q3 + q2

    with pytest.raises(DimensionalityTypeError):
        q1 + q3

    with pytest.raises(DimensionalityTypeError):
        q2 + q3

    s = Q(25, 'm')

    # this dimensionality means something specific,
    # it cannot be added to a normal length
    class DistanceAlongPath(Dimensionality):
        dimensions = Length.dimensions

    # need to override the overload based on unit "km"
    # this is not very elegant
    d = Q[DistanceAlongPath](25, str('km'))
    d2 = Q[DistanceAlongPath](5, str('km'))

    d + d
    d - d
    d + d2
    d2 - d

    with pytest.raises(DimensionalityTypeError):
        s + d

    with pytest.raises(DimensionalityTypeError):
        d - s

    assert str(
        (Q(25, 'MSEK/GWh') * Q(25, 'kWh')).to_reduced_units()
    ) == '0.000625 MSEK'

    assert str(
        (Q(25, 'MSEK/GWh') * Q(25, 'kWh')).to_base_units()
    ) == '625.0 currency'

    # if the _distinct class attribute is True, an unspecified
    # dimensionality will default to this
    # for example, HeatingValue has _distinct=True even though
    # it shares dimensions with other dimensionalities like SpecificEnthalpy

    # q4 will be become Quantity[HeatingValue] by default

    q4 = Q(25, 'kJ/kg')

    # override the Literal['kJ/kg'] overload
    q5 = Q[SpecificEnthalpy](25, str('kJ/kg'))
    q6 = Q[HeatingValue](25, str('kJ/kg'))

    with pytest.raises(DimensionalityTypeError):
        q4 - q5

    with pytest.raises(DimensionalityTypeError):
        q5 - q4

    q4 + q6
    q6 - q4


def test_distinct_dimensionality():

    unit = 'm**6/kg**2'
    uc = UnitsContainer({'[length]': 6, '[mass]': -2})

    class Indistinct(Dimensionality):
        dimensions = uc
        _distinct = False

    class Distinct(Dimensionality):
        dimensions = uc
        _distinct = True

    assert type(Q(1, unit)) is Q[Distinct]
    assert type(Q[Distinct](1, unit)) is Q[Distinct]
    assert type(Q[Indistinct](1, unit)) is Q[Indistinct]


def test_literal_units():

    for d, units in get_registered_units().items():

        for u in units:

            decoded = decode(
                {
                    'type': 'Quantity',
                    'dimensionality': d,
                    'data': [1, u]
                }
            )

            assert decoded._dimensionality_type.__name__ == d


def test_indexing():

    qs = Q([1, 2, 3], 'kg')

    assert isinstance(qs, Q[Mass])

    qi = qs[1]

    assert isinstance(qi, Q[Mass])
    assert qi == Q(2, 'kg')


def test_plus_minus():

    l = Q(2, 'm')

    # TODO: add type hints for this
    l_e = l.plus_minus(Q(1, 'cm'))

    l2_e = (l_e**2).to('km**2')

    assert l2_e.error == Q(4e-8, 'km**2')
    assert isinstance(l2_e.error, Q[Area])


def test_round():

    q = Q(25.12312312312, 'kg/s')

    q_r = round(q, 1)

    assert q_r.m == 25.1

    q = Q([25.12312312312, 25.12312312312], 'kg/s')

    q_r = round(q, 1)

    assert q_r.m[0] == 25.1
    assert q_r.m[1] == 25.1


def test_abs():

    q = Q(-25, 'kg/s')

    q_a = abs(q)

    assert q_a.m == 25

    q = Q([-25, -25], 'kg/s')

    q_a = abs(q)

    assert q_a.m[0] == 25
    assert q_a.m[1] == 25


def test_pandas_is_list_like():

    # scalar magnitude is not list like

    assert pandas_is_list_like(Q([25]))
    assert pandas_is_list_like(Q([25, 25]))
    assert pandas_is_list_like(Q(np.linspace(0, 1), 'kg'))

    assert not pandas_is_list_like(Q(25))
    assert not pandas_is_list_like(Q(0.2))
    assert not pandas_is_list_like(Q(0.2, 'kg/s'))


def test_pandas_integration():

    index = pd.date_range('2020-01-01', '2020-01-02', freq='h')
    df = pd.DataFrame(index=index)

    df['input'] = np.linspace(0, 1, len(df))

    q_vector = Q(df['input'], 'm/s')

    # assigns a float array, as expected
    df['A'] = q_vector.to('kmh')

    q_scalar = Q(25, 'ton/h')

    # assigns a repeated array of Quantity objects
    df['B'] = q_scalar

    # identical to the previous assignment
    df['C'] = [q_scalar] * len(df)

    # this will be correctly broadcasted to a repeated array
    df['D'] = q_scalar.m

    assert df.dtypes.A == np.float64
    assert df.dtypes.B == object
    assert df.dtypes.C == object
    assert df.dtypes.D == np.int64

    assert isinstance(df.A.values[0], np.float64)
    assert isinstance(df.B.values[-1], Q[MassFlow])
    assert isinstance(df.C.values[0], Q[MassFlow])
    assert isinstance(df.D.values[0], np.int64)


def test_unit_compatibility():

    # the ureg registry object contains unit attributes
    # that can be multiplied and divided by a magnitude
    # to create Quantity instances

    assert isinstance(ureg.m * 1, Q[Length])
    assert isinstance(1 * ureg.m / ureg.s, Q[Velocity])
    assert isinstance([1, 2, 3] * ureg.m / ureg.s, Q[Velocity])
    assert isinstance((1, 2, 3) * ureg.m / ureg.s, Q[Velocity])
    assert isinstance(np.array([1, 2, 3]) * ureg.m / ureg.s, Q[Velocity])



def test_decimal():

    # decimal.Decimal works, but it's not included in the type hints

    assert Q(Decimal('1.5'), 'MSEK').to('SEK').m == Decimal('1500000.00')

    q = Q([Decimal('1.5'), Decimal('1.5')], 'kg')

    with pytest.raises(TypeError):

        # this does not work with pint's internal API
        # unsupported operand type(s) for *: 'decimal.Decimal' and 'float'
        q.to('g')
