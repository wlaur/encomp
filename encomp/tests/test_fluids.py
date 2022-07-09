import pytest
import numpy as np

from encomp.units import Quantity as Q
from encomp.fluids import Fluid, HumidAir, Water
from encomp.utypes import Density


def test_Fluid():

    fld = Fluid('R123', P=Q(2, 'bar'), T=Q(25, '°C'))

    repr(fld)

    assert fld.__getattr__('S') == Q(1087.7758824621442, 'J/(K kg)')
    assert fld.S == Q(1087.7758824621442, 'J/(K kg)')

    assert fld.D == fld.__getattr__('D')

    water = Fluid('water', P=Q(2, 'bar'), T=Q(25, '°C'))
    assert water.T.u == Q.get_unit('degC')
    assert water.T.m == 25

    HumidAir(T=Q(25, 'degC'), P=Q(125, 'kPa'), R=Q(0.2, 'dimensionless'))

    Water(P=Q(1, 'bar'), Q=Q(0.9, ''))
    Water(P=Q(1, 'bar'), T=Q(0.9, 'degC'))
    Water(T=Q(1, 'bar'), Q=Q(0.9, ''))

    with pytest.raises(ValueError):

        # cannot fix all of P, T, Q
        Water(P=Q(1, 'bar'), T=Q(150, 'degC'), Q=(0.4, ''))

    with pytest.raises(ValueError):

        # incorrect argument name
        Water(T=Q(1, 'bar'), p=Q(9, 'degC'))

    Fluid('water', T=Q([25, 95], 'C'), P=Q([1, 2], 'bar')).H
    Fluid('water', T=Q([25, np.nan], 'C'), P=Q([1, 2], 'bar')).H
    Fluid('water', T=Q([np.nan, np.nan], 'C'), P=Q([1, 2], 'bar')).H
    Fluid('water', T=Q([np.nan, np.nan], 'C'), P=Q([np.nan, np.nan], 'bar')).H
    Fluid('water', T=Q(23, 'C'), P=Q([1, 2], 'bar')).H
    Fluid('water', T=Q(23, 'C'), P=Q([1], 'bar')).H
    Fluid('water', T=Q([23, 25], 'C'), P=Q([1], 'bar')).H
    Fluid('water', T=Q([23, 25], 'C'), P=Q(np.nan, 'bar')).H
    Fluid('water', T=Q([23, 25], 'C'), P=Q([1, np.nan], 'bar')).H

    Water(T=Q([25, 25, 63], 'C'), Q=Q([np.nan, np.nan, 0.4], '')).H
    Water(T=Q([25, np.nan, 63], 'C'), Q=Q([np.nan, 0.2, 0.5], '')).H
    Water(T=Q([25, np.nan, np.nan], 'C'), Q=Q([np.nan, 0.2, np.nan], '')).H

    # returns empty array (not nan)
    ret = Fluid('water', T=Q([], 'C'), P=Q([], 'bar')).H.m
    assert isinstance(ret, np.ndarray) and ret.size == 0
    ret = Fluid('water', T=Q([], 'C'), P=Q((), 'bar')).H.m
    assert isinstance(ret, np.ndarray) and ret.size == 0
    ret = Fluid('water', T=Q([], 'C'), P=Q(np.array([]), 'bar')).H.m
    assert isinstance(ret, np.ndarray) and ret.size == 0

    # 1-element list or array works in the same way as scalar,
    # except that the output is also a 1-element list or array
    ret = Water(
        P=Q([2, 3], 'bar'),
        Q=Q([0.5])
    ).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 2

    ret = Water(
        P=Q([2, 3], 'bar'),
        Q=Q(0.5)
    ).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 2

    ret = Water(
        P=Q([2], 'bar'),
        Q=Q([0.5])
    ).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 1

    ret = Water(
        P=Q([2], 'bar'),
        Q=Q(0.5)
    ).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 1

    ret = Water(
        P=Q(2, 'bar'),
        Q=Q([0.5])
    ).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 1

    ret = Water(
        P=Q(2, 'bar'),
        Q=Q(0.5)
    ).D.m

    assert isinstance(ret, float)

    ret = Water(
        P=Q([], 'bar'),
        Q=Q([0.5])
    ).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 0

    ret = Water(
        P=Q([], 'bar'),
        Q=Q([])
    ).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 0

    ret = Water(
        P=Q(np.array([]), 'bar'),
        Q=Q(np.array([]))
    ).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 0

    # returns 1-element list
    assert isinstance(Fluid('water', T=Q([23], 'C'), P=Q([1], 'bar')).H.m,
                      np.ndarray)

    assert isinstance(Fluid('water', T=Q(23, 'C'), P=Q([1], 'bar')).H.m,
                      np.ndarray)

    assert isinstance(Fluid('water', T=Q([23], 'C'), P=Q(1, 'bar')).H.m,
                      np.ndarray)

    # returns float
    assert isinstance(Fluid('water', T=Q(23, 'C'), P=Q(1, 'bar')).H.m,
                      float)

    with pytest.raises(ValueError):

        Fluid('water', T=Q([np.nan, np.nan], 'C'),
              P=Q([np.nan, np.nan, np.nan], 'bar')).H

    with pytest.raises(ValueError):

        Fluid('water', T=Q([np.nan, np.nan], 'C'), P=Q([], 'bar')).H


def test_Water():

    water_single = Water(
        T=Q(25, '°C'),
        P=Q(5, 'bar')
    )

    repr(water_single)

    water_multi = Water(
        T=Q(np.linspace(25, 50), '°C'),
        P=Q(5, 'bar')
    )

    repr(water_multi)

    water_mixed_phase = Water(
        T=Q(np.linspace(25, 500, 10), '°C'),
        P=Q(np.linspace(0.5, 10, 10), 'bar')
    )

    repr(water_mixed_phase)

    with pytest.raises(Exception):

        # mismatching sizes
        # must access an attribute before it's actually evaluated
        Water(
            T=Q(np.linspace(25, 500, 10), '°C'),
            P=Q(np.linspace(0.5, 10, 50), 'bar')
        ).P


def test_HumidAir():
    T = Q(20, 'C')
    P = Q(20, 'bar')
    R = Q(20, '%')

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([25, 34], 'C')
    P = Q(20, 'bar')
    R = Q(20, '%')

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([25, 34], 'C')
    P = Q([20, 30], 'bar')
    R = Q([20, 40], '%')

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([25, 34], 'C')
    P = Q([20, 30], 'bar')
    R = Q([20, np.nan], '%')

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([np.nan, 34], 'C')
    P = Q([np.nan, 30], 'bar')
    R = Q([20, np.nan], '%')

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([20, 40], 'C')
    P = Q([20, 1], 'bar')
    R = Q([20, 101], '%')

    ha = HumidAir(T=T, P=P, R=R)
    val = ha.V.m
    assert not np.isnan(val[0])
    assert np.isnan(val[1])


def test_shapes():

    N = 16

    T = Q(np.linspace(50, 60, N).reshape(4, 4), 'C')
    P = Q(np.linspace(2, 4, N).reshape(4, 4), 'bar')

    water = Fluid('water', T=T, P=P)

    assert water.D.m.shape == P.m.shape
    assert water.D.m.shape == T.m.shape

    N = 27

    T = Q(np.linspace(50, 60, N).reshape(3, 3, 3), 'C')
    P = Q(np.linspace(2, 4, N).reshape(3, 3, 3), 'bar')

    water = Fluid('water', T=T, P=P)

    assert water.D.m.shape == P.m.shape
    assert water.D.m.shape == T.m.shape


def test_invalid_areas():

    N = 10
    T = Q(np.linspace(-100, -50, N), 'K')
    P = Q(np.linspace(-1, -2, N), 'bar')

    water = Fluid('water', T=T, P=P)

    assert water.D.check(Density)
    assert isinstance(water.D.m, np.ndarray)

    T = Q(np.linspace(-100, 300, N), 'K')
    P = Q(np.linspace(-1, 2, N), 'bar')

    water = Fluid('water', T=T, P=P)

    assert water.D.check(Density)
    assert isinstance(water.D.m, np.ndarray)
    assert np.isnan(water.D.m[0])
    assert not np.isnan(water.D.m[-1])

    arr1 = np.linspace(-100, 400, N)
    arr2 = np.linspace(-1, 2, N)

    arr1[-2] = np.nan
    arr2[-1] = np.nan
    arr2[-3] = np.nan

    T = Q(arr1, 'K')
    P = Q(arr2, 'bar')

    water = Fluid('water', T=T, P=P)

    assert water.D.m.size == N


def test_properties_Fluid():

    props = Fluid.ALL_PROPERTIES

    fluid_names = ['water', 'methane', 'R134a']

    Ts = [
        25, 0, -1, -100, np.nan,
        [25, 30], [np.nan, 25], [np.nan, np.nan], [np.inf, np.nan],
        np.linspace(0, 10, 10), np.linspace(-10, 10, 10)
    ]

    Ps = [
        1, 0, -1, -100, np.nan,
        [3, 4], [np.nan, 3], [np.nan, np.nan], [np.inf, np.nan],
        np.linspace(0, 10, 10), np.linspace(-10, 10, 10)
    ]

    for fluid_name in fluid_names:
        for T, P in zip(Ts, Ps):

            fluid = Fluid(fluid_name, T=Q(T, 'C'), P=Q(P, 'bar'))
            repr(fluid)

            for p in props:
                getattr(fluid, p)


def test_properties_HumidAir():

    props = HumidAir.ALL_PROPERTIES

    Ts = [
        25, 0, -1, -100, np.nan,
        [25, 30], [np.nan, 25], [np.nan, np.nan], [np.inf, np.nan],
        np.linspace(0, 10, 10), np.linspace(-10, 10, 10)
    ]

    Ps = [
        1, 0, -1, -100, np.nan,
        [3, 4], [np.nan, 3], [np.nan, np.nan], [np.inf, np.nan],
        np.linspace(0, 10, 10), np.linspace(-10, 10, 10)
    ]

    Rs = [
        0.5, 0.1, -1, -100, np.nan, -0.5, 0.00001, -0.0001, 0.99999, 1, 1.00001,
        [0.3, 0.4], [np.nan, 0.3], [np.nan, np.nan], [np.inf, np.nan],
        np.linspace(0, 1, 10), np.linspace(-0.5, 0.5, 10)
    ]

    for T, P, R in zip(Ts, Ps, Rs):

        ha = HumidAir(T=Q(T, 'C'), P=Q(P, 'bar'), R=Q(R))
        repr(ha)

        for p in props:
            getattr(ha, p)
