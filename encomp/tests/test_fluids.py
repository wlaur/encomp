import pytest
import numpy as np

from encomp.units import Q
from encomp.fluids import Fluid, HumidAir, Water


def test_Fluid():

    fld = Fluid('R123', P=Q(2, 'bar'), T=Q(25, '°C'))

    assert fld.get('S') == Q(1087.7758824621442, 'J/(K kg)')
    assert fld.D == fld.get('D')

    water = Fluid('water', P=Q(2, 'bar'), T=Q(25, '°C'))
    assert water.T.u == Q.get_unit('degC')
    assert water.T.m == 25

    HumidAir(T=Q(25, 'degC'), P=Q(125, 'kPa'), R=Q(0.2, 'dimensionless'))

    Water(P=Q(1, 'bar'), Q=Q(0.9, ''))
    Water(P=Q(1, 'bar'), T=Q(0.9, 'degC'))
    Water(T=Q(1, 'bar'), Q=Q(0.9, ''))

    with pytest.raises(Exception):

        # cannot fix all of P, T, Q
        Water(P=Q(1, 'bar'), T=Q(150, 'degC'), Q=(0.4, ''))

        # incorrect argument name
        Water(T=Q(1, 'bar'), P=Q(9, 'degC'))

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

    # returns NaN (not empty array)
    assert np.isnan(Fluid('water', T=Q([], 'C'), P=Q([], 'bar')).H.m)

    # returns single float (not 1-element list)
    assert not Fluid('water', T=Q([23], 'C'), P=Q([1], 'bar')).H.m.shape

    with pytest.raises(ValueError):
        Fluid('water', T=Q([np.nan, np.nan], 'C'),
              P=Q([np.nan, np.nan, np.nan], 'bar')).H
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
