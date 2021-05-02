import pytest

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
