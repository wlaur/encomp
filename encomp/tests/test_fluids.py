from encomp.units import Q
from encomp.fluids import Fluid


def test_Fluid():

    fld = Fluid('R123', P=Q(2, 'bar'), T=Q(25, '°C'))

    assert fld.get('S') == Q(1087.7758824621442, 'J/(K kg)')
    assert fld.D == fld.get('D')

    water = Fluid('water', P=Q(2, 'bar'), T=Q(25, '°C'))
    assert water.T.u == Q.get_unit('degC')
    assert water.T.m == 25
