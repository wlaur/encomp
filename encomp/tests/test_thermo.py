
from encomp.units import Q
from encomp.thermo import heat_balance


def test_heat_balance():

    assert isinstance(heat_balance(
        Q(2, 'kg/s'), Q(2, 'kJ/s')), Q['Temperature'])

    assert isinstance(heat_balance(
        Q(2, 'K'), Q(2, 'kJ/s')), Q['MassFlow'])

    assert isinstance(heat_balance(
        Q(2, 'kg'), Q(2, 'delta_degF')), Q['Energy'])

    assert isinstance(heat_balance(
        Q(2, 'kg/s'), Q(2, 'delta_degF')), Q['Power'])
