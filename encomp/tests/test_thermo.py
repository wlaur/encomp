
from encomp.units import Q
from encomp.utypes import Temperature
from encomp.thermo import heat_balance, intermediate_temperatures


def test_heat_balance():

    assert isinstance(heat_balance(
        Q(2, 'kg/s'), Q(2, 'kJ/s')), Q['Temperature'])

    assert isinstance(heat_balance(
        Q(2, 'K'), Q(2, 'kJ/s')), Q['MassFlow'])

    assert isinstance(heat_balance(
        Q(2, 'kg'), Q(2, 'delta_degF')), Q['Energy'])

    assert isinstance(heat_balance(
        Q(2, 'kg/s'), Q(2, 'delta_degF')), Q['Power'])


def test_intermediate_temperatures():

    T1, T2 = intermediate_temperatures(Q(25, 'degC'), Q(10, 'degC'), Q(0.05, 'W/m/K'),
                                       Q(10, 'cm'), Q(1, 'W/m²/K'), Q(2, 'W/m²/K'), 0.7)

    assert isinstance(T1, Q['Temperature'])
    assert isinstance(T2, Q[Temperature])

    assert T2.check(Temperature)
    assert T2.check('K')
