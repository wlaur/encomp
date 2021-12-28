from encomp.units import Q

from encomp.gases import convert_gas_volume


def test_convert_gas_volume():

    ret = convert_gas_volume(
        Q(1, 'm3'),
        'N',
        (Q(2, 'bar'), Q(25, 'degC'))
    )

    assert ret.check(Q(0, 'liter'))
