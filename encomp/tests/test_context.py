from encomp.units import Q
from encomp.context import silence_stdout, quantity_format

def test_context():


    with silence_stdout():
        print('silenced')

    with quantity_format('~Lx'):

        s = str(Q(1, 'kPa'))
        assert s == '\\SI[]{1}{\\kilo\\pascal}'

    s = str(Q(1, 'kPa'))
    assert s == '1 kPa'
