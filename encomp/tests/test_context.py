from encomp.units import Quantity
from encomp.context import silence_stdout, quantity_format

def test_context():


    with silence_stdout():
        print('silenced')

    with quantity_format('~Lx'):

        s = str(Quantity(1, 'kPa'))
        assert s == '\\SI[]{1}{\\kilo\\pascal}'

    s = str(Quantity(1, 'kPa'))
    assert s == '1 kPa'
