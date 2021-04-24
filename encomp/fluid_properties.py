from encomp.units import *


def test(a: Quantity[Length], b: Quantity[Pressure]) -> Quantity[Length]:

    if b > Q(1, 'bar'):
        return a * 3

    return Q(2, 'mm')
