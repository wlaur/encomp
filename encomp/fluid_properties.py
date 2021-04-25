from encomp.units import Q
from encomp.utypes import Length, Pressure


def test(a: Q[Length], b: Q[Pressure]) -> Q[Length]:

    if b > Q(1, 'bar'):
        return a * 3

    return Q(2, 'm')
