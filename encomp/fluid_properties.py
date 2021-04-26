"""
Classes and functions relating to fluid properties.
Uses CoolProp as backend.
"""

from encomp.units import Q


def test(a: Q['Length'], b: Q['Pressure']) -> Q['Length']:

    if b > Q(1, 'bar'):
        return a * 3

    return Q(2, 'm')
