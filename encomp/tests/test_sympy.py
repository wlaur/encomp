import numpy as np
from encomp.units import Quantity as Q
from encomp.sympy import sp, to_identifier, get_args, get_function


def test_sympy():
    x = sp.Symbol('x', positive=True)

    x._('n,text')
    x._('n,text,j')
    x.__('n,text')
    x.__('n,text,j')

    x._('n').append('text')
    x.__('n').append('text', where='sup')
    x.__('A').append('text', where='sub')


def test_to_identifier():

    x = sp.Symbol('x', positive=True)

    s = to_identifier(x._('subscript').__('superscript'))

    assert s.isidentifier()


def test_get_args():

    x = sp.Symbol('x', positive=True)
    y = sp.Symbol('y', positive=True)

    e = x**2 + sp.sin(sp.sqrt(y))

    assert set(get_args(e)) == {'x', 'y'}


def test_decorate():

    n = sp.Symbol('n')

    n.decorate(prefix=r'\sum', prefix_sub='2', suffix_sup='i', suffix=r'\ldots')
    n._('H_2O').__('out')


def test_Quantity_integration():
    x, y, z = sp.symbols('x, y, z')

    expr = x * y / z

    result = expr.subs({
        x: Q(235, 'yard'),
        y: Q(98, 'K/m²'),
        z: Q(0.4, 'minutes')
    })

    Q.from_expr(result).dimensionality


def test_get_function():

    x, y, z = sp.symbols('x, y, z')

    expr = 25 * x * y / z

    fcn = get_function(expr, units=True)

    result_arr = fcn({
        x: Q(np.array([235, 335]), 'yard'),
        y: Q(np.array([2, 5]), 'm²'),
        z: Q(0.4, 'm³/kg')
    })
