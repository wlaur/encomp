from encomp.sympy import sp, to_identifier, get_args


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
