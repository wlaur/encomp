from encomp.sympy import sp


def test_sympy():
    x = sp.Symbol('x', positive=True)

    x._('n,text')
    x._('n,text,j')
    x.__('n,text')
    x.__('n,text,j')

    x._('n').append('text')
    x.__('n').append('text', where='sup')
    x.__('A').append('text', where='sub')
