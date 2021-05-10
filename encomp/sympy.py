"""
Imports and extends the ``sympy`` library for symbolic mathematics.
"""

from typing import Optional, Union, Literal
import re
import sympy as sp

from encomp.settings import SETTINGS


def typeset_chemical(s: str) -> str:

    # TODO: improve this function

    parts = []

    for n in re.sub('[A-Z]_\d', '|\g<0>|', s).split('|'):

        if re.match('[A-Z]_\d', n):
            parts.extend([f'{n[:-2]}', '}',  f'_{n[-1]}', '\\text{'])
        else:
            parts.append(n)

    parts = ['\\text{'] + [n for n in parts if n]

    if parts[-1] == '\\text{':
        parts = parts[:-1]

    ret = ''.join(parts)

    if ret.count('{') == ret.count('}') + 1:
        ret += '}'

    return ret


def typeset(x: Optional[str]) -> Optional[str]:
    """
    Does some additional typesetting for the input
    Latex string, for example ``\\text{}`` around
    strings and upper-case characters.

    Use comma ``,`` to separate parts of the input, for example

    .. code-block::

        input,i

    will be typeset as ``\\text{input},i``: the ``i`` is
    a separate part and is typeset with a math font.
    Spaces around commans will be removed to make sub- and
    superscripts more compact.
    Use ``~`` before a single upper-case letter to typeset it
    with a math font.

    Uses flags from ``encomp.settings`` to determine
    how to typeset the input.

    Parameters
    ----------
    x : Optional[str]
        Input Latex string

    Returns
    -------
    Optional[str]
        Output Latex string
    """

    if x is None:
        return None

    x = str(x)

    if not SETTINGS.typeset_symbol_scripts:
        return x

    parts = [n.strip() for n in x.split(',')]

    for i, p in enumerate(parts):

        # avoid typesetting single upper-case letters as text
        # if they start with ~
        if p.startswith('~') and p[1].isupper():
            parts[i] = p[1]
            continue

        # only typeset single words, also ignore Latex code
        if ' ' in p or '\\' in p:
            continue

        alpha_str = ''.join(n for n in p if n.isalpha())

        # typeset everything except 1-letter lower case as text
        typeset_text = len(alpha_str) >= 2 or len(p) == 1 and p.isupper()

        if typeset_text:

            # handle chemical compounds
            if re.match('[A-Z]', p):
                p = typeset_chemical(p)
                parts[i] = p
            else:
                parts[i] = '\\text{' + p + '}'

    return ','.join(parts)


def decorate(self,
             prefix: Optional[Union[str, int]] = None,
             suffix: Optional[Union[str, int]] = None,
             prefix_sub: Optional[Union[str, int]] = None,
             prefix_sup: Optional[Union[str, int]] = None,
             suffix_sub: Optional[Union[str, int]] = None,
             suffix_sup: Optional[Union[str, int]] = None
             ) -> sp.Symbol:
    """
    Method for ``sp.Symbol`` that decorates a symbol with
    subscripts and/or superscripts before or after the symbol.
    Returns a new symbol object with the same assumptions (i.e. real,
    positive, complex, etc...) as the input.

    Using LaTeX syntax supported by ``sympy``:

    .. code-block::

        {prefix}^{prefix_sub}_{prefix_sup}{symbol}_{suffix_sub}^{suffix_sup}{suffix}

    ``symbol`` is the input symbol.
    The ``prefix`` and ``suffix`` parts are added without ``_`` or ``^``.
    Each of the parts (except ``symbol``) can be empty.

    The decorations can be string or integer, floats are not allowed.
    In case the input symbol already contains sub- or superscripts,
    the decorations are not appended to those. Instead, a new level
    is introduced. To keep things simple, make sure that the input symbol
    is a simple symbol.

    Use the ``append`` method (with ``where='sub'|'sup'`) to append
    to an existing sub- or superscript in the suffix.

    Parameters
    ----------
    prefix : Optional[Union[str, int]], optional
        Prefix added before the symbol, by default None
    suffix : Optional[Union[str, int]], optional
        Suffix added after the symbol, by default None
    prefix_sub : Optional[Union[str, int]], optional
        Subscript prefix before the symbol and after ``prefix``, by default None
    prefix_sup : Optional[Union[str, int]], optional
        Superscript prefix before the symbol and after ``prefix``, by default None
    suffix_sub : Optional[Union[str, int]], optional
        Subscript suffix after the symbol and before ``suffix``, by default None
    suffix_sup : Optional[Union[str, int]], optional
        Superscript suffix after the symbol and before ``suffix``, by default None

    Returns
    -------
    sp.Symbol
        A new symbol with the same assumptions as the input, with added decorations
    """

    parts = [
        prefix,
        typeset(prefix_sup),
        typeset(prefix_sub),
        self.name,
        typeset(suffix_sub),
        typeset(suffix_sup),
        suffix
    ]

    delimiters = ['', '^', '_', '', '_', '^', '']

    decorated_parts = []

    for p, d in zip(parts, delimiters):

        if p is None:
            continue

        p = str(p)

        # don't add extra braces around the base symbol
        if p != self.name:
            p = '{' + p + '}'

        decorated_parts.append(d + p)

    decorated_symbol = ''.join(decorated_parts)

    # assumptions0 contains assumptions that are not None
    return sp.Symbol(decorated_symbol, **self.assumptions0)


def append(self, s: Union[str, int],
           where: Literal['sub', 'sup'] = 'sub') -> sp.Symbol:
    """
    Adds the input ``s`` to an existing sub- or superscript.
    Does not append to prefixes.
    Creates the sub- or superscript if it does not exist.

    Parameters
    ----------
    s : Union[str, int]
        Text or index to be added
    where : Literal['sub', 'sup'], optional
        Whether to append to the subscript or superscript, by default 'sub'

    Returns
    -------
    sp.Symbol
        A new symbol with the same assumptions as the input, with updated sub- or superscript
    """

    if where == 'sub':
        delimiter = '_'
    else:
        delimiter = '^'

    symbol = self.name

    s = typeset(str(s))

    if delimiter not in symbol:
        decorated_parts = [symbol, delimiter, '{' + s + '}']

    else:
        *base_symbol, existing_suffix = symbol.split(delimiter)

        base_symbol = ''.join(base_symbol)

        # assume that the input Latex symbol is correct, don't deal with unbalanced braces
        # this is a builtin str method in Python 3.9+
        if existing_suffix.startswith('{') and existing_suffix.endswith('}'):
            existing_suffix = existing_suffix[1:-1]

        existing_suffix += str(s)

        decorated_parts = [base_symbol, delimiter, '{' + existing_suffix + '}']

    decorated_symbol = ''.join(decorated_parts)
    return sp.Symbol(decorated_symbol, **self.assumptions0)


# additional methods for sp.Symbol
# these methods (potentially) return a new sp.Symbol object
# sympy keeps an internal symbol register: two symbols with the same
# name and assumptions refer to the same Python object
sp.Symbol.decorate = decorate
sp.Symbol.append = append

# shorthand to add suffixes to an existing symbol
sp.Symbol._ = lambda s, x: s.append(x, where='sub')
sp.Symbol.__ = lambda s, x: s.append(x, where='sup')

# shorthand to add Delta before a symbol
sp.Symbol.delta = lambda s: s.decorate(prefix='\\Delta')


def simplify_exponents(expr: sp.Basic) -> sp.Basic:
    """
    Simplifies an expression by combining float and int exponents.
    This is not done automatically by Sympy.

    Adapted from
    https://stackoverflow.com/questions/54243832/sympy-wont-simplify-or-expand-exponential-with-decimals.

    Parameters
    ----------
    expr : sp.Basic
        A Sympy expression, potentially containing mixed float and int exponents

    Returns
    -------
    sp.Basic
        Simplified expression with float and int exponents combined
    """

    def rewrite(expr, new_args):

        new_args = list(new_args)
        pow_val = new_args[1]
        pow_val_int = int(new_args[1])

        if pow_val.epsilon_eq(pow_val_int):
            new_args[1] = sp.Integer(pow_val_int)

        return type(expr)(*new_args)

    def isfloatpow(expr):
        return expr.is_Pow and expr.args[1].is_Float

    if not expr.args:
        return expr

    else:
        new_args = tuple(simplify_exponents(a) for a in expr.args)

        if isfloatpow(expr):
            return rewrite(expr, new_args)
        else:
            return type(expr)(*new_args)
