"""
Imports and extends the ``sympy`` library for symbolic mathematics.
Contains tools for converting Sympy expressions to Python modules and functions.
"""

import re
from typing import Callable, Optional, Union, Literal
from functools import lru_cache

import numpy as np
import sympy as sp
from sympy import default_sort_key

try:
    from sympy.utilities.lambdify import lambdify, lambdastr
except ImportError:

    def lambdify(*args, **kwargs) -> None:
        raise NotImplementedError()

    lambdastr = lambdify

from symbolic_equation import Eq as Eq_symbolic

from encomp.settings import SETTINGS
from encomp.units import Quantity


# TODO: what is this map used for?
# to_identifier already uses lru_cache
_IDENTIFIER_MAP: dict[str, str] = {}


@lru_cache()
def to_identifier(s: Union[sp.Symbol, str]) -> str:
    """
    Converts a Sympy symbol to a valid Python identifier.
    This function will only remove special characters.

    The Latex command ``\\text{}`` is replaced with ``T``. This is
    done to differentiate between symbols ``\\text{kg}`` (returns ``Tkg``)
    and ``kg`` (returns ``kg``).

    Parameters
    ----------
    s : Union[sp.Symbol, str])
        Input symbol or string representation

    Returns
    -------
    str
        Valid Python identifier created from the input symbol
    """

    s_inp = s

    if isinstance(s, str) and s in _IDENTIFIER_MAP:
        return _IDENTIFIER_MAP[s]

    # assume that input strings are already identifiers
    if isinstance(s, str):
        return s

    s = s.name

    s_orig = s

    s = s.replace(',', '_')
    s = s.replace('^', '__')
    s = s.replace("'", 'prime')

    # need to differentiate between symbols "\text{m}" and "m"
    # the string "text" is a bit long, replace with "T"
    s = s.replace('text', 'T')

    # the substring "lambda" cannot exist in the identifier
    s = s.replace('lambda', 'lam')

    # remove all non-alphanumeric or _
    s = re.sub(r'\W+', '', s)

    if not s.isidentifier():
        raise ValueError(
            f'Symbol could not be converted to a valid Python identifer: {s_orig}')

    _IDENTIFIER_MAP[str(s_inp)] = s

    return s


@lru_cache()
def get_args(e: sp.Basic) -> list[str]:
    """
    Returns a sorted list of identifiers for
    each free symbol in the input expression.
    The sort order is according to the string outputs
    from :py:func:`encomp.sympy.to_identifier`.

    Parameters
    ----------
    e : sp.Basic
        Input expression

    Returns
    -------
    list[str]
        Sorted list of identifiers for each free symbol
    """

    return sorted(list(map(to_identifier,
                           e.free_symbols)))


def recursive_subs(e: sp.Basic,
                   replacements: list[tuple[sp.Symbol, sp.Basic]]) -> sp.Basic:
    """
    Substitute the expressions in ``replacements`` recursively.
    This might not be necessary in all cases, Sympy's builtin
    ``subs()`` method should also do this recursively.

    .. note::
        The order of the tuples in ``replacements`` might matter,
        make sure to order these sensibly in case the expression contains
        a lot of nested substitutions.

    Parameters
    ----------
    e : sp.Basic
        Input expression
    replacements : list[tuple[sp.Symbol, sp.Basic]]
        List of replacements: ``symbol, replace``

    Returns
    -------
    sp.Basic
        Substituted expression
    """

    for _ in range(0, len(replacements) + 1):

        new_e = e.subs(replacements)

        if new_e == e:
            return new_e
        else:
            e = new_e

    return new_e


def simplify_exponents(e: sp.Basic) -> sp.Basic:
    """
    Simplifies an expression by combining float and int exponents.
    This is not done automatically by Sympy.

    Adapted from
    https://stackoverflow.com/questions/54243832/sympy-wont-simplify-or-expand-exponential-with-decimals

    Parameters
    ----------
    e : sp.Basic
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

    def is_float_pow(expr):
        return expr.is_Pow and expr.args[1].is_Float

    if not e.args:
        return e

    else:
        new_args = tuple(simplify_exponents(a) for a in e.args)

        if is_float_pow(e):
            return rewrite(e, new_args)
        else:
            return type(e)(*new_args)


def get_sol_expr(eqns: Union[sp.Equality, list[sp.Equality]],
                 symbol: sp.Symbol,
                 avoid: Optional[set[sp.Symbol]] = None) -> Optional[sp.Basic]:
    """
    Wrapper around ``sp.solve`` that returns the solution expression
    for a *single* symbol, or None in case Sympy could not solve for the specified symbol.
    Only considers equations in the input list that actually contains the symbol.
    Prefers to use equations that contain ``symbol`` on the LHS.

    Parameters
    ----------
    eqns : Union[sp.Equality, list[sp.Equality]]
        List of equations or a single equation
    symbol : sp.Symbol
        Symbol to solve for (isolate)
    avoid : Optional[set[sp.Symbol]], optional
        Set of symbols to avoid in the substitution expressions, by default None

    Returns
    -------
    Optional[sp.Basic]
        Expression that equals ``symbol``, or None in case the
        equation(s) could not be solved
    """

    if avoid is None:
        avoid = set()

    if isinstance(eqns, sp.Equality):
        eqns = [eqns]

    # only include unique equations that actually contains the symbol,
    # preferably on the LHS
    # this might leave multiple equations, there's no guarantee
    # that the equations can be solved
    # sort by the number of free symbols, use default_sort_key as the
    # secondary sort key to make sure that the order is consistent
    def eqn_simplicity(eqn):
        return len(eqn.lhs.free_symbols), default_sort_key(eqn)

    eqns = sorted(set(filter(lambda eqn: symbol in eqn.free_symbols, eqns)),
                  key=eqn_simplicity)

    # in case there are multiple equations containing the requested symbol,
    # first check if any of the equations directly contain the symbol on the LHS
    if len(eqns) > 1:
        for eqn in eqns:

            if symbol in eqn.lhs.free_symbols:

                ret = get_sol_expr(eqn, symbol)

                # don't return an expression that contains symbols to be avoided
                if ret is not None and not (avoid & ret.free_symbols):
                    return ret

    # if the symbol could not be solved directly from the LHS of a single equation,
    # try to solve all relevant (i.e. containing the requested symbol) equations instead
    # use dict=True to avoid inconsistent return types from sp.solve
    # make sure to define the assumptions correctly for all symbols, otherwise the
    # Sympy solver might not be able to find an explicit solution
    sol = sp.solve(eqns, symbol, dict=True)

    if not sol:
        return None

    # sp.solve() returns a list of dict, there should only be one element
    # since we solved for a single variable
    sol = sol[0]

    # hopefully, there is only be a single key in this dict
    # (quadratic equations might have multiple solutions, etc...)
    # sort with default_sort_key to keep the output consistent
    # Sympy might otherwise order expressions randomly
    return sorted(sol.values(),
                  key=default_sort_key)[0]


def get_lambda_kwargs(value_map: dict[Union[sp.Symbol, str], Union[Quantity, np.ndarray]],
                      include: Optional[list[Union[sp.Symbol, str]]] = None, *,
                      units: bool = False) -> dict[str, Union[Quantity, np.ndarray]]:
    """
    Returns a mapping from identifier to value (Quantity or float)
    based on the input value map (Symbol to value).
    If ``include`` is a list, only these symbols will be included.

    Parameters
    ----------
    value_map : dict[Union[sp.Symbol, str], Union[Quantity, np.ndarray]]
        Mapping from symbol or symbol identifier to value
    include : Optional[list[Union[sp.Symbol, str]]], optional
        Optional list of symbols or symbol identifiers to include, by default None
    units : bool, optional
        Whether to keep the units, if False Quantity is converted
        to float (after calling ``to_base_units()``), by default False

    Returns
    -------
    dict[str, Union[Quantity, np.ndarray]]
        Mapping from identifier to value
    """

    if include is not None:
        include = [to_identifier(n) for n in include]

    def _get_val(x):

        if not isinstance(x, Quantity):
            return x

        if units:
            return x.to_base_units()
        else:
            return x.to_base_units().m

    return {to_identifier(a): _get_val(b)
            for a, b in value_map.items()
            if include is None or to_identifier(a) in include}


@lru_cache()
def get_lambda(e: sp.Basic, *,
               to_str: bool = False) -> tuple[Union[Callable, str], list[str]]:
    """
    Converts the input expression to a lambda function
    with valid identifiers as parameter names.

    Parameters
    ----------
    e : sp.Basic
        Input expression
    to_str : bool, optional
        Whether to return the string representation of the lambda function, by default False

    Returns
    -------
    Tuple[Union[Callable, str], list[str]]
        The lambda function (or string representation) and the list
        of parameters to the function
    """

    # sorted list of function parameters (as valid identifiers)
    args = get_args(e)

    # substitute the symbols with the identifier version,
    # otherwise they will be converted to dummy identifiers (even if dummify=False)
    e_identifiers = e.subs({n: sp.Symbol(to_identifier(n),
                                         **n.assumptions0)
                            for n in e.free_symbols})

    _lambda_func = lambdastr if to_str else lambdify
    fcn = _lambda_func(args, e_identifiers, dummify=False)

    return fcn, args


def get_lambda_matrix(M: sp.MutableDenseMatrix) -> tuple[str, list[str]]:
    """
    Converts the input matrix into a lambda function that returns
    an array.
    Converts the matrix to Python source, it is not possible to
    use in-memory lambda functions for this. Use ``eval(src)``
    on the output from this function to create a function object.

    Parameters
    ----------
    M : sp.MutableDenseMatrix
        Input matrix

    Returns
    -------
    tuple[str, list[str]]
        Python source code for the function and a list of parameters
    """

    args = set()

    nrows, ncols = M.shape
    arr = np.zeros((nrows, ncols), dtype=object)

    for i in range(nrows):
        for j in range(ncols):

            fcn_str, n_args = get_lambda(M[i, j], to_str=True)
            args |= set(n_args)

            if isinstance(fcn_str, str):

                # remove the "lambda x, y, x:" part and extra parens,
                # they are added back later
                fcn_str = (
                    fcn_str
                    .split(':', 1)[-1]
                    .strip()
                    .removeprefix('(')
                    .removesuffix(')')
                )

                arr[i, j] = fcn_str

    # remove quotes around strings, they are mathematical expressions
    funcs = str(arr.tolist()).replace("'", '').replace('"', '')

    # TODO: "VisibleDeprecationWarning: Creating an ndarray from ragged..."
    # when mixing input vectors and floats
    func_src = f'lambda {", ".join(args)}: np.array({funcs})'

    return func_src, sorted(args)


@lru_cache()
def get_function(e: sp.Basic, *, units: bool = False) -> Callable:
    """
    Wrapper around :py:func:`encomp.sympy.get_lambda` that
    handles inputs and potential units.

    Parameters
    ----------
    e : sp.Basic
        Input expression
    units : bool, optional
        Whether to keep the units, if False Quantity is converted
        to float (after calling ``to_base_units()``), by default False

    Returns
    -------
    Callable
        Function that evaluates the input expression, can be called
        with extra kwargs. The kwargs can be a dict with mapping
        from symbol to value.
    """

    fcn, args = get_lambda(e)

    def expr_func(params):
        return fcn(**get_lambda_kwargs(params, args, units=units))

    return expr_func


def evaluate(e: sp.Basic,
             value_map: dict[sp.Symbol, Union[Quantity, np.ndarray]], *,
             units: bool = False) -> Union[Quantity, np.ndarray]:
    """
    Evaluates the input expression, given the mapping of symbol to
    value in ``value_map``.

    Parameters
    ----------
    e : sp.Basic
        Input expression to evaluate
    value_map : dict[sp.Symbol, Union[Quantity, np.ndarray]]
        Mapping from symbol to value for all required symbols in ``e``,
        additional symbols may be present
    units : bool, optional
        Whether to keep the units, if False Quantity is converted
        to float (after calling ``to_base_units()``), by default False

    Returns
    -------
    Union[Quantity, np.ndarray]
        Value of the expression, Quantity if ``units=True`` otherwise float
    """

    fcn = get_function(e, units=units)
    return fcn(value_map)


def substitute_unknowns(e: sp.Basic,
                        knowns: set[sp.Symbol],
                        eqns: list[sp.Equality],
                        avoid: Optional[set[sp.Symbol]] = None) -> sp.Basic:
    """
    Uses the equations ``eqns`` to substitute the unknown symbols
    in the input expression. Uses recursion to deal with nested substitutions.

    Parameters
    ----------
    e : sp.Basic
        Input expression that potentially contains unknown symbols
    knowns : set[sp.Symbol]
        Set of known symbols
    eqns : list[sp.Equality]
        List of equations that define the unknown symbols in terms of known ones
    avoid : Optional[set[sp.Symbol]], optional
        Set of symbols to avoid in the substitution expressions, by default None

    Returns
    -------
    sp.Basic
        The substituted expression without any unknown symbols
    """

    if avoid is None:
        avoid = set()

    replacements: list[tuple[sp.Symbol, sp.Basic]] = []

    def _get_unknowns(expr):

        all_symbols: list[sp.Symbol] = sorted(expr.free_symbols,
                                              key=default_sort_key)

        already_replaced = [m[0] for m in replacements]

        return [n for n in all_symbols if n not in (knowns | avoid) and
                n not in already_replaced]

    unknowns_list = _get_unknowns(e)

    for n in unknowns_list:
        n_expr = get_sol_expr(eqns, n, avoid=avoid)

        if n_expr is None:
            raise ValueError(
                f'Symbol {n} could not be isolated based on the specified equations.')

        # check if the expression for n contains even more unknown symbols
        # extend the list that is iterated over to account for these symbols
        # this will not loop infinitely since the replacements are accounted
        # for in the _get_unknowns function
        additional_unknowns = _get_unknowns(n_expr)
        unknowns_list.extend(additional_unknowns)

        replacements.append((n, n_expr))

    # the replacements list is reversed, since the "deepest" level
    # of substitutions must be done first
    replacements = replacements[::-1]

    # recursively apply the substitutions until the expression no longer changes
    # since the replacements list is ordered from deep → shallow, this
    # will substitute everything correctly
    return recursive_subs(e, replacements)


def typeset_chemical(s: str) -> str:
    """
    Typesets chemical formulas using Latex.

    Parameters
    ----------
    s : str
        Input string

    Returns
    -------
    str
        Output string with chemical formulas typeset correctly
    """

    parts = []

    for n in re.sub(r'[A-Z]_\d', r'|\g<0>|', s).split('|'):

        if re.match(r'[A-Z]_\d', n):
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


def typeset(x: Union[str, int]) -> str:
    """
    Does some additional typesetting for the input
    Latex string, for example ``\\text{}`` around
    strings and upper-case characters.

    Use comma ``,`` to separate parts of the input, for example

    .. code:: none

        input,i

    will be typeset as ``\\text{input},i``: the ``i`` is
    a separate part and is typeset with a math font.
    Spaces around commas will be removed to make sub- and
    superscripts more compact.
    Use ``~`` before a single upper-case letter to typeset it
    with a math font.

    Uses flags from ``encomp.settings`` to determine
    how to typeset the input.

    Parameters
    ----------
    x : Union[str, int]
        Input string or int (will be converted to str)

    Returns
    -------
    str
        Output Latex string
    """

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
        typeset_text = len(alpha_str) >= 2 or (
            len(alpha_str) == 1 and alpha_str.isupper())

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

    .. code:: none

        {prefix}^{prefix_sub}_{prefix_sup}{symbol}_{suffix_sub}^{suffix_sup}{suffix}

    ``symbol`` is the input symbol.
    The ``prefix`` and ``suffix`` parts are added without ``_`` or ``^``.
    Each of the parts (except ``symbol``) can be empty.

    The decorations can be string or integer, floats are not allowed.
    In case the input symbol already contains sub- or superscripts,
    the decorations are not appended to those. Instead, a new level
    is introduced. To keep things simple, make sure that the input symbol
    is a simple symbol.

    Use the ``append`` method to append to an existing sub- or superscript in the suffix.

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
        typeset(prefix_sup) if prefix_sup is not None else None,
        typeset(prefix_sub) if prefix_sub is not None else None,
        self.name,
        typeset(suffix_sub) if suffix_sub is not None else None,
        typeset(suffix_sup) if suffix_sup is not None else None,
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

    s = typeset(s)

    if delimiter not in symbol:
        decorated_parts = [symbol, delimiter, '{' + s + '}']

    else:
        *base_symbol, existing_suffix = symbol.split(delimiter)

        base_symbol = ''.join(base_symbol)

        # assume that the input Latex symbol is correct, don't deal with unbalanced braces
        existing_suffix = existing_suffix.removeprefix('{').removesuffix('}')

        existing_suffix += str(s)

        decorated_parts = [base_symbol, delimiter, '{' + existing_suffix + '}']

    decorated_symbol = ''.join(decorated_parts)
    return sp.Symbol(decorated_symbol, **self.assumptions0)


# additional methods for sp.Symbol
# these methods (potentially) return a new sp.Symbol object
# sympy keeps an internal symbol register: two symbols with the same
# name and assumptions refer to the same Python object

sp.Symbol.decorate = decorate  # type: ignore
sp.Symbol.append = append  # type: ignore

# shorthand to add suffixes to an existing symbol
sp.Symbol._ = lambda s, x: s.append(x, where='sub')  # type: ignore
sp.Symbol.__ = lambda s, x: s.append(x, where='sup')  # type: ignore

# shorthand to add Delta before a symbol
sp.Symbol.delta = lambda s: s.decorate(prefix='\\Delta')  # type: ignore


def display_equation(eqn: sp.Equality,
                     tag: Optional[str] = None, **kwargs) -> None:
    """
    Displays a Sympy equation (``sp.Equality``) using
    the package ``symbolic_equation``, which displays
    multi-line equations.
    ``kwargs`` are passed to ``symbolic_equation.Eq``.
    Calls ``IPython.display.display``.

    Parameters
    ----------
    eqn : sp.Equality
        Equation to display
    tag : Optional[str], optional
        Equation tag displayed on the right side inside parens, by default None
    """
    from IPython.display import display

    eqn = Eq_symbolic(lhs=eqn.lhs,
                      rhs=eqn.rhs,
                      tag=tag, **kwargs)
    display(eqn)
