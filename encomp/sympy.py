"""
Imports and extends the ``sympy`` library for symbolic mathematics.
Contains helper functions for converting a Sympy expression to a lambda function.
"""


from typing import Callable, Optional, Union, Literal
import re
import sympy as sp
from sympy.utilities.lambdify import lambdify, lambdastr
from sympy import default_sort_key
from functools import lru_cache
import numpy.typing as npt
from symbolic_equation import Eq as Eq_symbolic


from encomp.settings import SETTINGS
from encomp.units import Quantity
from encomp.structures import flatten


# functions that don't take mutable inputs (list, dict) are
# decorated with @lru_cache to avoid re-evaluating the functions for identical inputs

@lru_cache
def to_identifier(s: Union[sp.Symbol, str]) -> str:
    """
    Converts a Sympy symbol to a valid Python identifier.
    This function will only remove special characters.

    Latex commands are not stripped: the symbol ``\\text{kg}`` returns ``textkg`` and
    ``kg`` returns ``kg``.

    Parameters
    ----------
    s : Union[sp.Symbol, str])
        Input symbol or string representation

    Returns
    -------
    str
        Valid Python identifier created from the input symbol
    """

    if isinstance(s, sp.Basic):
        s = s.name

    s = s.replace(',', '_')
    s = s.replace('^', '__')
    s = s.replace("'", 'prime')

    # TODO: just remove all non-alphanumeric?
    remove = ['\\', '{', '}', '.', ';', '"', '*', '+', '-', '/']

    for n in remove:
        s = s.replace(n, '')

    if not s.isidentifier():
        raise ValueError(
            f'Symbol could not be converted to a valid identifer: {s}')

    return s


@lru_cache
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
                   replacements: list[Union[list[sp.Basic], tuple[sp.Symbol, sp.Basic]]]) -> sp.Basic:
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
    replacements : list[Union[list[sp.Basic], tuple[sp.Symbol, sp.Basic]]]
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


def get_sol_expr(eqns: Union[sp.Equality, list[sp.Equality]],
                 symbol: sp.Symbol) -> Optional[sp.Basic]:
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

    Returns
    -------
    Optional[sp.Basic]
        Expression that equals ``symbol``, or None in case the
        equation(s) could not be solved
    """

    if isinstance(eqns, sp.Equality):
        eqns = [eqns]

    all_eqns = eqns

    # only include unique equations that actually contains the symbol
    # this might leave multiple equations, there's no guarantee
    # that the equations can be solved
    # use default_sort_key to make sure that this is consistent
    eqns = sorted(set(filter(lambda eqn: symbol in eqn.free_symbols, eqns)),
                  key=default_sort_key)

    # in case there are multiple equations, first check if any of the
    # equations directly contain the requested symbol on the LHS
    if len(all_eqns) > 1:
        for eqn in all_eqns:
            if symbol in eqn.lhs.free_symbols:
                ret = get_sol_expr(eqn, symbol)
                if ret is not None:
                    return ret

    # if the symbol could not be solved directly from the LHS,
    # try to solve all equations instead, this will likely not work
    # use dict=True to avoid inconsistent return types
    sol = sp.solve(eqns, symbol, dict=True)

    if not sol:
        return None

    # sp.solve() returns a list of dict, there should only be one element
    # since solved for a single variable
    sol = sol[0]

    # there should only be a single solution expression
    # sort with default_sort_key to keep the output consistent
    # sympy might otherwise order expressions randomly
    return sorted(sol.values(),
                  key=default_sort_key)[0]


def get_lambda_kwargs(value_map: dict[Union[sp.Symbol, str], Union[Quantity, npt.ArrayLike]],
                      include: Optional[list[Union[sp.Symbol, str]]] = None, *,
                      units: bool = False) -> dict[str, Union[Quantity, npt.ArrayLike]]:
    """
    Returns a mapping from identifier to value (Quantity or float)
    based on the input value map (Symbol to value).
    If ``include`` is a list, only these symbols will be included.

    Parameters
    ----------
    value_map : dict[Union[sp.Symbol, str], Union[Quantity, npt.ArrayLike]]
        Mapping from symbol or symbol identifier to value
    include : Optional[list[Union[sp.Symbol, str]]], optional
        Optional list of symbols or symbol identifiers to include, by default None
    units : bool, optional
        Whether to keep the units, if False Quantity is converted
        to float (after calling ``to_base_units()``), by default False

    Returns
    -------
    dict[str, Union[Quantity, npt.ArrayLike]]
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


@lru_cache
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


@lru_cache
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
             value_map: dict[sp.Symbol, Union[Quantity, npt.ArrayLike]], *,
             units: bool = False) -> Union[Quantity, npt.ArrayLike]:
    """
    Evaluates the input expression, given the mapping of symbol to
    value in ``value_map``.

    Parameters
    ----------
    e : sp.Basic
        Input expression to evaluate
    value_map : dict[sp.Symbol, Union[Quantity, npt.ArrayLike]]
        Mapping from symbol to value for all required symbols in ``e``,
        additional symbols may be present
    units : bool, optional
        Whether to keep the units, if False Quantity is converted
        to float (after calling ``to_base_units()``), by default False

    Returns
    -------
    Union[Quantity, npt.ArrayLike]
        Value of the expression, Quantity if ``units=True`` otherwise float
    """

    fcn = get_function(e, units=units)
    return fcn(value_map)


def substitute_unknowns(e: sp.Basic,
                        knowns: set[sp.Symbol],
                        avoid: set[sp.Symbol],
                        eqns: list[sp.Equality]) -> sp.Basic:
    """
    Uses the equations ``eqns`` to substitute the unknown symbols
    in the input expression. Uses recursion to deal with nested substitutions.

    Parameters
    ----------
    e : sp.Basic
        Input expression that potentially contains unknown symbols
    knowns : set[sp.Symbol]
        Set of known symbols
    avoid : set[sp.Symbol]
        Set of symbols to avoid in the substitution expressions
    eqns : list[sp.Equality]
        List of equations that define the unknown symbols in terms of known ones

    Returns
    -------
    sp.Basic
        The substituted expression without any unknown symbols
    """

    replacements = []

    def _get_unknowns(expr):
        all_symbols = sorted(expr.free_symbols, key=default_sort_key)
        return [n for n in all_symbols if n not in (knowns | avoid) and
                n not in [m[0] for m in replacements]]

    unknowns_list = _get_unknowns(e)

    for n in unknowns_list:
        n_expr = get_sol_expr(eqns, n)

        if n_expr is None:
            raise ValueError(
                f'Symbol {n} could not be isolated based on the specified equations. '
                'Define the secondary equations so that unknown symbols are on the LHS.')

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


def get_mapping(y_symbols: Union[sp.Symbol, list[sp.Symbol]],
                x_symbols: Union[sp.Symbol, list[sp.Symbol]],
                y_solution: list[Union[list[sp.Basic], tuple[sp.Symbol, sp.Basic]]],
                value_map: dict[sp.Symbol, Union[Quantity, npt.ArrayLike]], *,
                secondary_equations: Optional[list[sp.Equality]] = None,
                units: bool = False) -> Callable:
    """
    Returns a function that maps a set of :math:`x`-values to a set of :math:`y`-values.
    The function will have input parameters corresponding to the symbols in ``x_symbols``,
    and returns a dict with values for each symbol in ``y_symbols``.
    The list ``y_solution`` contains an explicit solution for all the specified :math:`y`-values
    in terms of :math:`x`-values.
    :math:`x`-values that are not part of ``x_symbols`` (i.e. mapping parameters) will have
    values specified in ``value_map``.

    Parameters
    ----------
    y_symbols : Union[sp.Symbol, list[sp.Symbol]]
        :math:`N` symbol(s) that will be return values from the mapping function.
        In case the symbol is not directly solved in ``y_solution``, it needs to
        be explicitly defined in ``secondary_equations``.
    x_symbols : Union[sp.Symbol, list[sp.Symbol]]
        :math:`M` symbol(s) that will be input parameters to the mapping function
    y_solution : list[Union[list[sp.Basic], tuple[sp.Symbol, sp.Basic]]]
        Explicit solution for all :math:`y`-values in terms of :math:`x`-values,
        list of 2-element lists or tuples with (symbol, expression)
    value_map : dict[sp.Symbol, Union[Quantity, npt.ArrayLike]]
        Values to use for additional parameters in the solution expressions
    secondary_equations : Optional[list[sp.Equality]], optional
        Secondary equations used to evaluate :math:`x`-values that are not
        specified in the mapping ``value_map``, by default None
    units : bool, optional
        Whether to keep the units, if False Quantity is converted
        to float (after calling ``to_base_units()``), by default False

    Returns
    -------
    Callable
        Mapping function that takes :math:`M` inputs (dict) and
        returns :math:`N` outputs (dict). In case ``units=True`` the
        outputs are Quantity, otherwise they are float. The inputs are always Quantity,
        if ``units=False`` they are converted to float with ``to_base_unit()``.
    """

    if isinstance(y_symbols, sp.Symbol):
        y_symbols = [y_symbols]

    if isinstance(x_symbols, sp.Symbol):
        x_symbols = [x_symbols]

    # in case the specified y-symbols are not directly defined in y_solution,
    # they must be defined in the secondary equations
    unknown_y_symbols = [n for n in y_symbols
                         if n not in [m[0] for m in y_solution]]

    if unknown_y_symbols and secondary_equations is None:
        raise ValueError('All y-symbols must have an explicit solution in terms of '
                         'x-values in the list y_solution or a corresponding secondary equation, '
                         'pass a list of secondary equations that define these (kwarg secondary_equations)')

    N = len(y_symbols)
    M = len(x_symbols)

    # in case there are x-values not defined in value_map, check if they
    # can be solved (isolated) from the secondary equations
    all_x_symbols = sorted(flatten(n[1].free_symbols for n in y_solution),
                           key=default_sort_key)

    # sympy symbols for the base SI units (kg, m, K, etc...)
    # this can also contain custom user-defined dimensions,
    # so it needs to be re-evaluated inside this function
    # update value_map with values for the dimension symbols
    # the units have the values Q(1, unit)
    # make a copy to avoid modifying the caller's object
    value_map = value_map.copy()
    value_map.update({
        a: Quantity(1, b) for a, b in Quantity.get_dimension_symbol_map().items()
    })

    known_x_symbols = set(value_map) | set(x_symbols)
    unknown_x_symbols = [n for n in all_x_symbols
                         if n not in known_x_symbols]

    # eliminate the unknown x-symbols and substitute them with known ones
    if unknown_x_symbols:

        if secondary_equations is None:
            raise ValueError(f'Solution expression contains unknown x-symbols: {unknown_x_symbols}, '
                             'pass a list of secondary equations that define these (kwarg secondary_equations)')

        # make a new list, don't modify the input to this function
        y_solution_ = []

        for lhs, rhs in y_solution:
            y_solution_.append((lhs, substitute_unknowns(rhs,
                                                         known_x_symbols,
                                                         set(y_symbols),
                                                         secondary_equations)))

        y_solution = y_solution_

    # only keep the expressions that are needed for the specified return values
    # use the same order as the input list of y-symbols
    y_solution_ = []

    for yi in y_symbols:
        yi_expr = [n[1] for n in y_solution if n[0] == yi]

        # in case this y-symbol is in unknown_y_symbols
        if not yi_expr:

            # the unknown y-symbol should be defined on the LHS of one
            # of the secondary equations
            yi_expr = get_sol_expr(secondary_equations, yi)

            if yi_expr is None:
                raise ValueError(
                    f'Symbol {yi} could not be isolated based on the specified equations. '
                    'Define the secondary equations so that unknown symbols are on the LHS.')

            # this function can only be evaluated after the known
            # y-symbols have been evaluated
            y_solution_.append((yi, get_function(yi_expr, units=units)))

        else:
            yi_expr = yi_expr[0]
            y_solution_.append((yi, yi_expr))

    y_solution = y_solution_

    # at this point, the y_solution list contains explicit expressions
    # for all the required y-symbols
    # all x-values are either known, dimensionality symbols (kg, m, K) or mapping parameters

    # these functions can be constructed beforehand, calling
    # the mapping function will only evaluate them
    fns = [(yi, get_function(e, units=units)) for yi, e in y_solution
           if isinstance(e, sp.Basic)]

    x_symbols_set = set(x_symbols)

    def mapping(params: Optional[dict] = None):

        if params is None:
            params = {}

        if set(params) != x_symbols_set:
            raise ValueError(f'Expected parameters\n{x_symbols_set}\n'
                             f'passed\n{set(params)}')

        params = value_map | params
        ret = {yi: fn(params) for yi, fn in fns}

        # evaluate the unknown y-values based on the evaluated ones
        for yi, fn in y_solution:
            if yi not in ret and isinstance(fn, Callable):
                ret[yi] = fn(params | ret)

        # sort the output in the same order as the list of y-symbols
        ret = dict(sorted(ret.items(), key=lambda x: y_symbols.index(x[0])))

        return ret

    return mapping


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
        existing_suffix = existing_suffix.removeprefix('{').removesuffix('}')

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
