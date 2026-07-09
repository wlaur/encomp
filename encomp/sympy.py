# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false
"""
Imports and extends the ``sympy`` library for symbolic mathematics.
Contains tools for converting SymPy expressions to Python modules and functions.
"""

import keyword
import re
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from typing import Any, Literal, Self, cast, overload

import numpy as np
import sympy as sp
from sympy import default_sort_key
from sympy.utilities.lambdify import lambdastr, lambdify

from .settings import SETTINGS
from .units import Quantity
from .utypes import Numpy1DArray

__all__ = [
    "Symbol",
    "evaluate",
    "get_args",
    "get_function",
    "get_lambda",
    "get_lambda_kwargs",
    "get_lambda_matrix",
    "get_sol_expr",
    "recursive_subs",
    "simplify_exponents",
    "sp",
    "substitute_unknowns",
    "symbols",
    "to_identifier",
    "typeset",
    "typeset_chemical",
]


@lru_cache
def to_identifier(s: object) -> str:
    """
    Converts a SymPy symbol to a valid Python identifier.
    This function will only remove special characters.

    The Latex command ``\\text{}`` is replaced with ``T``. This is
    done to differentiate between symbols ``\\text{kg}`` (returns ``Tkg``)
    and ``kg`` (returns ``kg``).

    Parameters
    ----------
    s : object
        Input symbol or string representation

    Returns
    -------
    str
        Valid Python identifier created from the input symbol
    """

    if isinstance(s, str):
        s_text = s
    else:
        name = getattr(s, "name", None)
        if not isinstance(name, str):
            raise TypeError(f"Expected a SymPy symbol or string, got {type(s).__name__}")
        s_text = name
    s_orig = s_text

    s_text = s_text.replace(",", "_")
    s_text = s_text.replace("^", "__")
    s_text = s_text.replace("'", "prime")

    # collapse the LaTeX \text{...} token (used to distinguish "\text{m}" from "m") to "T".
    # anchored to the backslash so a plain symbol merely CONTAINING "text" (e.g. "context")
    # is not corrupted
    s_text = s_text.replace(r"\text", "T")

    # remove all non-alphanumeric or _
    s_text = re.sub(r"\W+", "", s_text)

    # a Python keyword (e.g. "lambda", "class") is a valid identifier syntactically but
    # cannot be used as a variable/parameter name -- suffix it so it can. handles all
    # keywords, not just "lambda", and does not corrupt names that merely contain one
    if keyword.iskeyword(s_text):
        s_text = f"{s_text}_"

    if not s_text.isidentifier():
        raise ValueError(f"Symbol could not be converted to a valid Python identifier: {s_orig}")

    return s_text


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

    symbols = e.free_symbols
    identifiers = sorted(map(to_identifier, symbols))
    # to_identifier is a lossy per-symbol mapping, so two DISTINCT symbols can collapse to the
    # same identifier (e.g. a bare keyword "lambda" and its twin "lambda_"). Refuse rather than
    # silently emit a lambda with a duplicate parameter that merges the two symbols.
    if len(set(identifiers)) != len(symbols):
        raise ValueError(
            f"Symbols in {e} map to colliding identifiers {identifiers}; "
            "rename the symbols so their to_identifier() forms are distinct"
        )
    return identifiers


def recursive_subs(e: sp.Basic, replacements: list[tuple[sp.Symbol, sp.Basic]]) -> sp.Basic:
    """
    Substitute the expressions in ``replacements`` recursively.
    This might not be necessary in all cases, SymPy's builtin
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

    new_e = None

    for _ in range(0, len(replacements) + 1):
        new_e = e.subs(replacements)

        if new_e == e:
            return new_e
        else:
            e = new_e

    raise ValueError(f"Recursive substitution did not converge for {e=}, {replacements=}")


def simplify_exponents(e: sp.Basic) -> sp.Basic:
    """
    Simplifies an expression by combining float and int exponents.
    This is not done automatically by SymPy.

    Adapted from
    https://stackoverflow.com/questions/54243832/sympy-wont-simplify-or-expand-exponential-with-decimals

    Parameters
    ----------
    e : sp.Basic
        A SymPy expression, potentially containing mixed float and int exponents

    Returns
    -------
    sp.Basic
        Simplified expression with float and int exponents combined
    """

    def rewrite(expr: sp.Basic, new_args: tuple[sp.Basic, ...]) -> sp.Basic:
        new_args_list = list(new_args)
        pow_val = new_args_list[1]
        pow_val_int = round(float(cast(Any, new_args_list[1])))

        if cast(Any, pow_val).epsilon_eq(pow_val_int):
            new_args_list[1] = sp.Integer(pow_val_int)

        return type(expr)(*new_args_list)

    def is_float_pow(expr: sp.Basic) -> bool:
        return bool(cast(Any, expr).is_Pow and cast(Any, expr).args[1].is_Float)

    if not e.args:
        return e

    else:
        new_args = tuple(simplify_exponents(a) for a in e.args)

        if is_float_pow(e):
            return rewrite(e, new_args)
        else:
            return type(e)(*new_args)


def get_sol_expr(
    equations: sp.Equality | list[sp.Equality],
    symbol: sp.Symbol,
    avoid: set[sp.Symbol] | None = None,
) -> sp.Basic | None:
    """
    Wrapper around ``sp.solve`` that returns the solution expression
    for a *single* symbol, or None in case SymPy
    could not solve for the specified symbol.
    Only considers equations in the input list that actually contains the symbol.
    Prefers to use equations that contain ``symbol`` on the LHS.

    Parameters
    ----------
    equations : sp.Equality | list[sp.Equality]
        List of equations or a single equation
    symbol : sp.Symbol
        Symbol to solve for (isolate)
    avoid : set[sp.Symbol] | None, optional
        Set of symbols to avoid in the substitution expressions, by default None

    Returns
    -------
    sp.Basic | None
        Expression that equals ``symbol``, or None in case the
        equation(s) could not be solved
    """

    if avoid is None:
        avoid = set()

    if isinstance(equations, sp.Equality):
        equations = [equations]

    # only include unique equations that actually contains the symbol,
    # preferably on the LHS
    # this might leave multiple equations, there's no guarantee
    # that the equations can be solved
    # sort by the number of free symbols, use default_sort_key as the
    # secondary sort key to make sure that the order is consistent
    def eqn_simplicity(eqn: sp.Eq) -> tuple[int, tuple[Any, ...]]:
        return len(eqn.lhs.free_symbols), default_sort_key(eqn)

    equations = sorted(set(filter(lambda eqn: symbol in eqn.free_symbols, equations)), key=eqn_simplicity)

    # in case there are multiple equations containing the requested symbol,
    # first check if any of the equations directly contain the symbol on the LHS
    if len(equations) > 1:
        for eqn in equations:
            if symbol in eqn.lhs.free_symbols:
                ret = get_sol_expr(eqn, symbol)

                # don't return an expression that contains symbols to be avoided
                if ret is not None and not (avoid & ret.free_symbols):
                    return ret

    # if the symbol could not be solved directly from the LHS of a single equation,
    # try to solve all relevant (i.e. containing the requested symbol) equations instead
    # use dict=True to avoid inconsistent return types from sp.solve
    # make sure to define the assumptions correctly for all symbols, otherwise the
    # SymPy solver might not be able to find an explicit solution
    sol = sp.solve(equations, symbol, dict=True)

    if not sol:
        return None

    # sp.solve() returns a list of dict, there should only be one element
    # since we solved for a single variable
    sol = sol[0]

    # hopefully, there is only be a single key in this dict
    # (quadratic equations might have multiple solutions, etc...)
    # sort with default_sort_key to keep the output consistent
    # SymPy might otherwise order expressions randomly
    return cast(sp.Basic, sorted(sol.values(), key=default_sort_key)[0])


def get_lambda_kwargs(
    value_map: dict[sp.Symbol | str, Quantity | np.ndarray],
    include: Sequence[sp.Symbol | str] | None = None,
    *,
    units: bool = False,
) -> dict[str, Quantity | np.ndarray | float]:
    """
    Returns a mapping from identifier to value (Quantity or float)
    based on the input value map (Symbol to value).
    If ``include`` is a list, only these symbols will be included.

    Parameters
    ----------
    value_map : dict[sp.Symbol | str, Quantity | np.ndarray]
        Mapping from symbol or symbol identifier to value
    include : Sequence[sp.Symbol | str] | None, optional
        Optional sequence of symbols or symbol identifiers to include, by default None
    units : bool, optional
        Whether to keep the units, if False Quantity is converted
        to float (after calling ``to_base_units()``), by default False

    Returns
    -------
    dict[str, Quantity | np.ndarray | float]
        Mapping from identifier to value
    """

    if include is not None:
        include = [to_identifier(n) for n in include]

    def _get_val(
        x: Quantity[Any, Numpy1DArray] | Numpy1DArray,
    ) -> Quantity | Numpy1DArray | float:
        if not isinstance(x, Quantity):
            return x

        if units:
            return x.to_base_units()
        else:
            return x.to_base_units().m

    return {
        to_identifier(a): _get_val(b) for a, b in value_map.items() if include is None or to_identifier(a) in include
    }


@overload
def get_lambda(e: sp.Basic, *, to_str: Literal[True]) -> tuple[str, list[str]]: ...


@overload
def get_lambda(e: sp.Basic, *, to_str: Literal[False]) -> tuple[Callable, list[str]]: ...


@overload
def get_lambda(e: sp.Basic) -> tuple[Callable, list[str]]: ...


@lru_cache
def get_lambda(e: sp.Basic, *, to_str: bool = False) -> tuple[Callable | str, list[str]]:
    """
    Converts the input expression to a lambda function
    with valid identifiers as parameter names.

    Parameters
    ----------
    e : sp.Basic
        Input expression
    to_str : bool, optional
        Whether to return the string representation of the lambda function,
        by default False

    Returns
    -------
    tuple[Callable | str, list[str]]
        The lambda function (or string representation) and the list
        of parameters to the function
    """

    # sorted list of function parameters (as valid identifiers)
    args = get_args(e)

    # substitute the symbols with the identifier version,
    # otherwise they will be converted to dummy identifiers (even if dummify=False)
    e_identifiers = e.subs({n: sp.Symbol(to_identifier(n), **n.assumptions0) for n in e.free_symbols})

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

    # catch identifier collisions across the WHOLE matrix up front: a per-cell get_lambda only
    # sees its own symbols, but two symbols in different cells could collapse to one identifier
    # and silently merge (the args set below would dedupe them without a duplicate-param error)
    matrix_symbols = cast("set[sp.Symbol]", M.free_symbols)
    if len({to_identifier(s) for s in matrix_symbols}) != len(matrix_symbols):
        raise ValueError(
            "Symbols in the matrix map to colliding identifiers; "
            "rename them so their to_identifier() forms are distinct"
        )

    args = set()

    nrows, ncols = M.shape
    arr = np.zeros((nrows, ncols), dtype=object)

    for i in range(nrows):
        for j in range(ncols):
            fcn_str, n_args = get_lambda(cast("sp.Basic", M[i, j]), to_str=True)
            args |= set(n_args)

            # remove the "lambda x, y, x:" part and extra parens,
            # they are added back later
            fcn_str = fcn_str.split(":", 1)[-1].strip().removeprefix("(").removesuffix(")")

            arr[i, j] = fcn_str

    # remove quotes around strings, they are mathematical expressions
    funcs = str(arr.tolist()).replace("'", "").replace('"', "")

    # TODO: "VisibleDeprecationWarning: Creating an ndarray from ragged..."
    # when mixing input vectors and floats
    # the signature MUST use the same order as the returned parameter list, otherwise a
    # caller binding the returned params positionally hits a different order than the lambda
    # expects (args is a set: its iteration order is arbitrary and varies with PYTHONHASHSEED)
    sorted_args = sorted(args)
    func_src = f"lambda {', '.join(sorted_args)}: np.array({funcs})"

    return func_src, sorted_args


@lru_cache
def get_function(e: sp.Basic, *, units: bool = False) -> Callable[[dict[Any, Any]], Any]:
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
        Function that evaluates the input expression from a mapping of
        symbols to values.
    """

    fcn, args = get_lambda(e)

    def expr_func(params: dict[Any, Any]) -> Any:  # noqa: ANN401
        return fcn(**get_lambda_kwargs(params, args, units=units))

    return expr_func


def evaluate(
    e: sp.Basic,
    value_map: dict[sp.Symbol, Quantity | np.ndarray],
    *,
    units: bool = False,
) -> Quantity | np.ndarray | float:
    """
    Evaluates the input expression, given the mapping of symbol to
    value in ``value_map``.

    Parameters
    ----------
    e : sp.Basic
        Input expression to evaluate
    value_map : dict[sp.Symbol, Quantity | np.ndarray]
        Mapping from symbol to value for all required symbols in ``e``,
        additional symbols may be present
    units : bool, optional
        Whether to keep the units, if False Quantity is converted
        to float (after calling ``to_base_units()``), by default False

    Returns
    -------
    Quantity | np.ndarray
        Value of the expression. Returns a ``Quantity`` if ``units=True``;
        otherwise returns the numeric magnitude as ``float`` or ``np.ndarray``.
    """

    fcn = get_function(e, units=units)
    return cast(Quantity | np.ndarray | float, fcn(value_map))


def substitute_unknowns(
    e: sp.Basic,
    knowns: set[sp.Symbol],
    eqns: list[sp.Equality],
    avoid: set[sp.Symbol] | None = None,
) -> sp.Basic:
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
    avoid : set[sp.Symbol] | None, optional
        Set of symbols to avoid in the substitution expressions, by default None

    Returns
    -------
    sp.Basic
        The substituted expression without any unknown symbols
    """

    if avoid is None:
        avoid = set()

    replacements: list[tuple[sp.Symbol, sp.Basic]] = []

    def _get_unknowns(expr: sp.Basic) -> list[sp.Symbol]:
        all_symbols = cast(list[sp.Symbol], sorted(expr.free_symbols, key=default_sort_key))

        already_replaced = [m[0] for m in replacements]

        return [n for n in all_symbols if n not in (knowns | avoid) and n not in already_replaced]

    unknowns_list = _get_unknowns(e)

    for n in unknowns_list:
        n_expr = get_sol_expr(eqns, n, avoid=avoid)

        if n_expr is None:
            raise ValueError(f"Symbol {n} could not be isolated based on the specified equations.")

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

    for n in re.sub(r"[A-Z]_\d", r"|\g<0>|", s).split("|"):
        if re.match(r"[A-Z]_\d", n):
            parts.extend([f"{n[:-2]}", "}", f"_{n[-1]}", "\\text{"])
        else:
            parts.append(n)

    parts = ["\\text{"] + [n for n in parts if n]

    if parts[-1] == "\\text{":
        parts = parts[:-1]

    ret = "".join(parts)

    if ret.count("{") == ret.count("}") + 1:
        ret += "}"

    return ret


def typeset(x: str | int) -> str:
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
    x : str | int
        Input string or int (will be converted to str)

    Returns
    -------
    str
        Output Latex string
    """

    x = str(x)

    if not SETTINGS.typeset_symbol_scripts:
        return x

    parts = [n.strip() for n in x.split(",")]

    for i, p in enumerate(parts):
        # avoid typesetting single upper-case letters as text
        # if they start with ~ (guard len so a bare "~" part does not IndexError)
        if p.startswith("~") and len(p) > 1 and p[1].isupper():
            parts[i] = p[1]
            continue

        # only typeset single words, also ignore Latex code
        if " " in p or "\\" in p:
            continue

        alpha_str = "".join(n for n in p if n.isalpha())

        # typeset everything except 1-letter lower case as text
        typeset_text = len(alpha_str) >= 2 or (len(alpha_str) == 1 and alpha_str.isupper())

        if typeset_text:
            # handle chemical compounds
            if re.match("[A-Z]", p):
                p = typeset_chemical(p)
                parts[i] = p
            else:
                parts[i] = "\\text{" + p + "}"

    return ",".join(parts)


class Symbol(sp.Symbol):
    """A ``sympy.Symbol`` extended with convenience methods for typeset sub- and
    superscripts (:meth:`_`, :meth:`__`, :meth:`decorate`). Each method returns a new
    symbol with the same assumptions as the original."""

    def decorate(
        self,
        prefix: str | int | None = None,
        suffix: str | int | None = None,
        prefix_sub: str | int | None = None,
        prefix_sup: str | int | None = None,
        suffix_sub: str | int | None = None,
        suffix_sup: str | int | None = None,
    ) -> Self:
        """
        Method that decorates a symbol with
        subscripts and/or superscripts before or after the symbol.
        Returns a new symbol object with the same assumptions (i.e. real,
        positive, complex, etc...) as the input.

        Using LaTeX syntax supported by ``sympy``:

        .. code:: none

            {prefix}^{prefix_sup}_{prefix_sub}{symbol}_{suffix_sub}^{suffix_sup}{suffix}

        ``symbol`` is the input symbol.
        The ``prefix`` and ``suffix`` parts are added without ``_`` or ``^``.
        Each of the parts (except ``symbol``) can be empty.

        The decorations can be string or integer, floats are not allowed.
        In case the input symbol already contains sub- or superscripts,
        the decorations are not appended to those. Instead, a new level
        is introduced. To keep things simple, make sure that the input symbol
        is a simple symbol.

        Use the ``append`` method to append to an existing sub-
        or superscript in the suffix.

        Parameters
        ----------
        prefix : str | int | None, optional
            Prefix added before the symbol, by default None
        suffix : str | int | None, optional
            Suffix added after the symbol, by default None
        prefix_sub : str | int | None, optional
            Subscript prefix before the symbol and after ``prefix``, by default None
        prefix_sup : str | int | None, optional
            Superscript prefix before the symbol and after ``prefix``, by default None
        suffix_sub : str | int | None, optional
            Subscript suffix after the symbol and before ``suffix``, by default None
        suffix_sup : str | int | None, optional
            Superscript suffix after the symbol and before ``suffix``, by default None

        Returns
        -------
        Symbol
            A new symbol with the same assumptions as the input, with added decorations
        """

        parts = [
            prefix,
            typeset(prefix_sup) if prefix_sup is not None else None,
            typeset(prefix_sub) if prefix_sub is not None else None,
            self.name,
            typeset(suffix_sub) if suffix_sub is not None else None,
            typeset(suffix_sup) if suffix_sup is not None else None,
            suffix,
        ]

        delimiters = ["", "^", "_", "", "_", "^", ""]

        decorated_parts = []

        for p, d in zip(parts, delimiters, strict=False):
            if p is None:
                continue

            p = str(p)

            # don't add extra braces around the base symbol
            if p != self.name:
                p = "{" + p + "}"

            decorated_parts.append(d + p)

        decorated_symbol = "".join(decorated_parts)

        # assumptions0 contains assumptions that are not None
        return self.__class__(decorated_symbol, **self.assumptions0)

    def append(self, s: str | int, where: Literal["sub", "sup"] = "sub") -> Self:
        """
        Adds the input ``s`` to an existing sub- or superscript.
        Does not append to prefixes.
        Creates the sub- or superscript if it does not exist.

        Parameters
        ----------
        s : str | int
            Text or index to be added
        where : Literal['sub', 'sup'], optional
            Whether to append to the subscript or superscript, by default 'sub'

        Returns
        -------
        Symbol
            A new symbol with the same assumptions as the input,
            with updated sub- or superscript
        """

        delimiter = "_" if where == "sub" else "^"

        symbol = self.name

        s = typeset(s)

        if delimiter not in symbol:
            decorated_parts = [symbol, delimiter, "{" + s + "}"]

        else:
            *base_symbol, existing_suffix = symbol.split(delimiter)

            base_symbol_str = "".join(base_symbol)

            # assume that the input Latex symbol is correct,
            # don't deal with unbalanced braces
            existing_suffix = existing_suffix.removeprefix("{").removesuffix("}")

            existing_suffix += s

            decorated_parts = [base_symbol_str, delimiter, "{" + existing_suffix + "}"]

        decorated_symbol = "".join(decorated_parts)
        return self.__class__(decorated_symbol, **self.assumptions0)

    def _(self, x: str) -> Self:
        """
        Add subscript ``x``.
        """
        return self.append(x, where="sub")

    def __(self, x: str) -> Self:
        """
        Add superscript ``x``.
        """
        return self.append(x, where="sup")

    def delta(self) -> Self:
        """
        Add ``\\Delta`` prefix.
        """
        return self.decorate(prefix="\\Delta")


def _patch_symbol_class(dest: type[sp.Symbol] | sp.Symbol) -> None:
    for n in ["decorate", "append", "delta", "_", "__"]:
        if not hasattr(dest, n):
            method = getattr(Symbol, n)
            setattr(dest, n, method)


def symbols(inp: str, **kwargs: Any) -> list[Symbol]:  # noqa: ANN401
    """Like ``sympy.symbols``, but typed as a list and always returning more than one symbol.

    ``kwargs`` are SymPy assumptions (``positive=True``, ``integer=True``, ...).
    """

    ret = sp.symbols(inp, **kwargs)

    if not isinstance(ret, Iterable):
        raise ValueError(
            f"Expected more than one input symbol, use sp.Symbol('{inp}') directly to create a single symbol"
        )
    for n in ret:
        _patch_symbol_class(n)

    return cast(list[Symbol], list(ret))


_patch_symbol_class(sp.Symbol)
