"""
Implements a general class for defining, manipulating and solving *balance equations*.
Uses Sympy as backend, and outputs a normal Python function that evaluates
unknown variables based on balance relationships and known parameters.

Balances can be based on energy, mass or stoichiometry (chemical). One balance equation is required for
each unknown variable.
"""


from typing import Optional, Literal, Callable, Union
from symbolic_equation import Eq as Eq_symbolic
from sympy import default_sort_key
import numpy.typing as npt


from encomp.structures import flatten, divide_chunks
from encomp.sympy import (sp, substitute_unknowns, get_sol_expr,
                          evaluate, get_function, get_lambda)

from encomp.units import Quantity


class Balance:

    def __init__(self, eqn: sp.Equality,
                 balance_type: Literal['mass', 'energy', 'chemical'],
                 tag: Optional[str] = None):

        self.eqn = eqn
        self.balance_type = balance_type
        self.tag = tag

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        return self.eqn.free_symbols

    def _repr_html_(self) -> str:

        if self.tag is None:
            tag_repr = ''
        else:
            tag_repr = f': {self.tag}'

        return Eq_symbolic(lhs=self.eqn.lhs,
                           rhs=self.eqn.rhs,
                           tag=f'{self.balance_type.title()} balance{tag_repr}')._repr_latex_()

    def to_json(self) -> dict:

        from encomp.serialize import serialize

        attrs = ['eqn',
                 'balance_type',
                 'tag']

        return {'type': 'Balance',
                'data': {a: serialize(getattr(self, a)) for a in attrs}}

    @classmethod
    def from_dict(cls, d: dict) -> 'Balance':

        from encomp.serialize import decode

        return cls(**{a: decode(b) for a, b in d.items()})


class BalancedSystem:

    def __init__(self,
                 balances: list[Balance],
                 unknowns: list[sp.Symbol],
                 secondary_equations: Optional[list[sp.Eq]] = None):

        # check that all unknowns are present in the balances
        # it is not supported to define the unknowns via secondary equations
        undefined_unknowns = set(unknowns) - \
            set(flatten(n.free_symbols for n in balances))
        if undefined_unknowns:
            raise ValueError(
                f'Unknown variables not included in balance equations:\n{undefined_unknowns}\n')

        self.balances = balances.copy()
        self.unknowns = unknowns.copy()

        if secondary_equations is None:
            secondary_equations = []

        self.secondary_equations = secondary_equations.copy()

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        s1 = set(flatten(n.free_symbols for n in self.balances))
        s2 = set(flatten(n.free_symbols for n in self.secondary_equations))
        return s1 | s2

    def get_matrix(self) -> sp.Matrix:
        """
        Find the matrix :math:`[A, b]` that corresponds to the
        linear system :math:`A x = b`.

        Returns
        -------
        sp.Matrix
            System matrix
        """

        eqns = [n.eqn for n in self.balances]
        A, b = sp.linear_eq_to_matrix(eqns, *self.unknowns)
        M = sp.Matrix.hstack(A, b)

        return M

    def mapping(self,
                value_map: dict[Union[sp.Symbol, str], Union[Quantity, npt.ArrayLike]],
                iterative_symbols: Optional[list[sp.Symbol]] = None,
                callback: Optional[Callable] = None,
                to_str: bool = False,
                mapping_name: str = 'mapping') -> Union[Callable, str]:
        """
        .. todo::
            Implement a method that constructs a mapping from x → y variables.
        """

        raise NotImplementedError()

    def _get_expr(self, symbol: sp.Symbol, knowns: set[sp.Symbol]) -> sp.Basic:

        if symbol not in self.free_symbols:
            raise ValueError(
                f'Symbol {symbol} is not included in the balances or secondary equations')

        expr = get_sol_expr(self.secondary_equations, symbol)

        # substitute as if the unknowns were known, they can be evaluated if necessary
        expr_subs = substitute_unknowns(expr,
                                        knowns,
                                        self.secondary_equations)
        if expr_subs is not None:
            expr = expr_subs

        return expr

    def _get_eval_func(self, symbol: sp.Symbol,
                       value_map: dict[sp.Symbol, Union[Quantity, npt.ArrayLike]],
                       mapping: Optional[Callable] = None,
                       units: bool = True,
                       to_str: bool = False) -> Union[Callable, tuple[str, list[str]]]:
        """
        Get a callable function that takes ``value_map`` as input and
        returns the value for ``symbol``.

        Parameters
        ----------
        symbol : sp.Symbol
            Symbol to evaluate
        value_map : dict[sp.Symbol, Union[Quantity, npt.ArrayLike]]
            Mapping from symbol to value
        mapping : Optional[Callable]
            Callable that evaluates the unknown variables, in case this is
            None it will be created (if necessary), by default None
        units : bool
            Whether to use units for the return value, by default True
        to_str : bool
            Whether to reuturn a string representation of the function, by default False

        Returns
        -------
        Union[Callable, tuple[str, list[str]]]
            Function that returns the specified value with ``value_map`` as input,
            or a string representation and a list of args in case ``to_str=True``
        """

        expr = self._get_expr(symbol, set(value_map) | set(self.unknowns))

        if to_str:
            fcn_str, args = get_lambda(expr, to_str=True)
            return fcn_str, args

        return get_function(expr, units=units)

    def evaluate(self, symbol: sp.Symbol,
                 value_map: dict[sp.Symbol, Union[Quantity, npt.ArrayLike]],
                 mapping: Optional[Callable] = None,
                 n_iter: int = 10,
                 units: bool = True) -> Quantity:
        """
        Evaluates the specified symbol based on the values in ``value_map``.

        .. note::
            This method is very slow, do not evaluate this in a loop.
            Pass Numpy array in ``value_map`` to evalute multiple values at once.

        Parameters
        ----------
        symbol : sp.Symbol
            Symbol to evaluate
        value_map : dict[sp.Symbol, Union[Quantity, npt.ArrayLike]]
            Mapping from symbol to value
        mapping : Optional[Callable]
            Callable that evaluates the unknown variables, in case this is
            None it will be created (if necessary), by default None
        n_iter : int, optional
            Number of iterations in case the mapping function needs to be evaluated
            and there are iterative symbols, by default 10
        units : bool
            Whether to use units for the return value, by default True

        Returns
        -------
        Quantity
            The evaluated value of the symbol, given the specified ``value_map``
        """

        # in case the symbol to be evaluated exists in value_map, remove it
        # we don't want to simply look it up from this dict
        if symbol in value_map:
            value_map = value_map.copy()
            value_map.pop(symbol)

        expr = self._get_expr(symbol, set(value_map) | set(self.unknowns))

        try:
            return evaluate(expr, value_map, units=units)

        # the generated lambda function will have missing positional arguments in case
        # the specificed symbol is calculated from an unknown variable
        # evaluate the unknowns and try again
        except TypeError:

            if mapping is None:
                # generating the explicit forms for all unknown variables will take a while
                # this does not consider iterative symbols, the user needs to pass the mapping
                # object explicitly in that case
                mapping = self.mapping(value_map, units=units)

            # we don't know if the mapping takes kwarg n_iter or not
            try:
                unknown_value_map = mapping(n_iter=n_iter)
            except:
                unknown_value_map = mapping()

            return evaluate(expr, value_map | unknown_value_map, units=units)

    def _repr_html_(self) -> str:

        def symbols_repr(symbols):

            symb_repr = '\n\\begin{equation}\n'

            symb_repr += '\n\\\\\n'.join('\\quad'.join(n._repr_latex_()[1:-1] for n in chunk)
                                         for chunk in divide_chunks(symbols, 4))

            symb_repr += '\n\\end{equation}\n\n'

            return symb_repr

        s = (f'Balanced system with {len(self.balances)} balance '
             f'equation{"s" if len(self.balances) != 1 else ""}, '
             f'{len(self.unknowns)} unknown{"s" if len(self.unknowns) != 1 else ""} '
             f'and {len(self.secondary_equations)} secondary '
             f'equation{"s" if len(self.secondary_equations) != 1 else ""}:\n')

        s += ''.join(b._repr_html_() for b in self.balances)

        s += 'Unknown symbols:\n\n'

        s += symbols_repr(self.unknowns)

        free_symbols_balances = sorted(set(flatten([n.free_symbols for n in self.balances])) - set(self.unknowns),
                                       key=default_sort_key)

        s += 'Additional symbols in balance equations:\n\n'

        s += symbols_repr(free_symbols_balances)

        if self.secondary_equations:
            s += 'Additional symbols in secondary equations:\n\n'

            free_symbols_secondary = sorted(set(flatten([n.free_symbols
                                                        for n in self.secondary_equations])) - set(free_symbols_balances),
                                            key=default_sort_key)

            s += symbols_repr(free_symbols_secondary)

        return s

    def to_json(self) -> dict:

        from encomp.serialize import serialize

        attrs = ['balances',
                 'unknowns',
                 'secondary_equations']

        return {'type': 'BalancedSystem',
                'data': {a: serialize(getattr(self, a)) for a in attrs}}

    @ classmethod
    def from_dict(cls, d: dict) -> 'BalancedSystem':

        from encomp.serialize import decode

        bs = cls(**{a: decode(b) for a, b in d.items()})

        return bs
