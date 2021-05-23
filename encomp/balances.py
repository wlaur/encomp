"""
Implements a general class for defining, manipulating and solving *balance equations*.
Uses Sympy as backend, and outputs a normal Python function that evaluates
unknown variables based on balance relationships and known parameters.

Balances can be based on energy, mass or stoichiometry (chemical). One balance equation is required for
each unknown variable.

.. warning::
    All balance equation must be *linear* with respect to the unknown variables.
"""


from typing import Optional, Literal, Callable, Union
from symbolic_equation import Eq as Eq_symbolic
from sympy import default_sort_key
from sympy.solvers.solveset import linsolve
import numpy.typing as npt


from encomp.structures import flatten, divide_chunks
from encomp.sympy import sp, get_mapping
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
                           tag=f'{self.balance_type.title()} balance{tag_repr.title()}')._repr_latex_()

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

        self.balances = balances.copy()
        self.unknowns = unknowns.copy()

        if secondary_equations is None:
            secondary_equations = []

        self.secondary_equations = secondary_equations.copy()
        self.sol = None

    def solve(self) -> None:
        """
        Solves the (linear) balance equations for the unknown symbols.
        Sets the ``sol`` attribute of this instance.
        """

        eqns = [n.eqn for n in self.balances]
        sol_set = linsolve(eqns, self.unknowns)

        self.sol = list(zip(self.unknowns, sol_set.args[0]))

    def mapping(self,
                value_map: dict[Union[sp.Symbol, str], Union[Quantity, npt.ArrayLike]],
                units: bool = False,
                to_str: bool = False) -> Union[Callable, str]:
        """
        Returns a mapping function (as object or string representation of a Python module)
        that returns evaluated values for unknown variables given the parameters defined
        in ``value_map``. The keys in ``value_map`` are symbols in the balance equations,
        and the values are default values that are used in case the mapping function
        did not receive an input for a certain symbol. Set the values in ``value_map`` to None
        to make them required (the mapping function will raise an error in case None is multiplied
        with a Quantity or ``np.ndarray``).

        Parameters
        ----------
        value_map : dict[Union[sp.Symbol, str], Union[Quantity, npt.ArrayLike]]
            Mapping from symbol or symbol identifier to value. Make sure the keys match
            the symbols used in the balanced system. The value can be set to None to
            make a key an required input to this mapping function
        units : bool
            Whether to use units for the return values. The inputs must be always Quantity, by default False
        to_str : bool, optional
            Whether to output a string representation of a Python module instead of a
            function object, by default False

        Returns
        -------
        Union[Callable, str]
            The mapping function, object or string representation of a Python module
        """

        if self.sol is None:
            self.solve()

        return get_mapping(
            self.unknowns,
            list(value_map),
            self.sol,
            value_map,
            secondary_equations=self.secondary_equations,
            units=units,
            to_str=to_str
        )

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
                 'secondary_equations',
                 'sol']

        return {'type': 'BalancedSystem',
                'data': {a: serialize(getattr(self, a)) for a in attrs}}

    @ classmethod
    def from_dict(cls, d: dict) -> 'BalancedSystem':

        from encomp.serialize import decode

        if 'sol' in d:
            sol = decode(d.pop('sol'))
        else:
            sol = None

        bs = cls(**{a: decode(b) for a, b in d.items()})
        bs.sol = sol

        return bs
