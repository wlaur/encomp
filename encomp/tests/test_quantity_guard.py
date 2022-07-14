import pytest

from typing import Union

from encomp.units import Quantity as Q
from encomp.utypes import *
from encomp.misc import isinstance_types

# it's important that the expected mypy output is a comment on the
# same line as the expression, disable autopep8 if necessary with
# autopep8: off
# ... some code above the line length limit
# autopep8: on


@pytest.mark.mypy_testing
def test_quantity_typeguard() -> None:

    # autopep8: off


    q: Union[Q[Length], Q[Velocity]]

    q = Q(25, 'kg')  # E: Incompatible types in assignment (expression has type "Quantity[Mass]", variable has type "Union[Quantity[Length], Quantity[Velocity]]")

    # avoid literal unit type inference
    unit = str('m')

    # Q(25, unit) has type Q[Unknown]
    q = Q(25, unit)  # type: ignore

    reveal_type(q)  # R: Union[encomp.units.Quantity[encomp.utypes.Length], encomp.units.Quantity[encomp.utypes.Velocity]]


    # TODO: this does not work with mypy, does work with pylance
    if isinstance_types(q, Q[Length]):

        # mypy infers Q[Unknown] from the type: ignore assignment
        reveal_type(q)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


    # this also does not work with mypy
    if isinstance_types(q2 := Q(25, 'm'), Q[Length]):
        reveal_type(q2)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


    # autopep8: on
