from typing import TYPE_CHECKING

import pytest

from ..misc import isinstance_types
from ..units import Quantity as Q
from ..utypes import Length, Velocity

if not TYPE_CHECKING:

    def reveal_type(x):
        return x


@pytest.mark.mypy_testing
def test_quantity_typeguard() -> None:
    return
    # fmt: off

    q: Q[Length] | Q[Velocity]

    q = Q(
        25, "kg"
    )  # E: Incompatible types in assignment (expression has type "Quantity[Mass]", variable has type "Union[Quantity[Length], Quantity[Velocity]]")  # noqa: E501

    # avoid literal unit type inference
    unit = str("m")

    # Q(25, unit) has type Q[Unknown]
    q = Q(25, unit)  # type: ignore

    reveal_type(
        q
    )  # R: Union[encomp.units.Quantity[encomp.utypes.Length], encomp.units.Quantity[encomp.utypes.Velocity]]

    # TODO: this does not work with mypy, does work with pyright
    if isinstance_types(q, Q[Length]):
        # mypy infers Q[Unknown] from the type: ignore assignment
        reveal_type(q)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # this also does not work with mypy
    if isinstance_types(q2 := Q(25, "m"), Q[Length]):
        reveal_type(q2)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # fmt: on
