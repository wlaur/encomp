from typing import TypeVar

import pytest

from encomp.units import DimensionalityError
from encomp.units import Quantity as Q
from encomp.utypes import *

# it's important that the expected mypy output is a comment on the
# same line as the expression, disable autopep8 if necessary with
# autopep8: off
# ... some code above the line length limit
# autopep8: on


@pytest.mark.mypy_testing
def test_simple_signatures() -> None:

    def f1(a: Q[Temperature]) -> Q[Pressure]:

        if a > Q(0, 'degC'):
            return Q(25, 'bar')
        else:
            return Q(15, 'bar')

    # autopep8: off

    reveal_type(f1(Q(25, 'degC')))  # R: encomp.units.Quantity[encomp.utypes.Pressure]

    with pytest.raises(DimensionalityError):
        f1(Q(25, 'm'))  # E: Argument 1 to "f1" has incompatible type "Quantity[Length]"; expected "Quantity[Temperature]"

    # missing return type hint
    # mypy cannot infer that return type is Q[Pressure] | None (pylance handles this)
    def f2(a: Q[Temperature]):


        if a > Q(0, 'degC'):
            return Q(25, 'bar')

        # implicit return None

    # autopep8: on


class Estimation(Dimensionality):
    pass


class EstimatedLength(Estimation):
    dimensions = Length.dimensions


class EstimatedMass(Estimation):
    dimensions = Mass.dimensions


@pytest.mark.mypy_testing
def test_custom_dimensionality_signatures() -> None:

    # refer to the custom dimensionalities defined before this function
    # this is an incorrect type annotation
    def f1(a: Q[Estimation]) -> Q[Estimation]:
        return a * 2

    # autopep8: off

    s = Q(25, 'm')
    e_s = Q[EstimatedLength](25, str('m'))
    m = Q(25, 'kg')
    e_m = Q[EstimatedMass](25, str('kg'))


    f1(s)  # E: Argument 1 to "f1" has incompatible type "Quantity[Length]"; expected "Quantity[Estimation]"
    f1(e_s)  # E: Argument 1 to "f1" has incompatible type "Quantity[EstimatedLength]"; expected "Quantity[Estimation]"


    # the annotation must use a bounded type variable
    T = TypeVar('T', bound=Estimation)

    def f2(a: Q[T]) -> Q[T]:
        return a * 2


    reveal_type(f2(e_s))  # R: encomp.units.Quantity[encomp.tests.test_function_signatures.EstimatedLength]
    reveal_type(f2(e_m))  # R: encomp.units.Quantity[encomp.tests.test_function_signatures.EstimatedMass]

    # the Length dimensionality does not match the type variable bound
    f2(s)  # E: Value of type variable "T" of "f2" cannot be "Length"
    f2(m)  # E: Value of type variable "T" of "f2" cannot be "Mass"


    # autopep8: on
