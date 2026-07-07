from textwrap import dedent
from typing import Any, cast

from ..misc import isinstance_types, name_assignments
from ..units import Quantity as Q
from ..utypes import (
    Dimensionless,
    Mass,
    Power,
    Temperature,
    UnknownDimensionality,
)

# This module tests the runtime isinstance_types helper itself, so it keeps
# direct isinstance_types assertions instead of replacing them with assert_type.


def test_name_assignments() -> None:
    s = "a = 5; b = [1, 2, 3]"

    assignments = name_assignments(s)

    assert len(assignments) == 2

    src = dedent(
        """

        a = 12
        b = a * 2

        c = b or none

    """
    )

    assert {n[0] for n in name_assignments(src)} == {"a", "b", "c"}


def test_isinstance_types() -> None:
    assert isinstance_types((1, 4), tuple)
    assert isinstance_types((1, 4), tuple[int, int])
    assert isinstance_types((1, 4), tuple[int, ...])

    assert not isinstance_types((1, 4), tuple[str, int])
    assert not isinstance_types((1, 4), tuple[int])

    # NOTE: only the first element in each collection is type checked at runtime
    d = {"dsa": [None, 1, 3.2, 4, 22], "asd": [1, 3, 4]}

    assert isinstance_types(d, dict[str, list[float | None]])
    assert not isinstance_types(d, dict[str, list[float]])

    x = (2, 2, 3)
    assert not (isinstance_types(x, tuple[int, int]) or isinstance_types(x, tuple[str, str]))
    assert isinstance_types(x, tuple[int, int, int]) or isinstance_types(x, tuple[str, str])

    y = (2, 2, "3")
    assert isinstance_types(y, tuple[int, int, str]) or isinstance_types(y, str)


def test_isinstance_types_invalid_expected() -> None:
    import pytest

    # a string "expected" is an unresolvable forward reference that typeguard
    # would silently pass -- it must be an explicit error, never a silent True
    with pytest.raises(TypeError, match="not a string"):
        isinstance_types(5, cast(Any, "not a type"))


def test_isinstance_types_quantity() -> None:
    q = Q(1)
    assert isinstance_types(q, Q)
    assert isinstance_types(q, Q[Dimensionless])
    assert isinstance_types(q, Q[Dimensionless, float])

    assert not isinstance_types(q, int)
    assert not (isinstance_types(q, int) or isinstance_types(q, float))

    q2 = Q(2, "kg")
    assert isinstance_types(q2, Q)
    assert isinstance_types(q2, Q[Mass])
    assert isinstance_types(q2, Q[Mass, float])

    assert isinstance_types(q2, Q[Mass]) or isinstance_types(q2, Q[Temperature])
    assert not (isinstance_types(q2, Q[Power]) or isinstance_types(q2, Q[Temperature]))


def test_isinstance_types_quantity_union() -> None:
    # a Quantity[UnknownDimensionality] union member matches ANY dimensionality, exactly as
    # it does as a lone type -- the union and single-type checks must be consistent (a plain
    # isinstance against the union would wrongly say False, since the unknown-dim subclass is
    # a runtime sibling of the concrete dimensionality subclasses, not a parent)
    q = Q(25.0, "kg")

    # the runtime UnionType value is built dynamically, so the argument is cast to Any (the
    # detailed union handling under test lives in isinstance_types, not the caller's types)
    def _check(expected: object) -> bool:
        return isinstance_types(q, cast(Any, expected))

    assert isinstance_types(q, Q[UnknownDimensionality])
    assert _check(Q[UnknownDimensionality] | Q[Power])
    assert _check(Q[Power] | Q[UnknownDimensionality])

    # a union of concrete dimensionalities still narrows correctly
    assert _check(Q[Mass] | Q[Power])
    assert not _check(Q[Power] | Q[Temperature])
