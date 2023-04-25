from textwrap import dedent

from ..misc import grid_dimensions, isinstance_types, name_assignments
from ..units import Quantity as Q
from ..utypes import Dimensionless, Mass, Power, Temperature


def test_grid_dimensions():
    assert [
        grid_dimensions(1, 2, 3),
        grid_dimensions(-1, -1, 3),
        grid_dimensions(-1, 2, 3),
        grid_dimensions(2, -1, 3),
    ] == [(2, 3), (0, 3), (2, 3), (1, 3)]


def test_name_assignments():
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


def test_isinstance_types():
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
    assert not isinstance_types(x, tuple[int, int] | tuple[str, str])
    assert isinstance_types(x, tuple[int, int, int] | tuple[str, str])

    y = (2, 2, "3")
    assert isinstance_types(y, tuple[int, int, str] | str)


def test_isinstance_types_quantity():
    q = Q(1)
    assert isinstance_types(q, Q)
    assert isinstance_types(q, Q[Dimensionless])
    assert isinstance_types(q, Q[Dimensionless, float])

    assert not isinstance_types(q, int)
    assert not isinstance_types(q, int | float)

    q2 = Q(2, "kg")
    assert isinstance_types(q2, Q)
    assert isinstance_types(q2, Q[Mass])
    assert isinstance_types(q2, Q[Mass, float])

    assert isinstance_types(q2, Q[Mass] | Q[Temperature])
    assert not isinstance_types(q2, Q[Power] | Q[Temperature])
