from typing import Optional
from textwrap import dedent

from encomp.misc import name_assignments, isinstance_types, name_assignments


def test_name_assignments():

    s = 'a = 5; b = [1, 2, 3]'

    assignments = name_assignments(s)

    assert len(assignments) == 2

    src = dedent("""

        a = 12
        b = a * 2

        c = b or none

    """)

    assert {n[0] for n in name_assignments(src)} == {'a', 'b', 'c'}


def test_isinstance_types():

    assert isinstance_types((1, 4), tuple)
    assert isinstance_types((1, 4), tuple[int, int])
    assert isinstance_types((1, 4), tuple[int, ...])

    assert not isinstance_types((1, 4), tuple[str, int])
    assert not isinstance_types((1, 4), tuple[int])

    d = {
        'asd': [1, 3, 4],
        'dsa': [1, 3.2, 4, 22, None]
    }

    assert isinstance_types(d, dict[str, list[Optional[float]]])
    assert not isinstance_types(d, dict[str, list[Optional[int]]])
    assert not isinstance_types(d, dict[str, list[float]])
