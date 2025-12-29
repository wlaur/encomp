from typing import Any

from ..units import Quantity as Q


def _q1() -> Q[Any, Any]:
    return Q(2)


def _q2(inp: Q[Any, Any]) -> Q[Any, Any]:
    return inp * 2


def test_transforms() -> None:
    _ = _q2(_q1())
