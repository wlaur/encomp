from typing import Any, assert_type

from typeguard import typechecked

from ..units import Quantity as Q


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def _q1() -> Q[Any, Any]:
    return Q(2)


def _q2(inp: Q[Any, Any]) -> Q[Any, Any]:
    return inp * 2


# TODO: test with typeguard.typechecked
# install typechecked on all funcs (no need to test the @decorator version)
def test_transforms() -> None:
    _ = typechecked(_q2)(typechecked(_q1)())
