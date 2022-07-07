from __future__ import annotations
from typing import Generic, TypeVar, overload


class Base:
    pass


class First(Base):
    pass


class Second(Base):
    pass


class Unknown(Base):
    pass


T = TypeVar('T', bound=Base)


class Container(Generic[T]):

    def __new__(cls, *args, d: type[T] = Unknown,  # type: ignore
                **kwargs) -> Container[T]:
        return super().__new__(cls)

    @overload
    def __mul__(self, other: Container[Unknown]) -> Container[Unknown]:
        ...

    @overload
    def __mul__(self: Container[Unknown], other) -> Container[Unknown]:
        ...

    @overload
    def __mul__(self: Container[First], other) -> Container[First]:
        ...

    @overload
    def __mul__(self: Container[Second], other) -> Container[Second]:
        ...

    def __mul__(self, other):
        return f'{self.val} * {other.val}'


c1 = Container('c1')
c2 = Container[First]('c2')
c3 = Container[Second]('c3')

reveal_type(c1)
reveal_type(c2)
reveal_type(c3)

reveal_type(c2 * c3)
reveal_type(c3 * c2)


reveal_type(c1 * c2)
reveal_type(c1 * c3)

reveal_type(c1 * c1)
reveal_type(c2 * c1)
reveal_type(c3 * c1)
