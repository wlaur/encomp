"""
Data structures and related functions.
"""

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, overload

import numpy as np
import polars as pl

from .units import Quantity


@overload
def divide_chunks[T](container: list[T], N: int) -> Iterator[list[T]]: ...


@overload
def divide_chunks[T](container: tuple[T], N: int) -> Iterator[tuple[T]]: ...


@overload
def divide_chunks[T](container: Sequence[T], N: int) -> Iterator[Sequence[T]]: ...


@overload
def divide_chunks(container: np.ndarray, N: int) -> Iterator[np.ndarray]: ...


def divide_chunks(container: Any, N: int) -> Any:
    if not len(container):
        raise ValueError("Cannot chunk empty container")

    if N < 1:
        raise ValueError(f"Cannot split container with into {N} chunks")

    for i in range(0, len(container), N):
        yield container[i : i + N]


def flatten(container: Iterable[Any], max_depth: int | None = None, _depth: int = 0) -> Iterator[Any]:
    if max_depth is not None and _depth >= max_depth:
        yield container
        return

    for obj in container:
        if isinstance(obj, (str, Quantity, np.ndarray, pl.Series, pl.Expr)):
            yield obj
            continue

        # check if this object can be flattened further
        if isinstance(obj, Iterable):
            yield from flatten(obj, max_depth=max_depth, _depth=_depth + 1)  # pyright: ignore[reportUnknownArgumentType]

            continue

        yield obj
