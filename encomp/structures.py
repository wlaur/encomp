from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast, overload

import numpy as np
import polars as pl

from .units import Quantity


@overload
def divide_chunks[T](container: list[T], N: int) -> Iterator[list[T]]: ...


@overload
def divide_chunks[T](container: tuple[T, ...], N: int) -> Iterator[tuple[T, ...]]: ...


@overload
def divide_chunks[T](container: Sequence[T], N: int) -> Iterator[Sequence[T]]: ...


@overload
def divide_chunks(container: np.ndarray, N: int) -> Iterator[np.ndarray]: ...


def divide_chunks(container: Any, N: int) -> Any:
    # validate eagerly: a generator body would defer these errors to the first
    # next() call, far from the call site
    if not len(container):
        raise ValueError("Cannot chunk empty container")

    if N < 1:
        raise ValueError(f"Chunk size must be at least 1, passed {N}")

    def _chunks() -> Iterator[Any]:
        for i in range(0, len(container), N):
            yield container[i : i + N]

    return _chunks()


def flatten(container: Iterable[Any], max_depth: int | None = None) -> Iterator[Any]:
    def _flatten(items: Iterable[Any], depth: int) -> Iterator[Any]:
        if max_depth is not None and depth >= max_depth:
            yield items
            return

        for obj in items:
            # atomic despite being iterable: iterating a dict would silently drop its
            # values (keys only), and bytes would flatten into individual integers
            if isinstance(obj, (str, bytes, dict, Quantity, np.ndarray, pl.Series, pl.Expr)):
                yield obj
                continue

            if isinstance(obj, Iterable):
                yield from _flatten(cast("Iterable[Any]", obj), depth + 1)  # pyrefly: ignore[redundant-cast]  # cast required by pyright
                continue

            yield obj

    yield from _flatten(container, 0)
