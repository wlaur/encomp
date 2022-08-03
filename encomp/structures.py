"""
Data structures and related functions.
"""

from typing import Sequence, Iterator, Optional, Any, Iterable, TypeVar, overload

import numpy as np
import pandas as pd

from encomp.units import Quantity

T = TypeVar('T')


_BASE_TYPES = (str, Quantity, pd.Series, pd.DataFrame, np.ndarray)


@overload
def divide_chunks(container: list[T], N: int) -> Iterator[list[T]]:
    ...


@overload
def divide_chunks(container: tuple[T], N: int) -> Iterator[tuple[T]]:
    ...


@overload
def divide_chunks(container: Sequence[T], N: int) -> Iterator[Sequence[T]]:
    ...


@overload
def divide_chunks(container: np.ndarray, N: int) -> Iterator[np.ndarray]:
    ...


def divide_chunks(container, N):
    """
    Generator that divides a container into chunks with length ``N``.
    The last chunk might not have ``N`` elements.

    .. code-block:: python

        parts_with_3_elements = list(divide_chunks(container, 3))


    Parameters
    ----------
    container : Sequence[T]
        The container that will be split into chunks. Since sets are unordered,
        it does not make sense to accept set inputs here.
    N : int
        Number of element for one chunk (last chunk might be shorter)

    Yields
    -------
    Iterator[Sequence[T]]
        Generator of chunks
    """

    if not len(container):
        raise ValueError('Cannot chunk empty container')

    if N < 1:
        raise ValueError(
            f'Cannot split container with into {N} chunks')

    for i in range(0, len(container), N):
        yield container[i:i + N]


def flatten(container: Iterable[Any],
            max_depth: Optional[int] = None,
            _depth: int = 0) -> Iterator[Any]:
    """
    Generator that flattens a nested container.

    Usage:

    .. code-block:: python

        flat_list = list(flatten(nested_list))

    This function will flatten arbitrarily deeply nested lists or tuples recursively.
    If ``max_depth`` is ``None``, recurse until no more nested structures remain,
    otherwise flatten until the specified max depth.
    The base types ``str``, :py:class:`encomp.units.Quantity`, ``pd.Series``, ``pd.DataFrame``,
    ``np.ndarray`` will not be flattened.


    Parameters
    ----------
    container : Iterable[Any]
        The container to be flattened
    max_depth : int, optional
        The maximum level to flatten to, by default None (flatten all)

    Yields
    -------
    Iterator[Any]
        Generator of non-nested objects
    """

    if max_depth is not None and _depth >= max_depth:
        yield container
        return

    for obj in container:

        if isinstance(obj, _BASE_TYPES):
            yield obj
            continue

        # check if this object can be flattened further
        if isinstance(obj, Iterable):

            for sub_obj in flatten(obj, max_depth=max_depth, _depth=_depth + 1):
                yield sub_obj

            continue

        yield obj
