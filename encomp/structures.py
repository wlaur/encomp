"""
Data structures and related functions.
"""

from typing import Sequence, Iterator, Optional, Any, Union


def divide_chunks(container: Sequence[Any], N: int) -> Iterator[Any]:
    """
    Generator that divides a container into chunks with length ``N``.
    The last chunk might not have ``N`` elements.

    .. code-block:: python

        parts_with_3_elements = list(divide_chunks(container, 3))


    Parameters
    ----------
    container : Sequence[Any]
        The container that will be split into chunks. Since sets are unordered,
        it does not make sense to accept set inputs here.
    N : int
        Number of element for one chunk (last chunk might be shorter)

    Yields
    -------
    Iterator[Any]
        Generator of chunks
    """

    if not len(container):
        raise ValueError('Cannot chunk empty container')

    if N < 1:
        raise ValueError(f'Cannot split container into {N} chunks')

    for i in range(0, len(container), N):
        yield container[i:i + N]


def flatten(container: Sequence[Union[Any, Sequence[Any]]],
            max_depth: Optional[int] = None, *,
            depth: int = 0) -> Iterator[Any]:
    """
    Generator that flattens a nested container.

    Usage:

    .. code-block:: python

        flat_list = list(flatten(nested_list), max_depth=3)

    This function will flatten arbitrarily deeply nested lists or tuples recursively.
    If ``max_depth`` is ``None``, recurse until no more nested structures remain,
    otherwise flatten until the specified max depth.

    .. note::
        Uses ``isinstance(obj, typing.Iterable)`` to determine if a sub-object is iterable.
        Strings are not considered iterables.

    Parameters
    ----------
    container : Sequence[Union[Any, Sequence[Any]]]
        The container to be flattened. Note that it is not possible
        to construct nested sets, so the ``Sequence`` type is appropriate here.
    max_depth : int, optional
        The maximum level to flatten to, by default None (flatten all)
    depth : int
        The current depth used by the recursive algorithm, do not pass this explicitly

    Yields
    -------
    Iterator[Any]
        Generator of non-nested objects
    """

    # in case the max depth was reached
    if max_depth is not None and depth >= max_depth:
        yield container

    depth += 1

    for obj in container:

        if isinstance(obj, str):
            yield obj
            continue

        # check if this object can be flattened further
        if isinstance(obj, Sequence):

            for sub_obj in flatten(obj, max_depth=max_depth, depth=depth):
                yield sub_obj

            continue

        yield obj
