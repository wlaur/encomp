"""
Miscellaneous functions that do not fit anywhere else.
"""
import ast
from types import UnionType
from typing import Any, Type, TypeVar, _GenericAlias, _TypedDictMeta, overload  # type: ignore

import asttokens
from typeguard import check_type
from typing_extensions import TypeGuard

T = TypeVar("T")


# NOTE: these overloads are a workaround to avoid issues with type[T] -> T
# signatures with mypy


@overload
def isinstance_types(obj: Any, expected: type[T]) -> TypeGuard[T]:
    ...


@overload
def isinstance_types(obj: Any, expected: T) -> bool:
    ...


def isinstance_types(obj: Any, expected: _GenericAlias | type) -> bool:
    """
    Checks if the input object matches the expected type.
    This function also supports complex type annotations that cannot
    be checked with the builtin ``isinstance()``.
    Uses ``typeguard.check_type`` for runtime checks of complex types.

    .. note::

        For mappings (e.g. ``dict[str, list[float]]``), only the
        first item is checked. Use custom validation logic to ensure
        that all elements are checked.

    .. todo::

        Return type hint should be a ``TypeGuard`` that helps static type checkers
        to narrow down the type of the input object.

        This does not work with complex types using ``mypy`` (https://github.com/python/mypy/issues/9003).
        However, it does work with Pylance.
        The current implementation is a workaround to avoid ``mypy`` errors when calling
        this function. The type guard does not work with ``mypy`` (the type will not be narrowed at all).

        ``mypy`` and Pylance do not support type negation using ``TypeGuard``.
        This means that the following does not work as expected (compare with behavior for
        the builtin ``isinstance()``):

        .. code-block:: python

            a: str | int = ...

            if isinstance_types(a, int):
                reveal_type(a)  # int
            else:
                reveal_type(a)  # str | int, should be str

            # this works with builtin isinstance()
            if isinstance(a, int):
                reveal_type(a)  # int
            else:
                reveal_type(a)  # str


    Parameters
    ----------
    obj : Any
        Object to check
    expected : _GenericAlias | type
        Expected type or type alias

    Returns
    -------
    bool
        Whether the input object matches the expected type
    """

    # normal types are checked with isinstance()
    # note: this check must use typing.Type, not the builtin type (lower case)
    if isinstance(expected, Type):  # type: ignore
        # typing.TypedDict is a special case
        if not isinstance(expected, _TypedDictMeta):
            return isinstance(obj, expected)

    if type(expected) is UnionType:
        try:
            return isinstance(obj, expected)
        except TypeError:
            return any(isinstance_types(obj, n) for n in expected.__args__)

    try:
        # this function raises an exception in case the object type
        # does not match the expected type
        check_type(obj, expected)
        return True

    except Exception:
        return False


def grid_dimensions(N: int, nrows: int, ncols: int) -> tuple[int, int]:
    """
    Returns image grid dimensions (rows and columns) based
    on the total number of items ``N``.

    Parameters
    ----------
    N : int
        Total number of items
    nrows : int
        Number or rows, -1 means determined by the number of items
    ncols : int
        Number of columns, -1 means determined by the number of items

    Returns
    -------
    tuple[int, int]
        Number of rows and columns for a grid that fits all items
    """

    if nrows == -1 and ncols == -1:
        # start with a square grid
        nrows = ncols = int(N**0.5)

        # increase the number of cols until all items fit in the grid
        while nrows * ncols < N:
            ncols += 1

        return nrows, ncols

    if nrows == -1:
        if N % ncols == 0:
            nrows = N // ncols
        else:
            nrows = N // ncols + 1

    elif ncols == -1:
        if N % nrows == 0:
            ncols = N // nrows
        else:
            ncols = N // nrows + 1

    else:
        if nrows * ncols < N:
            raise ValueError(f"{N} items cannot be placed in a {nrows} x {ncols} grid")

    return nrows, ncols


def name_assignments(src: str) -> list[tuple[str, str]]:
    """
    Finds all names that are assigned in the input Python source code.

    Parameters
    ----------
    src : str
        Python source code

    Returns
    -------
    list[tuple[str, str]]
        List of names and the assignment statements
    """

    assigned_names = []

    atok = asttokens.ASTTokens(src, parse=True)

    for node in ast.walk(atok.tree):
        if hasattr(node, "lineno"):
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name):
                    start = node.first_token.startpos  # type: ignore
                    end = node.last_token.endpos  # type: ignore
                    assignment_src = atok.text[start:end]

                    assigned_names.append((node.targets[0].id, assignment_src))

    return assigned_names
