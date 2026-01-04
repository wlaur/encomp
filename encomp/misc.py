import ast
from types import UnionType
from typing import Any, TypeIs, get_args, get_origin

import asttokens
from typeguard import check_type


def isinstance_types[T](obj: Any, expected: type[T]) -> TypeIs[T]:  # noqa: ANN401
    from .units import Quantity
    from .utypes import UnknownDimensionality

    if get_origin(expected) is UnionType:
        try:
            return isinstance(obj, expected)
        except TypeError:
            return any(isinstance_types(obj, n) for n in get_args(expected))

    if isinstance(obj, Quantity) and isinstance(expected, type) and (issubclass(expected, Quantity)):
        if expected is Quantity:
            return isinstance(obj, expected)

        expected_dt = getattr(expected, "_dimensionality_type", None)
        expected_mt = getattr(expected, "_magnitude_type", None)

        if expected_dt == UnknownDimensionality:
            if expected_mt is None:
                return True

            return isinstance_types(obj.m, expected_mt)

        if expected_dt is not None and obj._dimensionality_type is not expected_dt:
            return False

        return not (expected_mt is not None and not isinstance_types(obj.m, expected_mt))

    try:
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
        nrows = N // ncols if N % ncols == 0 else N // ncols + 1

    elif ncols == -1:
        ncols = N // nrows if N % nrows == 0 else N // nrows + 1

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

    assigned_names: list[tuple[str, str]] = []

    atok = asttokens.ASTTokens(src, parse=True)

    if atok.tree is None:
        return assigned_names

    for node in ast.walk(atok.tree):
        if hasattr(node, "lineno") and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            start = node.first_token.startpos  # type: ignore
            end = node.last_token.endpos  # type: ignore
            assignment_src = atok.text[start:end]

            assigned_names.append((node.targets[0].id, assignment_src))

    return assigned_names
