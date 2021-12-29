"""
Miscellaneous functions that do not fit anywhere else.
"""

import ast
import asttokens
import numpy as np

from typing import Any, _GenericAlias, Union, Type  # type: ignore
from typeguard import check_type


def isinstance_types(obj: Any,
                     expected: Union[_GenericAlias, Type]) -> bool:
    """
    Checks if the input object matches the expected type.
    This function also supports complex type annotations that cannot
    be checked with the builtin ``isinstance()``.

    Uses ``typeguard.check_type``.

    Parameters
    ----------
    obj : Any
        Object to check
    expected : Union[_GenericAlias, type]
        Expected type or generic type alias

    Returns
    -------
    bool
        Whether the input object matches the expected type
    """

    # normal types are checked with isinstance()
    if isinstance(expected, Type):  # type: ignore
        return isinstance(obj, expected)

    try:
        check_type('obj', obj, expected)
        return True

    except TypeError:
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

            raise ValueError(
                f'{N} items cannot be placed in a {nrows} × {ncols} grid')

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
        if hasattr(node, 'lineno'):
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name):

                    start = node.first_token.startpos  # type: ignore
                    end = node.last_token.endpos  # type: ignore
                    assignment_src = atok.text[start:end]

                    assigned_names.append((node.targets[0].id, assignment_src))

    return assigned_names


def pad_2D_array(arr: np.ndarray, N: int) -> np.ndarray:
    """
    Pads a ragged 2D Numpy arrays

    Parameters
    ----------
    arr : np.ndarray
        Input array
    N : int
        Goal length, padding is added to reach this

    Returns
    -------
    np.ndarray
        Output array, without ragged sequences
    """

    arr = arr.copy()

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):

            val = arr[i, j]

            if isinstance(val, (int, float)):
                arr[i, j] = np.repeat(val, N)

    return arr


def is_multiple_element_array(x: Any) -> bool:
    """
    Checks whether the input is a Numpy array
    or list-like with more than one element.

    Parameters
    ----------
    x : Any
        Input object

    Returns
    -------
    bool
        Whether ``x`` is an array or list-like
        with more than one element
    """

    list_like = (list, tuple, np.ndarray)

    if not isinstance(x, list_like):
        return False

    if isinstance(x, np.ndarray):
        return x.size > 1

    try:
        return len(x) > 1
    except TypeError:
        return False
