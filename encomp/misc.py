"""
Miscellaneous functions that do not fit anywhere else.
"""

from typing import Any, _GenericAlias, Union, Type, Tuple
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
    if isinstance(expected, Type):
        return isinstance(obj, expected)

    try:
        check_type('obj', obj, expected)
        return True

    except TypeError:
        return False


def grid_dimensions(N: int, nrows: int, ncols: int) -> Tuple[int, int]:
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
    Tuple[int, int]
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
