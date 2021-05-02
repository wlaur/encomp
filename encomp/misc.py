"""
Miscellaneous functions that do not fit anywhere else.
"""

from typing import Any, _GenericAlias, Union, Type
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
