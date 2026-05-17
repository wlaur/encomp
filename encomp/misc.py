import ast
from types import UnionType
from typing import Any, Protocol, TypeIs, cast, get_args, get_origin

import asttokens
from typeguard import check_type
from typing_extensions import TypeForm


def _is_quantity_subclass(expected: object) -> bool:
    from .units import Quantity

    return isinstance(expected, type) and issubclass(expected, Quantity)


def isinstance_types[T](obj: Any, expected: TypeForm[T]) -> TypeIs[T]:  # noqa: ANN401
    from .units import Quantity
    from .utypes import UnknownDimensionality

    if get_origin(expected) is UnionType:
        # narrowed to a UnionType by the check above, which isinstance accepts
        try:
            return isinstance(obj, cast(UnionType, expected))
        except TypeError:
            return any(isinstance_types(obj, n) for n in get_args(expected))

    if isinstance(obj, Quantity) and _is_quantity_subclass(expected):
        if expected is Quantity:
            return True

        obj_q = cast("Quantity[Any, Any]", obj)  # pyrefly: ignore[redundant-cast]  # cast required by pyright

        expected_dt: type | None = getattr(expected, "_dimensionality_type", None)
        expected_mt: type | None = getattr(expected, "_magnitude_type", None)
        obj_m: Any = obj_q.m

        if expected_dt == UnknownDimensionality:
            if expected_mt is None:
                return True

            return isinstance_types(obj_m, expected_mt)

        obj_dt: type | None = getattr(obj_q, "_dimensionality_type", None)

        if expected_dt is not None and obj_dt is not expected_dt:
            return False

        return not (expected_mt is not None and not isinstance_types(obj_m, expected_mt))

    try:
        check_type(cast(Any, obj), expected)
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


class _ASTTokens(Protocol):
    tree: ast.Module | None

    def get_text(self, node: ast.AST) -> str: ...


def name_assignments(src: str) -> list[tuple[str, str]]:
    assigned_names: list[tuple[str, str]] = []

    atok = cast(_ASTTokens, asttokens.ASTTokens(src, parse=True))

    if atok.tree is None:
        return assigned_names

    for node in ast.walk(atok.tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            assigned_names.append((node.targets[0].id, atok.get_text(node)))

    return assigned_names
