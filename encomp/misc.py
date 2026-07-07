import ast
from types import UnionType
from typing import Any, TypeIs, Union, cast, get_args, get_origin

from typeguard import TypeCheckError, check_type
from typing_extensions import TypeForm


def _is_quantity_subclass(expected: object) -> bool:
    from .units import Quantity

    return isinstance(expected, type) and issubclass(expected, Quantity)


def isinstance_types[T](obj: Any, expected: TypeForm[T]) -> TypeIs[T]:  # noqa: ANN401
    from .units import Quantity
    from .utypes import UnknownDimensionality

    # typeguard treats a plain string as a forward reference it cannot resolve
    # here: it would emit a TypeHintWarning and *pass*, silently answering True
    # for any obj. Reject it explicitly instead
    if isinstance(expected, str):
        raise TypeError(f"expected must be a type or type form, not a string: {expected!r}")

    origin = get_origin(expected)

    if origin in (UnionType, Union):
        # a Quantity must be routed through the detailed per-member logic below: a plain
        # isinstance against the union misclassifies a Quantity[UnknownDimensionality, ...]
        # member (it matches ANY dimensionality here, but is a *sibling* class at runtime, so
        # isinstance says False) -- decompose so single-type and union checks stay consistent.
        if not isinstance(obj, Quantity) and origin is UnionType:
            # narrowed to a UnionType by the check above, which isinstance accepts
            try:
                return isinstance(obj, cast(UnionType, expected))
            except TypeError:
                pass
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

    if _is_quantity_subclass(expected):
        # obj is not a Quantity instance (that case is handled above), so it can
        # never match -- and delegating to check_type could recurse forever
        # through a custom typeguard checker that routes Quantity checks back here
        return False

    try:
        check_type(cast(Any, obj), expected)
        return True
    except TypeCheckError:
        # only a genuine type mismatch answers False -- an invalid ``expected``
        # (e.g. an unsupported type form) propagates instead of being swallowed
        return False


def name_assignments(src: str) -> list[tuple[str, str]]:
    assigned_names: list[tuple[str, str]] = []

    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            source_segment = ast.get_source_segment(src, node) or ast.unparse(node)
            assigned_names.append((node.targets[0].id, source_segment))

    return assigned_names
