from inspect import isclass
from typing import Any

from typeguard import TypeCheckMemo
from typeguard._checkers import checker_lookup_functions
from typeguard._exceptions import TypeCheckError
from typeguard._utils import qualified_name

from ..misc import isinstance_types
from ..units import Quantity


def quantity_checker(value: Any, origin_type: Any, args: tuple[Any, ...], memo: TypeCheckMemo) -> None:  # noqa: ANN401, ARG001
    annotation = origin_type[args] if args else origin_type

    if not isinstance_types(value, annotation):
        raise TypeCheckError(f"is not an instance of {qualified_name(annotation)}")


def quantity_checker_lookup(origin_type: Any, args: tuple[Any, ...], extras: tuple[Any, ...]) -> Any:  # noqa: ANN401, ARG001
    if isclass(origin_type) and issubclass(origin_type, Quantity):
        return quantity_checker
    return None


def pytest_configure(config: Any) -> None:  # noqa: ANN401, ARG001
    checker_lookup_functions.insert(0, quantity_checker_lookup)
