"""Tests for the ``typeguard`` checker that :mod:`encomp.units` registers for ``Quantity``.

The checker is what makes ``@typeguard.typechecked`` and the nested type forms inside
:func:`encomp.misc.isinstance_types` compare *dimensionality* and *magnitude type* rather
than falling back to a plain ``isinstance``. It is registered on import of
``encomp.units``, so these tests exercise the same behavior library users get.
"""

from typing import Any, TypedDict, cast

import numpy as np
import polars as pl
import pytest
from typeguard import TypeCheckError, typechecked

from ..misc import isinstance_types
from ..units import Quantity as Q
from ..utypes import (
    EnergyPerMass,
    HeatingValue,
    Length,
    Numpy1DArray,
    Pressure,
    Temperature,
    TemperatureDifference,
    UnknownDimensionality,
)

PRESSURE = Q(1.0, "bar")
LENGTH = Q(1.0, "m")
MASS = Q(1.0, "kg")
PRESSURE_ARRAY = Q(np.array([1.0, 2.0]), "bar")
PRESSURE_SERIES = Q(pl.Series([1.0, 2.0]), "bar")
PRESSURE_EXPR = Q(pl.col("p"), "bar")


# --------------------------------------------------------------------------------------
# @typechecked: dimensionality
# --------------------------------------------------------------------------------------


@typechecked
def _takes_pressure(q: Q[Pressure, float]) -> None: ...


@typechecked
def _takes_any_quantity(q: Q[Any, Any]) -> None: ...


# a bare Quantity annotation: both type parameters default, and at runtime the checker sees
# origin_type=Quantity with no args, so it accepts any dimensionality and magnitude
@typechecked
def _takes_bare_quantity(q: Q) -> None: ...


@typechecked
def _takes_unknown_dimensionality(q: Q[UnknownDimensionality, Any]) -> None: ...


def test_typechecked_accepts_matching_dimensionality() -> None:
    _takes_pressure(PRESSURE)


def test_typechecked_rejects_wrong_dimensionality() -> None:
    with pytest.raises(TypeCheckError, match="Quantity"):
        _takes_pressure(cast(Any, LENGTH))


def test_typechecked_error_message_names_both_quantity_types() -> None:
    # the message format the README documents
    with pytest.raises(TypeCheckError) as excinfo:
        _takes_pressure(cast(Any, Q(26.0, "kW")))

    message = str(excinfo.value)

    assert "encomp.units.Quantity[Power, float]" in message
    assert "is not an instance of encomp.units.Quantity[Pressure, float]" in message


@pytest.mark.parametrize("quantity", [PRESSURE, LENGTH, MASS, PRESSURE_ARRAY, PRESSURE_SERIES, PRESSURE_EXPR])
def test_typechecked_any_quantity_accepts_every_dimensionality_and_magnitude(quantity: Q[Any, Any]) -> None:
    # Q[Any, Any] resolves to Quantity[UnknownDimensionality] at runtime, which is a *sibling*
    # of every concrete dimensionality, not a base class -- plain isinstance answers False here
    _takes_any_quantity(quantity)
    _takes_unknown_dimensionality(quantity)
    _takes_bare_quantity(quantity)


def test_typechecked_rejects_a_non_quantity() -> None:
    for value in (1.0, "bar", None, np.array([1.0])):
        with pytest.raises(TypeCheckError):
            _takes_any_quantity(cast(Any, value))


# --------------------------------------------------------------------------------------
# @typechecked: magnitude type
# --------------------------------------------------------------------------------------


@typechecked
def _takes_scalar_pressure(q: Q[Pressure, float]) -> None: ...


@typechecked
def _takes_array_pressure(q: Q[Pressure, Numpy1DArray]) -> None: ...


@typechecked
def _takes_series_pressure(q: Q[Pressure, pl.Series]) -> None: ...


@typechecked
def _takes_expr_pressure(q: Q[Pressure, pl.Expr]) -> None: ...


def test_typechecked_discriminates_magnitude_type() -> None:
    _takes_scalar_pressure(PRESSURE)
    _takes_array_pressure(PRESSURE_ARRAY)
    _takes_series_pressure(PRESSURE_SERIES)
    _takes_expr_pressure(PRESSURE_EXPR)


@pytest.mark.parametrize(
    ("func", "wrong"),
    [
        (_takes_scalar_pressure, PRESSURE_ARRAY),
        (_takes_scalar_pressure, PRESSURE_SERIES),
        (_takes_scalar_pressure, PRESSURE_EXPR),
        (_takes_array_pressure, PRESSURE),
        (_takes_array_pressure, PRESSURE_SERIES),
        (_takes_series_pressure, PRESSURE),
        (_takes_series_pressure, PRESSURE_ARRAY),
        (_takes_expr_pressure, PRESSURE),
        (_takes_expr_pressure, PRESSURE_SERIES),
    ],
)
def test_typechecked_rejects_wrong_magnitude_type(func: Any, wrong: Q[Any, Any]) -> None:  # noqa: ANN401
    with pytest.raises(TypeCheckError):
        func(cast(Any, wrong))


def test_typechecked_any_magnitude_accepts_every_container() -> None:
    @typechecked
    def takes_any_magnitude(q: Q[Pressure, Any]) -> None: ...

    for quantity in (PRESSURE, PRESSURE_ARRAY, PRESSURE_SERIES, PRESSURE_EXPR):
        takes_any_magnitude(quantity)


# --------------------------------------------------------------------------------------
# @typechecked: sibling dimensionalities that share the same physical dimensions
# --------------------------------------------------------------------------------------


def test_typechecked_separates_temperature_from_temperature_difference() -> None:
    @typechecked
    def takes_temperature(q: Q[Temperature, float]) -> None: ...

    @typechecked
    def takes_difference(q: Q[TemperatureDifference, float]) -> None: ...

    takes_temperature(Q(300.0, "K"))
    takes_difference(Q(5.0, "delta_degC"))

    # both have dimensions [temperature], so only the dimensionality *class* separates them
    with pytest.raises(TypeCheckError):
        takes_temperature(cast(Any, Q(5.0, "delta_degC")))

    with pytest.raises(TypeCheckError):
        takes_difference(cast(Any, Q(300.0, "K")))


def test_typechecked_separates_sibling_energy_per_mass_dimensionalities() -> None:
    @typechecked
    def takes_heating_value(q: Q[HeatingValue, float]) -> None: ...

    energy_per_mass = Q(10.0, "MJ/kg")

    assert isinstance_types(energy_per_mass, Q[EnergyPerMass, float])
    takes_heating_value(energy_per_mass.asdim(HeatingValue))

    with pytest.raises(TypeCheckError):
        takes_heating_value(cast(Any, energy_per_mass))


# --------------------------------------------------------------------------------------
# @typechecked: return values, containers, unions, TypedDict
# --------------------------------------------------------------------------------------


def test_typechecked_checks_the_return_value() -> None:
    @typechecked
    def returns_pressure() -> Q[Pressure, float]:
        return cast(Any, LENGTH)

    with pytest.raises(TypeCheckError):
        returns_pressure()


def test_typechecked_checks_quantities_nested_in_containers() -> None:
    @typechecked
    def takes_list(qs: list[Q[Pressure, float]]) -> None: ...

    @typechecked
    def takes_dict(qs: dict[str, Q[Pressure, float]]) -> None: ...

    @typechecked
    def takes_tuple(qs: tuple[Q[Pressure, float], Q[Length, float]]) -> None: ...

    takes_list([PRESSURE])
    takes_dict({"p": PRESSURE})
    takes_tuple((PRESSURE, LENGTH))

    with pytest.raises(TypeCheckError):
        takes_list(cast(Any, [LENGTH]))

    with pytest.raises(TypeCheckError):
        takes_dict(cast(Any, {"p": LENGTH}))

    with pytest.raises(TypeCheckError):
        takes_tuple(cast(Any, (LENGTH, PRESSURE)))


def test_typechecked_checks_optional_and_union_quantities() -> None:
    @typechecked
    def takes_optional(q: Q[Pressure, float] | None) -> None: ...

    @typechecked
    def takes_union(q: Q[Pressure, float] | Q[Length, float]) -> None: ...

    takes_optional(None)
    takes_optional(PRESSURE)
    takes_union(PRESSURE)
    takes_union(LENGTH)

    with pytest.raises(TypeCheckError):
        takes_optional(cast(Any, LENGTH))

    with pytest.raises(TypeCheckError):
        takes_union(cast(Any, MASS))


def test_typechecked_checks_typeddict_values() -> None:
    class State(TypedDict):
        P: Q[Pressure, Any]
        T: Q[Temperature, Any]

    @typechecked
    def returns_state(swap: bool) -> State:
        if swap:
            return {"P": Q(1.0, "bar"), "T": cast(Any, Q(1.0, "m"))}
        return {"P": Q(1.0, "bar"), "T": Q(300.0, "K")}

    returns_state(swap=False)

    with pytest.raises(TypeCheckError):
        returns_state(swap=True)


# --------------------------------------------------------------------------------------
# The checker must not hijack non-Quantity annotations
# --------------------------------------------------------------------------------------


def test_typechecked_still_checks_plain_types() -> None:
    @typechecked
    def takes_int(value: int) -> None: ...

    @typechecked
    def takes_list_of_str(values: list[str]) -> None: ...

    takes_int(1)
    takes_list_of_str(["a"])

    with pytest.raises(TypeCheckError):
        takes_int(cast(Any, "1"))

    with pytest.raises(TypeCheckError):
        takes_list_of_str(cast(Any, [1]))


def test_unit_objects_are_not_treated_as_quantities() -> None:
    @typechecked
    def takes_unit(unit: Any) -> None: ...  # noqa: ANN401

    # Unit is a separate class hierarchy; the lookup must not claim it
    takes_unit(PRESSURE.u)

    with pytest.raises(TypeCheckError):
        _takes_any_quantity(cast(Any, PRESSURE.u))


# --------------------------------------------------------------------------------------
# isinstance_types and @typechecked must agree
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "annotation", "expected"),
    [
        (PRESSURE, Q[Pressure, float], True),
        (PRESSURE, Q[Pressure, Any], True),
        (PRESSURE, Q[Any, Any], True),
        (PRESSURE, Q[UnknownDimensionality, float], True),
        (PRESSURE, Q[UnknownDimensionality, Numpy1DArray], False),
        (PRESSURE, Q[Length, float], False),
        (PRESSURE, Q[Pressure, Numpy1DArray], False),
        (PRESSURE_ARRAY, Q[Pressure, Numpy1DArray], True),
        (PRESSURE_ARRAY, Q[Pressure, float], False),
        (PRESSURE_SERIES, Q[Pressure, pl.Series], True),
        (PRESSURE_EXPR, Q[Pressure, pl.Expr], True),
    ],
)
def test_isinstance_types_direct_forms(value: Q[Any, Any], annotation: Any, expected: bool) -> None:  # noqa: ANN401
    assert isinstance_types(value, annotation) is expected


@pytest.mark.parametrize(
    ("value", "annotation", "expected"),
    [
        ([PRESSURE], list[Q[Pressure, Any]], True),
        ([PRESSURE], list[Q[Any, Any]], True),
        ([PRESSURE], list[Q[UnknownDimensionality, Any]], True),
        ([PRESSURE], list[Q[Length, Any]], False),
        ([PRESSURE], list[Q[Pressure, Numpy1DArray]], False),
        ({1: PRESSURE}, dict[int, Q[Any, Any]], True),
        ({1: PRESSURE}, dict[int, Q[Length, Any]], False),
        ((PRESSURE,), tuple[Q[Any, Any], ...], True),
        ((PRESSURE, LENGTH), tuple[Q[Pressure, Any], Q[Length, Any]], True),
        ((PRESSURE, LENGTH), tuple[Q[Length, Any], Q[Pressure, Any]], False),
    ],
)
def test_isinstance_types_nested_forms_use_the_checker(value: Any, annotation: Any, expected: bool) -> None:  # noqa: ANN401
    # a nested Quantity is resolved by typeguard, which reaches the registered checker; without
    # it, Quantity[UnknownDimensionality] would fail the plain isinstance typeguard falls back on
    assert isinstance_types(value, annotation) is expected


def test_nested_and_direct_forms_agree() -> None:
    annotations: list[Any] = [
        Q[Any, Any],
        Q[UnknownDimensionality, Any],
        Q[Pressure, Any],
        Q[Pressure, float],
        Q[Length, Any],
        Q[Pressure, Numpy1DArray],
    ]

    for annotation in annotations:
        direct = isinstance_types(PRESSURE, annotation)
        in_list = isinstance_types([PRESSURE], list[annotation])
        in_dict = isinstance_types({"a": PRESSURE}, dict[str, annotation])
        in_tuple = isinstance_types((PRESSURE,), tuple[annotation, ...])

        assert direct == in_list == in_dict == in_tuple, f"inconsistent nesting for {annotation}"
