from typing import Literal, get_origin

import pytest

from .. import utypes as ut

_EXPLICIT_UTYPES_EXPORTS = {
    "AllUnits",
    "BASE_SI_UNITS",
    "DT",
    "DT_",
    "MT",
    "MT_",
    "Numpy1DArray",
    "Numpy1DBoolArray",
    "UnitsContainer",
    "get_registered_units",
}


def test_utypes_all_is_complete() -> None:
    expected = set(_EXPLICIT_UTYPES_EXPORTS)

    for name, value in vars(ut).items():
        if name.startswith("_"):
            continue

        if get_origin(value) is Literal and name.endswith("Units"):
            expected.add(name)

        if isinstance(value, type) and issubclass(value, ut.Dimensionality):
            expected.add(name)

    assert set(ut.__all__) == expected


def test_dimensionality_requires_dimensions() -> None:
    # a non-intermediate Dimensionality subclass must define the `dimensions` attribute.
    # Building the subclass via type() (rather than a class statement) keeps the failing
    # definition out of the module namespace; either way it raises inside
    # __init_subclass__, before the class is ever registered, so the process-wide
    # dimensionality registry is never mutated.
    with pytest.raises(TypeError, match="'dimensions' is not defined"):
        type("_CoverageNoDimensions", (ut.Dimensionality,), {})


def test_dimensionality_subclass_dimensions_must_match_parent() -> None:
    # subclassing a concrete dimensionality with a different `dimensions` is rejected
    with pytest.raises(TypeError, match="do not match"):
        type("_CoverageBadChild", (ut.Length,), {"dimensions": ut.Mass.dimensions})
