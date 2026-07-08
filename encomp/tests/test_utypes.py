from typing import Literal, get_origin

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
