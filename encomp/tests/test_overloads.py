from typing import reveal_type

import polars as pl

from encomp.units import Quantity as Q
from encomp.utypes import MassUnits


def test_overloads() -> None:
    # Test with explicit float
    v1 = Q(1.0, "m")
    reveal_type(v1)

    # Test with int
    v2 = Q([1, 2, 3], "kg")
    reveal_type(v2)

    Q(pl.col.asd, "kg")
    Q(pl.DataFrame({"test": []})["test"], "kg")

    # Test with typed unit
    unit: MassUnits = "kg"
    v3 = Q(1.0, unit)
    reveal_type(v3)
