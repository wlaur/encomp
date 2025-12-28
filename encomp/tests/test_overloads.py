from typing import reveal_type

import polars as pl

from encomp.units import Quantity as Q


def test_overloads() -> None:
    reveal_type(Q(1.0, "m"))
    reveal_type(Q(1, "m"))

    reveal_type(Q(1, str("m")))  # noqa: UP018

    reveal_type(Q([1, 2, 3], "kg"))

    reveal_type(Q(pl.col.asd, "kg"))
    reveal_type(Q(pl.DataFrame({"test": []})["test"], "kg"))
