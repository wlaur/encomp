from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel

from ..units import Quantity


def test_model_serialize() -> None:
    class M(BaseModel):
        qty: Quantity[Any]

    for m in [
        2,
        2.5,
        [1, 2, 3],
        [1.0, 2.0, 3.5],
        pl.Series([2, 34, 5]),
        pl.Series([2, 34, 5], dtype=pl.Int64),
        pl.Series([2, 34, 5], dtype=pl.Int32),
        pl.Series([2, 34, 5], dtype=pl.Int16),
        pl.Series([2, 34, 5], dtype=pl.Float32),
        pl.Series([2, 34, 5], dtype=pl.Float64),
        np.array([[1, 2, 3], [3, 2, 1], [2, 2, 2]]).ravel(),
    ]:
        qty = Quantity(m, "kg")
        serialized = M(qty=qty).model_dump_json()

        deserialized = M.model_validate_json(serialized)

        deserialized.qty.to_base_units().to(qty.u)

        if isinstance(qty.m, np.ndarray) or isinstance(deserialized.qty.m, np.ndarray):
            assert np.array_equal(qty.to_base_units().m, deserialized.qty.to_base_units().m)
        elif isinstance(qty.m, pl.Series) or isinstance(deserialized.qty.m, pl.Series):
            assert (qty.to_base_units().m == deserialized.qty.to_base_units().m).all()
        else:
            assert qty == deserialized.qty

        if isinstance(m, float | int):
            assert deserialized.qty.m == m
        elif isinstance(m, list):
            m = np.array(m)
        else:
            assert type(deserialized.qty.m) is type(m)

    assert isinstance(M.model_json_schema(), dict)
