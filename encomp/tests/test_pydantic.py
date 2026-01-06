from typing import Any, assert_type, cast

import numpy as np
import polars as pl
from pydantic import BaseModel

from ..units import Quantity
from ..utypes import Mass, Numpy1DArray, UnknownDimensionality


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_model_serialize() -> None:
    class M(BaseModel):
        qty: Quantity[Mass, Any]

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
        cast(Numpy1DArray, np.array([[1, 2, 3], [3, 2, 1], [2, 2, 2]]).ravel()),
    ]:
        qty = cast("Quantity[Mass, Any]", Quantity(m, "kg"))  # pyright: ignore[reportCallIssue, reportArgumentType]
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
            assert deserialized.qty.m == m  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
        elif isinstance(m, list):
            m = np.array(m)
        else:
            assert type(deserialized.qty.m) is type(m)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    assert isinstance(M.model_json_schema(), dict)


def test_unknown_pydantic_field() -> None:
    class A(BaseModel):
        v: Quantity[UnknownDimensionality, float]

    assert_type(A(v=Quantity(2, "kg").asdim(UnknownDimensionality)).v, Quantity[UnknownDimensionality, float])
