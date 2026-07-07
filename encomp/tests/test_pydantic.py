import json
from typing import Any, assert_type, cast

import numpy as np
import polars as pl
import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from ..units import Quantity, set_quantity_format
from ..utypes import Length, Mass, UnknownDimensionality


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_model_serialize() -> None:
    class M(BaseModel):
        qty: Quantity[Mass, Any]

    # heterogeneous magnitude inputs, typed as a list of Any
    magnitudes = cast(
        "list[Any]",
        [
            2,
            2.5,
            [1, 2, 3],
            [1.0, 2.0, 3.5],
            pl.Series([2, 34, 5]),
            pl.Series([2, 34, 5], dtype=pl.Int64),
            pl.Series([2, 34, 5], dtype=pl.Int32),
            pl.Series([2, 34, 5], dtype=pl.Int16),
            pl.Series([2, 34, 5], dtype=pl.UInt8),
            pl.Series([2, 34, 5], dtype=pl.Float32),
            pl.Series([2, 34, 5], dtype=pl.Float64),
            np.array([[1, 2, 3], [3, 2, 1], [2, 2, 2]]).ravel(),
        ],
    )

    for m in magnitudes:
        qty = cast("Quantity[Mass, Any]", Quantity(m, "kg"))
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
            assert cast(Any, deserialized.qty).m == m
        elif isinstance(m, list):
            m = cast(Any, np.array(cast(Any, m)))
        else:
            assert type(cast(Any, deserialized.qty).m) is type(m)

    schema = M.model_json_schema()
    assert isinstance(schema, dict)
    value_schema = schema["properties"]["qty"]["properties"]["value"]["anyOf"]
    assert {"type": "null"} not in value_schema


def test_model_dump_python_mode_keeps_quantity() -> None:
    class M(BaseModel):
        qty: Quantity[Mass, float]

    qty = Quantity(2.0, "kg")
    model = M(qty=qty)

    assert model.model_dump() == {"qty": qty}
    assert model.model_dump(mode="json") == {"qty": {"unit": "kilogram", "value": 2.0, "magnitude_type": "float"}}


def test_json_unit_serialization_ignores_display_format() -> None:
    class M(BaseModel):
        qty: Quantity[Mass, float]

    try:
        set_quantity_format("~L")
        dumped = M(qty=Quantity(2.0, "kg")).model_dump_json()
    finally:
        set_quantity_format("compact")

    assert r"\mathrm" not in dumped
    assert '"kilogram"' in dumped
    assert M.model_validate_json(dumped).qty == Quantity(2.0, "kg")


def test_pydantic_validation_errors_are_collected() -> None:
    class M(BaseModel):
        length: Quantity[Length, float]
        mass: Quantity[Mass, float]

    with pytest.raises(ValidationError) as exc_info:
        M(
            length=cast(Any, Quantity(1.0, "kg")),
            mass=cast(Any, {"unit": "kg", "value": None, "magnitude_type": "float"}),
        )

    errors = exc_info.value.errors()
    assert {error["loc"] for error in errors} == {("length",), ("mass",)}
    assert {error["type"] for error in errors} == {"quantity_dimensionality", "quantity_validation"}


def test_pydantic_json_payload_enforces_magnitude_type() -> None:
    class M(BaseModel):
        qty: Quantity[Mass, float]

    payload = {"qty": {"unit": "kg", "value": [1.0], "magnitude_type": "list"}}

    with pytest.raises(ValidationError) as exc_info:
        M.model_validate_json(json.dumps(payload))

    errors = exc_info.value.errors()
    assert errors[0]["loc"] == ("qty",)
    assert errors[0]["type"] == "quantity_magnitude_type"


def test_pydantic_expr_serialization_error() -> None:
    class M(BaseModel):
        qty: Quantity[Mass, Any]

    with pytest.raises(PydanticSerializationError, match="expression holds no data"):
        M(qty=Quantity(pl.col("x"), "kg")).model_dump_json()


def test_type_adapter_round_trip_and_validation() -> None:
    adapter = TypeAdapter(Quantity[Mass, float])
    qty = Quantity(2.0, "kg")

    dumped = adapter.dump_json(qty)
    assert adapter.validate_json(dumped) == qty
    assert adapter.dump_python(qty) is qty

    with pytest.raises(ValidationError) as exc_info:
        adapter.validate_python(Quantity(2.0, "m"))

    assert exc_info.value.errors()[0]["type"] == "quantity_dimensionality"


def test_unknown_pydantic_field() -> None:
    class A(BaseModel):
        v: Quantity[UnknownDimensionality, float]

    assert_type(A(v=Quantity(2, "kg").asdim(UnknownDimensionality)).v, Quantity[UnknownDimensionality, float])
