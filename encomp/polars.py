"""Polars integration: a unit-carrying extension data type.

Importing this module registers the ``"encomp.unit"`` extension type with polars.
After that, plain polars I/O (``pl.read_parquet``, ``pl.scan_parquet``,
``pl.sink_parquet``, ...) round-trips unit-typed columns with no encomp-specific
read/write functions: the unit is column-level Arrow field metadata
(``ARROW:extension:name`` / ``ARROW:extension:metadata``) in the file, readable from
pyarrow, DuckDB and Spark with stock APIs. Consumers without encomp installed load the
plain storage values.

Polars refuses arithmetic on extension-typed columns (there is no third-party kernel
or supertype resolution), so the dtype is deliberately *only* a persistence and
guardrail layer: unit algebra lives in :class:`encomp.units.Quantity`, which is
constructed from unit-typed columns via :func:`quantities` and written back via
:func:`dataframe` or ``Expr.ext.to(UnitDType(...))``.

.. warning::
    The underlying polars extension-type API is marked unstable by polars. encomp
    pins the observed behavior in its test suite so a breaking polars bump fails
    loudly rather than silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, overload

import numpy as np
import polars as pl

from .units import Quantity, Unit
from .utypes import UnknownDimensionality

__all__ = [
    "EXTENSION_NAME",
    "UnitDType",
    "dataframe",
    "quantities",
    "units_of",
    "with_units",
]

EXTENSION_NAME = "encomp.unit"
"""Extension-type name registered with polars, and the public, stable identifier under
which the unit string is stored as Arrow field metadata in Parquet/IPC files."""


def _validate_storage(storage: pl.DataType) -> None:
    # bool magnitudes are always a mistake (same runtime invariant as Quantity), and
    # non-numeric dtypes cannot be magnitudes at all. An already unit-typed column is
    # also rejected here (its dtype is an extension type, not a numeric one).
    if not (storage.is_float() or storage.is_integer()):
        raise TypeError(
            f"unit dtype storage must be a float or integer dtype, got {storage!r}. "
            "Boolean and non-numeric columns cannot carry a unit, and a column that "
            "already has a unit dtype cannot be re-wrapped."
        )


class UnitDType(pl.BaseExtension):
    """Polars extension data type carrying a physical unit as column metadata.

    A dtype instance is (extension name ``"encomp.unit"``, numeric storage dtype, unit
    string). The unit string is normalized through the unit registry on construction,
    so different spellings of the same unit (``"m^3"``, ``"m³"``) produce equal
    dtypes; an unknown unit raises immediately.

    Polars refuses arithmetic on extension-typed columns, so a unit-typed column
    cannot silently take part in unitless math: the escape hatches are an explicit
    ``.ext.storage()`` unwrap or a :class:`encomp.units.Quantity` (see
    :func:`quantities`).
    """

    def __init__(self, unit: str | Unit[Any], storage: pl.DataType | None = None) -> None:
        if storage is None:
            storage = pl.Float64()
        _validate_storage(storage)
        canonical = str(Unit(unit) if isinstance(unit, str) else unit)
        super().__init__(EXTENSION_NAME, storage, canonical)

    @property
    def unit(self) -> Unit[Any]:
        """The column's unit, parsed from the dtype metadata.

        For a dtype loaded from a file this re-validates the stored unit string
        against the registry; a file carrying an unparseable unit raises here, not at
        scan time.
        """
        metadata = self.ext_metadata()
        if metadata is None:
            raise ValueError(f"{EXTENSION_NAME} dtype without a unit string in its metadata: {self!r}")
        return Unit(metadata)

    def _string_repr(self) -> str:
        return f"unit[{self.ext_metadata()}]"


def units_of(frame: pl.DataFrame | pl.LazyFrame) -> dict[str, Unit[Any]]:
    """Units of every unit-typed column in the frame, read from the schema.

    Reads only schema metadata: no data pass, and a ``LazyFrame`` (e.g. a
    ``pl.scan_parquet``) is not collected. Columns without a unit dtype are omitted.
    """
    return {name: dtype.unit for name, dtype in frame.collect_schema().items() if isinstance(dtype, UnitDType)}


@overload
def with_units(frame: pl.DataFrame, units: Mapping[str, str | Unit[Any]]) -> pl.DataFrame: ...


@overload
def with_units(frame: pl.LazyFrame, units: Mapping[str, str | Unit[Any]]) -> pl.LazyFrame: ...


def with_units(frame: pl.DataFrame | pl.LazyFrame, units: Mapping[str, str | Unit[Any]]) -> pl.DataFrame | pl.LazyFrame:
    """Attach unit dtypes to existing numeric columns.

    The magnitudes are not touched and the storage dtype is preserved
    (``Float32`` stays ``Float32``). The unit schema is always spelled explicitly by
    the caller — it is never inferred from data. A key that is not a column, a
    non-numeric column, or an unknown unit string raises.
    """
    schema = frame.collect_schema()
    missing = sorted(set(units) - set(schema))
    if missing:
        raise ValueError(f"unit schema keys are not columns of the frame: {missing}")
    return frame.with_columns(
        pl.col(name).ext.to(UnitDType(unit, storage=schema[name])) for name, unit in units.items()
    )


@overload
def quantities(frame: pl.DataFrame) -> dict[str, Quantity[UnknownDimensionality, pl.Series]]: ...


@overload
def quantities(frame: pl.LazyFrame) -> dict[str, Quantity[UnknownDimensionality, pl.Expr]]: ...


def quantities(
    frame: pl.DataFrame | pl.LazyFrame,
) -> dict[str, Quantity[UnknownDimensionality, Any]]:
    """Quantities for every unit-typed column: unit from the dtype, magnitude unwrapped.

    For a ``DataFrame`` the magnitudes are ``pl.Series``; for a ``LazyFrame`` they are
    ``pl.col(name)`` expressions, so the result composes into the lazy plan. The
    dimensionality is unknown statically (the file schema is runtime data) — assert it
    once at the boundary with ``.asdim(...)``.
    """
    units = units_of(frame)
    if isinstance(frame, pl.LazyFrame):
        return {name: Quantity(pl.col(name).ext.storage(), unit) for name, unit in units.items()}
    return {name: Quantity(frame.get_column(name).ext.storage(), unit) for name, unit in units.items()}


def dataframe(
    quantities: Mapping[str, Quantity[Any, pl.Series] | Quantity[Any, Any]],
    *,
    to: Mapping[str, str | Unit[Any]] | None = None,
) -> pl.DataFrame:
    """Build a ``DataFrame`` of unit-typed columns from quantities.

    Each quantity is converted to the target unit from ``to`` (if given), its
    magnitude becomes the column values, and its (converted) unit becomes the column's
    unit dtype. Magnitudes must be data (``pl.Series``, 1-D numpy array or scalar
    ``float``) — a ``pl.Expr`` magnitude is a deferred plan, not data, and raises.
    """
    columns: list[pl.Series] = []
    for name, quantity in quantities.items():
        if to is not None and name in to:
            quantity = quantity.to(to[name])
        magnitude = quantity.m
        if isinstance(magnitude, pl.Expr):
            raise TypeError(
                f"column {name!r}: a pl.Expr magnitude is a deferred plan, not data; evaluate it in a "
                "polars context (e.g. df.with_columns(q.m.ext.to(UnitDType(...)))) instead"
            )
        if isinstance(magnitude, pl.Series):
            series = magnitude.alias(name)
        else:
            series = pl.Series(name, np.atleast_1d(magnitude))
        columns.append(series.ext.to(UnitDType(quantity.u, storage=series.dtype)))
    return pl.DataFrame(columns)


# Registration happens at import so that plain polars I/O reconstructs UnitDType columns
# in any process that has imported encomp.polars; processes without it load the storage
# type (with a polars warning that names the extension type).
pl.register_extension_type(EXTENSION_NAME, UnitDType)
