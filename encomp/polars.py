"""Polars integration: a unit-carrying extension data type.

Importing this module registers the ``"encomp.unit"`` extension type with polars.
After that, plain polars I/O (``pl.read_parquet``, ``pl.scan_parquet``,
``pl.sink_parquet``, ...) round-trips unit-typed columns with no encomp-specific
read/write functions: the unit is column-level Arrow field metadata
(``ARROW:extension:name`` / ``ARROW:extension:metadata``) in the file. Metadata-aware
Arrow consumers can inspect those standard field keys with their ordinary schema APIs.
Unregistered Polars 1.x readers warn and load storage values by default; readers using
``POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_extension`` preserve a generic
extension (the planned Polars 2.0 default).

Polars refuses arithmetic on extension-typed columns (there is no third-party kernel
or supertype resolution), so the dtype is deliberately *only* a persistence and
guardrail layer: unit algebra lives in :class:`encomp.units.Quantity`, which is
exposed by validated :class:`QuantityFrame` descriptors and written back through
:meth:`QuantityFrame.derive`. The :func:`with_units`, :func:`quantities`, and
:func:`dataframe` helpers remain available for schema-less bulk interop.

.. warning::
    The underlying polars extension-type API is marked unstable by polars. encomp
    pins the observed behavior in its test suite so a breaking polars bump fails
    loudly rather than silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self, cast, overload

import numpy as np
import polars as pl

from . import utypes as _ut
from ._polars_dtype import EXTENSION_NAME, UnitDType, canonical_unit_string
from .units import Quantity, Unit
from .utypes import Dimensionality, UnknownDimensionality

__all__ = [
    "EXTENSION_NAME",
    "Assignment",
    "Column",
    "QuantityFrame",
    "UnitDType",
    "canonical_unit_string",
    "dataframe",
    "quantities",
    "unit",
    "units_of",
    "with_units",
]


class Column[DT: Dimensionality]:
    """A unit-bearing column declaration on a :class:`QuantityFrame`.

    Access through the schema class returns the declaration. Access through a
    validated schema instance returns a ``Quantity[DT, pl.Expr]``.
    """

    def __init__(self, value: str | Unit[Any], dimensionality: type[DT], *, name: str | None = None) -> None:
        validated = Quantity(1.0, value).asdim(dimensionality)
        self.unit = validated.u
        self.dimensionality = dimensionality
        self._explicit_name = name
        self._attribute_name: str | None = None

    def __set_name__(self, owner: type[object], name: str) -> None:
        self._attribute_name = name

    @property
    def name(self) -> str:
        """Physical Polars column name, defaulting to the schema attribute name."""
        if self._explicit_name is not None:
            return self._explicit_name
        if self._attribute_name is None:
            raise RuntimeError("unit column declaration is not bound to a QuantityFrame class")
        return self._attribute_name

    @overload
    def __get__(self, instance: None, owner: type[object] | None = None) -> Column[DT]: ...

    @overload
    def __get__(self, instance: QuantityFrame, owner: type[object] | None = None) -> Quantity[DT, pl.Expr]: ...

    def __get__(
        self, instance: QuantityFrame | None, owner: type[object] | None = None
    ) -> Column[DT] | Quantity[DT, pl.Expr]:
        if instance is None:
            return self
        return Quantity(pl.col(self.name).ext.storage(), self.unit).asdim(self.dimensionality)

    def assign(self, value: Quantity[DT, pl.Expr]) -> Assignment[DT]:
        """Bind a typed expression quantity to this declared output column."""
        return Assignment(self, value)


def _validate_assignment_value(value: object) -> None:
    if not isinstance(value, Quantity):
        raise TypeError("quantity-column assignments require a Quantity with a pl.Expr magnitude")
    typed = cast("Quantity[Any, Any]", value)
    if not isinstance(typed.m, pl.Expr):
        raise TypeError("quantity-column assignments require a Quantity with a pl.Expr magnitude")


class Assignment[DT: Dimensionality]:
    """A dimension-checked output binding created by :meth:`Column.assign`."""

    def __init__(self, column: Column[DT], value: Quantity[DT, pl.Expr]) -> None:
        _validate_assignment_value(value)
        value.to(column.unit)
        self.column = column
        self.value = value


@overload
def unit(  # pyright: ignore[reportOverlappingOverload]  # narrow literals intentionally precede open-unit fallback
    value: _ut.DimensionlessUnits, *, name: str | None = None
) -> Column[_ut.Dimensionless]: ...


@overload
def unit(value: _ut.CurrencyUnits, *, name: str | None = None) -> Column[_ut.Currency]: ...


@overload
def unit(value: _ut.CurrencyPerEnergyUnits, *, name: str | None = None) -> Column[_ut.CurrencyPerEnergy]: ...


@overload
def unit(value: _ut.CurrencyPerMassUnits, *, name: str | None = None) -> Column[_ut.CurrencyPerMass]: ...


@overload
def unit(value: _ut.CurrencyPerVolumeUnits, *, name: str | None = None) -> Column[_ut.CurrencyPerVolume]: ...


@overload
def unit(value: _ut.CurrencyPerTimeUnits, *, name: str | None = None) -> Column[_ut.CurrencyPerTime]: ...


@overload
def unit(value: _ut.LengthUnits, *, name: str | None = None) -> Column[_ut.Length]: ...


@overload
def unit(value: _ut.MassUnits, *, name: str | None = None) -> Column[_ut.Mass]: ...


@overload
def unit(value: _ut.TimeUnits, *, name: str | None = None) -> Column[_ut.Time]: ...


@overload
def unit(value: _ut.FrequencyUnits, *, name: str | None = None) -> Column[_ut.Frequency]: ...


@overload
def unit(value: _ut.TemperatureUnits, *, name: str | None = None) -> Column[_ut.Temperature]: ...


@overload
def unit(value: _ut.TemperatureDifferenceUnits, *, name: str | None = None) -> Column[_ut.TemperatureDifference]: ...


@overload
def unit(value: _ut.SubstanceUnits, *, name: str | None = None) -> Column[_ut.Substance]: ...


@overload
def unit(value: _ut.MolarMassUnits, *, name: str | None = None) -> Column[_ut.MolarMass]: ...


@overload
def unit(value: _ut.SubstancePerMassUnits, *, name: str | None = None) -> Column[_ut.SubstancePerMass]: ...


@overload
def unit(value: _ut.CurrentUnits, *, name: str | None = None) -> Column[_ut.Current]: ...


@overload
def unit(value: _ut.LuminosityUnits, *, name: str | None = None) -> Column[_ut.Luminosity]: ...


@overload
def unit(value: _ut.AreaUnits, *, name: str | None = None) -> Column[_ut.Area]: ...


@overload
def unit(value: _ut.VolumeUnits, *, name: str | None = None) -> Column[_ut.Volume]: ...


@overload
def unit(value: _ut.NormalVolumeUnits, *, name: str | None = None) -> Column[_ut.NormalVolume]: ...


@overload
def unit(value: _ut.PressureUnits, *, name: str | None = None) -> Column[_ut.Pressure]: ...


@overload
def unit(value: _ut.MassFlowUnits, *, name: str | None = None) -> Column[_ut.MassFlow]: ...


@overload
def unit(value: _ut.VolumeFlowUnits, *, name: str | None = None) -> Column[_ut.VolumeFlow]: ...


@overload
def unit(value: _ut.NormalVolumeFlowUnits, *, name: str | None = None) -> Column[_ut.NormalVolumeFlow]: ...


@overload
def unit(value: _ut.DensityUnits, *, name: str | None = None) -> Column[_ut.Density]: ...


@overload
def unit(value: _ut.MolarDensityUnits, *, name: str | None = None) -> Column[_ut.MolarDensity]: ...


@overload
def unit(value: _ut.SpecificVolumeUnits, *, name: str | None = None) -> Column[_ut.SpecificVolume]: ...


@overload
def unit(value: _ut.NormalVolumePerMassUnits, *, name: str | None = None) -> Column[_ut.NormalVolumePerMass]: ...


@overload
def unit(value: _ut.MassPerNormalVolumeUnits, *, name: str | None = None) -> Column[_ut.MassPerNormalVolume]: ...


@overload
def unit(value: _ut.EnergyUnits, *, name: str | None = None) -> Column[_ut.Energy]: ...


@overload
def unit(value: _ut.PowerUnits, *, name: str | None = None) -> Column[_ut.Power]: ...


@overload
def unit(value: _ut.VelocityUnits, *, name: str | None = None) -> Column[_ut.Velocity]: ...


@overload
def unit(value: _ut.ForceUnits, *, name: str | None = None) -> Column[_ut.Force]: ...


@overload
def unit(value: _ut.DynamicViscosityUnits, *, name: str | None = None) -> Column[_ut.DynamicViscosity]: ...


@overload
def unit(value: _ut.KinematicViscosityUnits, *, name: str | None = None) -> Column[_ut.KinematicViscosity]: ...


@overload
def unit(value: _ut.EnergyPerMassUnits, *, name: str | None = None) -> Column[_ut.EnergyPerMass]: ...


@overload
def unit(value: _ut.MolarSpecificEnthalpyUnits, *, name: str | None = None) -> Column[_ut.MolarSpecificEnthalpy]: ...


@overload
def unit(value: _ut.SpecificHeatCapacityUnits, *, name: str | None = None) -> Column[_ut.SpecificHeatCapacity]: ...


@overload
def unit(value: _ut.ThermalConductivityUnits, *, name: str | None = None) -> Column[_ut.ThermalConductivity]: ...


@overload
def unit(value: _ut.PowerPerAreaUnits, *, name: str | None = None) -> Column[_ut.PowerPerArea]: ...


@overload
def unit(
    value: _ut.HeatTransferCoefficientUnits, *, name: str | None = None
) -> Column[_ut.HeatTransferCoefficient]: ...


@overload
def unit[DT: Dimensionality](value: Unit[DT], *, name: str | None = None) -> Column[DT]: ...


@overload
def unit[DT: Dimensionality](value: str | Unit[Any], *, name: str | None = None, asdim: type[DT]) -> Column[DT]: ...


@overload
def unit(value: str | Unit[Any], *, name: str | None = None, asdim: None = None) -> Column[UnknownDimensionality]: ...


def unit(
    value: str | Unit[Any],
    *,
    name: str | None = None,
    asdim: type[Dimensionality] | None = None,
) -> Column[Any]:
    """Declare a quantity column, inferring dimensionality from its unit.

    Registered literal spellings are also inferred by static type checkers. Any
    other valid Pint unit is inferred at runtime and is statically
    ``UnknownDimensionality`` unless ``asdim=`` is supplied. The override is
    validated; it never performs an unchecked cast.
    """
    probe = Quantity(1.0, value)
    dimensionality = probe.dt if asdim is None else asdim
    return Column(probe.u, dimensionality, name=name)


class QuantityFrame:
    """Validated, lazy Polars frame with typed quantity-column descriptors."""

    _unit_columns: ClassVar[dict[str, Column[Any]]] = {}
    lf: pl.LazyFrame

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        declarations: dict[str, Column[Any]] = {}
        for base in reversed(cls.__mro__[1:]):
            declarations.update(cast("dict[str, Column[Any]]", getattr(base, "_unit_columns", {})))
        declarations.update(
            {
                attribute: cast("Column[Any]", value)  # pyrefly: ignore[redundant-cast]
                for attribute, value in cls.__dict__.items()
                if isinstance(value, Column)
            }
        )

        physical_names: dict[str, str] = {}
        for attribute, declaration in declarations.items():
            if declaration.name in physical_names:
                other = physical_names[declaration.name]
                raise TypeError(
                    f"{cls.__name__} declares physical column {declaration.name!r} twice: {other!r} and {attribute!r}"
                )
            physical_names[declaration.name] = attribute
        cls._unit_columns = declarations

    def __init__(self, frame: pl.DataFrame | pl.LazyFrame) -> None:
        lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
        schema = lf.collect_schema()
        conversions: list[pl.Expr] = []

        for declaration in self._unit_columns.values():
            if declaration.name not in schema:
                raise ValueError(f"missing declared quantity column {declaration.name!r}")
            dtype = schema[declaration.name]
            if not isinstance(dtype, UnitDType):
                raise TypeError(
                    f"column {declaration.name!r} does not carry an {EXTENSION_NAME!r} unit dtype; "
                    f"use {type(self).__name__}.from_untyped(...) only when assigning the declared units is intentional"
                )
            if dtype.unit == declaration.unit:
                continue

            source = Quantity(pl.col(declaration.name).ext.storage(), dtype.unit).asdim(declaration.dimensionality)
            converted = source.to(declaration.unit).m.alias(declaration.name)
            storage = lf.select(converted).collect_schema()[declaration.name]
            conversions.append(converted.ext.to(UnitDType(declaration.unit, storage=storage)).alias(declaration.name))

        self.lf = lf.with_columns(conversions)

    @classmethod
    def from_untyped(cls, frame: pl.DataFrame | pl.LazyFrame) -> Self:
        """Assign the declared units to bare numeric columns, then validate."""
        schema = frame.collect_schema()
        declared_units: dict[str, Unit[Any]] = {}
        for declaration in cls._unit_columns.values():
            if declaration.name not in schema:
                raise ValueError(f"missing declared quantity column {declaration.name!r}")
            if isinstance(schema[declaration.name], UnitDType):
                raise TypeError(
                    f"column {declaration.name!r} is already unit-typed; construct {cls.__name__}(frame) to validate it"
                )
            declared_units[declaration.name] = declaration.unit
        return cls(with_units(frame, declared_units))

    @classmethod
    def derive(
        cls,
        frame: QuantityFrame | pl.DataFrame | pl.LazyFrame,
        *assignments: Assignment[Any],
    ) -> Self:
        """Derive declared quantity columns from typed expression assignments."""
        lf = (
            frame.lf
            if isinstance(frame, QuantityFrame)
            else (frame.lazy() if isinstance(frame, pl.DataFrame) else frame)
        )
        declared = tuple(cls._unit_columns.values())
        seen: set[str] = set()
        expressions: list[pl.Expr] = []

        for assignment in assignments:
            target = assignment.column
            if not any(target is candidate for candidate in declared):
                raise ValueError(f"column {target.name!r} is not declared by {cls.__name__}")
            if target.name in seen:
                raise ValueError(f"column {target.name!r} was assigned more than once")
            seen.add(target.name)

            converted = assignment.value.to(target.unit).m.alias(target.name)
            storage = lf.select(converted).collect_schema()[target.name]
            expressions.append(converted.ext.to(UnitDType(target.unit, storage=storage)).alias(target.name))

        return cls(lf.with_columns(expressions))


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
                "polars context with QuantityFrame.derive(...) instead"
            )
        if isinstance(magnitude, pl.Series):
            series = magnitude.alias(name)
        else:
            series = pl.Series(name, np.atleast_1d(magnitude))
        columns.append(series.ext.to(UnitDType(quantity.u, storage=series.dtype)))
    return pl.DataFrame(columns)
