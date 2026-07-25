"""Tests for the unit-carrying polars extension dtype (encomp.polars).

The polars extension-type API is marked unstable upstream; these tests deliberately pin
the behavior encomp relies on (I/O round-trip, arithmetic refusal, concat refusal) so a
polars bump that changes any of it fails loudly.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, assert_type, cast

import polars as pl
import pytest
from pint.errors import UndefinedUnitError
from polars.exceptions import InvalidOperationError, SchemaError
from pytest import raises

from ..fluids import Water
from ..polars import (
    EXTENSION_NAME,
    Column,
    QuantityFrame,
    UnitDType,
    unit,
    units_of,
    with_units,
)
from ..units import DimensionalityTypeError, Unit
from ..units import Quantity as Q
from ..utypes import (
    Density,
    Power,
    Pressure,
    Temperature,
    TemperatureDifference,
    UnknownDimensionality,
    Velocity,
    VolumeFlow,
)


def _sensor_df() -> pl.DataFrame:
    df = pl.DataFrame({"P": [1.0, 2.0, 3.0], "V": [10.0, 20.0, 30.0], "tag": ["a", "b", "c"]})
    return with_units(df, {"P": "bar", "V": "m^3/h"})


class Sensors(QuantityFrame):
    pressure = unit("bar")
    flow = unit("m³/h", name="Volume flow")


class FallbackUnits(QuantityFrame):
    speed = unit("furlong/fortnight")
    pressure = unit("kg/(m*s**2)", asdim=Pressure)


class Report(QuantityFrame):
    power = unit("kW", name="Hydraulic power")


class OtherReport(QuantityFrame):
    power = unit("kW")


class SensorsReport(Sensors, Report):
    pass


class FluidInputs(QuantityFrame):
    pressure = unit("bar", name="P")
    temperature = unit("degC", name="T")


class FluidOutputs(QuantityFrame):
    density = unit("kg/m³", name="rho")


assert_type(Sensors.pressure, Column[Pressure])
assert_type(Sensors.flow, Column[VolumeFlow])
assert_type(FallbackUnits.speed, Column[UnknownDimensionality])
assert_type(FallbackUnits.pressure, Column[Pressure])


def test_quantity_frame_assigns_declared_units_to_untyped_input() -> None:
    sensors = Sensors.from_untyped(
        pl.DataFrame({"pressure": [1.0, 2.0], "Volume flow": [10.0, 20.0], "tag": ["a", "b"]})
    )

    assert_type(sensors.pressure, Q[Pressure, pl.Expr])
    assert_type(sensors.flow, Q[VolumeFlow, pl.Expr])
    assert units_of(sensors.lf) == {"pressure": Unit("bar"), "Volume flow": Unit("m³/h")}
    assert sensors.lf.select(sensors.pressure.m).collect().to_series().to_list() == [1.0, 2.0]


def test_quantity_frame_validates_and_normalizes_stored_units() -> None:
    stored = with_units(
        pl.DataFrame({"pressure": [100.0, 200.0], "Volume flow": [10.0, 20.0]}),
        {"pressure": "kPa", "Volume flow": "m³/h"},
    )
    sensors = Sensors(stored)

    assert units_of(sensors.lf) == {"pressure": Unit("bar"), "Volume flow": Unit("m³/h")}
    assert sensors.lf.select(sensors.pressure.m).collect().to_series().to_list() == [1.0, 2.0]

    with raises(Exception, match=r"Cannot convert"):
        Sensors(
            with_units(
                pl.DataFrame({"pressure": [1.0], "Volume flow": [1.0]}),
                {"pressure": "m", "Volume flow": "m³/h"},
            )
        )


def test_quantity_frame_refuses_temperature_reinterpretation() -> None:
    class Absolute(QuantityFrame):
        value = unit("degC")

    class KelvinAbsolute(QuantityFrame):
        value = unit("K")

    class RankineAbsolute(QuantityFrame):
        value = unit("degR")

    class Difference(QuantityFrame):
        value = unit("delta_degC")

    class ExplicitAbsolute(QuantityFrame):
        value = unit("K", asdim=Temperature)

    class KelvinDifference(QuantityFrame):
        value = unit("K", asdim=TemperatureDifference)

    absolute = with_units(pl.DataFrame({"value": [25.0]}), {"value": "degC"})
    difference = with_units(pl.DataFrame({"value": [5.0]}), {"value": "delta_degC"})
    ambiguous_kelvin = with_units(pl.DataFrame({"value": [5.0]}), {"value": "K"})

    with raises(DimensionalityTypeError, match=r"[Tt]emperature"):
        Difference(absolute)
    with raises(DimensionalityTypeError, match=r"[Tt]emperature"):
        Absolute(difference)
    with raises(DimensionalityTypeError, match=r"TemperatureDifference.*Temperature"):
        KelvinAbsolute(difference)
    with raises(DimensionalityTypeError, match=r"TemperatureDifference.*Temperature"):
        RankineAbsolute(difference)

    kelvin = KelvinAbsolute(absolute)
    assert kelvin.lf.select(kelvin.value.m).collect().item() == pytest.approx(298.15)

    explicitly_reinterpreted = ExplicitAbsolute(difference)
    assert explicitly_reinterpreted.lf.select(explicitly_reinterpreted.value.m).collect().item() == 5.0

    assert KelvinAbsolute(ambiguous_kelvin).value.dt is Temperature
    assert KelvinDifference(ambiguous_kelvin).value.dt is TemperatureDifference
    assert KelvinDifference(difference).lf.select(pl.col("value").ext.storage()).collect().item() == 5.0

    assert_type(Absolute.value, Column[Temperature])
    assert_type(Difference.value, Column[TemperatureDifference])


def test_declaring_a_difference_with_an_offset_unit_rewrites_the_stored_unit() -> None:
    # unit(..., asdim=...) goes through Quantity.asdim, which moves an offset unit onto the
    # matching delta scale -- so the persisted metadata can never say "absolute °C" for a
    # column declared to hold differences
    class Differences(QuantityFrame):
        value = unit("degC", asdim=TemperatureDifference)

    assert Differences.value.unit == Unit("delta_degC")
    assert Differences.value.dimensionality is TemperatureDifference

    typed = Differences.from_untyped(pl.DataFrame({"value": [5.0]}))
    assert units_of(typed.lf) == {"value": Unit("delta_degC")}
    assert typed.value.dt is TemperatureDifference
    assert typed.lf.collect()["value"].ext.storage().to_list() == [5.0]


def test_quantity_frame_boundaries_are_explicit() -> None:
    with raises(TypeError, match="from_untyped"):
        Sensors(pl.DataFrame({"pressure": [1.0], "Volume flow": [1.0]}))
    with raises(TypeError, match="already unit-typed"):
        Sensors.from_untyped(
            with_units(
                pl.DataFrame({"pressure": [1.0], "Volume flow": [1.0]}),
                {"pressure": "bar", "Volume flow": "m³/h"},
            )
        )
    with raises(ValueError, match="missing declared"):
        Sensors.from_untyped(pl.DataFrame({"pressure": [1.0]}))


def test_unit_runtime_fallback_and_validated_asdim() -> None:
    assert FallbackUnits.speed.dimensionality is Velocity
    assert FallbackUnits.pressure.dimensionality is Pressure
    with raises(Exception, match=r"[Dd]imension"):
        unit("m³/h", asdim=Pressure)
    with raises(Exception, match="not defined"):
        unit("not_a_real_unit")


def test_quantity_frame_rejects_duplicate_physical_names() -> None:
    with raises(TypeError, match=r"declares physical column.*twice"):
        _ = type(
            "DuplicateColumns",
            (QuantityFrame,),
            {"first": unit("bar", name="P"), "second": unit("kPa", name="P")},
        )


def test_unit_dtype_normalization() -> None:
    # different spellings of the same unit give equal dtypes: the canonical registry
    # rendering is what lands in the file metadata
    assert UnitDType("m^3") == UnitDType("m³")
    assert UnitDType("m**3/hour") == UnitDType("m³/h")
    assert UnitDType("bar").unit == Unit("bar")
    # Canonicalization normalizes syntax, not physical equivalence. This conservative
    # identity is what lets the Rust plugin compare metadata without embedding Pint.
    assert UnitDType("Pa") != UnitDType("N/m²")
    assert UnitDType("Nm3/h") == UnitDType("Nm³/h")
    assert UnitDType("-").unit == Unit("")

    # an unknown unit fails at dtype construction, not at attach or write time
    with raises(Exception, match="asdfgh"):
        UnitDType("asdfgh")
    with raises(UndefinedUnitError):
        UnitDType("(")


def test_unit_dtype_metadata_ignores_display_format() -> None:
    # the dtype metadata is an on-disk, cross-process contract: it must not follow the
    # process-wide display format, which e.g. renders LaTeX under "siunitx"
    from ..units import set_quantity_format

    try:
        set_quantity_format("siunitx")
        assert UnitDType("m^3/h").ext_metadata() == "m³/h"
        assert UnitDType("degC").ext_metadata() == "°C"
    finally:
        set_quantity_format("compact")


def test_unit_dtype_storage_validation() -> None:
    with raises(TypeError, match="float or integer"):
        UnitDType("bar", storage=pl.Boolean())
    with raises(TypeError, match="float or integer"):
        UnitDType("bar", storage=pl.String())

    # already unit-typed storage cannot be re-wrapped
    with raises(TypeError, match="float or integer"):
        UnitDType("bar", storage=UnitDType("kPa"))

    assert UnitDType("bar", storage=pl.Float32()).ext_storage() == pl.Float32()


def test_with_units() -> None:
    df = _sensor_df()
    assert isinstance(df.schema["P"], UnitDType)
    assert units_of(df) == {"P": Unit("bar"), "V": Unit("m³/h")}

    # storage dtype is preserved, magnitudes untouched
    df32 = pl.DataFrame({"P": pl.Series([1.0, 2.0], dtype=pl.Float32)})
    out = with_units(df32, {"P": "bar"})
    dtype32 = out.schema["P"]
    assert isinstance(dtype32, UnitDType)
    assert dtype32.ext_storage() == pl.Float32()
    assert out["P"].ext.storage().to_list() == [1.0, 2.0]

    # lazy variant returns a LazyFrame and does not collect
    lf = with_units(df32.lazy(), {"P": "bar"})
    assert isinstance(lf, pl.LazyFrame)
    assert units_of(lf) == {"P": Unit("bar")}

    with raises(ValueError, match="not columns"):
        with_units(df32, {"nope": "bar"})
    with raises(TypeError, match="float or integer"):
        with_units(_sensor_df().select("tag"), {"tag": "bar"})


def test_parquet_round_trip(tmp_path: Path) -> None:
    df = _sensor_df()
    path = tmp_path / "sensors.parquet"
    df.write_parquet(path)

    # plain polars I/O reconstructs the dtype: no encomp read function involved
    back = pl.read_parquet(path)
    assert back.schema == df.schema
    assert units_of(back) == {"P": Unit("bar"), "V": Unit("m³/h")}

    # schema-only read on a lazy scan, and a full lazy round-trip through sink_parquet
    lf = pl.scan_parquet(path)
    assert units_of(lf) == {"P": Unit("bar"), "V": Unit("m³/h")}
    sunk = tmp_path / "sunk.parquet"
    lf.sink_parquet(sunk)
    assert units_of(pl.read_parquet(sunk)) == {"P": Unit("bar"), "V": Unit("m³/h")}


def test_parquet_round_trip_preserves_null_and_nan_distinctly(tmp_path: Path) -> None:
    # null (missing) and NaN (a float value) are different things in polars, and the unit
    # dtype must not collapse one into the other -- not even through an on-read conversion
    path = tmp_path / "temperatures.parquet"
    stored = with_units(pl.DataFrame({"value": [25.0, None, float("nan")]}), {"value": "degC"})
    stored.write_parquet(path)

    back = pl.read_parquet(path)["value"].ext.storage()
    assert back.to_list()[0] == 25.0
    assert back.null_count() == 1
    assert back.is_nan().to_list() == [False, None, True]

    class Kelvin(QuantityFrame):
        value = unit("K")

    converted = Kelvin.scan_parquet(path).lf.collect()["value"].ext.storage()
    assert converted.to_list()[0] == pytest.approx(298.15)
    assert converted.null_count() == 1
    assert converted.is_nan().to_list() == [False, None, True]


def test_coolprop_import_registers_before_parquet_read(tmp_path: Path) -> None:
    path = tmp_path / "bar.parquet"
    with_units(pl.DataFrame({"P": [5.0], "T": [400.0]}), {"P": "bar", "T": "K"}).write_parquet(path)
    script = f"""
import polars as pl
from encomp import coolprop as cp

df = pl.read_parquet({str(path)!r})
dtype = df.schema["P"]
print(type(dtype).__name__, dtype.ext_metadata())
try:
    df.lazy().select(cp.water("D", "P", "T")).collect_schema()
except Exception as exc:
    print("carries unit 'bar'" in str(exc))
else:
    print(False)
"""

    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    assert result.stdout.splitlines() == ["UnitDType bar", "True"]
    assert "not registered" not in result.stderr


def test_unregistered_reader_modes_are_explicit(tmp_path: Path) -> None:
    path = tmp_path / "bar.parquet"
    with_units(pl.DataFrame({"P": [5.0]}), {"P": "bar"}).write_parquet(path)
    script = f"""
import polars as pl

df = pl.read_parquet({str(path)!r})
print(df.schema["P"])
"""

    stripped = subprocess.run([sys.executable, "-Wdefault", "-c", script], check=True, capture_output=True, text=True)
    assert stripped.stdout.strip() == "Float64"
    assert "not registered" in stripped.stderr

    env = {**os.environ, "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_extension"}
    preserved_script = f"""
import polars as pl

df = pl.read_parquet({str(path)!r})
print(df.schema["P"])
from encomp import coolprop as cp
try:
    df.lazy().select(cp.water("D", "P", pl.lit(400.0).alias("T"))).collect_schema()
except Exception as exc:
    print("carries unit 'bar'" in str(exc))
else:
    print(False)
"""
    preserved = subprocess.run(
        [sys.executable, "-Wdefault", "-c", preserved_script], check=True, capture_output=True, text=True, env=env
    )
    assert preserved.stdout.splitlines() == ["Extension('encomp.unit', Float64, 'bar')", "True"]
    assert "not registered" not in preserved.stderr


def test_arrow_field_metadata(tmp_path: Path) -> None:
    # Pin the actual cross-tool contract through PyArrow, not merely by asking Polars
    # to deserialize its own file representation.
    parquet: Any = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "meta.parquet"
    _sensor_df().write_parquet(path)
    field = parquet.read_schema(path).field("P")
    assert field.metadata == {
        b"ARROW:extension:metadata": b"bar",
        b"ARROW:extension:name": EXTENSION_NAME.encode(),
    }


def test_ipc_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sensors.arrow"
    df = _sensor_df()
    df.write_ipc(path)
    assert pl.read_ipc(path).schema == df.schema
    assert units_of(pl.scan_ipc(path)) == {"P": Unit("bar"), "V": Unit("m³/h")}


def test_passthrough_operations_preserve_unit_dtype() -> None:
    df = _sensor_df()
    grouped = df.group_by("P").agg(pl.len())
    assert units_of(grouped) == {"P": Unit("bar")}

    lookup = df.select("P", code=pl.int_range(pl.len()))
    joined = df.join(lookup, on="P")
    assert units_of(joined) == {"P": Unit("bar"), "V": Unit("m³/h")}


def test_arithmetic_is_refused() -> None:
    # no supertype/kernel hooks for extension types: unitless math on a unit-typed
    # column is a loud error instead of a silent bug. Quantity (or an explicit
    # .ext.storage() unwrap) is the only way to compute.
    df = _sensor_df()

    with raises(InvalidOperationError):
        df.select(pl.col("P") + pl.col("P"))
    with raises(SchemaError, match="supertype"):
        df.select(pl.col("P") * pl.col("V"))
    with raises(SchemaError, match="supertype"):
        df.select(pl.col("P") * 2)
    with raises(InvalidOperationError):
        df.select(pl.col("P").mean())

    # comparisons and passthrough operations keep working
    assert df.filter(pl.col("P") > 1.5).height == 2
    assert df.sort("P", descending=True)["P"].ext.storage().to_list() == [3.0, 2.0, 1.0]


def test_concat_unit_mismatch_is_refused() -> None:
    bar = with_units(pl.DataFrame({"P": [1.0]}), {"P": "bar"})
    kpa = with_units(pl.DataFrame({"P": [2.0]}), {"P": "kPa"})
    with raises(SchemaError):
        pl.concat([bar, kpa])
    assert pl.concat([bar, bar]).height == 2


def test_quantity_frame_derive_is_typed_and_float32_safe() -> None:
    df = pl.DataFrame(
        {
            "pressure": pl.Series([1.0, 2.0], dtype=pl.Float32),
            "Volume flow": pl.Series([10.0, 20.0], dtype=pl.Float32),
        }
    )
    sensors = Sensors.from_untyped(df)
    power = (sensors.pressure * sensors.flow).to("kW")
    assert_type(power, Q[Power, pl.Expr])

    target = Report.power
    assignment = target.assign(power)
    report = Report.derive(sensors, assignment)
    assert_type(report, Report)
    dtype = report.lf.collect_schema()["Hydraulic power"]
    assert isinstance(dtype, UnitDType)
    assert dtype.ext_storage() == pl.Float32()
    assert dtype.unit == Unit("kW")
    assert report.lf.collect()["Hydraulic power"].ext.storage().to_list() == pytest.approx([0.2778, 1.1111], rel=1e-3)

    combined = SensorsReport(report.lf)
    assert_type(combined, SensorsReport)
    assert combined.lf.collect_schema()["Hydraulic power"] == UnitDType("kW", storage=pl.Float32())


def test_quantity_frame_derive_validates_assignments() -> None:
    sensors = Sensors.from_untyped(pl.DataFrame({"pressure": [1.0], "Volume flow": [10.0]}))
    power = (sensors.pressure * sensors.flow).to("kW")

    with raises(ValueError, match="not declared"):
        Report.derive(sensors, OtherReport.power.assign(power))
    with raises(ValueError, match="more than once"):
        assignment = Report.power.assign(power)
        Report.derive(sensors, assignment, assignment)
    with raises(TypeError, match=r"pl\.Expr"):
        Report.power.assign(cast(Any, Q(pl.Series([1.0]), "kW")))
    with raises(Exception, match="Cannot convert"):
        Report.power.assign(cast(Any, sensors.pressure))


def test_scan_parquet_survives_predicate_pushdown(tmp_path: Path) -> None:
    # polars' parquet statistics reader has no implementation for extension dtypes and
    # aborts a worker thread with a Rust panic when a pushed-down predicate makes it decode
    # them (upstream; reproduced on 1.42 and 1.43). QuantityFrame.scan_parquet inserts an
    # identity re-wrap that the optimizer will not push a predicate through
    path = tmp_path / "sensors.parquet"
    frame = pl.DataFrame({"pressure": [10.0, 60.0, 100.0], "Volume flow": [1.0, 2.0, 3.0]})
    Sensors.from_untyped(frame).lf.sink_parquet(path)

    scanned = Sensors.scan_parquet(path)
    filtered = scanned.lf.filter(pl.col("pressure").ext.storage() > 50.0).collect()

    assert filtered["pressure"].ext.storage().to_list() == [60.0, 100.0]
    assert filtered.schema["pressure"] == UnitDType("bar")

    # is_null() decodes statistics too, and a non-declared unit column is shielded as well
    assert scanned.lf.filter(pl.col("Volume flow").ext.storage().is_null()).collect().height == 0

    # pin the upstream bug itself: when it is fixed, this fails and the shield (and the
    # note in QuantityFrame.scan_parquet) can be revisited
    with raises(BaseException, match=r"not yet implemented|statistics"):
        pl.scan_parquet(path).filter(pl.col("pressure").ext.storage() > 50.0).collect()


def test_units_of_degrades_per_column_for_unparseable_metadata() -> None:
    # EXTENSION_NAME is a cross-process contract: another producer can write metadata pint
    # cannot parse. That must not make the well-formed columns of the frame uninspectable
    good = with_units(pl.DataFrame({"good": [1.0]}), {"good": "bar"})
    bad_dtype = pl.BaseExtension(EXTENSION_NAME, pl.Float64(), "totally_bogus")
    frame = good.with_columns(pl.Series("bad", [2.0]).cast(pl.Float64).ext.to(bad_dtype))

    with pytest.warns(UserWarning, match="totally_bogus"):
        assert units_of(frame) == {"good": Unit("bar")}


def test_typed_read_errors_name_the_column() -> None:
    stored = with_units(pl.DataFrame({"pressure": [300.0], "Volume flow": [1.0]}), {"pressure": "degC"})

    with raises(DimensionalityTypeError, match="column 'pressure'"):
        Sensors(stored)

    with raises(TypeError, match="float or integer") as excinfo:
        with_units(pl.DataFrame({"tag": ["a"]}), {"tag": "bar"})

    assert any("'tag'" in note for note in excinfo.value.__notes__)

    with raises(UndefinedUnitError) as unit_excinfo:
        with_units(pl.DataFrame({"P": [1.0]}), {"P": "bogus_unit"})

    assert any("'P'" in note for note in unit_excinfo.value.__notes__)


def test_quantity_frame_repr_summarizes_declarations() -> None:
    sensors = Sensors.from_untyped(pl.DataFrame({"pressure": [1.0], "Volume flow": [10.0]}))

    # a collection-free summary: the repr of a lazy frame must not evaluate it
    assert repr(sensors) == "<Sensors: pressure [bar], Volume flow [m³/h] (lazy)>"


def test_quantity_frame_round_trips_fluid_expression_output(tmp_path: Path) -> None:
    source_path = tmp_path / "states.parquet"
    result_path = tmp_path / "properties.parquet"
    FluidInputs.from_untyped(pl.DataFrame({"P": [5.0, 5.0], "T": [150.0, 250.0]})).lf.sink_parquet(source_path)

    inputs = FluidInputs.scan_parquet(source_path)
    assert_type(inputs, FluidInputs)
    water = Water[pl.Expr](P=inputs.pressure, T=inputs.temperature)
    density = water.D
    assert_type(inputs.pressure, Q[Pressure, pl.Expr])
    assert_type(inputs.temperature, Q[Temperature, pl.Expr])
    assert_type(density, Q[Density, pl.Expr])

    outputs = FluidOutputs.derive(inputs, FluidOutputs.density.assign(density))
    outputs.lf.sink_parquet(result_path)

    restored = FluidOutputs.scan_parquet(result_path)
    assert units_of(restored.lf)["rho"] == Unit("kg/m³")
    expected = [
        Water(P=Q(5.0, "bar"), T=Q(150.0, "degC")).D.m,
        Water(P=Q(5.0, "bar"), T=Q(250.0, "degC")).D.m,
    ]
    assert restored.lf.collect()["rho"].ext.storage().to_list() == pytest.approx(expected)
