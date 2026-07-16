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

import numpy as np
import polars as pl
import pytest
from polars.exceptions import InvalidOperationError, SchemaError
from pytest import raises

from ..polars import EXTENSION_NAME, UnitDType, attach, dataframe, quantities, quantity, units_of, with_units
from ..units import Quantity as Q
from ..units import Unit
from ..utypes import Power, Pressure, VolumeFlow


def _sensor_df() -> pl.DataFrame:
    df = pl.DataFrame({"P": [1.0, 2.0, 3.0], "V": [10.0, 20.0, 30.0], "tag": ["a", "b", "c"]})
    return with_units(df, {"P": "bar", "V": "m^3/h"})


def test_unit_dtype_normalization() -> None:
    # different spellings of the same unit give equal dtypes: the canonical registry
    # rendering is what lands in the file metadata
    assert UnitDType("m^3") == UnitDType("m³")
    assert UnitDType("m**3/hour") == UnitDType("m³/h")
    assert UnitDType("bar").unit == Unit("bar")

    # an unknown unit fails at dtype construction, not at attach or write time
    with raises(Exception, match="asdfgh"):
        UnitDType("asdfgh")


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
    assert preserved.stderr == ""


def test_arrow_field_metadata(tmp_path: Path) -> None:
    # the persisted form is the standard Arrow extension convention, under the
    # documented EXTENSION_NAME -- this is the cross-tool (pyarrow/DuckDB/Spark)
    # contract, pinned here via the parquet footer polars itself reads back
    path = tmp_path / "meta.parquet"
    _sensor_df().write_parquet(path)
    dtype = pl.read_parquet(path).schema["P"]
    assert isinstance(dtype, UnitDType)
    assert dtype.ext_name() == EXTENSION_NAME
    assert dtype.ext_metadata() == "bar"


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


def test_quantities_eager_compute() -> None:
    qs = quantities(_sensor_df())
    assert set(qs) == {"P", "V"}

    pressure = qs["P"].asdim(Pressure)
    flow = qs["V"].asdim(VolumeFlow)
    power: Q[Power, pl.Series] = (pressure * flow).to("kW")
    # 1 bar * 10 m³/h = 1e5 Pa * (10/3600) m³/s = 277.78 W
    assert power.m.to_list() == pytest.approx([0.2778, 1.1111, 2.5], rel=1e-3)

    # wrong dimensionality assertion fails at the boundary
    with raises(Exception, match=r"[Dd]imension"):
        qs["P"].asdim(VolumeFlow)


def test_quantities_lazy_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sensors.parquet"
    _sensor_df().write_parquet(path)

    lf = pl.scan_parquet(path)
    qs = quantities(lf)
    assert isinstance(qs["P"].m, pl.Expr)

    power = (qs["P"].asdim(Pressure) * qs["V"].asdim(VolumeFlow)).to("kW")
    out = lf.with_columns(power.m.ext.to(UnitDType("kW")).alias("W")).collect()
    assert units_of(out)["W"] == Unit("kW")
    assert out["W"].ext.storage().to_list() == pytest.approx([0.2778, 1.1111, 2.5], rel=1e-3)

    # unit errors surface at plan-BUILD time, before any data is collected
    with raises(Exception, match=r"[Dd]imension"):
        _ = qs["P"] + qs["V"]


def test_quantity_is_typed_compute_bridge() -> None:
    df = _sensor_df()
    pressure = quantity(df, "P", Pressure)
    assert_type(pressure, Q[Pressure, pl.Series])
    assert pressure.m.to_list() == [1.0, 2.0, 3.0]
    assert pressure.u == Unit("bar")

    lf = df.lazy()
    lazy_pressure = quantity(lf, "P", Pressure)
    assert_type(lazy_pressure, Q[Pressure, pl.Expr])
    assert isinstance(lazy_pressure.m, pl.Expr)

    with raises(Exception, match=r"[Dd]imension"):
        quantity(lf, "P", VolumeFlow)
    with raises(ValueError, match="not present"):
        quantity(lf, "missing", Pressure)
    with raises(TypeError, match="with_units"):
        quantity(pl.DataFrame({"P": [1.0]}).lazy(), "P", Pressure)


def test_attach_is_float32_safe_lazy_write_back() -> None:
    df = pl.DataFrame(
        {
            "P": pl.Series([1.0, 2.0], dtype=pl.Float32),
            "V": pl.Series([10.0, 20.0], dtype=pl.Float32),
        }
    )
    lf = with_units(df.lazy(), {"P": "bar", "V": "m³/h"})
    power = (quantity(lf, "P", Pressure) * quantity(lf, "V", VolumeFlow)).to("kW")

    out_lf = attach(lf, W=power)
    assert_type(out_lf, pl.LazyFrame)
    dtype = out_lf.collect_schema()["W"]
    assert isinstance(dtype, UnitDType)
    assert dtype.ext_storage() == pl.Float32()
    assert dtype.unit == Unit("kW")
    assert out_lf.collect()["W"].ext.storage().to_list() == pytest.approx([0.2778, 1.1111], rel=1e-3)


def test_attach_eager_and_validation() -> None:
    df = _sensor_df()
    pressure = quantity(df, "P", Pressure).to("kPa")
    out = attach(df, {"converted pressure": pressure})
    assert_type(out, pl.DataFrame)
    assert units_of(out)["converted pressure"] == Unit("kPa")
    assert out["converted pressure"].ext.storage().to_list() == [100.0, 200.0, 300.0]

    with raises(ValueError, match="provided twice"):
        attach(df, {"P2": pressure}, P2=pressure)
    with raises(TypeError, match=r"pl\.Series is eager"):
        attach(df.lazy(), P2=pressure)
    with raises(TypeError, match=r"dataframe\(\.\.\.\)"):
        attach(df, scalar=Q(1.0, "bar"))
    with raises(TypeError, match="must be Quantity"):
        attach(df, {"bad": cast(Any, "bar")})


def test_dataframe_inverse() -> None:
    df = dataframe(
        {
            "P": Q(pl.Series([1.0, 2.0]), "bar"),
            "rho": Q(np.array([997.0, 998.0]), "kg/m^3"),
        },
        to={"P": "kPa"},
    )
    assert units_of(df) == {"P": Unit("kPa"), "rho": Unit("kg/m³")}
    assert df["P"].ext.storage().to_list() == [100.0, 200.0]

    with raises(TypeError, match="deferred plan"):
        dataframe({"P": Q(pl.col("P"), "bar")})


def test_quantity_round_trip_preserves_values() -> None:
    # with_units -> quantities -> dataframe is lossless for the magnitudes
    df = _sensor_df()
    back = dataframe(quantities(df))
    assert back["P"].ext.storage().to_list() == df["P"].ext.storage().to_list()
    assert units_of(back) == units_of(df)
