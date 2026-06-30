"""Tests for the encomp.coolprop Polars plugin API.

These exercise ``cp.fluid`` / ``cp.humid_air`` directly on Polars expressions,
WITHOUT going through encomp's ``Fluid`` class. Reference values come from
CoolProp's PropsSI / HAPropsSI / AbstractState.
"""

from __future__ import annotations

from typing import Any

import CoolProp.CoolProp as _CP
import numpy as np
import polars as pl
import pytest

from encomp import coolprop as cp

CP: Any = _CP

RTOL = 1e-5


def test_self_check_version_and_lib_path() -> None:
    assert cp.self_check()
    assert cp.lib_version() == "8.0.0"
    assert cp.lib_path().endswith((".dylib", ".so", ".dll"))


def test_fluid_defaults_if97_water() -> None:
    df = pl.DataFrame({"P": [50e5, 101325.0], "T": [400.0, 300.0]})
    out = df.select(rho=cp.fluid("DMASS", "P", "T"))  # defaults: IF97, Water
    ref = CP.PropsSI("DMASS", "P", df["P"].to_numpy(), "T", df["T"].to_numpy(), "IF97::Water")
    assert np.allclose(out["rho"].to_numpy(), ref, rtol=RTOL)


def test_fluid_heos_multiple_properties_one_select() -> None:
    # independent properties in a single select run in parallel (GIL-free)
    df = pl.DataFrame({"P": np.full(6, 40e5), "T": np.linspace(300.0, 500.0, 6)})
    out = df.select(
        d=cp.fluid("DMASS", "P", "T", name="HEOS::CarbonDioxide"),
        h=cp.fluid("HMASS", "P", "T", name="HEOS::CarbonDioxide"),
        s=cp.fluid("SMASS", "P", "T", name="HEOS::CarbonDioxide"),
    )
    for prop, col in [("DMASS", "d"), ("HMASS", "h"), ("SMASS", "s")]:
        ref = CP.PropsSI(prop, "P", df["P"].to_numpy(), "T", df["T"].to_numpy(), "HEOS::CarbonDioxide")
        assert np.allclose(out[col].to_numpy(), ref, rtol=RTOL), prop


def test_fluid_non_pt_input_pair() -> None:
    # any input pair: density from (P, H), with name2 naming the second column
    p = np.full(5, 50e5)
    t = np.linspace(320.0, 500.0, 5)
    h = CP.PropsSI("HMASS", "P", p, "T", t, "HEOS::Water")
    df = pl.DataFrame({"P": p, "H": h})
    out = df.select(d=cp.fluid("DMASS", "P", "H", name="HEOS::Water"))
    ref = CP.PropsSI("DMASS", "P", p, "HMASS", h, "HEOS::Water")
    assert np.allclose(out["d"].to_numpy(), ref, rtol=RTOL)


def test_fluid_mixture_composition() -> None:
    # composition is a {species: mole fraction} dict, like encomp.fluids
    df = pl.DataFrame({"P": np.full(4, 60e5), "T": np.linspace(320.0, 460.0, 4)})
    out = df.select(
        d=cp.fluid(
            "DMASS", "P", "T", name="HEOS::CarbonDioxide&Oxygen", composition={"CarbonDioxide": 0.7, "Oxygen": 0.3}
        )
    )
    state = CP.AbstractState("HEOS", "CarbonDioxide&Oxygen")
    state.set_mole_fractions([0.7, 0.3])
    ref: list[float] = []
    for pressure, temperature in zip(df["P"].to_list(), df["T"].to_list(), strict=True):
        state.update(CP.PT_INPUTS, pressure, temperature)
        ref.append(float(state.rhomass()))
    assert np.allclose(out["d"].to_numpy(), ref, rtol=RTOL)

    # fractions can also be folded into the name; a non-unit composition is renormalised
    # by default. All three spellings agree.
    by_name = df.select(d=cp.fluid("DMASS", "P", "T", name="HEOS::CarbonDioxide[0.7]&Oxygen[0.3]"))
    assert np.allclose(by_name["d"].to_numpy(), ref, rtol=RTOL)
    normed = df.select(d=cp.fluid("DMASS", "P", "T", name="HEOS", composition={"CarbonDioxide": 7.0, "Oxygen": 3.0}))
    assert np.allclose(normed["d"].to_numpy(), ref, rtol=RTOL)


def test_fluid_assume_phase() -> None:
    # assume_phase takes the encomp.fluids phase names ("gas"/"liquid"/...), pinning the
    # phase (here a HEOS subcritical liquid) so it matches an explicit specify_phase
    df = pl.DataFrame({"P": np.full(3, 5e5), "T": np.linspace(290.0, 320.0, 3)})
    out = df.select(d=cp.fluid("DMASS", "P", "T", name="HEOS::Water", assume_phase="liquid"))
    state = CP.AbstractState("HEOS", "Water")
    state.specify_phase(CP.iphase_liquid)
    ref: list[float] = []
    for pressure, temperature in zip(df["P"].to_list(), df["T"].to_list(), strict=True):
        state.update(CP.PT_INPUTS, pressure, temperature)
        ref.append(float(state.rhomass()))
    assert np.allclose(out["d"].to_numpy(), ref, rtol=RTOL)


def test_invalid_inputs_become_null() -> None:
    # the plugin emits null (encomp's single missing-value sentinel, never NaN) for
    # invalid inputs, so the Fluid wrapper needs no fill_nan(None); T = -5 K is invalid
    df = pl.DataFrame({"P": [50e5, 50e5], "T": [400.0, -5.0]})
    rho = df.select(rho=cp.fluid("DMASS", "P", "T"))["rho"]
    assert rho[0] is not None and np.isfinite(rho[0])
    assert rho[1] is None  # null, not NaN
    assert rho.null_count() == 1


def test_null_inputs_become_null() -> None:
    # NULL input cells (not just out-of-range values) yield NULL outputs, matching the
    # eager numpy/NaN path -- the lazy plugin must not hard-error on the first null
    # (nulls are ubiquitous in real frames: joins, sensor dropouts)
    df = pl.DataFrame({"P": [50e5, None, 60e5], "T": [400.0, 450.0, None]})
    rho = df.select(rho=cp.fluid("DMASS", "P", "T"))["rho"]
    assert rho[0] is not None and np.isfinite(rho[0])
    assert rho[1] is None and rho[2] is None  # null P, then null T -> null
    # humid air path handles nulls the same way
    df2 = pl.DataFrame({"P": [101325.0, None], "T": [293.15, 300.0], "R": [0.5, 0.5]})
    w = df2.select(w=cp.humid_air("W", "P", "T", "R"))["w"]
    assert w[0] is not None and w[1] is None


def test_humid_air() -> None:
    df = pl.DataFrame({"P": np.full(3, 101325.0), "T": [293.15, 303.15, 313.15], "R": [0.5, 0.4, 0.3]})
    out = df.select(w=cp.humid_air("W", "P", "T", "R"))
    ref = [
        CP.HAPropsSI("W", "P", pressure, "T", temperature, "R", humidity)
        for pressure, temperature, humidity in zip(df["P"].to_list(), df["T"].to_list(), df["R"].to_list(), strict=True)
    ]
    assert np.allclose(out["w"].to_numpy(), ref, rtol=RTOL)


def test_eager_and_lazy_agree() -> None:
    df = pl.DataFrame({"P": np.full(100, 50e5), "T": np.linspace(300.0, 500.0, 100)})
    expr = {"d": cp.fluid("DMASS", "P", "T"), "h": cp.fluid("HMASS", "P", "T")}
    eager = df.select(**expr)
    lazy = df.lazy().select(**expr).collect()
    assert eager.equals(lazy)


def test_inputs_named_by_property() -> None:
    # the property of each input is its name: a string, or an expression's output
    # name. Differently-named columns must be aliased to the CoolProp input name.
    df = pl.DataFrame({"pressure": [50e5], "temp": [400.0]})
    out = df.select(rho=cp.fluid("DMASS", pl.col("pressure").alias("P"), pl.col("temp").alias("T")))
    ref = CP.PropsSI("DMASS", "P", 50e5, "T", 400.0, "IF97::Water")
    assert np.isclose(out["rho"][0], ref, rtol=RTOL)


def test_input_not_named_after_state_input_raises() -> None:
    # an input whose name is not a CoolProp state input is rejected at build time
    with pytest.raises(ValueError, match="state input"):
        cp.fluid("DMASS", pl.col("pressure"), "T")
    with pytest.raises(ValueError, match="state input"):
        cp.humid_air("W", pl.col("rel_hum"), "T", "P")


def test_fluid_composition_validation_matches_fluids() -> None:
    # bare cp.fluid reconciles name + composition the same way encomp.fluids.Fluid does
    with pytest.raises(ValueError, match="do not match"):  # name species != dict keys
        cp.fluid("DMASS", "P", "T", name="HEOS::CarbonDioxide&Oxygen", composition={"CH4": 0.7, "N2": 0.3})
    with pytest.raises(ValueError, match="both"):  # fractions in name AND dict
        cp.fluid("DMASS", "P", "T", name="HEOS::CO2[0.5]&O2[0.5]", composition={"CO2": 0.5, "O2": 0.5})
    with pytest.raises(ValueError, match="IF97"):  # IF97 is not a mixture backend
        cp.fluid("DMASS", "P", "T", name="IF97::Water", composition={"Water": 0.5, "Ethanol": 0.5})
    with pytest.raises(ValueError, match="two species"):  # a composition is a mixture
        cp.fluid("DMASS", "P", "T", name="HEOS", composition={"Water": 1.0})
    with pytest.raises(ValueError, match="non-negative"):  # invalid fraction
        cp.fluid("DMASS", "P", "T", name="HEOS", composition={"CO2": -0.7, "O2": 0.3})


def test_humid_air_invalid_output_raises() -> None:
    # HAPropsSI returns _HUGE -> null for an unknown output, so a typo would otherwise
    # yield a silent all-null column; humid_air validates the output up front
    with pytest.raises(ValueError, match="HAPropsSI"):
        cp.humid_air("NOTAPROP", "P", "T", "R")


def test_typeguards() -> None:
    assert cp.is_fluid_param("DMASS")
    assert not cp.is_fluid_param("DEFINITELY_NOT_A_PROP")
    assert cp.is_humid_air_param("W")
    assert not cp.is_humid_air_param("DMASS_is_not_humid")
    assert cp.is_backend("HEOS")
    assert not cp.is_backend("NOPE")
    assert cp.is_phase("phase_liquid")
    assert not cp.is_phase("phase_nonsense")
    assert cp.is_assumed_phase("gas")
    assert not cp.is_assumed_phase("phase_gas")  # the CoolProp string is not an AssumedPhase


def test_plugin_output_dtype_preserves_input_precision() -> None:
    # the plugin's output dtype follows the input column precision: Float32 only when
    # every non-scalar input is Float32, else Float64 (the supertype for mixed). CoolProp
    # still computes in f64, so the value is correct, just cast to the result dtype.
    f32: pl.DataType = pl.Float32()
    f64: pl.DataType = pl.Float64()

    def fluid_dtype(p_dt: pl.DataType, t_dt: pl.DataType) -> pl.DataType:
        df = pl.DataFrame({"P": pl.Series([50e5], dtype=p_dt), "T": pl.Series([400.0], dtype=t_dt)})
        return df.select(cp.fluid("DMASS", "P", "T").alias("d"))["d"].dtype

    assert fluid_dtype(f32, f32) == f32
    assert fluid_dtype(f64, f64) == f64
    assert fluid_dtype(f32, f64) == f64  # mixed -> highest
    assert fluid_dtype(f64, f32) == f64

    # a length-1 literal is neutral: a Float64 scalar alongside a Float32 column -> Float32
    df = pl.DataFrame({"T": pl.Series([400.0], dtype=pl.Float32)})
    out = df.select(cp.fluid("DMASS", pl.lit(50e5).alias("P"), "T").alias("d"))["d"]
    assert out.dtype == pl.Float32
    assert out[0] == pytest.approx(939.90625, rel=RTOL)  # value still correct

    # humid air preserves precision the same way
    dfa = pl.DataFrame(
        {
            "P": pl.Series([101325.0], dtype=pl.Float32),
            "T": pl.Series([300.0], dtype=pl.Float32),
            "R": pl.Series([0.5], dtype=pl.Float32),
        }
    )
    assert dfa.select(cp.humid_air("W", "P", "T", "R").alias("w"))["w"].dtype == pl.Float32
    dfa64 = dfa.with_columns(pl.all().cast(pl.Float64))
    assert dfa64.select(cp.humid_air("W", "P", "T", "R").alias("w"))["w"].dtype == pl.Float64
