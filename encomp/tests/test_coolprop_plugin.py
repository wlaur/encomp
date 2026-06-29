"""Tests for the standalone encomp_coolprop Polars plugin API.

These exercise ``cp.fluid`` / ``cp.humid_air`` directly on Polars expressions,
WITHOUT going through encomp's ``Fluid`` class -- the package is meant to be usable
on its own. Reference values come from CoolProp's PropsSI / HAPropsSI / AbstractState.
"""

from __future__ import annotations

from typing import Any

import CoolProp.CoolProp as _CP
import encomp_coolprop as cp
import numpy as np
import polars as pl

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
        d=cp.fluid("DMASS", "P", "T", backend="HEOS", fluid="CarbonDioxide"),
        h=cp.fluid("HMASS", "P", "T", backend="HEOS", fluid="CarbonDioxide"),
        s=cp.fluid("SMASS", "P", "T", backend="HEOS", fluid="CarbonDioxide"),
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
    out = df.select(d=cp.fluid("DMASS", "P", "H", name2="HMASS", backend="HEOS", fluid="Water"))
    ref = CP.PropsSI("DMASS", "P", p, "HMASS", h, "HEOS::Water")
    assert np.allclose(out["d"].to_numpy(), ref, rtol=RTOL)


def test_fluid_mixture_mole_fractions() -> None:
    df = pl.DataFrame({"P": np.full(4, 60e5), "T": np.linspace(320.0, 460.0, 4)})
    out = df.select(
        d=cp.fluid("DMASS", "P", "T", backend="HEOS", fluid="CarbonDioxide&Oxygen", mole_fractions=[0.7, 0.3])
    )
    state = CP.AbstractState("HEOS", "CarbonDioxide&Oxygen")
    state.set_mole_fractions([0.7, 0.3])
    ref: list[float] = []
    for pressure, temperature in zip(df["P"].to_list(), df["T"].to_list(), strict=True):
        state.update(CP.PT_INPUTS, pressure, temperature)
        ref.append(float(state.rhomass()))
    assert np.allclose(out["d"].to_numpy(), ref, rtol=RTOL)


def test_invalid_inputs_become_nan() -> None:
    # the standalone plugin returns NaN for invalid inputs (encomp's Fluid wrapper is
    # what converts NaN -> null); T = -5 K is invalid
    df = pl.DataFrame({"P": [50e5, 50e5], "T": [400.0, -5.0]})
    vals = df.select(rho=cp.fluid("DMASS", "P", "T"))["rho"].to_numpy()
    assert np.isfinite(vals[0])
    assert np.isnan(vals[1])


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


def test_typeguards() -> None:
    assert cp.is_fluid_param("DMASS")
    assert not cp.is_fluid_param("DEFINITELY_NOT_A_PROP")
    assert cp.is_humid_air_param("W")
    assert not cp.is_humid_air_param("DMASS_is_not_humid")
    assert cp.is_backend("HEOS")
    assert not cp.is_backend("NOPE")
    assert cp.is_phase("phase_liquid")
    assert not cp.is_phase("phase_nonsense")
