"""Tests for the encomp.coolprop Polars plugin API.

These exercise ``cp.fluid`` / ``cp.humid_air`` directly on Polars expressions,
WITHOUT going through encomp's ``Fluid`` class. Reference values come from
CoolProp's PropsSI / HAPropsSI / AbstractState.
"""

from __future__ import annotations

from typing import Any, cast

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

    # fractions can also be folded into the name; both spellings agree
    by_name = df.select(d=cp.fluid("DMASS", "P", "T", name="HEOS::CarbonDioxide[0.7]&Oxygen[0.3]"))
    assert np.allclose(by_name["d"].to_numpy(), ref, rtol=RTOL)


def test_fluid_incompressible_mixture() -> None:
    # an incompressible mixture carries its concentration in the name (INCOMP::MEG[0.5]); the
    # basis is fluid-specific -- mass for glycols/brines, volume for the antifreezes -- and
    # CoolProp's set_fractions picks it, so the plugin matches PropsSI for both bases
    for name, temp in [("INCOMP::MEG[0.5]", 300.0), ("INCOMP::MITSW[0.035]", 290.0), ("INCOMP::AEG[0.4]", 300.0)]:
        df = pl.DataFrame({"P": np.full(5, 3e5), "T": np.linspace(temp - 10.0, temp + 10.0, 5)})
        out = df.select(d=cp.fluid("DMASS", "P", "T", name=name))
        ref = CP.PropsSI("DMASS", "P", df["P"].to_numpy(), "T", df["T"].to_numpy(), name)
        assert np.allclose(out["d"].to_numpy(), ref, rtol=RTOL), name


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


# Worker for the cold-start parallel warmup test below. Run in a FRESH process (via
# subprocess) so CoolProp's process-global init and every (backend, fluid)'s backend-specific
# init actually happen from cold, concurrently, under the plugin's per-config warmup.
_COLD_PARALLEL_WORKER = r"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, polars as pl
from encomp import coolprop as cp
import CoolProp.CoolProp as CP

rng = np.random.default_rng(0)
n = 300  # the warmup race is on concurrent first-use of distinct configs, not row count
P = rng.uniform(10e5, 50e5, n)
T = rng.uniform(300.0, 400.0, n)
Ta = rng.uniform(285.0, 320.0, n)
Pa = np.full(n, 101325.0)
Rh = rng.uniform(0.1, 0.9, n)
df = pl.DataFrame({"P": P, "T": T, "Pa": Pa, "Ta": Ta, "R": Rh})

# many DISTINCT (backend, fluid) first-uses + humid air, ALL in one parallel select so their
# warmups race; humid-air inputs are aliased to their HAPropsSI names (Pa->P, Ta->T)
exprs = {
    "if97": cp.fluid("DMASS", "P", "T", name="IF97::Water"),
    "heos": cp.fluid("DMASS", "P", "T", name="HEOS::Water"),
    "co2": cp.fluid("DMASS", "P", "T", name="HEOS::CarbonDioxide"),
    "n2": cp.fluid("DMASS", "P", "T", name="HEOS::Nitrogen"),
    "o2": cp.fluid("HMASS", "P", "T", name="HEOS::Oxygen"),
    "ch4": cp.fluid("HMASS", "P", "T", name="HEOS::Methane"),
    "mix": cp.fluid("DMASS", "P", "T", name="HEOS::CO2&O2", composition={"CO2": 0.7, "O2": 0.3}),
    "meg": cp.fluid("DMASS", "P", "T", name="INCOMP::MEG[0.5]"),
    "ha_w": cp.humid_air("W", pl.col("Pa").alias("P"), pl.col("Ta").alias("T"), "R"),
    "ha_h": cp.humid_air("Hha", pl.col("Pa").alias("P"), pl.col("Ta").alias("T"), "R"),
}
out = df.select(**exprs)

refs = {
    "if97": CP.PropsSI("DMASS", "P", P, "T", T, "IF97::Water"),
    "heos": CP.PropsSI("DMASS", "P", P, "T", T, "HEOS::Water"),
    "co2": CP.PropsSI("DMASS", "P", P, "T", T, "HEOS::CarbonDioxide"),
    "n2": CP.PropsSI("DMASS", "P", P, "T", T, "HEOS::Nitrogen"),
    "o2": CP.PropsSI("HMASS", "P", P, "T", T, "HEOS::Oxygen"),
    "ch4": CP.PropsSI("HMASS", "P", P, "T", T, "HEOS::Methane"),
    "meg": CP.PropsSI("DMASS", "P", P, "T", T, "INCOMP::MEG[0.5]"),
    "ha_w": CP.HAPropsSI("W", "P", Pa, "T", Ta, "R", Rh),
    "ha_h": CP.HAPropsSI("Hha", "P", Pa, "T", Ta, "R", Rh),
}
st = CP.AbstractState("HEOS", "CO2&O2")
st.set_mole_fractions([0.7, 0.3])
mix = np.empty(n)
for i in range(n):
    st.update(CP.PT_INPUTS, P[i], T[i])
    mix[i] = st.rhomass()
refs["mix"] = mix

bad = []
for k, ref in refs.items():
    got = out[k].to_numpy().astype(float)
    ref = np.asarray(ref, dtype=float)
    if not np.array_equal(np.isfinite(got), np.isfinite(ref)):
        bad.append(k + ": nan-mask differs")
        continue
    m = np.isfinite(got) & np.isfinite(ref)
    rel = np.abs(got[m] - ref[m]) / np.maximum(np.abs(ref[m]), 1e-9)
    if rel.size and rel.max() > 1e-4:
        bad.append(k + ": max rel " + format(rel.max(), ".2e"))
if bad:
    print("FAIL", bad)
    raise SystemExit(1)
print("OK")
"""


def test_cold_start_parallel_warmup() -> None:
    # from a COLD process, evaluate many distinct (backend, fluid) configs + humid air in ONE
    # parallel select so their first-use warmups happen concurrently. This exercises the
    # per-config lazy warmup: process-global init and each backend's init must run correctly
    # (once, single-threaded) under Polars' thread pool. Several fresh runs -- a warmup race
    # would be probabilistic, not deterministic.
    import subprocess
    import sys

    for i in range(2):
        r = subprocess.run([sys.executable, "-c", _COLD_PARALLEL_WORKER], capture_output=True, text=True)
        assert r.returncode == 0, f"cold-start parallel run {i} failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        assert "OK" in r.stdout


def test_fluid_assume_phase_changes_value() -> None:
    # a DISCRIMINATING state: 1 bar, 110 C -- auto-phase water is GAS (steam, ~0.57 kg/m3).
    # Forcing "liquid" must return the metastable subcooled-liquid root (~950 kg/m3), i.e. a
    # value that (a) differs from auto by orders of magnitude and (b) equals an explicit
    # specify_phase(iphase_liquid). This is what proves assume_phase is not silently a no-op.
    df = pl.DataFrame({"P": [1e5], "T": [383.15]})
    auto = df.select(d=cp.fluid("DMASS", "P", "T", name="HEOS::Water"))["d"][0]
    forced = df.select(d=cp.fluid("DMASS", "P", "T", name="HEOS::Water", assume_phase="liquid"))["d"][0]

    state = CP.AbstractState("HEOS", "Water")
    state.specify_phase(CP.iphase_liquid)
    state.update(CP.PT_INPUTS, 1e5, 383.15)
    ref_liquid = float(state.rhomass())

    assert forced == pytest.approx(ref_liquid, rel=RTOL)
    assert forced > 900.0 and auto < 10.0  # liquid vs steam: unmistakably different roots
    assert abs(forced - auto) / auto > 100.0


def test_fluid_input_pair_order_invariant() -> None:
    # the plugin canonicalizes input order via generate_update_pair, so evaluating the SAME
    # state with the two inputs swapped must give identical results (guards the swap flag)
    p = np.full(5, 50e5)
    t = np.linspace(320.0, 500.0, 5)
    h = CP.PropsSI("HMASS", "P", p, "T", t, "HEOS::Water")
    cases: list[tuple[cp.FluidInput, cp.FluidInput, dict[str, Any]]] = [
        ("P", "T", {"P": p, "T": t}),
        ("P", "H", {"P": p, "H": h}),
    ]
    for a, b, cols in cases:
        df = pl.DataFrame(cols)
        forward = df.select(d=cp.fluid("DMASS", a, b, name="HEOS::Water"))["d"].to_numpy()
        reverse = df.select(d=cp.fluid("DMASS", b, a, name="HEOS::Water"))["d"].to_numpy()
        assert np.allclose(forward, reverse, rtol=RTOL, equal_nan=True), (a, b)


def test_fluid_quality_input_pair() -> None:
    # two-phase quality (P, Q) through the plugin -- a core input pair never otherwise
    # exercised on the plugin path -- matches PropsSI, both orders
    p = np.linspace(1e5, 20e5, 5)
    q = np.full(5, 0.5)
    df = pl.DataFrame({"P": p, "Q": q})
    ref = CP.PropsSI("DMASS", "P", p, "Q", q, "HEOS::Water")
    pq = df.select(d=cp.fluid("DMASS", "P", "Q", name="HEOS::Water"))["d"].to_numpy()
    qp = df.select(d=cp.fluid("DMASS", "Q", "P", name="HEOS::Water"))["d"].to_numpy()
    assert np.allclose(pq, ref, rtol=RTOL)
    assert np.allclose(qp, ref, rtol=RTOL)


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


def test_composition_non_unit_sum_raises() -> None:
    # a composition that does not sum to 1 is an error, not a silent renormalisation
    with pytest.raises(ValueError, match="sum to 1"):
        cp.fluid("DMASS", "P", "T", name="HEOS", composition={"CO2": 0.3, "O2": 0.3})


def test_humid_air_invalid_output_raises() -> None:
    # HAPropsSI returns _HUGE -> null for an unknown output, so a typo would otherwise
    # yield a silent all-null column; humid_air validates the output up front
    with pytest.raises(ValueError, match="HAPropsSI"):
        cp.humid_air(cast(Any, "NOTAPROP"), "P", "T", "R")


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


# results must be BIT-identical regardless of the polars thread count: the flash is
# pure per-row math on per-thread states, and chunk boundaries only affect scheduling,
# never values. POLARS_MAX_THREADS is read at polars import, so each count runs in a
# fresh subprocess; the digest covers every byte of every output column (nulls -> NaN).
_THREAD_PARITY_WORKER = r"""
import hashlib
import numpy as np
import polars as pl
from encomp import coolprop as cp

rng = np.random.default_rng(1234)
n = 5_000
P = rng.uniform(1e5, 200e5, n)
T = rng.uniform(280.0, 900.0, n)
P[0] = np.nan  # null-propagation must also be thread-count-invariant
df = pl.DataFrame({"P": P, "T": T})
out = df.select(
    cp.fluid("DMASS", "P", "T", name="IF97::Water").alias("d"),
    cp.fluid("HMASS", "P", "T", name="IF97::Water").alias("h"),
    cp.fluid(
        "SMASS", "P", "T", name="HEOS::CO2&O2",
        composition={"CO2": 0.5, "O2": 0.5}, assume_phase="gas",
    ).alias("s_mix"),
    cp.fluid("VISCOSITY", "P", "T", name="HEOS::Nitrogen").alias("v"),
)
digest = hashlib.blake2s()
for name in out.columns:
    arr = out[name].to_numpy()
    digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
print(pl.thread_pool_size(), digest.hexdigest())
"""


def test_thread_count_parity() -> None:
    import os
    import subprocess
    import sys

    digests: dict[str, str] = {}
    for threads in ("1", "4", "8"):
        env = dict(os.environ, POLARS_MAX_THREADS=threads)
        r = subprocess.run(
            [sys.executable, "-c", _THREAD_PARITY_WORKER], capture_output=True, text=True, env=env, check=False
        )
        assert r.returncode == 0, f"thread-parity worker ({threads} threads) failed:\n{r.stdout}\n{r.stderr}"
        reported, digest = r.stdout.split()
        # the pin must actually take effect, otherwise this test compares nothing
        assert reported == threads, f"expected a {threads}-thread pool, got {reported}"
        digests[threads] = digest

    assert len(set(digests.values())) == 1, f"results differ across thread counts: {digests}"
