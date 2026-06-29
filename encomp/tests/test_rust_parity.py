"""Comprehensive parity between the rust and python CoolProp backends.

Both backends evaluate CoolProp 8.0 and cast to Float32, so for the same fluid /
backend / property / inputs the results should match to float precision. We
compare across many fluids (IF97 water, a broad set of HEOS pure fluids, and a
mixture), several properties, and (P,T) grids spanning liquid / gas / supercritical
(reduced coordinates from each fluid's critical point), plus a few non-PT input
pairs. The finite overlap is compared with a relative tolerance.
"""

from __future__ import annotations

import CoolProp.CoolProp as CP
import numpy as np
import polars as pl
import pytest

from encomp.fluids import Fluid, clear_expr_evaluation_cache
from encomp.units import Quantity as Q

# both backends produce Float32; this comfortably allows any float-precision diff
RTOL = 1e-4

PURE = [
    "Water", "CarbonDioxide", "Nitrogen", "Oxygen", "Methane", "Ammonia",
    "R134a", "n-Propane", "Hydrogen", "Argon", "Ethane", "CarbonMonoxide",
]  # fmt: skip
PROPS = ["D", "H", "S", "U", "C"]

_CONFIGS: list[tuple[str, str, dict]] = (
    [("IF97::Water", "IF97::Water", {})]
    + [(f"HEOS::{f}", f"HEOS::{f}", {}) for f in PURE]
    + [("HEOS CO2&O2 (0.7/0.3)", "HEOS", {"composition": {"CarbonDioxide": 0.7, "Oxygen": 0.3}})]
)


def _grid(name: str, kwargs: dict) -> pl.DataFrame:
    base = name.split("::")[-1].split("&")[0] if "::" in name else next(iter(kwargs["composition"]))
    try:
        tc = CP.PropsSI("Tcrit", base)
        pc = CP.PropsSI("Pcrit", base)
    except Exception:
        tc, pc = 600.0, 5e6
    tr = np.array([0.7, 0.85, 1.0, 1.2, 1.6, 2.5])
    pr = np.array([0.4, 1.0, 2.0, 5.0])
    tt, pp = np.meshgrid(tr * tc, pr * pc)
    return pl.DataFrame({"P": pp.ravel(), "T": tt.ravel()})


def _evaluate(
    name: str, kwargs: dict, prop: str, backend: str, df: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> np.ndarray:
    monkeypatch.setattr("encomp.fluids.SETTINGS.coolprop_backend", backend)
    clear_expr_evaluation_cache()
    fluid = Fluid(name, P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"), **kwargs)
    expr = getattr(fluid, prop).m.alias("x")
    return df.select(expr)["x"].to_numpy().astype(float)


@pytest.mark.parametrize(("label", "name", "kwargs"), _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_rust_python_parity(label: str, name: str, kwargs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    encomp_coolprop = pytest.importorskip("encomp_coolprop")
    if not encomp_coolprop.self_check():
        pytest.skip("encomp_coolprop plugin unavailable")

    df = _grid(name, kwargs)
    for prop in PROPS:
        py = _evaluate(name, kwargs, prop, "python", df, monkeypatch)
        ru = _evaluate(name, kwargs, prop, "rust", df, monkeypatch)
        assert py.shape == ru.shape

        both = np.isfinite(py) & np.isfinite(ru)
        assert both.sum() >= 8, f"{label}.{prop}: only {int(both.sum())} finite-overlap points"
        rel = np.abs(ru[both] - py[both]) / np.maximum(np.abs(py[both]), 1e-9)
        worst = int(np.argmax(rel))
        assert rel.max() <= RTOL, (
            f"{label}.{prop}: max rel {rel.max():.2e} (rust {ru[both][worst]:.6g} vs python {py[both][worst]:.6g})"
        )


def test_rust_python_parity_input_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parity for non-PT input pairs (P,H and P,S) -- the rust path resolves the
    canonical pair + column order, the python path uses PropsSI."""
    encomp_coolprop = pytest.importorskip("encomp_coolprop")
    if not encomp_coolprop.self_check():
        pytest.skip("encomp_coolprop plugin unavailable")

    p = np.geomspace(2e5, 200e5, 25)
    t = np.linspace(300.0, 620.0, 25)
    h = CP.PropsSI("HMASS", "P", p, "T", t, "HEOS::Water")
    s = CP.PropsSI("SMASS", "P", p, "T", t, "HEOS::Water")

    for second, unit, vals in [("HMASS", "J/kg", h), ("SMASS", "J/kg/K", s)]:
        df = pl.DataFrame({"P": p, second: vals})

        def density(backend: str, second: str = second, unit: str = unit, df: pl.DataFrame = df) -> np.ndarray:
            monkeypatch.setattr("encomp.fluids.SETTINGS.coolprop_backend", backend)
            clear_expr_evaluation_cache()
            fluid = Fluid("HEOS::Water", P=Q(pl.col("P"), "Pa"), **{second: Q(pl.col(second), unit)})
            return df.select(fluid.D.m.alias("d"))["d"].to_numpy().astype(float)

        py = density("python")
        ru = density("rust")
        both = np.isfinite(py) & np.isfinite(ru)
        assert both.sum() >= 15, f"P,{second}: only {int(both.sum())} finite points"
        rel = np.abs(ru[both] - py[both]) / np.maximum(np.abs(py[both]), 1e-9)
        assert rel.max() <= RTOL, f"P,{second}: max rel {rel.max():.2e}"
