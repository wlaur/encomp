"""Comprehensive parity between the rust plugin and the Python CoolProp path.

``pl.Expr`` (lazy) inputs are evaluated by the encomp.coolprop rust plugin; eager
numpy inputs go through the Python CoolProp path. Both evaluate CoolProp 8.0 and
cast to Float32, so for the same fluid / property / inputs the results should match
to float precision. We compare across many fluids (IF97 water, a broad set of HEOS
pure fluids, and a mixture), several properties, and (P,T) grids spanning liquid /
gas / supercritical (reduced coordinates from each fluid's critical point), plus a
few non-PT input pairs. The finite overlap is compared with a relative tolerance.
"""

from __future__ import annotations

from typing import Any

import CoolProp.CoolProp as _CP
import numpy as np
import polars as pl
import pytest

from encomp import coolprop as encomp_coolprop
from encomp.fluids import Fluid, clear_expr_evaluation_cache
from encomp.units import Quantity as Q

# CoolProp.CoolProp is a compiled, untyped extension module; alias it as Any so
# its dynamic functions (PropsSI, ...) do not surface unknown-type errors.
CP: Any = _CP


# skip (not fail) when the plugin binaries simply have not been built in this
# checkout (fresh clone before `python scripts/build_libcoolprop.py` + `maturin
# develop`). Present-but-broken binaries still run the tests and fail loudly.
def _plugin_built() -> bool:
    from pathlib import Path

    here = Path(encomp_coolprop.__file__).parent
    has_lib = any(
        (here / n).exists() for n in ("libCoolProp.dylib", "libCoolProp.so", "CoolProp.dll", "libCoolProp.dll")
    )
    return has_lib and bool(list(here.glob("_internal*")))


if not _plugin_built():
    pytest.skip(
        "encomp.coolprop plugin binaries are not built (run scripts/build_libcoolprop.py + maturin develop)",
        allow_module_level=True,
    )


# both paths produce Float32; this comfortably allows any float-precision diff
RTOL = 1e-4

PURE = [
    "Water", "CarbonDioxide", "Nitrogen", "Oxygen", "Methane", "Ammonia",
    "R134a", "n-Propane", "Hydrogen", "Argon", "Ethane", "CarbonMonoxide",
]  # fmt: skip
PROPS = ["D", "H", "S", "U", "C"]

_CONFIGS: list[tuple[str, str, dict[str, Any]]] = [
    ("IF97::Water", "IF97::Water", {}),
    *((f"HEOS::{f}", f"HEOS::{f}", {}) for f in PURE),
    ("HEOS CO2&O2 (0.7/0.3)", "HEOS", {"composition": {"CarbonDioxide": 0.7, "Oxygen": 0.3}}),
]


def _grid(name: str, kwargs: dict[str, Any]) -> pl.DataFrame:
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


def _evaluate(name: str, kwargs: dict[str, Any], prop: str, mode: str, df: pl.DataFrame) -> np.ndarray:
    clear_expr_evaluation_cache()
    if mode == "rust":  # pl.Expr inputs -> rust plugin
        fluid = Fluid(name, P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"), **kwargs)
        return df.select(getattr(fluid, prop).m.alias("x"))["x"].to_numpy().astype(float)
    # python reference: eager numpy inputs -> Python CoolProp path
    fluid = Fluid(name, P=Q(df["P"].to_numpy(), "Pa"), T=Q(df["T"].to_numpy(), "K"), **kwargs)
    return np.asarray(getattr(fluid, prop).m, dtype=float)


@pytest.mark.parametrize(("label", "name", "kwargs"), _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_rust_python_parity(label: str, name: str, kwargs: dict[str, Any]) -> None:
    assert encomp_coolprop.self_check()

    df = _grid(name, kwargs)
    for prop in PROPS:
        py = _evaluate(name, kwargs, prop, "python", df)
        ru = _evaluate(name, kwargs, prop, "rust", df)
        assert py.shape == ru.shape

        # the two paths must agree on WHICH points are finite -- otherwise a path that
        # returns NaN where the other returns a value would be silently excluded below
        finite_py, finite_ru = np.isfinite(py), np.isfinite(ru)
        assert np.array_equal(finite_py, finite_ru), (
            f"{label}.{prop}: finite/NaN masks differ ({int((finite_py != finite_ru).sum())} points)"
        )
        both = finite_py & finite_ru
        assert both.sum() >= 8, f"{label}.{prop}: only {int(both.sum())} finite-overlap points"
        rel = np.abs(ru[both] - py[both]) / np.maximum(np.abs(py[both]), 1e-9)
        worst = int(np.argmax(rel))
        assert rel.max() <= RTOL, (
            f"{label}.{prop}: max rel {rel.max():.2e} (rust {ru[both][worst]:.6g} vs python {py[both][worst]:.6g})"
        )


def test_rust_python_parity_input_pairs() -> None:
    """Parity for non-PT input pairs (P,H and P,S) -- the rust path resolves the
    canonical pair + column order, the python path uses PropsSI."""
    assert encomp_coolprop.self_check()

    p = np.geomspace(2e5, 200e5, 25)
    t = np.linspace(300.0, 620.0, 25)
    h = CP.PropsSI("HMASS", "P", p, "T", t, "HEOS::Water")
    s = CP.PropsSI("SMASS", "P", p, "T", t, "HEOS::Water")

    for second, unit, vals in [("HMASS", "J/kg", h), ("SMASS", "J/kg/K", s)]:
        df = pl.DataFrame({"P": p, second: vals})

        def density(mode: str, second: str = second, unit: str = unit, df: pl.DataFrame = df) -> np.ndarray:
            clear_expr_evaluation_cache()
            second_point: dict[str, Any]
            if mode == "rust":
                second_point = {second: Q(pl.col(second), unit)}
                fluid = Fluid("HEOS::Water", P=Q(pl.col("P"), "Pa"), **second_point)
                return df.select(fluid.D.m.alias("d"))["d"].to_numpy().astype(float)
            second_point = {second: Q(df[second].to_numpy(), unit)}
            fluid = Fluid("HEOS::Water", P=Q(df["P"].to_numpy(), "Pa"), **second_point)
            return np.asarray(fluid.D.m, dtype=float)

        py = density("python")
        ru = density("rust")
        finite_py, finite_ru = np.isfinite(py), np.isfinite(ru)
        assert np.array_equal(finite_py, finite_ru), (
            f"P,{second}: finite/NaN masks differ ({int((finite_py != finite_ru).sum())} points)"
        )
        both = finite_py & finite_ru
        assert both.sum() >= 15, f"P,{second}: only {int(both.sum())} finite points"
        rel = np.abs(ru[both] - py[both]) / np.maximum(np.abs(py[both]), 1e-9)
        assert rel.max() <= RTOL, f"P,{second}: max rel {rel.max():.2e}"


def test_eager_rust_non_pt_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The large-eager numpy path (_rust_eager, n >= EAGER_PLUGIN_MIN_SIZE) with a NON-PT
    pair must match the Python PropsSI path -- the eager-plugin parity test is PT-only, so
    this covers the eager canonical-pair/column-order resolution for P,H."""
    assert encomp_coolprop.self_check()

    n = 1200  # >= EAGER_PLUGIN_MIN_SIZE (1000) -> eager rust plugin
    p = np.geomspace(2e5, 200e5, n)
    t = np.linspace(300.0, 620.0, n)
    h = CP.PropsSI("HMASS", "P", p, "T", t, "HEOS::Water")

    eager = np.asarray(Fluid("HEOS::Water", P=Q(p, "Pa"), H=Q(h, "J/kg")).D.m, dtype=float)

    monkeypatch.setattr("encomp.fluids.EAGER_PLUGIN_MIN_SIZE", 10**18)  # force Python PropsSI
    reference = np.asarray(Fluid("HEOS::Water", P=Q(p, "Pa"), H=Q(h, "J/kg")).D.m, dtype=float)

    assert np.array_equal(np.isfinite(eager), np.isfinite(reference))
    both = np.isfinite(eager) & np.isfinite(reference)
    assert both.sum() >= n // 2
    rel = np.abs(eager[both] - reference[both]) / np.maximum(np.abs(reference[both]), 1e-9)
    assert rel.max() <= RTOL, f"eager non-PT: max rel {rel.max():.2e}"
