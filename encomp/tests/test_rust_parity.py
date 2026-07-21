"""Comprehensive parity between encomp's native paths and Python CoolProp oracle.

``pl.Expr`` inputs are evaluated by the encomp.coolprop Rust plugin; eager arrays
use that same batch path. The optional Python package is only a numerical oracle.
Both evaluate CoolProp 8.0, so results should match closely across IF97 and HEOS
pure fluids, a mixture, several properties, and (P,T) grids spanning liquid /
gas / supercritical (reduced coordinates from each fluid's critical point), plus a
few non-PT input pairs. The finite overlap is compared with a relative tolerance.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from encomp import coolprop as encomp_coolprop
from encomp.fluids import Fluid, clear_expr_evaluation_cache
from encomp.units import Quantity as Q

# Optional test-only oracle; absent from runtime and installed-wheel environments.
CP: Any = pytest.importorskip("CoolProp.CoolProp", reason="Python CoolProp oracle group is not installed")


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


def _evaluate_native(name: str, kwargs: dict[str, Any], prop: str, df: pl.DataFrame) -> np.ndarray:
    clear_expr_evaluation_cache()
    expression = cast(Any, encomp_coolprop.fluid)(prop, "P", "T", name=name, **kwargs)
    return df.select(expression.alias("x"))["x"].to_numpy().astype(float)


def _evaluate_oracle(name: str, kwargs: dict[str, Any], prop: str, df: pl.DataFrame) -> np.ndarray:
    # Independent test-only Python CoolProp numerical reference. Explicit
    # composition is rendered into the equivalent name syntax for PropsSI.
    composition = kwargs.get("composition")
    if composition is not None:
        components = "&".join(f"{species}[{fraction}]" for species, fraction in composition.items())
        name = f"{name}::{components}"
    return np.asarray(CP.PropsSI(prop, "P", df["P"].to_numpy(), "T", df["T"].to_numpy(), name), dtype=float)


@pytest.mark.parametrize(("label", "name", "kwargs"), _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_native_python_oracle_parity(label: str, name: str, kwargs: dict[str, Any]) -> None:
    assert encomp_coolprop.self_check()

    df = _grid(name, kwargs)
    for prop in PROPS:
        oracle = _evaluate_oracle(name, kwargs, prop, df)
        native = _evaluate_native(name, kwargs, prop, df)
        assert oracle.shape == native.shape

        # the two paths must agree on WHICH points are finite -- otherwise a path that
        # returns NaN where the other returns a value would be silently excluded below
        finite_oracle, finite_native = np.isfinite(oracle), np.isfinite(native)
        assert np.array_equal(finite_oracle, finite_native), (
            f"{label}.{prop}: finite/NaN masks differ ({int((finite_oracle != finite_native).sum())} points)"
        )
        both = finite_oracle & finite_native
        assert both.sum() >= 8, f"{label}.{prop}: only {int(both.sum())} finite-overlap points"
        rel = np.abs(native[both] - oracle[both]) / np.maximum(np.abs(oracle[both]), 1e-9)
        worst = int(np.argmax(rel))
        assert rel.max() <= RTOL, (
            f"{label}.{prop}: max rel {rel.max():.2e} "
            f"(native {native[both][worst]:.6g} vs oracle {oracle[both][worst]:.6g})"
        )


_INPUT_PAIR_PARAMETERS = (
    ("Q", "T"),
    ("Qmass", "T"),
    ("P", "Q"),
    ("P", "Qmass"),
    ("P", "T"),
    ("Dmolar", "T"),
    ("Dmass", "T"),
    ("Hmolar", "T"),
    ("Hmass", "T"),
    ("Smolar", "T"),
    ("Smass", "T"),
    ("T", "Umolar"),
    ("T", "Umass"),
    ("Dmass", "Hmass"),
    ("Dmolar", "Hmolar"),
    ("Dmass", "Smass"),
    ("Dmolar", "Smolar"),
    ("Dmass", "Umass"),
    ("Dmolar", "Umolar"),
    ("Dmass", "P"),
    ("Dmolar", "P"),
    ("Dmass", "Q"),
    ("Dmass", "Qmass"),
    ("Dmolar", "Q"),
    ("Dmolar", "Qmass"),
    ("Hmass", "P"),
    ("Hmolar", "P"),
    ("P", "Smass"),
    ("P", "Smolar"),
    ("P", "Umass"),
    ("P", "Umolar"),
    ("Hmass", "Smass"),
    ("Hmolar", "Smolar"),
    ("Smass", "Umass"),
    ("Smolar", "Umolar"),
)


@pytest.mark.parametrize(("name1", "name2"), _INPUT_PAIR_PARAMETERS)
def test_native_input_pair_table_matches_python_oracle(name1: str, name2: str) -> None:
    """Every reviewed native pair matches CoolProp's generate_update_pair table."""

    first, second = 1.25, 2.5
    oracle_pair, oracle_first, oracle_second = CP.generate_update_pair(
        CP.get_parameter_index(name1),
        first,
        CP.get_parameter_index(name2),
        second,
    )
    native_pair, swap = encomp_coolprop._native().resolve_input_pair(name1, name2)
    native_values = (second, first) if swap else (first, second)
    assert native_pair == int(oracle_pair)
    assert native_values == (oracle_first, oracle_second)


@pytest.mark.parametrize(
    "name",
    (
        "Water",
        "IF97::Water",
        "HEOS::CO2[0.5]&O2[0.5]",
        "HEOS::CO2[0]&O2[1]",
        "INCOMP::MEG[0.5]",
        "INCOMP::EG-20%",
    ),
)
def test_native_fraction_parser_matches_python_oracle(name: str) -> None:
    """The native fallback parser matches CoolProp's documented Python parser."""

    oracle_backend, fraction_spec = CP.extract_backend(name)
    oracle_fluids, oracle_fractions = CP.extract_fractions(fraction_spec)
    expected = (
        "HEOS" if oracle_backend == "?" else oracle_backend,
        "&".join(oracle_fluids),
        list(oracle_fractions) or None,
    )
    assert encomp_coolprop._native().resolve_fluid_name(name) == expected


def test_native_python_oracle_parity_input_pairs() -> None:
    """Parity for non-PT pairs where native resolves order and the oracle uses PropsSI."""
    assert encomp_coolprop.self_check()

    p = np.geomspace(2e5, 200e5, 25)
    t = np.linspace(300.0, 620.0, 25)
    h = CP.PropsSI("HMASS", "P", p, "T", t, "HEOS::Water")
    s = CP.PropsSI("SMASS", "P", p, "T", t, "HEOS::Water")

    for second, unit, vals in [("HMASS", "J/kg", h), ("SMASS", "J/kg/K", s)]:
        df = pl.DataFrame({"P": p, second: vals})
        clear_expr_evaluation_cache()
        second_point: dict[str, Any] = {second: Q(pl.col(second), unit)}
        fluid = Fluid("HEOS::Water", P=Q(pl.col("P"), "Pa"), **second_point)
        native = df.select(fluid.D.m.alias("d"))["d"].to_numpy().astype(float)
        oracle = np.asarray(
            CP.PropsSI("DMASS", "P", df["P"].to_numpy(), second, df[second].to_numpy(), "HEOS::Water"),
            dtype=float,
        )
        finite_oracle, finite_native = np.isfinite(oracle), np.isfinite(native)
        assert np.array_equal(finite_oracle, finite_native), (
            f"P,{second}: finite/NaN masks differ ({int((finite_oracle != finite_native).sum())} points)"
        )
        both = finite_oracle & finite_native
        assert both.sum() >= 15, f"P,{second}: only {int(both.sum())} finite points"
        rel = np.abs(native[both] - oracle[both]) / np.maximum(np.abs(oracle[both]), 1e-9)
        assert rel.max() <= RTOL, f"P,{second}: max rel {rel.max():.2e}"


def test_eager_native_non_pt_parity() -> None:
    """The eager plugin's non-PT canonical order matches the oracle PropsSI path."""
    assert encomp_coolprop.self_check()

    n = 1200
    p = np.geomspace(2e5, 200e5, n)
    t = np.linspace(300.0, 620.0, n)
    h = CP.PropsSI("HMASS", "P", p, "T", t, "HEOS::Water")

    eager = np.asarray(Fluid("HEOS::Water", P=Q(p, "Pa"), H=Q(h, "J/kg")).D.m, dtype=float)

    reference = np.asarray(CP.PropsSI("DMASS", "P", p, "HMASS", h, "HEOS::Water"), dtype=float)

    assert np.array_equal(np.isfinite(eager), np.isfinite(reference))
    both = np.isfinite(eager) & np.isfinite(reference)
    assert both.sum() >= n // 2
    rel = np.abs(eager[both] - reference[both]) / np.maximum(np.abs(reference[both]), 1e-9)
    assert rel.max() <= RTOL, f"eager non-PT: max rel {rel.max():.2e}"
