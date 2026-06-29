"""Parallel CoolProp property evaluation as Polars expression plugins.

Usable directly on any Polars expression -- independent of encomp:

    import polars as pl
    import encomp_coolprop as cp

    df.select(
        cp.fluid("DMASS", "P", "T").alias("rho"),   # defaults: backend IF97, fluid Water
        cp.fluid("HMASS", "P", "T").alias("h"),
    )                       # independent properties run in PARALLEL (no GIL)

    df.select(cp.fluid("DMASS", "P", "H", name2="HMASS", backend="HEOS"))  # any input pair
    df.select(cp.humid_air("W", "P", "T", "R"))                             # humid air
"""

from __future__ import annotations

from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Literal, get_args

import CoolProp.CoolProp as _CoolProp
import polars as pl
from polars.plugins import register_plugin_function

# CoolProp ships incomplete type stubs; treat the module as Any (mirrors encomp).
_cp: Any = _CoolProp

__all__ = [
    "BACKENDS",
    "FLUID_PARAMS",
    "HUMID_AIR_PARAMS",
    "PHASES",
    "Backend",
    "FluidParam",
    "HumidAirParam",
    "Phase",
    "fluid",
    "humid_air",
    "lib_path",
    "lib_version",
]

# Each name set is defined ONCE as a Literal (static typing) and exposed as a
# matching runtime `frozenset[<that Literal>]` via typing.get_args() -- one source,
# reusable for both type hints and O(1) membership checks, with the precise element
# type preserved. CoolProp's param_index is the ultimate runtime authority: names
# outside these sets still work, they just aren't statically typed.

# CoolProp AbstractState backends (HEOS is the general-purpose mixture EOS; IF97 is water/steam).
Backend = Literal["HEOS", "IF97", "REFPROP", "SRK", "PR", "PCSAFT", "VTPR", "INCOMP", "BICUBIC&HEOS", "TTSE&HEOS"]
BACKENDS: frozenset[Backend] = frozenset(get_args(Backend))

# CoolProp phase strings for specify_phase / assume_phase.
Phase = Literal[
    "phase_liquid",
    "phase_gas",
    "phase_twophase",
    "phase_supercritical",
    "phase_supercritical_gas",
    "phase_supercritical_liquid",
    "phase_critical_point",
]
PHASES: frozenset[Phase] = frozenset(get_args(Phase))

# CoolProp fluid parameters (outputs / the two state-input names): canonical names
# plus the common single-letter aliases.
FluidParam = Literal[
    "P",
    "T",
    "Q",
    "D",
    "DMASS",
    "DMOLAR",
    "H",
    "HMASS",
    "HMOLAR",
    "S",
    "SMASS",
    "SMOLAR",
    "U",
    "UMASS",
    "UMOLAR",
    "C",
    "CPMASS",
    "CPMOLAR",
    "CP0MASS",
    "CP0MOLAR",
    "CVMASS",
    "CVMOLAR",
    "G",
    "GMASS",
    "GMOLAR",
    "HELMHOLTZMASS",
    "HELMHOLTZMOLAR",
    "A",
    "SPEED_OF_SOUND",
    "V",
    "VISCOSITY",
    "L",
    "CONDUCTIVITY",
    "PRANDTL",
    "Z",
    "PHASE",
    "M",
    "MOLARMASS",
    "GAS_CONSTANT",
    "SURFACE_TENSION",
    "ISOBARIC_EXPANSION_COEFFICIENT",
    "ISOTHERMAL_COMPRESSIBILITY",
    "ISENTROPIC_EXPANSION_COEFFICIENT",
    "DIPOLE_MOMENT",
    "PCRIT",
    "P_CRITICAL",
    "TCRIT",
    "T_CRITICAL",
    "RHOMASS_CRITICAL",
    "PTRIPLE",
    "TTRIPLE",
    "PMAX",
    "PMIN",
    "TMAX",
    "TMIN",
    "T_FREEZE",
]
FLUID_PARAMS: frozenset[FluidParam] = frozenset(get_args(FluidParam))

# Common HumidAir (HAPropsSI) parameters.
HumidAirParam = Literal[
    "W",
    "Omega",
    "T",
    "Tdb",
    "Twb",
    "B",
    "Tdp",
    "D",
    "Tdew",
    "P",
    "R",
    "RH",
    "H",
    "Hda",
    "Hha",
    "S",
    "Sda",
    "Sha",
    "V",
    "Vda",
    "Vha",
    "C",
    "Cha",
    "CV",
    "Visc",
    "mu",
    "K",
    "Conductivity",
]
HUMID_AIR_PARAMS: frozenset[HumidAirParam] = frozenset(get_args(HumidAirParam))

_HERE = Path(__file__).parent


@lru_cache(maxsize=1)
def lib_path() -> str:
    """Absolute path to the bundled CoolProp dynamic library."""
    for name in ("libCoolProp.dylib", "libCoolProp.so", "CoolProp.dll", "libCoolProp.dll"):
        p = _HERE / name
        if p.exists():
            return str(p)
    raise RuntimeError(f"bundled CoolProp library not found in {_HERE}")


@cache
def _resolve_pair(name1: str, name2: str) -> tuple[int, bool]:
    # CoolProp's generate_update_pair gives the canonical input_pairs index and
    # value order; the swap decision is value-independent, so resolve it once.
    pair, a, _ = _cp.generate_update_pair(_cp.get_parameter_index(name1), 1.0, _cp.get_parameter_index(name2), 2.0)
    return int(pair), (a == 2.0)


def _as_expr(x: str | pl.Expr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def fluid(
    output: FluidParam,
    input1: str | pl.Expr,
    input2: str | pl.Expr,
    *,
    name1: FluidParam = "P",
    name2: FluidParam = "T",
    backend: Backend = "IF97",
    fluid: str = "Water",
    phase: Phase | None = None,
    mole_fractions: list[float] | None = None,
) -> pl.Expr:
    """A CoolProp fluid property (``output``) as a parallel Polars expression.

    Defaults to IF97 water; for other fluids/mixtures pass ``backend="HEOS"`` (and
    ``fluid="CO2&O2"`` + ``mole_fractions``). ``input1``/``input2`` may be in any
    order -- name them with ``name1``/``name2`` (any CoolProp pair: PT, PH, PQ, ...).
    """
    pair_idx, swap = _resolve_pair(name1, name2)
    a, b = (input2, input1) if swap else (input1, input2)  # canonical order
    return register_plugin_function(
        plugin_path=_HERE,
        function_name="cp_evaluate",
        args=[_as_expr(a).cast(pl.Float64), _as_expr(b).cast(pl.Float64)],
        kwargs={
            "lib_path": lib_path(),
            "backend": backend,
            "fluid": fluid,
            "input_pair": pair_idx,
            "output": output,
            "phase": phase,
            "mole_fractions": mole_fractions,
        },
        is_elementwise=True,
        use_abs_path=True,
    )


def humid_air(
    output: HumidAirParam,
    input1: str | pl.Expr,
    input2: str | pl.Expr,
    input3: str | pl.Expr,
    *,
    name1: HumidAirParam = "P",
    name2: HumidAirParam = "T",
    name3: HumidAirParam = "R",
) -> pl.Expr:
    """A humid-air (HAPropsSI) property as a Polars expression."""
    return register_plugin_function(
        plugin_path=_HERE,
        function_name="ha_evaluate",
        args=[_as_expr(input1).cast(pl.Float64), _as_expr(input2).cast(pl.Float64), _as_expr(input3).cast(pl.Float64)],
        kwargs={"lib_path": lib_path(), "output": output, "name1": name1, "name2": name2, "name3": name3},
        is_elementwise=True,
        use_abs_path=True,
    )


def lib_version() -> str:
    """CoolProp version of the bundled library."""
    import ctypes

    lib = ctypes.CDLL(lib_path())
    buf = ctypes.create_string_buffer(256)
    lib.get_global_param_string(b"version", buf, 256)
    return buf.value.decode()


@lru_cache(maxsize=1)
def self_check() -> bool:
    """True if the plugin loads and evaluates one known value correctly (cached)."""
    try:
        df = pl.DataFrame({"P": [50e5], "T": [400.0]})
        v = df.select(fluid("DMASS", "P", "T", backend="IF97", fluid="Water"))[0, 0]
        return v is not None and abs(v - 939.906) < 1.0
    except Exception:
        return False
