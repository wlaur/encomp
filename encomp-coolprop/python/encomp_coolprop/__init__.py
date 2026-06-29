"""Parallel CoolProp property evaluation as Polars expression plugins.

Usable directly on any Polars expression -- independent of encomp:

    import polars as pl
    import encomp_coolprop as cp

    df.select(
        cp.fluid("DMASS", "P", "T").alias("rho"),   # defaults: backend IF97, fluid Water
        cp.fluid("HMASS", "P", "T").alias("h"),
    )                       # independent properties run in PARALLEL (no GIL)

    df.select(cp.fluid("DMASS", "P", "H", backend="HEOS"))   # any input pair (named by inputs)
    df.select(cp.humid_air("W", "P", "T", "R"))              # humid air
    # for differently-named columns, alias them: cp.fluid("DMASS", pl.col("p").alias("P"), "T")
"""

from __future__ import annotations

import sys
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Literal, TypeIs, get_args

import CoolProp.CoolProp as _CoolProp
import polars as pl
from polars.plugins import register_plugin_function

# CoolProp ships incomplete type stubs; treat the module as Any (mirrors encomp).
_cp: Any = _CoolProp

__all__ = [
    "BACKENDS",
    "FLUID_INPUTS",
    "FLUID_PARAMS",
    "HUMID_AIR_INPUTS",
    "HUMID_AIR_PARAMS",
    "PHASES",
    "Backend",
    "FluidInput",
    "FluidParam",
    "HumidAirInput",
    "HumidAirParam",
    "Phase",
    "fluid",
    "humid_air",
    "is_backend",
    "is_fluid_input",
    "is_fluid_param",
    "is_humid_air_input",
    "is_humid_air_param",
    "is_phase",
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
    "A",
    "C",
    "CONDUCTIVITY",
    "CP0MASS",
    "CP0MOLAR",
    "CPMASS",
    "CPMOLAR",
    "CVMASS",
    "CVMOLAR",
    "Cp0mass",
    "Cp0molar",
    "Cpmass",
    "Cpmolar",
    "Cvmass",
    "Cvmolar",
    "D",
    "DELTA",
    "DIPOLE_MOMENT",
    "DMASS",
    "DMOLAR",
    "Delta",
    "Dmass",
    "Dmolar",
    "G",
    "GMASS",
    "GAS_CONSTANT",
    "GMOLAR",
    "GMOLAR_RESIDUAL",
    "Gmass",
    "Gmolar",
    "Gmolar_residual",
    "H",
    "HELMHOLTZMASS",
    "HELMHOLTZMOLAR",
    "HMASS",
    "HMOLAR",
    "HMOLAR_RESIDUAL",
    "Helmholtzmass",
    "Helmholtzmolar",
    "Hmass",
    "Hmolar",
    "Hmolar_residual",
    "I",
    "ISENTROPIC_EXPANSION_COEFFICIENT",
    "ISOBARIC_EXPANSION_COEFFICIENT",
    "ISOTHERMAL_COMPRESSIBILITY",
    "L",
    "M",
    "MOLARMASS",
    "MOLAR_MASS",
    "MOLEMASS",
    "O",
    "P",
    "PCRIT",
    "PHASE",
    "PMAX",
    "PMIN",
    "PRANDTL",
    "PTRIPLE",
    "P_CRITICAL",
    "P_MAX",
    "P_MIN",
    "P_REDUCING",
    "P_TRIPLE",
    "P_max",
    "P_min",
    "Pcrit",
    "Phase",
    "Prandtl",
    "Q",
    "RHOCRIT",
    "RHOMASS_CRITICAL",
    "RHOMASS_REDUCING",
    "RHOMOLAR_CRITICAL",
    "RHOMOLAR_REDUCING",
    "S",
    "SMASS",
    "SMOLAR",
    "SMOLAR_RESIDUAL",
    "SPEED_OF_SOUND",
    "SURFACE_TENSION",
    "Smass",
    "Smolar",
    "Smolar_residual",
    "T",
    "TAU",
    "TCRIT",
    "TMAX",
    "TMIN",
    "TTRIPLE",
    "T_CRITICAL",
    "T_FREEZE",
    "T_MAX",
    "T_MIN",
    "T_REDUCING",
    "T_TRIPLE",
    "T_critical",
    "T_freeze",
    "T_max",
    "T_min",
    "T_reducing",
    "T_triple",
    "Tau",
    "Tcrit",
    "Tmax",
    "Tmin",
    "Ttriple",
    "U",
    "UMASS",
    "UMOLAR",
    "Umass",
    "Umolar",
    "V",
    "VISCOSITY",
    "Z",
    "conductivity",
    "dipole_moment",
    "gas_constant",
    "isentropic_expansion_coefficient",
    "isobaric_expansion_coefficient",
    "isothermal_compressibility",
    "molar_mass",
    "molarmass",
    "molemass",
    "p_critical",
    "p_reducing",
    "p_triple",
    "pcrit",
    "pmax",
    "pmin",
    "ptriple",
    "rhocrit",
    "rhomass_critical",
    "rhomass_reducing",
    "rhomolar_critical",
    "rhomolar_reducing",
    "speed_of_sound",
    "surface_tension",
    "viscosity",
]
FLUID_PARAMS: frozenset[FluidParam] = frozenset(get_args(FluidParam))

# CoolProp fluid STATE-INPUT properties: the subset of FluidParam valid as the two
# inputs that fix a state -- pressure/temperature/quality + density/enthalpy/entropy/
# internal-energy (mass + molar + aliases). `output` can be any FluidParam, but the
# two inputs (and their `name1`/`name2`) must come from this set.
FluidInput = Literal[
    "P", "T", "Q",
    "D", "DMASS", "DMOLAR", "Dmass", "Dmolar",
    "H", "HMASS", "HMOLAR", "Hmass", "Hmolar",
    "S", "SMASS", "SMOLAR", "Smass", "Smolar",
    "U", "UMASS", "UMOLAR", "Umass", "Umolar",
]  # fmt: skip
FLUID_INPUTS: frozenset[FluidInput] = frozenset(get_args(FluidInput))

# Common HumidAir (HAPropsSI) parameters.
HumidAirParam = Literal[
    "B",
    "C",
    "CV",
    "CVha",
    "Cha",
    "Conductivity",
    "D",
    "DewPoint",
    "Enthalpy",
    "Entropy",
    "H",
    "Hda",
    "Hha",
    "HumRat",
    "K",
    "M",
    "Omega",
    "P",
    "P_w",
    "R",
    "RH",
    "RelHum",
    "S",
    "Sda",
    "Sha",
    "T",
    "T_db",
    "T_dp",
    "T_wb",
    "Tdb",
    "Tdp",
    "Twb",
    "V",
    "Vda",
    "Vha",
    "Visc",
    "W",
    "WetBulb",
    "Y",
    "Z",
    "cp",
    "cp_ha",
    "cv_ha",
    "k",
    "mu",
    "psi_w",
]
HUMID_AIR_PARAMS: frozenset[HumidAirParam] = frozenset(get_args(HumidAirParam))

# HumidAir (HAPropsSI) STATE-INPUT properties: the subset valid as the three inputs.
HumidAirInput = Literal[
    "T", "Tdb", "T_db", "B", "Twb", "T_wb", "WetBulb",
    "D", "Tdp", "T_dp", "DewPoint", "W", "Omega", "HumRat", "psi_w",
    "R", "RH", "RelHum", "H", "Hda", "Hha", "Enthalpy",
    "S", "Sda", "Sha", "Entropy", "V", "Vda", "Vha", "P", "P_w", "Y",
]  # fmt: skip
HUMID_AIR_INPUTS: frozenset[HumidAirInput] = frozenset(get_args(HumidAirInput))


# TypeIs predicates (PEP 742): narrow a runtime ``str`` to the corresponding strict
# Literal without a cast, in both branches (and reject unknown names). Use these
# wherever a property/backend/phase name arrives as ``str`` (e.g. from **kwargs,
# whose keys are always ``str``).
def is_fluid_param(name: str) -> TypeIs[FluidParam]:
    return name in FLUID_PARAMS


def is_humid_air_param(name: str) -> TypeIs[HumidAirParam]:
    return name in HUMID_AIR_PARAMS


def is_backend(name: str) -> TypeIs[Backend]:
    return name in BACKENDS


def is_phase(name: str) -> TypeIs[Phase]:
    return name in PHASES


def is_fluid_input(name: str) -> TypeIs[FluidInput]:
    return name in FLUID_INPUTS


def is_humid_air_input(name: str) -> TypeIs[HumidAirInput]:
    return name in HUMID_AIR_INPUTS


_HERE = Path(__file__).parent


# platform -> bundled CoolProp library name(s), in preference order
_LIB_NAMES = {
    "darwin": ("libCoolProp.dylib",),
    "win32": ("CoolProp.dll", "libCoolProp.dll"),
}.get(sys.platform, ("libCoolProp.so",))


@lru_cache(maxsize=1)
def lib_path() -> str:
    """Absolute path to the bundled CoolProp dynamic library for this platform."""
    for name in _LIB_NAMES:
        p = _HERE / name
        if p.exists():
            return str(p)
    raise RuntimeError(f"bundled CoolProp library ({', '.join(_LIB_NAMES)}) not found in {_HERE}")


@cache
def _resolve_pair(name1: str, name2: str) -> tuple[int, bool]:
    # CoolProp's generate_update_pair gives the canonical input_pairs index and
    # value order; the swap decision is value-independent, so resolve it once.
    pair, a, _ = _cp.generate_update_pair(_cp.get_parameter_index(name1), 1.0, _cp.get_parameter_index(name2), 2.0)
    return int(pair), (a == 2.0)


def _as_expr(x: str | pl.Expr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def _input_name(x: str | pl.Expr) -> str:
    # a state input identifies its property by NAME: the string itself, or the
    # output name of the expression (e.g. pl.col("P") or pl.col("p").alias("P")).
    return x if isinstance(x, str) else x.meta.output_name()


def fluid(
    output: FluidParam,
    input1: FluidInput | pl.Expr,
    input2: FluidInput | pl.Expr,
    *,
    backend: Backend = "IF97",
    fluid: str = "Water",
    phase: Phase | None = None,
    mole_fractions: list[float] | None = None,
) -> pl.Expr:
    """A CoolProp fluid property (``output``) as a parallel Polars expression.

    Each input identifies its property by NAME -- a string is the property and the
    column read from it; an expression by its output name (``pl.col("P")`` or
    ``pl.col("p").alias("P")``). Both must be CoolProp state inputs (P/T/Q/D/H/S/U,
    any pair: PT, PH, PQ, ...). Defaults to IF97 water; for other fluids/mixtures
    pass ``backend="HEOS"`` (and ``fluid="CO2&O2"`` + ``mole_fractions``).
    """
    name1, name2 = _input_name(input1), _input_name(input2)
    for name in (name1, name2):
        if not is_fluid_input(name):
            raise ValueError(
                f"fluid input must be named after a CoolProp state input "
                f"(P, T, Q, D, H, S, U, ...); got {name!r}. Alias the column, e.g. "
                f'pl.col("pressure").alias("P").'
            )
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
    input1: HumidAirInput | pl.Expr,
    input2: HumidAirInput | pl.Expr,
    input3: HumidAirInput | pl.Expr,
) -> pl.Expr:
    """A humid-air (HAPropsSI) property as a Polars expression.

    Each input identifies its property by name (the string, or the expression's
    output name); all three must be HAPropsSI state inputs (T, P, R, W, B, ...).
    """
    name1, name2, name3 = _input_name(input1), _input_name(input2), _input_name(input3)
    for name in (name1, name2, name3):
        if not is_humid_air_input(name):
            raise ValueError(
                f"humid-air input must be named after a HAPropsSI state input "
                f"(T, P, R, W, B, ...); got {name!r}. Alias the column, e.g. "
                f'pl.col("rel_hum").alias("R").'
            )
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
