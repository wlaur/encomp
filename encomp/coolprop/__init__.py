"""Parallel CoolProp property evaluation as Polars expression plugins.

The ``fluid`` / ``humid_air`` API mirrors :mod:`encomp.fluids`: a fluid is identified
by its ``name`` (with the backend folded in, e.g. ``"HEOS::CarbonDioxide"``), a mixture
by a ``composition`` dict, and a fixed phase by ``assume_phase``.

    import polars as pl
    from encomp import coolprop as cp

    df.select(
        cp.fluid("DMASS", "P", "T").alias("rho"),   # default: IF97 water
        cp.fluid("HMASS", "P", "T").alias("h"),
    )                       # independent properties run in PARALLEL (no GIL)

    df.select(cp.fluid("DMASS", "P", "H", name="HEOS::Water"))   # any input pair (named by inputs)
    df.select(cp.fluid("DMASS", "P", "T", name="HEOS::CO2&O2",   # mixture, mole fractions
                       composition={"CO2": 0.7, "O2": 0.3}))
    df.select(cp.humid_air("W", "P", "T", "R"))                  # humid air
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
    "ASSUMED_PHASES",
    "BACKENDS",
    "FLUID_INPUTS",
    "FLUID_PARAMS",
    "HUMID_AIR_INPUTS",
    "HUMID_AIR_PARAMS",
    "PHASES",
    "AssumedPhase",
    "Backend",
    "CName",
    "CommonFluidName",
    "Composition",
    "FluidInput",
    "FluidParam",
    "FractionValue",
    "HumidAirInput",
    "HumidAirParam",
    "Phase",
    "fluid",
    "humid_air",
    "is_assumed_phase",
    "is_backend",
    "is_fluid_input",
    "is_fluid_param",
    "is_humid_air_input",
    "is_humid_air_param",
    "is_phase",
    "lib_path",
    "lib_version",
    "resolve_fluid_spec",
]

# Each name set is defined ONCE as a Literal (static typing) and exposed as a
# matching runtime `frozenset[<that Literal>]` via typing.get_args() -- one source,
# reusable for both type hints and O(1) membership checks, with the precise element
# type preserved. CoolProp's param_index is the ultimate runtime authority: names
# outside these sets still work, they just aren't statically typed.

# CoolProp AbstractState backends (HEOS is the general-purpose mixture EOS; IF97 is water/steam).
Backend = Literal["HEOS", "IF97", "REFPROP", "SRK", "PR", "PCSAFT", "VTPR", "INCOMP", "BICUBIC&HEOS", "TTSE&HEOS"]
BACKENDS: frozenset[Backend] = frozenset(get_args(Backend))

# A few common CoolProp fluid names surfaced as Literals so editors can suggest them;
# ANY CoolProp name string is still accepted (the union widens to ``str``). The name may
# fold in the backend ("IF97::Water", "HEOS::CO2&O2") and fixed mole fractions
# ("HEOS::CO2[0.5]&O2[0.5]"); a bare "Water" defaults to the HEOS (IAPWS-95) backend.
# Single source of truth -- re-exported by encomp.fluids.
CommonFluidName = Literal[
    "Water",
    "IF97::Water",
    "HEOS::Water",
    "HEOS",
    "Air",
    "Nitrogen",
    "Oxygen",
    "CarbonDioxide",
    "Hydrogen",
    "Methane",
    "Ammonia",
    "Argon",
    "R134a",
]
CName = CommonFluidName | str

FractionValue = float | int
# Fixed mixture composition as ``{species: mole fraction}``, mirroring encomp.fluids.
Composition = dict[CName, FractionValue]

# CoolProp phase strings for AbstractState.specify_phase (the value passed to the
# plugin). Internal/low-level; the user-facing name is ``AssumedPhase`` below.
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

# User-facing assumed-phase names (matches encomp.fluids.Fluid.assume_phase). Each maps
# to the CoolProp ``phase_*`` string (the ``Phase`` literal) via ``_phase_from_assumed``;
# there is no assumed-phase counterpart for ``phase_critical_point``.
AssumedPhase = Literal[
    "gas",
    "liquid",
    "supercritical",
    "supercritical_gas",
    "supercritical_liquid",
    "twophase",
]
ASSUMED_PHASES: frozenset[AssumedPhase] = frozenset(get_args(AssumedPhase))

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


def is_assumed_phase(name: str) -> TypeIs[AssumedPhase]:
    return name in ASSUMED_PHASES


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


def _is_scalar(expr: pl.Expr) -> bool:
    # a length-1 literal (pl.lit, or a lifted python float) depends on no column, so
    # its dtype is "weak" and must not force the output precision up (e.g. a Float64
    # literal alongside a Float32 column still yields Float32). Columns have root names.
    return not expr.meta.root_names()


def _input_name(x: str | pl.Expr) -> str:
    # a state input identifies its property by NAME: the string itself, or the
    # output name of the expression (e.g. pl.col("P") or pl.col("p").alias("P")).
    return x if isinstance(x, str) else x.meta.output_name()


def _phase_from_assumed(assume_phase: AssumedPhase) -> Phase:
    """Map a user-facing assumed phase (``"gas"``) to the CoolProp phase string the
    plugin's ``specify_phase`` expects (``"phase_gas"``)."""
    phase = f"phase_{assume_phase}"
    if not is_phase(phase):
        raise ValueError(f"unknown assumed phase {assume_phase!r}; expected one of {sorted(ASSUMED_PHASES)}")
    return phase


def resolve_fluid_spec(
    name: CName,
    composition: Composition | None = None,
    normalize: bool = True,
) -> tuple[str, str, list[float] | None]:
    """Resolve a CoolProp fluid ``name`` (+ optional explicit ``composition``) into the
    ``(backend, fluids, mole_fractions)`` the plugin's ``AbstractState`` needs. Mirrors
    :class:`encomp.fluids.Fluid` (which delegates here for its rust path), so both layers
    parse names identically.

    The backend may be folded into ``name`` (``"IF97::Water"``, ``"HEOS::CO2&O2"``) or,
    when a ``composition`` dict is given, passed as the bare backend (``"HEOS"``); a name
    without a backend (``"Water"``, ``"CO2&O2"``) defaults to HEOS. Mole fractions come
    from ``composition`` (``{species: fraction}``) or from the name
    (``"HEOS::CO2[0.5]&O2[0.5]"``), renormalised to sum to 1 unless ``normalize=False``.
    ``mole_fractions`` is ``None`` for a pure fluid.
    """
    if composition is not None:
        backend = name.split("::", 1)[0] if "::" in name else name
        species = list(composition)
        fluids = "&".join(species)
        fractions: list[float] | None = [float(v) for v in composition.values()]
    else:
        backend_raw, fluid_str = _cp.extract_backend(name)
        backend = "HEOS" if backend_raw == "?" else backend_raw
        species, fracs = _cp.extract_fractions(fluid_str)
        fluids = "&".join(species)
        fractions = [float(x) for x in fracs] if len(species) > 1 else None
    if fractions is not None and normalize:
        total = sum(fractions)
        if total:
            fractions = [x / total for x in fractions]
    return backend, fluids, fractions


def fluid(
    output: FluidParam,
    input1: FluidInput | pl.Expr,
    input2: FluidInput | pl.Expr,
    *,
    name: CName = "IF97::Water",
    assume_phase: AssumedPhase | None = None,
    composition: Composition | None = None,
    normalize: bool = True,
) -> pl.Expr:
    """A CoolProp fluid property (``output``) as a parallel Polars expression.

    Mirrors :class:`encomp.fluids.Fluid`. The fluid is identified by ``name`` with the
    backend folded in (``"HEOS::CarbonDioxide"``, ``"IF97::Water"``; a bare ``"Water"``
    defaults to HEOS/IAPWS-95). A mixture is given either by fractions in the name
    (``"HEOS::CO2[0.5]&O2[0.5]"``) or a ``composition={species: mole fraction}`` dict
    (renormalised unless ``normalize=False``). ``assume_phase`` pins the phase, skipping
    CoolProp's phase-stability search (a speed tool; honoured by HEOS/GERG, ignored by
    region-explicit backends like IF97).

    Each input identifies its property by NAME -- a string is the property and the
    column read from it; an expression by its output name (``pl.col("P")`` or
    ``pl.col("p").alias("P")``). Both must be CoolProp state inputs (P/T/Q/D/H/S/U,
    any pair: PT, PH, PQ, ...).
    """
    name1, name2 = _input_name(input1), _input_name(input2)
    for nm in (name1, name2):
        if not is_fluid_input(nm):
            raise ValueError(
                f"fluid input must be named after a CoolProp state input "
                f"(P, T, Q, D, H, S, U, ...); got {nm!r}. Alias the column, e.g. "
                f'pl.col("pressure").alias("P").'
            )
    pair_idx, swap = _resolve_pair(name1, name2)
    a, b = (input2, input1) if swap else (input1, input2)  # canonical order
    a_expr, b_expr = _as_expr(a), _as_expr(b)
    backend, fluids, mole_fractions = resolve_fluid_spec(name, composition, normalize)
    phase = _phase_from_assumed(assume_phase) if assume_phase is not None else None
    # inputs are passed at their own dtype (the plugin casts to f64 internally); the
    # output dtype preserves the input precision (Float32 in -> Float32 out).
    return register_plugin_function(
        plugin_path=_HERE,
        function_name="cp_evaluate",
        args=[a_expr, b_expr],
        kwargs={
            "lib_path": lib_path(),
            "backend": backend,
            "fluid": fluids,
            "input_pair": pair_idx,
            "output": output,
            "phase": phase,
            "mole_fractions": mole_fractions,
            "scalar_mask": [_is_scalar(a_expr), _is_scalar(b_expr)],
        },
        is_elementwise=True,
        use_abs_path=True,
    ).alias(output)  # name the result after the computed property, not the first input


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
    # Unlike the fluid path (validated by CoolProp's param_index in the plugin), HAPropsSI
    # returns _HUGE -> null for an unknown output, so a typo'd output would silently yield
    # an all-null column. Validate it here against the known HAPropsSI parameter set.
    if not is_humid_air_param(output):
        raise ValueError(f"humid-air output must be a HAPropsSI parameter (W, H, Twb, RH, ...); got {output!r}.")
    name1, name2, name3 = _input_name(input1), _input_name(input2), _input_name(input3)
    for name in (name1, name2, name3):
        if not is_humid_air_input(name):
            raise ValueError(
                f"humid-air input must be named after a HAPropsSI state input "
                f"(T, P, R, W, B, ...); got {name!r}. Alias the column, e.g. "
                f'pl.col("rel_hum").alias("R").'
            )
    e1, e2, e3 = _as_expr(input1), _as_expr(input2), _as_expr(input3)
    return register_plugin_function(
        plugin_path=_HERE,
        function_name="ha_evaluate",
        args=[e1, e2, e3],
        kwargs={
            "lib_path": lib_path(),
            "output": output,
            "name1": name1,
            "name2": name2,
            "name3": name3,
            "scalar_mask": [_is_scalar(e1), _is_scalar(e2), _is_scalar(e3)],
        },
        is_elementwise=True,
        use_abs_path=True,
    ).alias(output)  # name the result after the computed property, not the first input


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
        v = df.select(fluid("DMASS", "P", "T", name="IF97::Water"))[0, 0]
        return v is not None and abs(v - 939.906) < 1.0
    except Exception:
        return False
