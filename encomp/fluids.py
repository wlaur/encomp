"""
Classes and functions relating to fluid properties.
Uses CoolProp as backend.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from threading import Lock
from typing import Annotated, Any, ClassVar, Generic, Literal, Self, TypedDict, Unpack, cast

# CoolProp.CoolProp is a compiled extension module exporting both PropsSI and
# HAPropsSI. Importing the module (rather than the untyped pure-Python
# CoolProp.HumidAirProp) avoids a missing-stub warning. The functions are
# untyped, so they are exposed as typed aliases below.
import CoolProp.CoolProp as _CoolProp
import numpy as np
import polars as pl

from .coolprop import (
    FLUID_INPUTS,
    HUMID_AIR_INPUTS,
    AssumedPhase,
    Backend,
    CName,
    Composition,
    FluidParam,
    HumidAirParam,
    Phase,
    is_backend,
    is_fluid_param,
    is_humid_air_param,
    is_phase,
    resolve_fluid_spec,
)
from .settings import SETTINGS
from .structures import flatten
from .units import DimensionalityError, ExpectedDimensionalityError, Quantity, Unit
from .utypes import (
    MT,
    Density,
    Dimensionless,
    DynamicViscosity,
    MixtureEnthalpyPerDryAir,
    MixtureEnthalpyPerHumidAir,
    MixtureEntropyPerDryAir,
    MixtureEntropyPerHumidAir,
    MixtureVolumePerDryAir,
    MixtureVolumePerHumidAir,
    MolarDensity,
    MolarMass,
    MolarSpecificEnthalpy,
    MolarSpecificEntropy,
    MolarSpecificInternalEnergy,
    Numpy1DArray,
    Pressure,
    SpecificEnthalpy,
    SpecificEntropy,
    SpecificHeatCapacity,
    SpecificHeatPerDryAir,
    SpecificHeatPerHumidAir,
    SpecificInternalEnergy,
    Temperature,
    ThermalConductivity,
    Velocity,
)

_LOGGER = logging.getLogger(__name__)

_cp: Any = _CoolProp
PropsSI: Callable[..., float | Numpy1DArray] = _cp.PropsSI
HAPropsSI: Callable[..., float | Numpy1DArray] = _cp.HAPropsSI

# Strict Literals for CoolProp property names, single source of truth in the
# encomp.coolprop plugin (fluid + humid-air namespaces). The fluid-name / composition /
# assumed-phase types (CName, CommonFluidName, Composition, FractionValue, AssumedPhase)
# live there too; the ones used here are imported above.
CProperty = FluidParam | HumidAirParam
UnitString = Annotated[str, "Unit string"]


class FluidState(TypedDict, Generic[MT], total=False):  # noqa: UP046
    """Valid CoolProp fluid state-input property names for ``Fluid``/``Water``.

    Used with ``Unpack`` to statically check the ``**kwargs`` keys at the call
    site. The ``Quantity[Any, float]`` arm of each value type lets one fixed
    point be a plain scalar while another is vectorised, so a mixed
    ``float`` + ``MT`` call still infers the magnitude type correctly.
    """

    P: Quantity[Any, MT] | Quantity[Any, float]
    T: Quantity[Any, MT] | Quantity[Any, float]
    Q: Quantity[Any, MT] | Quantity[Any, float]
    D: Quantity[Any, MT] | Quantity[Any, float]
    DMASS: Quantity[Any, MT] | Quantity[Any, float]
    DMOLAR: Quantity[Any, MT] | Quantity[Any, float]
    Dmass: Quantity[Any, MT] | Quantity[Any, float]
    Dmolar: Quantity[Any, MT] | Quantity[Any, float]
    H: Quantity[Any, MT] | Quantity[Any, float]
    HMASS: Quantity[Any, MT] | Quantity[Any, float]
    HMOLAR: Quantity[Any, MT] | Quantity[Any, float]
    Hmass: Quantity[Any, MT] | Quantity[Any, float]
    Hmolar: Quantity[Any, MT] | Quantity[Any, float]
    S: Quantity[Any, MT] | Quantity[Any, float]
    SMASS: Quantity[Any, MT] | Quantity[Any, float]
    SMOLAR: Quantity[Any, MT] | Quantity[Any, float]
    Smass: Quantity[Any, MT] | Quantity[Any, float]
    Smolar: Quantity[Any, MT] | Quantity[Any, float]
    U: Quantity[Any, MT] | Quantity[Any, float]
    UMASS: Quantity[Any, MT] | Quantity[Any, float]
    UMOLAR: Quantity[Any, MT] | Quantity[Any, float]
    Umass: Quantity[Any, MT] | Quantity[Any, float]
    Umolar: Quantity[Any, MT] | Quantity[Any, float]


class HumidAirState(TypedDict, Generic[MT], total=False):  # noqa: UP046
    """Valid CoolProp humid-air (``HAPropsSI``) STATE-INPUT parameter names for
    ``HumidAir`` -- the subset that can fix a state (matches
    ``encomp.coolprop.HUMID_AIR_INPUTS``; output-only properties like ``Visc`` /
    ``Conductivity`` / heat capacities are excluded). Used with ``Unpack`` to
    statically check the ``**kwargs`` keys at the call site.
    """

    B: Quantity[Any, MT] | Quantity[Any, float]
    D: Quantity[Any, MT] | Quantity[Any, float]
    DewPoint: Quantity[Any, MT] | Quantity[Any, float]
    Enthalpy: Quantity[Any, MT] | Quantity[Any, float]
    Entropy: Quantity[Any, MT] | Quantity[Any, float]
    H: Quantity[Any, MT] | Quantity[Any, float]
    Hda: Quantity[Any, MT] | Quantity[Any, float]
    Hha: Quantity[Any, MT] | Quantity[Any, float]
    HumRat: Quantity[Any, MT] | Quantity[Any, float]
    Omega: Quantity[Any, MT] | Quantity[Any, float]
    P: Quantity[Any, MT] | Quantity[Any, float]
    P_w: Quantity[Any, MT] | Quantity[Any, float]
    R: Quantity[Any, MT] | Quantity[Any, float]
    RH: Quantity[Any, MT] | Quantity[Any, float]
    RelHum: Quantity[Any, MT] | Quantity[Any, float]
    S: Quantity[Any, MT] | Quantity[Any, float]
    Sda: Quantity[Any, MT] | Quantity[Any, float]
    Sha: Quantity[Any, MT] | Quantity[Any, float]
    T: Quantity[Any, MT] | Quantity[Any, float]
    T_db: Quantity[Any, MT] | Quantity[Any, float]
    T_dp: Quantity[Any, MT] | Quantity[Any, float]
    T_wb: Quantity[Any, MT] | Quantity[Any, float]
    Tdb: Quantity[Any, MT] | Quantity[Any, float]
    Tdp: Quantity[Any, MT] | Quantity[Any, float]
    Twb: Quantity[Any, MT] | Quantity[Any, float]
    V: Quantity[Any, MT] | Quantity[Any, float]
    Vda: Quantity[Any, MT] | Quantity[Any, float]
    Vha: Quantity[Any, MT] | Quantity[Any, float]
    W: Quantity[Any, MT] | Quantity[Any, float]
    WetBulb: Quantity[Any, MT] | Quantity[Any, float]
    Y: Quantity[Any, MT] | Quantity[Any, float]
    psi_w: Quantity[Any, MT] | Quantity[Any, float]


# user-facing phase name (AssumedPhase, imported from encomp.coolprop) -> CoolProp phase
# index for the low-level AbstractState.specify_phase path. (The rust plugin path instead
# uses the ``phase_*`` string, resolved by encomp.coolprop._phase_from_assumed.)
_ASSUMED_PHASE_MAP: dict[str, Any] = {
    "gas": _cp.iphase_gas,
    "liquid": _cp.iphase_liquid,
    "supercritical": _cp.iphase_supercritical,
    "supercritical_gas": _cp.iphase_supercritical_gas,
    "supercritical_liquid": _cp.iphase_supercritical_liquid,
    "twophase": _cp.iphase_twophase,
}

# backends that ignore AbstractState.specify_phase (region-explicit), so assuming a
# phase has no effect and should not switch evaluation to the slower low-level loop
_PHASE_IGNORING_BACKENDS = frozenset({"IF97"})

# Eager numpy / pl.Series inputs with at least this many elements are evaluated
# through the GIL-free rust plugin (faster and lower peak memory than CoolProp's
# vectorized PropsSI, and much faster than the low-level per-row Python loop for an
# assumed phase / composition). Scalars and smaller arrays stay on the Python
# CoolProp path, where the plugin's fixed dispatch overhead would dominate. The two
# paths are verified to agree on dtype, value, and NaN/inf handling (test_fluids).
EAGER_PLUGIN_MIN_SIZE = 1000

# Caches the CONSTRUCTED pl.Expr per (fluid config, output, input-expr digests), so
# repeated evaluations of the same property return the IDENTICAL expression object.
# That object identity is the point: callers that deduplicate expressions by id()
# (e.g. DAG builders that level-batch shared nodes into with_columns stages) see one
# node instead of many. Polars' own CSE is no help here -- it never touches plugin
# expressions (measured: duplicate plugin nodes each run their own flash, whether
# they are the same object or structurally identical fresh ones; plain expressions
# in the same plan do get CSE'd).
_EXPR_EVALUATION_CACHE_MAX_SIZE = 1024
_EXPR_EVALUATION_CACHE: OrderedDict[tuple[Any, ...], pl.Expr] = OrderedDict()
_EXPR_EVALUATION_CACHE_LOCK = Lock()


def clear_expr_evaluation_cache() -> None:
    with _EXPR_EVALUATION_CACHE_LOCK:
        _EXPR_EVALUATION_CACHE.clear()


def _expr_cache_digest(expr: pl.Expr) -> str:
    serialized = expr.meta.undo_aliases().meta.serialize(format="json")
    return hashlib.blake2s(serialized.encode(), digest_size=16).hexdigest()


def _get_expr_evaluation_cache_key(
    fluid: "CoolPropFluid[Any]",
    output: CProperty,
    points: tuple[tuple[CProperty, pl.Expr], ...],
) -> tuple[Any, ...]:
    fluid_any = cast(Any, fluid)

    comp = fluid_any._composition
    # _composition holds fixed float fractions (per-row varying composition is rejected
    # at construction), so repr is a stable, cheap cache key
    comp_key = None if comp is None else tuple((sp, repr(v)) for sp, v in comp.items())

    return (
        type(fluid),
        fluid.name,
        fluid_any._assumed_phase,
        comp_key,
        output,
        bool(fluid_any._append_name_to_cp_inputs),
        bool(fluid_any._evaluate_invalid_separately),
        id(type(fluid).BACKEND["backend"]),
        tuple((prop, _expr_cache_digest(expr)) for prop, expr in points),
    )


def _expr_evaluation_cache_get(key: tuple[Any, ...]) -> pl.Expr | None:
    with _EXPR_EVALUATION_CACHE_LOCK:
        expr = _EXPR_EVALUATION_CACHE.get(key)
        if expr is None:
            return None
        _EXPR_EVALUATION_CACHE.move_to_end(key)
        return expr


def _expr_evaluation_cache_set(key: tuple[Any, ...], expr: pl.Expr) -> None:
    with _EXPR_EVALUATION_CACHE_LOCK:
        _EXPR_EVALUATION_CACHE[key] = expr
        _EXPR_EVALUATION_CACHE.move_to_end(key)
        if len(_EXPR_EVALUATION_CACHE) > _EXPR_EVALUATION_CACHE_MAX_SIZE:
            _EXPR_EVALUATION_CACHE.popitem(last=False)


class CoolPropFluid(ABC, Generic[MT]):  # noqa: UP046
    name: CName
    points: list[tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]]]

    BACKEND: ClassVar[dict[Literal["backend"], Callable[..., float | Numpy1DArray]]] = {"backend": PropsSI}

    # PropsSI expects the fluid name as the first input, but not HAPropsSI
    _append_name_to_cp_inputs: bool = True

    # HAPropsSI fails if one or more inputs are incorrect,
    # PropsSI returns NaN for invalid inputs in case valid inputs are also present
    _evaluate_invalid_separately: bool = False

    # assumed phase (set via Fluid.assume_phase). When not None, property
    # evaluation switches from the high-level PropsSI backend to the low-level
    # AbstractState API with specify_phase, which skips the (for mixtures very
    # expensive) phase-stability search. None means CoolProp determines the phase.
    _assumed_phase: str | None = None

    # fixed mixture composition (set via Fluid(..., composition=...)) as MOLE
    # fractions, one float per species. When not None, evaluation uses the
    # low-level AbstractState API (PropsSI cannot take an explicit composition).
    # None means the composition is fixed in the fluid name, or the fluid is pure.
    _composition: dict[str, float] | None = None

    # True when an assumed phase or a composition is configured, i.e. evaluation
    # must use the low-level AbstractState path. A single flag so the normal
    # PropsSI path pays one attribute read, not several is-not-None checks.
    _lowlevel: bool = False

    # display only head of vector inputs in the __repr__ method
    _repr_cutoff: int = 3

    # substrings from the CoolProp error messages for when inputs are
    # not valid or not implemented (CoolProp will always raise ValueError,
    # no matter the error)
    # in case the error message does not match any of these, a warning is emitted
    COOLPROP_ERROR_MESSAGES = (
        "is not valid for keyed_output",
        "is not valid for trivial_keyed_output",
        "For now, we don't support",
        "is not implemented for this backend",
        "is only defined within the two-phase region",
        "failed ungracefully",
        "value to T_phase_determination_pure_or_pseudopure is invalid",
        "Brent's method f(b) is NAN",
        "do not bracket the root",
        "was unable to find a solution for",
        "is outside the range of validity",
    )

    PHASES: dict[float, str] = {
        0.0: "Liquid",
        5.0: "Gas",
        6.0: "Two-phase",
        3.0: "Supercritical liquid",  # P > P_crit
        2.0: "Supercritical gas",  # T > T_crit
        1.0: "Supercritical fluid",  # P > P_crit and T > T_crit
        4.0: "Critical point",
        7.0: "Unknown",
        8.0: "Not imposed",
    }

    # unit and description for properties in function PropsSI
    # (name1, name2, ...): (unit, description)
    # names are case-sensitive
    PROPERTY_MAP: dict[tuple[CProperty, ...], tuple[UnitString, str]] = {
        ("DELTA", "Delta"): ("dimensionless", "Reduced density (rho/rhoc)"),
        ("DMOLAR", "Dmolar"): ("mol/m³", "Molar density"),
        ("D", "DMASS", "Dmass"): ("kg/m³", "Mass density"),
        ("HMOLAR", "Hmolar"): ("J/mol", "Molar specific enthalpy"),
        ("H", "HMASS", "Hmass"): ("J/kg", "Mass specific enthalpy"),
        ("P",): ("Pa", "Pressure"),
        ("Q",): ("dimensionless", "Mass vapor quality"),
        ("SMOLAR", "Smolar"): ("J/mol/K", "Molar specific entropy"),
        ("S", "SMASS", "Smass"): ("J/kg/K", "Mass specific entropy"),
        ("TAU", "Tau"): ("dimensionless", "Reciprocal reduced temperature (Tc/T)"),
        ("T",): ("K", "Temperature"),
        ("UMOLAR", "Umolar"): ("J/mol", "Molar specific internal energy"),
        ("U", "UMASS", "Umass"): ("J/kg", "Mass specific internal energy"),
        ("A", "SPEED_OF_SOUND", "speed_of_sound"): ("m/s", "Speed of sound"),
        ("CONDUCTIVITY", "L", "conductivity"): ("W/m/K", "Thermal conductivity"),
        ("CP0MASS", "Cp0mass"): (
            "J/kg/K",
            "Ideal gas mass specific constant pressure specific heat",
        ),
        ("CP0MOLAR", "Cp0molar"): (
            "J/mol/K",
            "Ideal gas molar specific constant pressure specific heat",
        ),
        ("CPMOLAR", "Cpmolar"): (
            "J/mol/K",
            "Molar specific constant pressure specific heat",
        ),
        ("CVMASS", "Cvmass", "O"): (
            "J/kg/K",
            "Mass specific constant volume specific heat",
        ),
        ("CVMOLAR", "Cvmolar"): (
            "J/mol/K",
            "Molar specific constant volume specific heat",
        ),
        ("C", "CPMASS", "Cpmass"): (
            "J/kg/K",
            "Mass specific constant pressure specific heat",
        ),
        ("DIPOLE_MOMENT", "dipole_moment"): ("C*m", "Dipole moment"),
        ("GAS_CONSTANT", "gas_constant"): ("J/mol/K", "Molar gas constant"),
        ("GMOLAR_RESIDUAL", "Gmolar_residual"): (
            "J/mol",
            "Residual molar Gibbs energy",
        ),
        ("GMOLAR", "Gmolar"): ("J/mol", "Molar specific Gibbs energy"),
        ("G", "GMASS", "Gmass"): ("J/kg", "Mass specific Gibbs energy"),
        ("HELMHOLTZMASS", "Helmholtzmass"): ("J/kg", "Mass specific Helmholtz energy"),
        ("HELMHOLTZMOLAR", "Helmholtzmolar"): (
            "J/mol",
            "Molar specific Helmholtz energy",
        ),
        ("HMOLAR_RESIDUAL", "Hmolar_residual"): ("J/mol", "Residual molar enthalpy"),
        ("ISENTROPIC_EXPANSION_COEFFICIENT", "isentropic_expansion_coefficient"): (
            "dimensionless",
            "Isentropic expansion coefficient",
        ),
        ("ISOBARIC_EXPANSION_COEFFICIENT", "isobaric_expansion_coefficient"): (
            "1/K",
            "Isobaric expansion coefficient",
        ),
        ("ISOTHERMAL_COMPRESSIBILITY", "isothermal_compressibility"): (
            "1/Pa",
            "Isothermal compressibility",
        ),
        ("I", "SURFACE_TENSION", "surface_tension"): ("N/m", "Surface tension"),
        (
            "M",
            "MOLARMASS",
            "MOLAR_MASS",
            "MOLEMASS",
            "molar_mass",
            "molarmass",
            "molemass",
        ): ("kg/mol", "Molar mass"),
        ("PCRIT", "P_CRITICAL", "Pcrit", "p_critical", "pcrit"): (
            "Pa",
            "Pressure at the critical point",
        ),
        ("PHASE", "Phase"): ("dimensionless", "Phase index as a float"),
        ("PMAX", "P_MAX", "P_max", "pmax"): ("Pa", "Maximum pressure limit"),
        ("PMIN", "P_MIN", "P_min", "pmin"): ("Pa", "Minimum pressure limit"),
        ("PRANDTL", "Prandtl"): ("dimensionless", "Prandtl number"),
        ("PTRIPLE", "P_TRIPLE", "p_triple", "ptriple"): (
            "Pa",
            "Pressure at the triple point (pure only)",
        ),
        ("P_REDUCING", "p_reducing"): ("Pa", "Pressure at the reducing point"),
        ("RHOCRIT", "RHOMASS_CRITICAL", "rhocrit", "rhomass_critical"): (
            "kg/m³",
            "Mass density at critical point",
        ),
        ("RHOMASS_REDUCING", "rhomass_reducing"): (
            "kg/m³",
            "Mass density at reducing point",
        ),
        ("RHOMOLAR_CRITICAL", "rhomolar_critical"): (
            "mol/m³",
            "Molar density at critical point",
        ),
        ("RHOMOLAR_REDUCING", "rhomolar_reducing"): (
            "mol/m³",
            "Molar density at reducing point",
        ),
        ("SMOLAR_RESIDUAL", "Smolar_residual"): ("J/mol/K", "Residual molar entropy"),
        ("TCRIT", "T_CRITICAL", "T_critical", "Tcrit"): (
            "K",
            "Temperature at the critical point",
        ),
        ("TMAX", "T_MAX", "T_max", "Tmax"): ("K", "Maximum temperature limit"),
        ("TMIN", "T_MIN", "T_min", "Tmin"): ("K", "Minimum temperature limit"),
        ("TTRIPLE", "T_TRIPLE", "T_triple", "Ttriple"): (
            "K",
            "Temperature at the triple point",
        ),
        ("T_FREEZE", "T_freeze"): (
            "K",
            "Freezing temperature for incompressible solutions",
        ),
        ("T_REDUCING", "T_reducing"): ("K", "Temperature at the reducing point"),
        ("V", "VISCOSITY", "viscosity"): ("Pa*s", "Viscosity"),
        ("Z",): ("dimensionless", "Compressibility factor"),
    }

    ALL_PROPERTIES: set[CProperty] = set(flatten(list(PROPERTY_MAP)))
    STATE_INPUTS: ClassVar[frozenset[str]] = frozenset(FLUID_INPUTS)
    REPR_PROPERTIES: tuple[tuple[CProperty, str], ...] = (
        ("P", ".0f"),
        ("T", ".1f"),
        ("D", ".1f"),
        ("V", ".2g"),
    )

    # Preferred public return units. Values are evaluated in CoolProp's canonical
    # units from PROPERTY_MAP first, then converted here by canonical property key.
    RETURN_UNITS: dict[CProperty, UnitString] = {
        "P": "kPa",
        "PCRIT": "kPa",
        "PMAX": "kPa",
        "PMIN": "kPa",
        "PTRIPLE": "kPa",
        "P_REDUCING": "kPa",
        "T": "°C",
        "TCRIT": "°C",
        "TMAX": "°C",
        "TMIN": "°C",
        "TTRIPLE": "°C",
        "T_FREEZE": "°C",
        "T_REDUCING": "°C",
        "V": "cP",
        "H": "kJ/kg",
        "U": "kJ/kg",
        "S": "kJ/kg/K",
        "C": "kJ/kg/K",
    }

    @property
    def _mt(self) -> type[MT]:
        candidates = [type(n[1].m) for n in self.points]
        candidates = [n for n in candidates if n is not float]

        if not candidates:
            return cast(type[MT], float)

        magnitude_type = candidates[0]

        return cast(type[MT], magnitude_type)

    @abstractmethod
    def __init__(self, name: CName, **kwargs: Quantity[Any, MT] | Quantity[Any, float]) -> None:
        """
        Base class that represents a fluid (pure or mixture, gas or liquid).
        Uses *CoolProp* as backend to determine fluid properties.

        This class should not be used directly, since it does not contain a fixed
        point to determine fluid properties
        (temperature, pressure, enthalpy, entropy, ...).
        Define a subclass of :py:class:`encomp.fluids.CoolPropFluid` that implements
        the ``__init__`` method
        (this method must set instance attributes ``name`` and ``points``).

        Fluid names for pure fluids are not case-sensitive, but the mixture names are.
        The following fluid names are recognized by CoolProp:

        **Pure**

        .. code:: none

            1-Butene,Acetone,Air,Ammonia,Argon,Benzene,CarbonDioxide,CarbonMonoxide,
            CarbonylSulfide,CycloHexane,CycloPropane,Cyclopentane,D4,D5,D6,Deuterium,
            Dichloroethane,DiethylEther,DimethylCarbonate,DimethylEther,Ethane,
            Ethanol,EthylBenzene,Ethylene,EthyleneOxide,Fluorine,HFE143m,HeavyWater,
            Helium,Hydrogen,HydrogenChloride,HydrogenSulfide,IsoButane,IsoButene,
            Isohexane,Isopentane,Krypton,MD2M,MD3M,MD4M,MDM,MM,Methane,Methanol,
            MethylLinoleate,MethylLinolenate,MethylOleate,MethylPalmitate,MethylStearate,
            Neon,Neopentane,Nitrogen,NitrousOxide,Novec649,OrthoDeuterium,OrthoHydrogen,
            Oxygen,ParaDeuterium,ParaHydrogen,Propylene,Propyne,R11,R113,R114,R115,
            R116,R12,R123,R1233zd(E),R1234yf,R1234ze(E),R1234ze(Z),R124,R1243zf,
            R125,R13,R134a,R13I1,R14,R141b,R142b,R143a,R152A,R161,R21,R218,R22,R227EA,
            R23,R236EA,R236FA,R245ca,R245fa,R32,R365MFC,R40,R404A,R407C,R41,R410A,
            R507A,RC318,SES36,SulfurDioxide,SulfurHexafluoride,Toluene,Water,Xenon,
            cis-2-Butene,m-Xylene,n-Butane,n-Decane,n-Dodecane,n-Heptane,n-Hexane,
            n-Nonane,n-Octane,n-Pentane,n-Propane,n-Undecane,o-Xylene,p-Xylene,trans-2-Butene

        **Incompressible pure**

        .. code:: none

            INCOMP::AS10,INCOMP::AS20,INCOMP::AS30,INCOMP::AS40,INCOMP::AS55,INCOMP::DEB,
            INCOMP::DSF,INCOMP::DowJ,INCOMP::DowJ2,INCOMP::DowQ,INCOMP::DowQ2,INCOMP::HC10,
            INCOMP::HC20,INCOMP::HC30,INCOMP::HC40,INCOMP::HC50,INCOMP::HCB,INCOMP::HCM,
            INCOMP::HFE,INCOMP::HFE2,INCOMP::HY20,INCOMP::HY30,INCOMP::HY40,INCOMP::HY45,
            INCOMP::HY50,INCOMP::NBS,INCOMP::NaK,INCOMP::PBB,INCOMP::PCL,INCOMP::PCR,
            INCOMP::PGLT,INCOMP::PHE,INCOMP::PHR,INCOMP::PLR,INCOMP::PMR,INCOMP::PMS1,
            INCOMP::PMS2,INCOMP::PNF,INCOMP::PNF2,INCOMP::S800,INCOMP::SAB,INCOMP::T66,
            INCOMP::T72,INCOMP::TCO,INCOMP::TD12,INCOMP::TVP1,INCOMP::TVP1869,INCOMP::TX22,
            INCOMP::TY10,INCOMP::TY15,INCOMP::TY20,INCOMP::TY24,INCOMP::Water,INCOMP::XLT,
            INCOMP::XLT2,INCOMP::ZS10,INCOMP::ZS25,INCOMP::ZS40,INCOMP::ZS45,INCOMP::ZS55

        **Incompressible mixtures**

        .. code:: none

            INCOMP::FRE,INCOMP::IceEA,INCOMP::IceNA,INCOMP::IcePG,INCOMP::LiBr,INCOMP::MAM,
            INCOMP::MAM2,INCOMP::MCA,INCOMP::MCA2,INCOMP::MEA,INCOMP::MEA2,INCOMP::MEG,
            INCOMP::MEG2,INCOMP::MGL,INCOMP::MGL2,INCOMP::MITSW,INCOMP::MKA,INCOMP::MKA2,
            INCOMP::MKC,INCOMP::MKC2,INCOMP::MKF,INCOMP::MLI,INCOMP::MMA,INCOMP::MMA2,
            INCOMP::MMG,INCOMP::MMG2,INCOMP::MNA,INCOMP::MNA2,INCOMP::MPG,INCOMP::MPG2,
            INCOMP::VCA,INCOMP::VKC,INCOMP::VMA,INCOMP::VMG,INCOMP::VNA,INCOMP::AEG,
            INCOMP::AKF,INCOMP::AL,INCOMP::AN,INCOMP::APG,INCOMP::GKN,INCOMP::PK2,
            INCOMP::PKL,INCOMP::ZAC,INCOMP::ZFC,INCOMP::ZLC,INCOMP::ZM,INCOMP::ZMC

        **Mixtures**

        .. code:: none

            AIR.MIX,AMARILLO.MIX,Air.mix,Amarillo.mix,EKOFISK.MIX,Ekofisk.mix,GULFCOAST.MIX,
            GULFCOASTGAS(NIST1).MIX,GulfCoast.mix,GulfCoastGas(NIST1).mix,HIGHCO2.MIX,
            HIGHN2.MIX,HighCO2.mix,HighN2.mix,NATURALGASSAMPLE.MIX,NaturalGasSample.mix,
            R401A.MIX,R401A.mix,R401B.MIX,R401B.mix,R401C.MIX,R401C.mix,R402A.MIX,R402A.mix,
            R402B.MIX,R402B.mix,R403A.MIX,R403A.mix,R403B.MIX,R403B.mix,R404A.MIX,R404A.mix,
            R405A.MIX,R405A.mix,R406A.MIX,R406A.mix,R407A.MIX,R407A.mix,R407B.MIX,R407B.mix,
            R407C.MIX,R407C.mix,R407D.MIX,R407D.mix,R407E.MIX,R407E.mix,R407F.MIX,R407F.mix,
            R408A.MIX,R408A.mix,R409A.MIX,R409A.mix,R409B.MIX,R409B.mix,R410A.MIX,R410A.mix,
            R410B.MIX,R410B.mix,R411A.MIX,R411A.mix,R411B.MIX,R411B.mix,R412A.MIX,R412A.mix,
            R413A.MIX,R413A.mix,R414A.MIX,R414A.mix,R414B.MIX,R414B.mix,R415A.MIX,R415A.mix,
            R415B.MIX,R415B.mix,R416A.MIX,R416A.mix,R417A.MIX,R417A.mix,R417B.MIX,R417B.mix,
            R417C.MIX,R417C.mix,R418A.MIX,R418A.mix,R419A.MIX,R419A.mix,R419B.MIX,R419B.mix,
            R420A.MIX,R420A.mix,R421A.MIX,R421A.mix,R421B.MIX,R421B.mix,R422A.MIX,R422A.mix,
            R422B.MIX,R422B.mix,R422C.MIX,R422C.mix,R422D.MIX,R422D.mix,R422E.MIX,R422E.mix,
            R423A.MIX,R423A.mix,R424A.MIX,R424A.mix,R425A.MIX,R425A.mix,R426A.MIX,R426A.mix,
            R427A.MIX,R427A.mix,R428A.MIX,R428A.mix,R429A.MIX,R429A.mix,R430A.MIX,R430A.mix,
            R431A.MIX,R431A.mix,R432A.MIX,R432A.mix,R433A.MIX,R433A.mix,R433B.MIX,R433B.mix,
            R433C.MIX,R433C.mix,R434A.MIX,R434A.mix,R435A.MIX,R435A.mix,R436A.MIX,R436A.mix,
            R436B.MIX,R436B.mix,R437A.MIX,R437A.mix,R438A.MIX,R438A.mix,R439A.MIX,R439A.mix,
            R440A.MIX,R440A.mix,R441A.MIX,R441A.mix,R442A.MIX,R442A.mix,R443A.MIX,R443A.mix,
            R444A.MIX,R444A.mix,R444B.MIX,R444B.mix,R445A.MIX,R445A.mix,R446A.MIX,R446A.mix,
            R447A.MIX,R447A.mix,R448A.MIX,R448A.mix,R449A.MIX,R449A.mix,R449B.MIX,R449B.mix,
            R450A.MIX,R450A.mix,R451A.MIX,R451A.mix,R451B.MIX,R451B.mix,R452A.MIX,R452A.mix,
            R453A.MIX,R453A.mix,R454A.MIX,R454A.mix,R454B.MIX,R454B.mix,R500.MIX,R500.mix,
            R501.MIX,R501.mix,R502.MIX,R502.mix,R503.MIX,R503.mix,R504.MIX,R504.mix,R507A.MIX,
            R507A.mix,R508A.MIX,R508A.mix,R508B.MIX,R508B.mix,R509A.MIX,R509A.mix,R510A.MIX,
            R510A.mix,R511A.MIX,R511A.mix,R512A.MIX,R512A.mix,R513A.MIX,R513A.mix,
            TYPICALNATURALGAS.MIX,TypicalNaturalGas.mix


        Refer to the CoolProp documentation for more information:

        - http://www.coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids
        - http://www.coolprop.org/fluid_properties/Mixtures.html#binary-pairs
        - http://www.coolprop.org/fluid_properties/Incompressibles.html#the-different-fluids
        - table-of-inputs-outputs-to-hapropssi
        - http://www.coolprop.org/coolprop/HighLevelAPI.html
        - http://www.coolprop.org/fluid_properties/HumidAir.html


        The names ``Water`` and ``HEOS::Water``
        uses the formulation defined by IAPWS-95.
        Use the name ``IF97::Water`` to instead use the slightly faster
        (but less accurate) IAPWS-97 formulation.
        In most cases, the difference between IAPWS-95 and IAPWS-97 is negligible.
        Read CoolProp's `introduction
        <http://www.coolprop.org/fluid_properties/IF97.html>`_
        about water properties for more information.


        Parameters
        ----------
        name : CName
            The name of the fluid, same name as is used by CoolProp.
            Include the ``INCOMP::`` prefix and potential mixing ratio
            for incompressible mixtures.

            Examples:

                - ``INCOMP::MITSW[0.05]``: seawater with 5 mass-percent salt.
                - ``INCOMP::MPG[0.5]``: 50 % propylene glycol
                - ``INCOMP::T66``: Therminol 66 (https://www.therminol.com/product/71093438)

        """

    @classmethod
    def get_prop_key(cls, prop: str) -> tuple[CProperty, ...]:
        if prop not in cls.ALL_PROPERTIES:
            raise ValueError(f'Property "{prop}" is not a valid CoolProp property name')

        for names in cls.PROPERTY_MAP:
            if prop in names:
                return names

        raise ValueError(f'Property "{prop}" is not a valid CoolProp property name')

    @classmethod
    def get_coolprop_unit(cls, prop: CProperty) -> Unit:
        key = cls.get_prop_key(prop)

        if key in cls.PROPERTY_MAP:
            unit_str = cls.PROPERTY_MAP[key][0]
            return Quantity.get_unit(unit_str)

        raise ValueError(f'Could not get unit, key "{key}" does not exist')

    @classmethod
    def is_valid_prop(cls, prop: str) -> bool:
        try:
            cls.get_prop_key(prop)
            return True

        except ValueError:
            return False

    @classmethod
    def check_inputs(cls, kwargs: Mapping[str, object]) -> None:
        invalid = [key for key in kwargs if not cls.is_valid_prop(key)]

        if len(invalid):
            raise ValueError(
                f"Invalid CoolProp property name{'s' if len(invalid) > 1 else ''}: "
                f"{', '.join(invalid)}\n"
                f"Valid names:\n{', '.join(sorted(cls.ALL_PROPERTIES))}"
            )

        output_only = [key for key in kwargs if key not in cls.STATE_INPUTS]

        if len(output_only):
            raise ValueError(
                f"Invalid CoolProp state input{'s' if len(output_only) > 1 else ''}: "
                f"{', '.join(output_only)}\n"
                "These properties are outputs only and cannot fix a state.\n"
                f"Valid state inputs:\n{', '.join(sorted(cls.STATE_INPUTS))}"
            )

    def _build_points(
        self, kwargs: Mapping[str, object]
    ) -> list[tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]]]:
        """Validate raw ``**kwargs`` keys and narrow each to the strict ``CProperty``
        Literal union, producing the typed ``points`` list.

        The ``Unpack[TypedDict]`` ``**kwargs`` degrades its values to ``object``
        when iterated, so the property *name* is narrowed via the ``TypeGuard``
        (no cast) while the *value* is cast back to the ``Quantity`` union."""
        points: list[tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]]] = []
        for name, qty in kwargs.items():
            if is_fluid_param(name) or is_humid_air_param(name):
                points.append((name, cast("Quantity[Any, MT] | Quantity[Any, float]", qty)))
            else:
                raise ValueError(f'Invalid CoolProp property name: "{name}"')
        return points

    @classmethod
    def describe(cls, prop: CProperty) -> str:
        key = cls.get_prop_key(prop)

        if key in cls.PROPERTY_MAP:
            unit_str, description = cls.PROPERTY_MAP[key]
            unit = Quantity.get_unit(unit_str)
            unit_repr = f"{unit:~P}"

            if not unit_repr:
                unit_repr = "dimensionless"

            return f"{', '.join(key)}: {description} [{unit_repr}]"

        raise ValueError(f'Could not get description, key "{key}" does not exist')

    @classmethod
    def search(cls, inp: str) -> list[str]:
        matches: list[str] = []

        for key in cls.PROPERTY_MAP:
            description = cls.describe(key[0])
            if inp.lower() in description.lower():
                matches.append(description)

        return matches

    def _warn_coolprop_nan(self, prop: CProperty, msg: str) -> None:
        if not SETTINGS.ignore_coolprop_warnings:
            _LOGGER.warning(f'CoolProp could not calculate "{prop}" for fluid "{self.name}", output is NaN: {msg}')

    def check_exception(self, prop: CProperty, e: ValueError) -> None:
        msg = str(e)

        # this error occurs in case the input values are outside
        # the allowable range for this property
        # in this case the return value will be NaN, no exception is raised
        if "No outputs were able to be calculated" in msg or "is outside the range of validity" in msg:
            self._warn_coolprop_nan(prop, msg)
            return

        # if CoolProp has not implemented prop as output, return NaN
        if any(n in msg for n in self.COOLPROP_ERROR_MESSAGES):
            return

        if "Output string is invalid" in msg:
            return

        if "Initialize failed for backend" in msg:
            raise ValueError(
                f"Fluid '{self.name}' could not be initialized, ensure that the name is a valid CoolProp fluid name"
            ) from e

        self._warn_coolprop_nan(prop, msg)

    # ------------------------------------------------------------------ #
    # Low-level AbstractState path. Used whenever an assumed phase and/or a
    # ``composition=`` dict is set (PropsSI cannot do either). The composition is
    # fixed (baked into the name, or passed as floats), so the AbstractState is
    # built once and the loop only flashes each (P, T) row.
    # ------------------------------------------------------------------ #

    def _constant_state(self) -> Any:  # noqa: ANN401 - CoolProp AbstractState is untyped
        # name/composition parsing is delegated to resolve_fluid_spec (the single source);
        # here we just build the AbstractState and pin the assumed phase.
        backend, fluids, fractions = resolve_fluid_spec(self.name, self._composition)
        state = _cp.AbstractState(backend, fluids)
        if fractions is not None:
            state.set_mole_fractions(fractions)
        if self._assumed_phase is not None:
            state.specify_phase(_ASSUMED_PHASE_MAP[self._assumed_phase])
        return state

    def _lowlevel_loop_constant(
        self, output: CProperty, p1_name: CProperty, p1_arr: Numpy1DArray, p2_name: CProperty, p2_arr: Numpy1DArray
    ) -> Numpy1DArray:
        state = self._constant_state()
        k1 = _cp.get_parameter_index(p1_name)
        k2 = _cp.get_parameter_index(p2_name)
        out_idx = _cp.get_parameter_index(output)
        generate_update_pair = _cp.generate_update_pair
        update = state.update
        keyed_output = state.keyed_output

        out = np.full(p1_arr.size, np.nan)
        for i in range(p1_arr.size):
            v1 = p1_arr[i]
            v2 = p2_arr[i]
            if not (np.isfinite(v1) and np.isfinite(v2)):
                continue
            try:
                pair, a, b = generate_update_pair(k1, v1, k2, v2)
                update(pair, a, b)
                out[i] = keyed_output(out_idx)
            except ValueError as e:
                self.check_exception(output, e)

        out[~np.isfinite(out)] = np.nan
        return out

    def _evaluate_lowlevel(
        self, output: CProperty, points: tuple[tuple[CProperty, float | Numpy1DArray | pl.Expr], ...]
    ) -> float | Numpy1DArray | pl.Expr:
        p1_name = points[0][0]
        p2_name = points[1][0]

        def compute(p1: Numpy1DArray, p2: Numpy1DArray) -> Numpy1DArray:
            return self._lowlevel_loop_constant(output, p1_name, p1, p2_name, p2)

        return self._lowlevel_dispatch(output, points, compute)

    def _lowlevel_dispatch(
        self,
        output: CProperty,
        points: tuple[tuple[CProperty, float | Numpy1DArray | pl.Expr], ...],
        compute: Callable[[Numpy1DArray, Numpy1DArray], Numpy1DArray],
    ) -> float | Numpy1DArray | pl.Expr:
        points = tuple((name, self._reduce_single_element(mag)) for name, mag in points)
        p1 = points[0][1]
        p2 = points[1][1]
        all_vals: list[float | Numpy1DArray | pl.Expr] = [p1, p2]

        if any(isinstance(m, pl.Expr) for m in all_vals):
            if any(isinstance(m, np.ndarray) for m in all_vals):
                raise TypeError("cannot mix numpy array and pl.Expr inputs")
            return self._lowlevel_expr(output, points)

        if all(isinstance(m, float) for m in all_vals):
            out = compute(np.array([cast("float", p1)]), np.array([cast("float", p2)]))
            return float(out[0])

        arrays = [m for m in all_vals if isinstance(m, np.ndarray)]
        shape = arrays[0].shape
        n = arrays[0].size
        if any(a.shape != shape for a in arrays):
            raise ValueError("all array inputs must share the same shape")

        def to_arr(m: float | Numpy1DArray | pl.Expr) -> Numpy1DArray:
            if isinstance(m, np.ndarray):
                return m.flatten().astype(float)
            return np.full(n, cast("float", m))

        p1_arr, p2_arr = to_arr(p1), to_arr(p2)
        # large eager arrays: skip the per-row Python loop, use the rust plugin (it
        # honors the assumed phase / composition too).
        if n >= EAGER_PLUGIN_MIN_SIZE and self._rust_representable():
            named = ((points[0][0], p1_arr), (points[1][0], p2_arr))
            return self._rust_eager(output, named).reshape(shape)

        out = compute(p1_arr, p2_arr)
        return out.reshape(shape)

    def _rust_spec(self) -> tuple[Backend, str, list[float] | None, Phase | None] | None:
        """Plugin spec ``(backend, fluids, fractions, phase)``, or None if the rust plugin
        cannot represent it (an unsupported backend). Name/composition resolution is
        delegated to :func:`encomp.coolprop.resolve_fluid_spec` -- the same function
        :func:`~encomp.coolprop.fluid` uses -- so both agree."""
        backend, fluids, fractions = resolve_fluid_spec(self.name, self._composition)
        if not is_backend(backend):
            return None
        phase: Phase | None
        if self._assumed_phase is None:
            phase = None
        else:
            phase_candidate = f"phase_{self._assumed_phase}"
            if not is_phase(phase_candidate):
                return None
            phase = phase_candidate
        return backend, fluids, fractions, phase

    def _rust_representable(self) -> bool:
        """Whether the rust plugin can evaluate this fluid's config (the output name is
        resolved by CoolProp at runtime, so it never affects representability).

        Humid air is always representable (per-row HAPropsSI in the plugin); a Fluid
        is representable unless its backend is one the plugin's spec rejects.
        """
        if self.BACKEND["backend"] is HAPropsSI:
            return True
        return self._rust_spec() is not None

    def _rust_plugin_expr(self, output: CProperty, points: tuple[tuple[str, pl.Expr], ...]) -> pl.Expr:
        """The raw encomp.coolprop plugin expr (aliased to ``output``): a
        precision-preserving dtype (Float32 in -> Float32 out) with null (not NaN) for
        failed/out-of-range rows.

        Shared by the lazy ``pl.Expr`` path (:meth:`_rust_expr`) and the eager
        large-array path (:meth:`_rust_eager`); the latter reads it via ``to_numpy()``,
        which maps null back to NaN, so its numpy masking is unchanged. Output property
        names are resolved by
        CoolProp at runtime, so any valid name works regardless of the static
        ``FluidParam`` / ``HumidAirParam`` sets. Raises if the plugin is unavailable
        (there is no map_batches fallback) or cannot represent the request.
        """
        from . import coolprop as _cprust

        if not _cprust.self_check():
            raise RuntimeError(
                "the encomp.coolprop rust plugin failed to load, but it is required to "
                "evaluate pl.Expr (lazy) inputs and large eager arrays (there is no "
                "fallback). Reinstall encomp with its compiled plugin."
            )
        names = [p[0] for p in points]
        exprs = [p[1] for p in points]
        if self.BACKEND["backend"] is HAPropsSI:
            n1, n2, n3 = names
            # the plugin reads each input's property from its (aliased) name
            expr = _cprust.humid_air(
                cast("HumidAirParam", output),
                exprs[0].alias(n1),
                exprs[1].alias(n2),
                exprs[2].alias(n3),
            )
        else:
            if self._rust_spec() is None:
                raise TypeError(
                    f"the encomp.coolprop rust plugin cannot represent this evaluation for "
                    f"fluid {self.name!r} (an unsupported backend)."
                )
            n1, n2 = names
            # pass the high-level spec straight through; coolprop.fluid resolves the
            # name + composition the same way _rust_spec does (both via resolve_fluid_spec)
            expr = _cprust.fluid(
                cast("FluidParam", output),
                exprs[0].alias(n1),
                exprs[1].alias(n2),
                name=self.name,
                assume_phase=cast("AssumedPhase | None", self._assumed_phase),
                composition=cast("Composition | None", self._composition),
            )
        return expr.alias(output)

    def _rust_expr(self, output: CProperty, points: tuple[tuple[CProperty, pl.Expr], ...]) -> pl.Expr:
        """The plugin expr for lazy ``pl.Expr`` inputs. The plugin's output dtype already
        preserves the input precision (Float32 in -> Float32 out) and emits null (not
        NaN) for failed/out-of-range rows directly, so no ``fill_nan(None)`` wrapper is
        needed: that wrapper lowers to ``when(is_not_nan).then(x).otherwise(null)``, which
        re-evaluates the whole plugin subtree (no common-subexpression elimination) and
        runs the CoolProp flash 2-3x."""
        return self._rust_plugin_expr(output, points)

    def _rust_eager(self, output: CProperty, points: tuple[tuple[str, Numpy1DArray], ...]) -> Numpy1DArray:
        """Evaluate eager numpy ``points`` through the rust plugin, returning a 1-D
        Float64 array with the *same* invalid handling as :meth:`evaluate_multiple`
        (non-finite inputs and non-finite results -> NaN). ``points`` are equal-length
        arrays; the caller reshapes the result.
        """
        names = [name for name, _ in points]
        arrs = [np.ascontiguousarray(arr, dtype=float).ravel() for _, arr in points]
        # generic column names avoid any collision between the input property names
        frame = pl.DataFrame({f"__in{i}": a for i, a in enumerate(arrs)})
        plugin_points = tuple((names[i], pl.col(f"__in{i}")) for i in range(len(arrs)))
        expr = self._rust_plugin_expr(output, plugin_points)
        out = np.array(frame.select(expr)[output].to_numpy(), dtype=float)
        # match evaluate_multiple exactly: rows with any non-finite input are NaN, and
        # any inf/_HUGE that slipped through becomes NaN.
        finite_inputs = np.logical_and.reduce([np.isfinite(a) for a in arrs])
        out[~finite_inputs] = np.nan
        out[~np.isfinite(out)] = np.nan
        return out

    def _lowlevel_expr(
        self,
        output: CProperty,
        points: tuple[tuple[CProperty, float | Numpy1DArray | pl.Expr], ...],
    ) -> pl.Expr:
        # fixed composition and/or assumed phase with pl.Expr inputs: rust-plugin only.
        expr_points: tuple[tuple[CProperty, pl.Expr], ...] = tuple(
            (name, v if isinstance(v, pl.Expr) else pl.lit(v)) for name, v in points
        )
        key = _get_expr_evaluation_cache_key(self, output, expr_points)
        cached = _expr_evaluation_cache_get(key)
        if cached is not None:
            return cached

        expr = self._rust_expr(output, expr_points)
        _expr_evaluation_cache_set(key, expr)
        return expr

    @staticmethod
    def _reduce_single_element(x: float | Numpy1DArray | pl.Expr) -> float | Numpy1DArray | pl.Expr:
        if isinstance(x, np.ndarray) and x.size == 1:
            return float(x[0])
        return x

    def evaluate_single(self, output: CProperty, *points: tuple[CProperty, float]) -> float:
        inputs = list(flatten(points))

        if self._append_name_to_cp_inputs:
            inputs.append(self.name)

        try:
            val = self.BACKEND["backend"](output, *inputs)

            if not isinstance(val, float):
                raise TypeError(f"Unexpected value type: {type(val)}, expected float")

            if val == np.inf or val == -np.inf:
                val = np.nan

            return val

        except ValueError as e:
            self.check_exception(output, e)
            return np.nan

    def evaluate_expression(self, output: CProperty, *points: tuple[CProperty, pl.Expr]) -> pl.Expr:
        # pl.Expr (lazy) inputs are evaluated only through the GIL-free rust plugin;
        # _rust_expr raises if it is unavailable (no map_batches fallback).
        key = _get_expr_evaluation_cache_key(self, output, points)
        cached_expr = _expr_evaluation_cache_get(key)

        if cached_expr is not None:
            return cached_expr

        expr = self._rust_expr(output, points)
        _expr_evaluation_cache_set(key, expr)
        return expr

    def evaluate_multiple_separately(
        self,
        output: CProperty,
        props: list[CProperty],
        arrs_flat_masked: list[Numpy1DArray],
        N: int,
    ) -> Numpy1DArray:
        vals: list[float] = []

        for i in range(N):
            arrs_flat_masked_i = [n[i] for n in arrs_flat_masked]

            inputs_i = list(flatten(list(zip(props, arrs_flat_masked_i, strict=False))))

            if self._append_name_to_cp_inputs:
                inputs_i.append(self.name)

            try:
                val_i = self.BACKEND["backend"](output, *inputs_i)

                if not isinstance(val_i, float):
                    raise TypeError(f"Unexpected value type: {type(val_i)}, expected float")

            except ValueError as e:
                self.check_exception(output, e)
                val_i = np.nan

            vals.append(val_i)

        return np.array(vals)

    def evaluate_multiple(self, output: CProperty, *points: tuple[CProperty, Numpy1DArray]) -> Numpy1DArray:
        props: list[CProperty] = [pt[0] for pt in points]
        arrs = [pt[1] for pt in points]
        shape = arrs[0].shape

        arrs_flat = [n.flatten() for n in arrs]

        mask: np.ndarray = np.logical_and.reduce([np.isfinite(n) for n in arrs_flat])

        def get_empty_like(x: np.ndarray) -> np.ndarray:
            empty = np.empty_like(x).astype(float)
            empty[:] = np.nan
            return empty

        val = get_empty_like(arrs_flat[0])

        # number of finite (not nan, inf, ...) values
        N = mask.astype(int).sum()

        if N > 0:
            arrs_flat_masked = [n[mask] for n in arrs_flat]

            inputs = list(flatten(list(zip(props, arrs_flat_masked, strict=False))))

            if self._append_name_to_cp_inputs:
                inputs.append(self.name)

            # this can fail if the numeric values
            # are *all* incorrect, for example negative pressure
            try:
                val_masked = self.BACKEND["backend"](output, *inputs)

                if isinstance(val_masked, float):
                    raise TypeError(f"Unexpected value type: {type(val_masked)}, expected np.ndarray")

            except ValueError as e:
                self.check_exception(output, e)

                if self._evaluate_invalid_separately:
                    val_masked = self.evaluate_multiple_separately(output, props, arrs_flat_masked, N)
                else:
                    val_masked = get_empty_like(arrs_flat_masked[0])

            val[mask] = val_masked

        def validate_output(x: Numpy1DArray) -> Numpy1DArray:
            x[x == np.inf] = np.nan
            x[x == -np.inf] = np.nan
            return x.reshape(shape)

        return validate_output(val)

    def evaluate(
        self, output: CProperty, *points: tuple[CProperty, float | Numpy1DArray | pl.Expr]
    ) -> float | Numpy1DArray | pl.Expr:
        if self._lowlevel:
            return self._evaluate_lowlevel(output, points)

        if all(isinstance(pt[1], float) for pt in points):
            scalar_points = [cast("tuple[CProperty, float]", n) for n in points]
            return self.evaluate_single(output, *scalar_points)

        if any(isinstance(pt[1], pl.Expr) for pt in points):
            if not all(isinstance(pt[1], (float, pl.Expr)) for pt in points):
                raise TypeError(
                    "Only pl.Expr and float inputs are supported when one or more inputs is pl.Expr, "
                    f"passed types: {[(n[0], type(n[1])) for n in points]}"
                )

            expr_points: list[tuple[CProperty, pl.Expr]] = [
                (n[0], n[1] if isinstance(n[1], pl.Expr) else pl.lit(n[1])) for n in points
            ]
            return self.evaluate_expression(output, *expr_points)

        reduced_points: tuple[tuple[CProperty, float | Numpy1DArray | pl.Expr], ...] = tuple(
            (p, self._reduce_single_element(v))
            for p, v in cast("tuple[tuple[CProperty, float | Numpy1DArray]]", points)
        )

        sizes = [v.size for _, v in reduced_points if isinstance(v, np.ndarray)]
        shapes = [v.shape for _, v in reduced_points if isinstance(v, np.ndarray)]

        # the sizes list is empty if all inputs were 1-element vectors
        if len(sizes):
            n = sizes[0]
            shape = shapes[0]

            # 1-length vectors were converted to float, so this error will be relevant
            if len(set(sizes)) != 1:
                raise ValueError(f"All inputs must have the same size, passed {reduced_points} with sizes {sizes}")

            if len(set(shapes)) != 1:
                raise ValueError(f"All inputs must have the same shape, passed {reduced_points} with shapes {shapes}")
        else:
            n = 1
            shape = (1,)

        def expand_scalars(x: float | np.ndarray) -> Numpy1DArray:
            if isinstance(x, np.ndarray):
                return x

            return np.repeat(x, n).astype(float).reshape(shape)

        points_arr: tuple[tuple[CProperty, Numpy1DArray], ...] = tuple(
            (p, expand_scalars(cast(Any, v))) for p, v in reduced_points
        )

        # large eager arrays go through the GIL-free rust plugin (faster + leaner);
        # smaller arrays stay on the vectorized PropsSI path (lower fixed overhead).
        if n >= EAGER_PLUGIN_MIN_SIZE and self._rust_representable():
            return self._rust_eager(output, points_arr).reshape(shape)

        return self.evaluate_multiple(output, *points_arr)

    def _eager_series_output_dtype(self) -> pl.DataType:
        # eager pl.Series output preserves polars precision: Float32 only when every
        # pl.Series input is Float32, else Float64. Scalar (float) inputs are neutral.
        dtypes = [p[1].m.dtype for p in self.points if isinstance(p[1].m, pl.Series)]
        if dtypes and all(dt == pl.Float32 for dt in dtypes):
            return pl.Float32()
        return pl.Float64()

    def construct_quantity(
        self, val: float | Numpy1DArray | pl.Expr, output: CProperty, convert_magnitude: bool = True
    ) -> Quantity[Any, MT]:
        unit_output = self.get_coolprop_unit(output)

        # the dimensionality is not known until runtime
        qty = Quantity(cast(MT, val), unit_output)

        key = self.get_prop_key(output)

        if len(key) > 0 and key[0] in self.RETURN_UNITS:
            ret_unit = self.RETURN_UNITS[key[0]]
            qty.ito(ret_unit)

        if convert_magnitude:
            qty = qty.astype(self._mt)

        if isinstance(qty.m, pl.Series):
            # missing values surface as null, never NaN (the library's single sentinel)
            qty.m = qty.m.fill_nan(None)
            qty.m = cast(Any, qty.m).cast(self._eager_series_output_dtype())

        return cast("Quantity[Any, MT]", qty)

    def to_numeric_correct_unit(
        self, prop: CProperty, qty: Quantity[Any, MT] | Quantity[Any, float]
    ) -> float | Numpy1DArray | pl.Expr:
        unit = self.get_coolprop_unit(prop)

        try:
            m = qty.to(unit).m
        except DimensionalityError as e:
            raise ExpectedDimensionalityError(
                f'CoolProp input for property "{prop}" is incorrect. '
                f"expected {unit} ({unit.dimensionality}), but passed "
                f"{qty.u} ({qty.dimensionality})"
            ) from e

        if isinstance(m, (float, int, pl.Expr)):
            return m
        elif isinstance(m, pl.Series):
            return m.to_numpy()
        elif isinstance(m, list):
            return np.array(m)
        else:
            return m

    def get(
        self,
        output: CProperty,
        points: list[tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]]] | None = None,
        convert_magnitude: bool = True,
    ) -> Quantity[Any, MT]:
        """
        Wraps the CoolProp backend function (``PropsSI``, or ``HAPropsSI``
        for :py:class:`encomp.fluids.HumidAir`), handles input
        and output with :py:class:`encomp.units.Quantity` objects.

        Parameters
        ----------
        output : CProperty
            Name of the output property
        points : list[tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]]] | None
            Fixed state variables: name and value of the property.
            The number of points must match the number expected
            by the CoolProp backend function.
            If None, the points from the ``__init__`` method are used
        convert_magnitude : bool
            Whether to convert the output to the same magnitude type as the input,
            by default True

        Returns
        -------
        Quantity[Any, MT]
            Quantity representing the output property
        """

        if points is None:
            points = self.points

        points_numeric: list[tuple[CProperty, float | Numpy1DArray | pl.Expr]] = [
            (pt[0], self.to_numeric_correct_unit(*pt)) for pt in points
        ]
        val = self.evaluate(output, *points_numeric)

        return self.construct_quantity(val, output, convert_magnitude=convert_magnitude)

    def __getattr__(self, attr: str) -> Quantity[Any, MT]:
        # __getattr__ is Python's fallback for any attribute name (str by protocol);
        # narrow to a known property before delegating to the strictly-typed get().
        if (is_fluid_param(attr) or is_humid_air_param(attr)) and attr in self.ALL_PROPERTIES:
            return self.get(attr)
        raise AttributeError(attr)

    def _get_repr(self, prop: CProperty, fmt: str) -> str:
        if all(isinstance(n[1].m, (float, int)) for n in self.points):
            return f"{self.get(prop).astype(float):{fmt}}"

        if any(isinstance(n[1].m, pl.Expr) for n in self.points):
            return "<pl.Expr>"

        vector_inputs: list[tuple[CProperty, Quantity[Any, Numpy1DArray] | Quantity[Any, pl.Series]]] = [
            (n[0], cast(Quantity[Any, Numpy1DArray] | Quantity[Any, pl.Series], n[1]))
            for n in self.points
            if isinstance(n[1].m, (np.ndarray, pl.Series))
        ]

        is_cutoff = max(len(n[1].m) for n in vector_inputs) > self._repr_cutoff

        def _get_cutoff_qty(q: Quantity[Any, Any]) -> Quantity[Any, Any]:
            return Quantity(q.m[: self._repr_cutoff], q.u)

        vector_inputs_cutoff: list[tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]]] = [
            (n[0], _get_cutoff_qty(n[1])) for n in vector_inputs
        ]

        # add optional scalar points also
        covered = {name for name, _ in vector_inputs_cutoff}
        vector_inputs_cutoff += [pt for pt in self.points if pt[0] not in covered]

        qty = self.get(prop, points=vector_inputs_cutoff, convert_magnitude=False).astype(Numpy1DArray)

        qty_formatted = f"{qty:{fmt}}"

        if is_cutoff:
            head, tail = qty_formatted.split("]", 1)
            qty_formatted = f"{head} ...]{tail}"

        return qty_formatted

    def _repr_properties(self) -> str:
        try:
            return ", ".join(f"{p}={self._get_repr(p, fmt)}" for p, fmt in self.REPR_PROPERTIES)
        except Exception as e:
            return f"invalid: {e}"


class Fluid(CoolPropFluid[MT]):
    def __init__(
        self,
        name: CName,
        *,
        composition: Composition | None = None,
        **kwargs: Unpack[FluidState[MT]],
    ) -> None:
        """
        Represents a fluid at a fixed state, for example at a
        specific temperature and pressure.

        Parameters
        ----------
        name : CName
            Name of the fluid. With ``composition`` this is the backend only
            (e.g. ``"HEOS"``); otherwise the full CoolProp name, optionally with
            fixed fractions (e.g. ``"HEOS::CO2[0.5]&O2[0.5]"``).
        composition : Composition | None
            Fixed mixture composition as ``{species: mole fraction}`` (like CoolProp's
            ``"HEOS::CO2[0.5]&O2[0.5]"`` name syntax). Mole fractions must sum to 1.
            Cannot be combined with fractions baked into ``name``, and requires a mixture
            backend (e.g. HEOS). Pair with :meth:`assume_phase` for speed. Per-row varying
            composition is not supported -- loop and build one Fluid per composition.
            (For an incompressible mixture instead use the name concentration, e.g.
            ``"INCOMP::MEG[0.5]"`` -- 50 % glycol, on the fluid's own mass/volume basis.)
        kwargs: Quantity[Any, MT]
            Values for the two fixed points. The name of the keyword argument is the
            CoolProp property name.
        """

        self.check_inputs(kwargs)

        if len(kwargs) != 2:
            raise ValueError(f"Exactly two fixed points are required, passed {list(kwargs)}")

        if composition is not None:
            self._init_composition(name, composition)
        else:
            self.name = name

        points = self._build_points(kwargs)

        self.point_1: tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]] = points[0]
        self.point_2: tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]] = points[1]

        self.points = [self.point_1, self.point_2]

    def _init_composition(self, name: CName, composition: Composition) -> None:
        # all name + composition parsing/validation lives in resolve_fluid_spec (the single
        # source); store the canonical name + resolved fractions for the low-level/rust paths
        backend, fluids, fractions = resolve_fluid_spec(name, composition)
        assert fractions is not None  # a composition= dict always resolves to a mixture
        self.name = f"{backend}::{fluids}"
        self._composition = dict(zip(fluids.split("&"), fractions, strict=True))
        self._lowlevel = True

    def assume_phase(self, phase: AssumedPhase | None) -> Self:
        """Force the equation of state to assume ``phase``, skipping CoolProp's
        own phase determination. This is a *speed* tool, not a validation tool.

        With ``P, T`` inputs CoolProp normally runs a phase-stability search to
        decide whether the state is single- or two-phase. For HEOS/GERG mixtures
        that search dominates the cost (~5 ms/point); assuming a phase you
        already know skips it and switches evaluation to the low-level
        ``AbstractState`` API -- on the order of 100-1000x faster.

        Important caveats:

        * **Only the HEOS/GERG backends honour it.** ``IF97`` (the default for
          :class:`Water`) is region-explicit and ignores an assumed phase, so
          this call is a no-op there: it emits a warning and keeps the fast
          vectorized path (rather than switching to the slower per-point loop).
          Use ``Fluid("HEOS::Water", ...)`` if you need an assumed phase for water.
        * **The assumed phase must be correct for the operating domain.** Forcing
          a phase the fluid is not actually in does NOT raise -- on HEOS you get
          either ``NaN`` (deep in the wrong phase) or a finite but non-physical
          metastable root (near the saturation line). So it cannot be used to
          *detect* a wrong phase; for that, evaluate with auto-phase and check
          :attr:`phase` (or quality ``Q``) instead.

        Pass ``None`` to clear it and restore automatic determination. Mutates
        ``self`` and returns it, so it can be chained::

            Fluid("HEOS::CO2[0.5]&O2[0.5]", P=P, T=T).assume_phase("gas").D

        Parameters
        ----------
        phase : AssumedPhase | None
            One of ``"gas"``, ``"liquid"``, ``"supercritical"``,
            ``"supercritical_gas"``, ``"supercritical_liquid"``, ``"twophase"``,
            or ``None`` to clear.
        """
        if phase is not None and phase not in _ASSUMED_PHASE_MAP:
            raise ValueError(f"unknown phase {phase!r}, expected one of {sorted(_ASSUMED_PHASE_MAP)} or None")

        backend = self.name.split("::", 1)[0] if "::" in self.name else self.name
        if phase is not None and backend.upper() in _PHASE_IGNORING_BACKENDS:
            _LOGGER.warning(
                f"the {backend} backend is region-explicit and ignores an assumed phase; "
                f"assume_phase({phase!r}) is a no-op here (the fast vectorized path is kept). "
                "Use a HEOS-backed fluid (e.g. Fluid('HEOS::Water', ...)) to assume a phase."
            )
            return self

        self._assumed_phase = phase
        self._lowlevel = phase is not None or self._composition is not None
        return self

    @property
    def phase(self) -> str:
        if any(isinstance(n[1].m, pl.Expr) for n in self.points):
            return "Unknown"

        phase_idx = self.get("PHASE", convert_magnitude=False)
        phase_idx_val = phase_idx.m

        if isinstance(phase_idx_val, np.ndarray):
            if len(set(phase_idx_val)) == 1:
                phase_idx_val_element = float(phase_idx_val[0])
                return self.PHASES.get(phase_idx_val_element, "N/A")

            else:
                return "Variable"

        elif isinstance(phase_idx_val, float | int):
            return self.PHASES.get(float(phase_idx_val), "N/A")

        raise TypeError(f"Cannot determine phase of {type(self)} when {phase_idx=}")

    @property
    def PHASE(self) -> Quantity[Dimensionless, MT]:
        return self.get("PHASE").asdim(Dimensionless)

    @property
    def PRANDTL(self) -> Quantity[Dimensionless, MT]:
        return self.get("PRANDTL").asdim(Dimensionless)

    @property
    def P(self) -> Quantity[Pressure, MT]:
        return self.get("P").asdim(Pressure)

    @property
    def PCRIT(self) -> Quantity[Pressure, MT]:
        return self.get("PCRIT").asdim(Pressure)

    @property
    def PMAX(self) -> Quantity[Pressure, MT]:
        return self.get("PMAX").asdim(Pressure)

    @property
    def PMIN(self) -> Quantity[Pressure, MT]:
        return self.get("PMIN").asdim(Pressure)

    @property
    def PTRIPLE(self) -> Quantity[Pressure, MT]:
        return self.get("PTRIPLE").asdim(Pressure)

    @property
    def P_REDUCING(self) -> Quantity[Pressure, MT]:
        return self.get("P_REDUCING").asdim(Pressure)

    @property
    def T(self) -> Quantity[Temperature, MT]:
        return self.get("T").asdim(Temperature)

    @property
    def TCRIT(self) -> Quantity[Temperature, MT]:
        return self.get("TCRIT").asdim(Temperature)

    @property
    def TMAX(self) -> Quantity[Temperature, MT]:
        return self.get("TMAX").asdim(Temperature)

    @property
    def TMIN(self) -> Quantity[Temperature, MT]:
        return self.get("TMIN").asdim(Temperature)

    @property
    def TTRIPLE(self) -> Quantity[Temperature, MT]:
        return self.get("TTRIPLE").asdim(Temperature)

    @property
    def T_FREEZE(self) -> Quantity[Temperature, MT]:
        return self.get("T_FREEZE").asdim(Temperature)

    @property
    def T_REDUCING(self) -> Quantity[Temperature, MT]:
        return self.get("T_REDUCING").asdim(Temperature)

    @property
    def Q(self) -> Quantity[Dimensionless, MT]:
        return self.get("Q").asdim(Dimensionless)

    @property
    def H(self) -> Quantity[SpecificEnthalpy, MT]:
        return self.get("H").asdim(SpecificEnthalpy)

    @property
    def HMOLAR(self) -> Quantity[MolarSpecificEnthalpy, MT]:
        return self.get("HMOLAR").asdim(MolarSpecificEnthalpy)

    @property
    def S(self) -> Quantity[SpecificEntropy, MT]:
        return self.get("S").asdim(SpecificEntropy)

    @property
    def SMOLAR(self) -> Quantity[MolarSpecificEntropy, MT]:
        return self.get("SMOLAR").asdim(MolarSpecificEntropy)

    @property
    def U(self) -> Quantity[SpecificInternalEnergy, MT]:
        return self.get("U").asdim(SpecificInternalEnergy)

    @property
    def UMOLAR(self) -> Quantity[MolarSpecificInternalEnergy, MT]:
        return self.get("UMOLAR").asdim(MolarSpecificInternalEnergy)

    @property
    def V(self) -> Quantity[DynamicViscosity, MT]:
        return self.get("V").asdim(DynamicViscosity)

    @property
    def Z(self) -> Quantity[Dimensionless, MT]:
        return self.get("Z").asdim(Dimensionless)

    @property
    def DELTA(self) -> Quantity[Dimensionless, MT]:
        return self.get("DELTA").asdim(Dimensionless)

    @property
    def D(self) -> Quantity[Density, MT]:
        return self.get("D").asdim(Density)

    @property
    def RHOMASS_REDUCING(self) -> Quantity[Density, MT]:
        return self.get("RHOMASS_REDUCING").asdim(Density)

    @property
    def RHOMOLAR_CRITICAL(self) -> Quantity[MolarDensity, MT]:
        return self.get("RHOMOLAR_CRITICAL").asdim(MolarDensity)

    @property
    def RHOMOLAR_REDUCING(self) -> Quantity[MolarDensity, MT]:
        return self.get("RHOMOLAR_REDUCING").asdim(MolarDensity)

    @property
    def DMOLAR(self) -> Quantity[MolarDensity, MT]:
        return self.get("DMOLAR").asdim(MolarDensity)

    @property
    def A(self) -> Quantity[Velocity, MT]:
        return self.get("A").asdim(Velocity)

    @property
    def L(self) -> Quantity[ThermalConductivity, MT]:
        return self.get("L").asdim(ThermalConductivity)

    @property
    def C(self) -> Quantity[SpecificHeatCapacity, MT]:
        return self.get("C").asdim(SpecificHeatCapacity)

    @property
    def M(self) -> Quantity[MolarMass, MT]:
        return self.get("M").asdim(MolarMass)

    def __repr__(self) -> str:
        props_str = self._repr_properties()

        s = f'<{self.__class__.__name__} "{self.name}", {props_str}>'

        return s


class Water(Fluid[MT]):
    REPR_PROPERTIES: tuple[tuple[CProperty, str], ...] = (
        ("P", ".0f"),
        ("T", ".1f"),
        ("D", ".1f"),
        ("V", ".2g"),
    )

    def __init__(self, **kwargs: Unpack[FluidState[MT]]) -> None:
        """
        Convenience class to access water and steam properties via CoolProp.

        Parameters
        ----------
        kwargs: Quantity[Any, MT]
            Values for the two fixed points. The name of the keyword argument is the
            CoolProp property name.
        """

        # default IF97; use the name "Water" for IAPWS-95
        self.name = "IF97::Water"

        self.check_inputs(kwargs)

        if len(kwargs) != 2:
            if set(kwargs) == {"P", "T", "Q"}:
                raise ValueError(
                    "Cannot set both P, T and vapor quality Q. Remove one of P, T to get properties of saturated steam."
                )

            raise ValueError(f"Exactly two fixed points are required, passed {list(kwargs)}")

        points = self._build_points(kwargs)

        self.point_1 = points[0]
        self.point_2 = points[1]

        self.points = [self.point_1, self.point_2]

    def __repr__(self) -> str:
        try:
            phase = self.phase
        except Exception as e:
            return f"<{self.__class__.__name__}, invalid: {e}>"

        props_str = self._repr_properties()

        s = f"<{self.__class__.__name__} ({phase}), {props_str}>"

        return s


class HumidAir(CoolPropFluid[MT]):
    BACKEND = {"backend": HAPropsSI}
    _append_name_to_cp_inputs = False
    _evaluate_invalid_separately = True
    STATE_INPUTS: ClassVar[frozenset[str]] = frozenset(HUMID_AIR_INPUTS)

    # unit and description for properties in function HAPropsSI
    PROPERTY_MAP: dict[tuple[CProperty, ...], tuple[str, str]] = {
        ("B", "Twb", "T_wb", "WetBulb"): ("K", "Wet-Bulb Temperature"),
        ("C", "cp"): ("J/kg/K", "Mixture specific heat per unit dry air"),
        ("Cha", "cp_ha"): ("J/kg/K", "Mixture specific heat per unit humid air"),
        ("CV",): (
            "J/kg/K",
            "Mixture specific heat at constant volume per unit dry air",
        ),
        ("CVha", "cv_ha"): (
            "J/kg/K",
            "Mixture specific heat at constant volume per unit humid air",
        ),
        ("D", "Tdp", "DewPoint", "T_dp"): ("K", "Dew-Point Temperature"),
        ("H", "Hda", "Enthalpy"): ("J/kg", "Mixture enthalpy per dry air"),
        ("Hha",): ("J/kg", "Mixture enthalpy per humid air"),
        ("K", "k", "Conductivity"): ("W/m/K", "Mixture thermal conductivity"),
        ("M", "Visc", "mu"): ("Pa*s", "Mixture viscosity"),
        ("psi_w", "Y"): ("dimensionless", "Water mole fraction"),
        ("P",): ("Pa", "Pressure"),
        ("P_w",): ("Pa", "Partial pressure of water vapor"),
        ("R", "RH", "RelHum"): ("dimensionless", "Relative humidity in [0, 1]"),
        ("S", "Sda", "Entropy"): ("J/kg/K", "Mixture entropy per unit dry air"),
        ("Sha",): ("J/kg/K", "Mixture entropy per unit humid air"),
        ("T", "Tdb", "T_db"): ("K", "Dry-Bulb Temperature"),
        ("V", "Vda"): ("m³/kg", "Mixture volume per unit dry air"),
        ("Vha",): ("m³/kg", "Mixture volume per unit humid air"),
        ("W", "Omega", "HumRat"): (
            "dimensionless",
            "Humidity Rat mass water per mass dry air",
        ),
        ("Z",): ("dimensionless", "Compressibility factor"),
    }

    ALL_PROPERTIES: set[CProperty] = set(flatten(list(PROPERTY_MAP)))

    # HAPropsSI has different parameter names
    # density is not defined, need to use either Vda (volume per dry air)
    # or Vha (per humid air)
    RETURN_UNITS: dict[CProperty, str] = {
        "P": "kPa",
        "P_w": "kPa",
        "M": "cP",
        "T": "°C",
        "D": "°C",
        "B": "°C",
    }

    REPR_PROPERTIES: tuple[tuple[CProperty, str], ...] = (
        ("P", ".0f"),
        ("T", ".1f"),
        ("R", ".2f"),
        ("Vda", ".1f"),
        ("Vha", ".1f"),
        ("M", ".2g"),
    )

    def __init__(self, **kwargs: Unpack[HumidAirState[MT]]) -> None:
        """
        Interface to the CoolProp function for humid air,
        ``CoolProp.CoolProp.HAPropsSI``.
        Needs three fixed points instead of two.

        Parameters
        ----------
        kwargs: Quantity[Any, MT]
            Values for the three fixed points. The name of the keyword argument is the
            CoolProp property name.
        """

        self.name = "Humid air"

        self.check_inputs(kwargs)

        if len(kwargs) != 3:
            raise ValueError(f"Exactly three fixed points are required, passed {list(kwargs)}")

        points = self._build_points(kwargs)

        self.point_1: tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]] = points[0]
        self.point_2: tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]] = points[1]
        self.point_3: tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]] = points[2]

        self.points = [self.point_1, self.point_2, self.point_3]

    @property
    def psi_w(self) -> Quantity[Dimensionless, MT]:
        return self.get("psi_w").asdim(Dimensionless)

    @property
    def W(self) -> Quantity[Dimensionless, MT]:
        return self.get("W").asdim(Dimensionless)

    @property
    def Z(self) -> Quantity[Dimensionless, MT]:
        return self.get("Z").asdim(Dimensionless)

    @property
    def R(self) -> Quantity[Dimensionless, MT]:
        return self.get("R").asdim(Dimensionless)

    @property
    def P(self) -> Quantity[Pressure, MT]:
        return self.get("P").asdim(Pressure)

    @property
    def P_w(self) -> Quantity[Pressure, MT]:
        return self.get("P_w").asdim(Pressure)

    @property
    def B(self) -> Quantity[Temperature, MT]:
        return self.get("B").asdim(Temperature)

    @property
    def T(self) -> Quantity[Temperature, MT]:
        return self.get("T").asdim(Temperature)

    @property
    def D(self) -> Quantity[Temperature, MT]:
        return self.get("D").asdim(Temperature)

    @property
    def K(self) -> Quantity[ThermalConductivity, MT]:
        return self.get("K").asdim(ThermalConductivity)

    @property
    def M(self) -> Quantity[DynamicViscosity, MT]:
        return self.get("M").asdim(DynamicViscosity)

    @property
    def C(self) -> Quantity[SpecificHeatPerDryAir, MT]:
        return self.get("C").asdim(SpecificHeatPerDryAir)

    @property
    def Cha(self) -> Quantity[SpecificHeatPerHumidAir, MT]:
        return self.get("Cha").asdim(SpecificHeatPerHumidAir)

    @property
    def H(self) -> Quantity[MixtureEnthalpyPerDryAir, MT]:
        return self.get("H").asdim(MixtureEnthalpyPerDryAir)

    @property
    def Hha(self) -> Quantity[MixtureEnthalpyPerHumidAir, MT]:
        return self.get("Hha").asdim(MixtureEnthalpyPerHumidAir)

    @property
    def S(self) -> Quantity[MixtureEntropyPerDryAir, MT]:
        return self.get("S").asdim(MixtureEntropyPerDryAir)

    @property
    def Sha(self) -> Quantity[MixtureEntropyPerHumidAir, MT]:
        return self.get("Sha").asdim(MixtureEntropyPerHumidAir)

    @property
    def V(self) -> Quantity[MixtureVolumePerDryAir, MT]:
        return self.get("V").asdim(MixtureVolumePerDryAir)

    @property
    def Vha(self) -> Quantity[MixtureVolumePerHumidAir, MT]:
        return self.get("Vha").asdim(MixtureVolumePerHumidAir)

    def __repr__(self) -> str:
        props_str = self._repr_properties()

        s = f"<{self.__class__.__name__}, {props_str}>"

        return s
