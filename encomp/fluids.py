"""Fluid-property classes backed by encomp's bundled CoolProp library."""

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from functools import cache
from threading import Lock
from typing import Annotated, Any, ClassVar, Generic, Literal, Self, TypedDict, Unpack, cast

import numpy as np
import polars as pl

from .coolprop import (
    ASSUMED_PHASES,
    FLUID_INPUTS,
    HUMID_AIR_INPUTS,
    AssumedPhase,
    CName,
    Composition,
    FluidParam,
    HumidAirParam,
    _fluid_scalar,  # pyright: ignore[reportPrivateUsage]
    _humid_air_scalar,  # pyright: ignore[reportPrivateUsage]
    _native,  # pyright: ignore[reportPrivateUsage]
    is_fluid_param,
    is_humid_air_param,
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
    MolarSpecificHeatCapacity,
    MolarSpecificInternalEnergy,
    Numpy1DArray,
    Pressure,
    SpecificEnthalpy,
    SpecificEntropy,
    SpecificHeatCapacity,
    SpecificHeatPerDryAir,
    SpecificHeatPerHumidAir,
    SpecificInternalEnergy,
    SurfaceTension,
    Temperature,
    ThermalConductivity,
    Velocity,
)

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "EAGER_PLUGIN_MIN_SIZE",
    "CProperty",
    "CoolPropFluid",
    "Fluid",
    "FluidPhase",
    "FluidState",
    "HumidAir",
    "HumidAirState",
    "UnitString",
    "Water",
    "clear_expr_evaluation_cache",
]

# Strict Literals for CoolProp property names, single source of truth in the
# encomp.coolprop plugin (fluid + humid-air namespaces). The fluid-name / composition /
# assumed-phase types (CName, CommonFluidName, Composition, FractionValue, AssumedPhase)
# live there too; the ones used here are imported above.
CProperty = FluidParam | HumidAirParam
UnitString = Annotated[str, "Unit string"]
FluidPhase = Literal[
    "Liquid",
    "Gas",
    "Two-phase",
    "Supercritical liquid",
    "Supercritical gas",
    "Supercritical fluid",
    "Critical point",
    "Unknown",
    "Not imposed",
    "Variable",
    "N/A",
]
BackendKind = Literal["fluid", "humid_air"]


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


# backends that ignore AbstractState.specify_phase (region-explicit), so assuming a
# phase has no effect
_PHASE_IGNORING_BACKENDS = frozenset({"IF97"})

EAGER_PLUGIN_MIN_SIZE = 1000
"""Deprecated compatibility constant; now a no-op.

All eager arrays use the native Rust/Polars batch path, regardless of size. Scalars use
the direct PyO3 bridge, so there is no longer a Python CoolProp cutoff to configure.
"""

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
    """Drop the cache of constructed CoolProp ``pl.Expr`` nodes.

    Repeated evaluations of the same property, on the same fluid and inputs, return the
    *identical* expression object, so callers that deduplicate expressions by ``id()``
    see one node instead of many. Clearing the cache never changes a result; it only
    releases the cached nodes. Useful when benchmarking expression construction, or to
    free the (bounded) cache in a long-lived process that builds many distinct fluids.
    """

    with _EXPR_EVALUATION_CACHE_LOCK:
        _EXPR_EVALUATION_CACHE.clear()


@cache
def _resolve_fluid_name(backend: str, fluids: str, fractions: tuple[float, ...] | None) -> None:
    # Constructing the AbstractState is the only reliable way to ask CoolProp whether a
    # name resolves (it covers pure fluids, INCOMP, and mixtures alike). It costs tens of
    # microseconds for the usual backends -- and seconds the very first time a tabular
    # backend builds its tables, a cost that would otherwise be paid at the first property
    # access anyway. Cached per (backend, fluids), so the 1000th Fluid("Water", ...) pays
    # a dict lookup rather than another CoolProp initialization.
    #
    # functools.cache stores return values, never exceptions, so only a resolved name is
    # remembered: a failure (an invalid name, or a transient CoolProp error) is re-checked
    # on the next call instead of being cached as "this fluid does not exist".
    _native().validate_fluid(backend, fluids, None if fractions is None else list(fractions))


def _validate_fluid_name(name: CName, composition: Composition | None = None) -> None:
    backend, fluids, fractions = resolve_fluid_spec(name, composition)

    try:
        _resolve_fluid_name(backend, fluids, None if fractions is None else tuple(fractions))
    except Exception as e:
        raise ValueError(
            f"Fluid '{name}' could not be initialized, ensure that the name is a valid CoolProp fluid name"
        ) from e


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
        type(fluid).BACKEND_KIND,
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

    BACKEND_KIND: ClassVar[BackendKind] = "fluid"

    # Assumed phase passed to both native interfaces. None means CoolProp determines
    # the phase; a value skips the expensive mixture phase-stability search.
    _assumed_phase: str | None = None

    # Fixed mixture composition as mole fractions, one float per species. None means
    # the composition is fixed in the fluid name, or the fluid is pure.
    _composition: Composition | None = None

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
    )

    PHASES: dict[float, FluidPhase] = {
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

    # CoolProp fluid-property names, units, and descriptions
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

        if len(set(candidates)) > 1:
            names = ", ".join(sorted(t.__name__ for t in set(candidates)))
            raise TypeError(
                f"Mixed vector magnitude containers are not supported for one fluid state: {names}. "
                "Use one container type for all vector inputs."
            )

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
        A name may fold in the backend (``"IF97::Water"``, ``"HEOS::CO2&O2"``); a bare
        name defaults to the HEOS backend. Incompressible fluids and their mixtures use
        the ``INCOMP::`` prefix, optionally with a concentration
        (``"INCOMP::MEG[0.5]"``). An unknown name raises ``ValueError`` at construction.

        Examples recognized by the bundled build include ``Water``, ``Air``,
        ``Nitrogen``, ``CarbonDioxide``, ``Ammonia``,
        ``R134a``, ``Toluene``, ``INCOMP::T66``, ``INCOMP::MPG[0.5]``, ``R410A.mix``.

        Refer to the CoolProp documentation for more information:

        - http://www.coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids
        - http://www.coolprop.org/fluid_properties/Mixtures.html#binary-pairs
        - http://www.coolprop.org/fluid_properties/Incompressibles.html#the-different-fluids
        - http://www.coolprop.org/fluid_properties/HumidAir.html#table-of-inputs-outputs-to-hapropssi
        - http://www.coolprop.org/coolprop/HighLevelAPI.html
        - http://www.coolprop.org/fluid_properties/HumidAir.html


        The names ``Water`` and ``HEOS::Water``
        use the formulation defined by IAPWS-95.
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
        """Return the tuple of synonyms in :attr:`PROPERTY_MAP` that ``prop`` belongs to."""

        if prop not in cls.ALL_PROPERTIES:
            raise ValueError(f'Property "{prop}" is not a valid CoolProp property name')

        for names in cls.PROPERTY_MAP:
            if prop in names:
                return names

        raise ValueError(f'Property "{prop}" is not a valid CoolProp property name')

    @classmethod
    def get_coolprop_unit(cls, prop: CProperty) -> Unit:
        """Return the unit CoolProp reports ``prop`` in, before conversion to :attr:`RETURN_UNITS`."""

        key = cls.get_prop_key(prop)

        if key in cls.PROPERTY_MAP:
            unit_str = cls.PROPERTY_MAP[key][0]
            return Quantity.get_unit(unit_str)

        raise ValueError(f'Could not get unit, key "{key}" does not exist')

    @classmethod
    def is_valid_prop(cls, prop: str) -> bool:
        """Whether ``prop`` is a CoolProp property name known to this class."""

        try:
            cls.get_prop_key(prop)
            return True

        except ValueError:
            return False

    @classmethod
    def check_inputs(cls, kwargs: Mapping[str, object]) -> None:
        """Raise ``ValueError`` unless every key names a distinct CoolProp state input."""

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

        seen: dict[tuple[CProperty, ...], str] = {}
        for key in kwargs:
            prop_key = cls.get_prop_key(key)
            if previous := seen.get(prop_key):
                raise ValueError(
                    f"{cls.__name__} inputs must be distinct; got {previous!r} and {key!r} for the same state input"
                )
            seen[prop_key] = key

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
        """Return a one-line description of ``prop``: its synonyms, meaning, and unit."""

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
        """Return the :meth:`describe` lines of every property whose description contains ``inp``."""

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
        """Re-raise ``e``, or return so the caller can yield ``NaN`` for ``prop``.

        CoolProp signals both "out of range" and "not implemented" with ``ValueError``;
        those cases warn (or stay silent) and produce ``NaN``. Anything else re-raises.
        """

        msg = str(e)

        # this error occurs in case the input values are outside
        # the allowable range for this property
        # in this case the return value will be NaN, no exception is raised
        if "No outputs were able to be calculated" in msg:
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

    def _plugin_expr(self, output: CProperty, points: tuple[tuple[str, pl.Expr], ...]) -> pl.Expr:
        """Build an encomp.coolprop plugin expression aliased to ``output``: a
        precision-preserving dtype (Float32 in -> Float32 out) with null (not NaN) for
        failed/out-of-range rows.

        Shared by lazy expressions and eager arrays; the latter reads it via
        ``to_numpy()``, which maps null back to NaN. Output property names are resolved
        by CoolProp at runtime, so any valid name works regardless of the static
        ``FluidParam`` / ``HumidAirParam`` sets. Raises if the plugin is unavailable
        or cannot represent the request.
        """
        from . import coolprop as _cprust

        if not _cprust.self_check():
            raise RuntimeError(
                "the encomp.coolprop rust plugin failed to load, but it is required to "
                "evaluate arrays and pl.Expr inputs. Reinstall encomp with its compiled plugin."
            )
        names = [p[0] for p in points]
        exprs = [p[1] for p in points]
        if self.BACKEND_KIND == "humid_air":
            n1, n2, n3 = names
            # the plugin reads each input's property from its (aliased) name
            expr = _cprust.humid_air(
                cast("HumidAirParam", output),
                exprs[0].alias(n1),
                exprs[1].alias(n2),
                exprs[2].alias(n3),
            )
        else:
            n1, n2 = names
            # pass the high-level spec straight through; coolprop.fluid resolves the
            # name + composition through the bundled native implementation
            expr = _cprust.fluid(
                cast("FluidParam", output),
                exprs[0].alias(n1),
                exprs[1].alias(n2),
                name=self.name,
                assume_phase=cast("AssumedPhase | None", self._assumed_phase),
                composition=self._composition,
            )
        return expr.alias(output)

    def _evaluate_array(self, output: CProperty, points: tuple[tuple[str, Numpy1DArray], ...]) -> Numpy1DArray:
        """Evaluate eager NumPy ``points`` through the native plugin, returning a 1-D
        Float64 array with non-finite inputs and results mapped to NaN. ``points`` are
        equal-length arrays; the caller reshapes the result.
        """
        names = [name for name, _ in points]
        arrs = [np.ascontiguousarray(arr, dtype=float).ravel() for _, arr in points]
        # generic column names avoid any collision between the input property names
        frame = pl.DataFrame({f"__in{i}": a for i, a in enumerate(arrs)})
        plugin_points = tuple((names[i], pl.col(f"__in{i}")) for i in range(len(arrs)))
        expr = self._plugin_expr(output, plugin_points)
        out = np.array(frame.select(expr)[output].to_numpy(), dtype=float)
        # Rows with any non-finite input are NaN, and any inf/_HUGE that slipped
        # through becomes NaN.
        finite_inputs = np.logical_and.reduce([np.isfinite(a) for a in arrs])
        out[~finite_inputs] = np.nan
        out[~np.isfinite(out)] = np.nan
        return out

    @staticmethod
    def _reduce_single_element(x: float | Numpy1DArray | pl.Expr) -> float | Numpy1DArray | pl.Expr:
        if isinstance(x, np.ndarray) and x.size == 1:
            return float(x[0])  # ty: ignore[invalid-argument-type]
        return x

    def evaluate_single(self, output: CProperty, *points: tuple[CProperty, float]) -> float:
        """Evaluate a scalar through the direct native bridge, returning ``NaN`` on an invalid state."""

        # A non-finite input cannot fix a state, and CoolProp does not reliably say so: it
        # can otherwise return a phase sentinel, a state-independent constant, or echo an
        # input. Mask here so scalar and batch/plugin paths agree.
        if not all(np.isfinite(value) for _, value in points):
            return np.nan

        try:
            if self.BACKEND_KIND == "humid_air":
                (name1, value1), (name2, value2), (name3, value3) = points
                value = _humid_air_scalar(output, name1, value1, name2, value2, name3, value3)
            else:
                (name1, value1), (name2, value2) = points
                value = _fluid_scalar(
                    output,
                    name1,
                    value1,
                    name2,
                    value2,
                    name=self.name,
                    assume_phase=cast("AssumedPhase | None", self._assumed_phase),
                    composition=self._composition,
                )
            if not np.isfinite(value):
                self._warn_coolprop_nan(output, "native evaluation returned no finite result")
                return np.nan
            return value

        except ValueError as e:
            self.check_exception(output, e)
            return np.nan

    def evaluate_expression(self, output: CProperty, *points: tuple[CProperty, pl.Expr]) -> pl.Expr:
        """Build (or reuse) the cached ``pl.Expr`` plugin node that evaluates ``output``."""

        # Expressions are evaluated only through the GIL-free native plugin.
        key = _get_expr_evaluation_cache_key(self, output, points)
        cached_expr = _expr_evaluation_cache_get(key)

        if cached_expr is not None:
            return cached_expr

        # A fill_nan wrapper would duplicate the opaque plugin subtree because Polars
        # cannot apply common-subexpression elimination across plugin nodes.
        expr = self._plugin_expr(output, points)
        _expr_evaluation_cache_set(key, expr)
        return expr

    def evaluate(
        self, output: CProperty, *points: tuple[CProperty, float | Numpy1DArray | pl.Expr]
    ) -> float | Numpy1DArray | pl.Expr:
        """Evaluate ``output`` in CoolProp's own unit, dispatching on the magnitude type of ``points``."""

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
                return x  # ty: ignore[invalid-return-type]

            return np.repeat(x, n).astype(float).reshape(shape)

        points_arr: tuple[tuple[CProperty, Numpy1DArray], ...] = tuple(
            (p, expand_scalars(cast(Any, v))) for p, v in reduced_points
        )
        return self._evaluate_array(output, points_arr).reshape(shape)

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
        """Wrap a raw CoolProp result as a ``Quantity``, converted to the preferred :attr:`RETURN_UNITS`."""

        unit_output = self.get_coolprop_unit(output)

        # the dimensionality is not known until runtime
        qty = Quantity(cast(MT, val), unit_output)

        key = self.get_prop_key(output)

        if len(key) > 0 and key[0] in self.RETURN_UNITS:
            ret_unit = self.RETURN_UNITS[key[0]]
            qty.ito(ret_unit)

        if convert_magnitude:
            qty = qty.astype(self._mt)  # ty: ignore[no-matching-overload]

        if isinstance(qty.m, pl.Series):
            # missing values surface as null, never NaN (the library's single sentinel)
            qty.m = qty.m.fill_nan(None).cast(self._eager_series_output_dtype())

        return cast("Quantity[Any, MT]", qty)

    def to_numeric_correct_unit(
        self, prop: CProperty, qty: Quantity[Any, MT] | Quantity[Any, float]
    ) -> float | Numpy1DArray | pl.Expr:
        """Convert a state-input quantity to CoolProp's unit for ``prop`` and return its bare magnitude.

        This is where the dimensionality of a state input is validated: a mismatch raises
        :class:`encomp.units.ExpectedDimensionalityError`.
        """

        unit = self.get_coolprop_unit(prop)

        try:
            m = qty.to(unit).m
        except DimensionalityError as e:
            raise ExpectedDimensionalityError(
                f'CoolProp input for property "{prop}" is incorrect. '
                f"expected {unit} ({unit.dimensionality}), but passed "
                f"{qty.u} ({qty.dimensionality})"
            ) from e

        # a Quantity magnitude is float / ndarray / pl.Series / pl.Expr (a list input is
        # converted to an ndarray at construction, and .to() preserves the container)
        if isinstance(m, pl.Series):
            return m.to_numpy()

        return m

    def get(
        self,
        output: CProperty,
        points: list[tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]]] | None = None,
        convert_magnitude: bool = True,
    ) -> Quantity[Any, MT]:
        """
        Evaluates through encomp's bundled native CoolProp artifact and handles input
        and output with :py:class:`encomp.units.Quantity` objects.

        Parameters
        ----------
        output : CProperty
            Name of the output property
        points : list[tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]]] | None
            Fixed state variables: name and value of the property.
            The number of points must match the number expected
            by the fluid implementation.
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

        # dunder lookups reach here during copy/pickle/inspect protocol probing; those
        # must fail plainly, without suggesting a property lookup
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)

        raise AttributeError(
            f'"{attr}" is not a CoolProp property name for {type(self).__name__} '
            f'(property names are case-sensitive); use {type(self).__name__}.search("...") to look one up'
        )

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

        The fluid name is resolved eagerly: an unknown name raises ``ValueError`` here,
        not at the first property access. The dimensionality of each state input is still
        validated lazily, when a property is evaluated.

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

        _validate_fluid_name(self.name)

        points = self._build_points(kwargs)

        self.point_1: tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]] = points[0]
        self.point_2: tuple[CProperty, Quantity[Any, MT] | Quantity[Any, float]] = points[1]

        self.points = [self.point_1, self.point_2]

    def _init_composition(self, name: CName, composition: Composition) -> None:
        # all name + composition parsing/validation lives in resolve_fluid_spec (the single
        # source); store the canonical name + resolved fractions for both native paths
        backend, fluids, fractions = resolve_fluid_spec(name, composition)
        assert fractions is not None  # a composition= dict always resolves to a mixture
        self.name = f"{backend}::{fluids}"
        self._composition = dict(zip(fluids.split("&"), fractions, strict=True))

    def assume_phase(self, phase: AssumedPhase | None) -> Self:
        """Force the equation of state to assume ``phase``, skipping CoolProp's
        own phase determination. This is a *speed* tool, not a validation tool.

        With ``P, T`` inputs CoolProp normally runs a phase-stability search to
        decide whether the state is single- or two-phase. For HEOS/GERG mixtures
        that search dominates the cost (~5 ms/point); assuming a phase you
        already know skips it while retaining the same native evaluation API.

        Important caveats:

        * **Only the HEOS/GERG backends honour it.** ``IF97`` (the default for
          :class:`Water`) is region-explicit and ignores an assumed phase, so
          this call is a no-op there and emits a warning.
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
        if phase is not None and phase not in ASSUMED_PHASES:
            raise ValueError(f"unknown phase {phase!r}, expected one of {sorted(ASSUMED_PHASES)} or None")

        backend = self.name.split("::", 1)[0] if "::" in self.name else self.name
        if phase is not None and backend.upper() in _PHASE_IGNORING_BACKENDS:
            _LOGGER.warning(
                f"the {backend} backend is region-explicit and ignores an assumed phase; "
                f"assume_phase({phase!r}) is a no-op here. "
                "Use a HEOS-backed fluid (e.g. Fluid('HEOS::Water', ...)) to assume a phase."
            )
            return self

        self._assumed_phase = phase
        return self

    @property
    def phase(self) -> FluidPhase:
        if any(isinstance(n[1].m, pl.Expr) for n in self.points):
            return "Unknown"

        phase_idx = self.get("PHASE", convert_magnitude=False)
        phase_idx_val = phase_idx.m

        if isinstance(phase_idx_val, np.ndarray):
            # a NaN row is an invalid state, not a phase, so the phase is judged on the rows that
            # have one. Reducing over the raw array would instead count each NaN as its own
            # distinct phase, since nan != nan
            finite = cast("Numpy1DArray", phase_idx_val[np.isfinite(phase_idx_val)])  # ty: ignore[invalid-argument-type]

            if finite.size == 0:
                return "N/A"

            unique = np.unique(finite)

            if unique.size == 1:
                return self.PHASES.get(float(unique[0]), "N/A")

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
    def TAU(self) -> Quantity[Dimensionless, MT]:
        return self.get("TAU").asdim(Dimensionless)

    @property
    def D(self) -> Quantity[Density, MT]:
        return self.get("D").asdim(Density)

    @property
    def RHOCRIT(self) -> Quantity[Density, MT]:
        return self.get("RHOCRIT").asdim(Density)

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
    def CVMASS(self) -> Quantity[SpecificHeatCapacity, MT]:
        return self.get("CVMASS").asdim(SpecificHeatCapacity)

    @property
    def CPMOLAR(self) -> Quantity[MolarSpecificHeatCapacity, MT]:
        return self.get("CPMOLAR").asdim(MolarSpecificHeatCapacity)

    @property
    def CVMOLAR(self) -> Quantity[MolarSpecificHeatCapacity, MT]:
        return self.get("CVMOLAR").asdim(MolarSpecificHeatCapacity)

    @property
    def GAS_CONSTANT(self) -> Quantity[MolarSpecificHeatCapacity, MT]:
        """Molar gas constant, which shares its dimensions with a molar heat capacity."""

        return self.get("GAS_CONSTANT").asdim(MolarSpecificHeatCapacity)

    @property
    def I(self) -> Quantity[SurfaceTension, MT]:  # noqa: E743  # CoolProp's name for surface tension
        return self.get("I").asdim(SurfaceTension)

    @property
    def M(self) -> Quantity[MolarMass, MT]:
        return self.get("M").asdim(MolarMass)

    def __repr__(self) -> str:
        props_str = self._repr_properties()

        s = f'<{self.__class__.__name__} "{self.name}", {props_str}>'

        return s


class Water(Fluid[MT]):
    # REPR_PROPERTIES is inherited unchanged from CoolPropFluid (P, T, D, V); only the
    # __repr__ header differs (it shows the phase). HumidAir overrides it, Water does not.

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
    """Humid-air properties from encomp's bundled CoolProp implementation.

    ``HumidAir.M`` follows CoolProp's humid-air naming and returns dynamic
    viscosity. This differs from ``Fluid.M``, which returns molar mass.
    """

    BACKEND_KIND: ClassVar[BackendKind] = "humid_air"
    STATE_INPUTS: ClassVar[frozenset[str]] = frozenset(HUMID_AIR_INPUTS)

    # CoolProp humid-air property names, units, and descriptions
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
            "Humidity Ratio mass water per mass dry air",
        ),
        ("Z",): ("dimensionless", "Compressibility factor"),
    }

    ALL_PROPERTIES: set[CProperty] = set(flatten(list(PROPERTY_MAP)))

    # Humid air has a property namespace distinct from AbstractState fluids.
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
        Interface to the bundled CoolProp implementation for humid air.
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
    def CV(self) -> Quantity[SpecificHeatPerDryAir, MT]:
        return self.get("CV").asdim(SpecificHeatPerDryAir)

    @property
    def CVha(self) -> Quantity[SpecificHeatPerHumidAir, MT]:
        return self.get("CVha").asdim(SpecificHeatPerHumidAir)

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
