# ruff: noqa: UP046
"""
Imports and extends the ``pint`` library for physical units.
Always import this module when working with ``encomp`` (most other modules
will import this one).

Implements a type-aware system on top of ``pint`` that verifies
that the dimensionality of the unit is correct.
"""

from __future__ import annotations

import copy
import logging
import math
import numbers
import re
import sys
import warnings
from collections.abc import Iterable, Iterator, Sequence, Sized
from inspect import isclass
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    TypeVar,
    assert_never,
    cast,
    get_origin,
    overload,
)

import numpy as np
import pint
import polars as pl
from pint.errors import DimensionalityError, RedefinitionError, UnitStrippedWarning
from pint.facets.measurement.objects import MeasurementQuantity
from pint.facets.nonmultiplicative.objects import NonMultiplicativeQuantity
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain.quantity import PlainQuantity
from pint.facets.plain.unit import PlainUnit
from pint.registry import LazyRegistry, UnitRegistry
from pint.util import UnitsContainer
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema
from typeguard import TypeCheckError, TypeCheckMemo, checker_lookup_functions

from .settings import PINT_FORMATTING_SPECIFIERS, SETTINGS
from .utypes import (
    BASE_SI_UNITS,
    DT,
    DT_,
    MT,
    MT_,
    AllUnits,
    Area,
    AreaUnits,
    Currency,
    CurrencyPerEnergy,
    CurrencyPerEnergyUnits,
    CurrencyPerMass,
    CurrencyPerMassUnits,
    CurrencyPerTime,
    CurrencyPerTimeUnits,
    CurrencyPerVolume,
    CurrencyPerVolumeUnits,
    CurrencyUnits,
    Current,
    CurrentUnits,
    Density,
    DensityUnits,
    Dimensionality,
    Dimensionless,
    DimensionlessUnits,
    DynamicViscosity,
    DynamicViscosityUnits,
    Energy,
    EnergyPerMass,
    EnergyPerMassUnits,
    EnergyUnits,
    Force,
    ForceUnits,
    Frequency,
    FrequencyUnits,
    HeatTransferCoefficient,
    HeatTransferCoefficientUnits,
    KinematicViscosity,
    KinematicViscosityUnits,
    Length,
    LengthUnits,
    Luminosity,
    LuminosityUnits,
    Mass,
    MassFlow,
    MassFlowUnits,
    MassPerNormalVolume,
    MassPerNormalVolumeUnits,
    MassUnits,
    MolarDensity,
    MolarDensityUnits,
    MolarMass,
    MolarMassUnits,
    MolarSpecificEnthalpy,
    MolarSpecificEnthalpyUnits,
    NormalVolume,
    NormalVolumeFlow,
    NormalVolumeFlowUnits,
    NormalVolumePerMass,
    NormalVolumePerMassUnits,
    NormalVolumeUnits,
    Numpy1DArray,
    Numpy1DBoolArray,
    Numpy1DIntArray,
    Power,
    PowerPerArea,
    PowerPerAreaUnits,
    PowerUnits,
    Pressure,
    PressureUnits,
    SpecificHeatCapacity,
    SpecificHeatCapacityUnits,
    SpecificVolume,
    SpecificVolumeUnits,
    Substance,
    SubstancePerMass,
    SubstancePerMassUnits,
    SubstanceUnits,
    Temperature,
    TemperatureDifference,
    TemperatureDifferenceUnits,
    TemperatureUnits,
    ThermalConductivity,
    ThermalConductivityUnits,
    Time,
    TimeUnits,
    UnknownDimensionality,
    Velocity,
    VelocityUnits,
    Volume,
    VolumeFlow,
    VolumeFlowUnits,
    VolumeUnits,
)

__all__ = [
    "CUSTOM_DIMENSIONS",
    "UNIT_REGISTRY",
    "DimensionalityComparisonError",
    "DimensionalityError",
    "DimensionalityRedefinitionError",
    "DimensionalityTypeError",
    "ExpectedDimensionalityError",
    "Quantity",
    "Unit",
    "UnitStrippedWarning",
    "define_dimensionality",
    "set_quantity_format",
]

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = None


def _ensure_sympy() -> None:
    global sp
    if sp is None:
        import sympy as sp


_LOGGER = logging.getLogger(__name__)


def _is_close(
    a: float | Numpy1DArray,
    b: float | Numpy1DArray,
    rtol: float,
    atol: float,
) -> Any:  # noqa: ANN401
    """Symmetric closeness test for float / ndarray magnitudes.

    Implements the same predicate as ``math.isclose`` and ``polars`` ``is_close``::

        |a - b| <= max(rtol * max(|a|, |b|), atol)

    so that every magnitude type answers identically. ``np.isclose`` is deliberately not used:
    it evaluates the asymmetric ``atol + rtol * |b|``, treating ``b`` as a reference value, and
    an equality operator must be symmetric. The exact-equality term keeps ``+-inf`` close to
    itself, matching both the stdlib and polars.
    """

    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    with np.errstate(invalid="ignore"):
        # an infinite operand makes the tolerance infinite too, and `inf <= inf` would then call
        # any finite value close to infinity -- so the band applies only where both sides are
        # finite. NaN is never close to anything, including itself. inf is close only to itself,
        # via the exact-equality term (which also short-circuits the `inf - inf` NaN)
        tolerance = np.maximum(rtol * np.maximum(np.abs(a_arr), np.abs(b_arr)), atol)
        finite = np.isfinite(a_arr) & np.isfinite(b_arr)
        close = (np.abs(a_arr - b_arr) <= tolerance) & finite

    return close | (a_arr == b_arr)


DimensionalityTypeName = Annotated[str, "Dimensionality name"]
MagnitudeTypeName = Literal[
    "float",
    "ndarray",
    "pl.Series",
    "pl.Expr",
]

# custom errors inherit from pint.errors.DimensionalityError
# (which inherits from TypeError)
# this makes it possible to use
# try:
#     ...
# except DimensionalityError:
#     ...
# to catch all unit/dimensionality-related errors


class _DimensionalityError(DimensionalityError):
    msg: str

    def __init__(self, msg: str = "") -> None:
        self.msg = msg
        super().__init__(None, None, dim1="", dim2="", extra_msg=msg)

    def __str__(self) -> str:
        return self.msg


class ExpectedDimensionalityError(_DimensionalityError):
    """Raised when a quantity cannot be reinterpreted as a requested dimensionality."""


class DimensionalityTypeError(_DimensionalityError):
    """Raised when two semantic dimensionality classes are not compatible."""


class DimensionalityComparisonError(_DimensionalityError):
    """Raised when an ordering comparison has incompatible dimensionality."""


class DimensionalityRedefinitionError(ValueError):
    """Raised when defining a dimensionality that already exists."""


# NOTE: make sure to list all that are defined in defs/units.txt ("# custom dimensions")
CUSTOM_DIMENSIONS: list[str] = [
    "currency",
    "normal",
]
"""Names of the registered custom base dimensions. The entries defined out of the box
match the custom-dimension section of ``defs/units.txt``;
:func:`define_dimensionality` appends every dimensionality it defines."""


_REGISTRY_STATIC_OPTIONS = {
    # if False, degC must be explicitly converted to K when multiplying
    # this is False by default, there's no reason to set this to True
    "autoconvert_offset_to_baseunit": SETTINGS.autoconvert_offset_to_baseunit,
    # if this is True, scalar magnitude inputs will
    # be converted to 1-element arrays
    # tests are written with the assumption that this is False
    "force_ndarray_like": False,
    "force_ndarray": False,
}


class _UnitRegistry(UnitRegistry[Any]):
    # flipped once the pinned values in _REGISTRY_STATIC_OPTIONS have been applied;
    # writes before that are pint's own initialization, not user overrides
    _static_options_pinned: bool = False

    def __setattr__(self, key: str, value: Any) -> None:  # noqa: ANN401
        # ensure that static options cannot be overridden
        if key in _REGISTRY_STATIC_OPTIONS and value != _REGISTRY_STATIC_OPTIONS[key]:
            if self._static_options_pinned:
                # pint documents these as ordinary mutable options; encomp pins them,
                # so tell the caller instead of discarding the write in silence
                _LOGGER.warning(
                    "Ignoring assignment UNIT_REGISTRY.%s = %r: encomp pins this registry option to %r",
                    key,
                    value,
                    _REGISTRY_STATIC_OPTIONS[key],
                )
            return

        return super().__setattr__(key, value)


class _LazyRegistry(LazyRegistry[Any, Any]):
    def __init(self) -> None:  # pyright: ignore[reportUnusedFunction]  # invoked by pint via name mangling
        args, kwargs = self.__dict__["params"]
        kwargs["on_redefinition"] = "raise"

        # override the filename
        kwargs["filename"] = str(SETTINGS.units.resolve().absolute())

        setattr(self, "__class__", _UnitRegistry)  # noqa: B010
        cast(Any, self).__init__(*args, **kwargs)
        assert self._after_init != "raise"
        self._after_init()


# NOTE: no attribute docstring here: autodoc's signature formatter fails on the
# UnitRegistry[Any] value annotation, which would break the -W docs build
UNIT_REGISTRY = cast(UnitRegistry[Any], _LazyRegistry())

for _option_name, _option_value in _REGISTRY_STATIC_OPTIONS.items():
    setattr(UNIT_REGISTRY, _option_name, _option_value)

setattr(UNIT_REGISTRY, "_static_options_pinned", True)  # noqa: B010

# Make sure that UNIT_REGISTRY is the only registry that can be used. This DELIBERATELY
# reaches into pint internals and takes over its process-wide application registry: every
# Quantity in the process must come from encomp's registry, or the dimensionality
# subclasses, the custom dimensions (currency, normal) and on_redefinition="raise" would
# silently not apply. The trade-off is documented (README "Settings"): another pint-based
# library in the same process sees encomp's registry after `import encomp`. This is an
# intentional, settled design decision -- not an oversight to be re-flagged.
setattr(pint, "_DEFAULT_REGISTRY", UNIT_REGISTRY)  # noqa: B010
cast(Any, pint.application_registry).set(UNIT_REGISTRY)

# the default format must be set after Quantity and Unit are registered
UNIT_REGISTRY.formatter.default_format = SETTINGS.default_unit_format


def set_quantity_format(fmt: str = "compact") -> None:
    """Set the process-wide default format used when rendering quantities and units.

    ``"compact"``/``"normal"`` select Pint's compact pretty format (``"~P"``);
    ``"siunitx"`` selects the LaTeX siunitx format (``"~Lx"``). Any value in
    :attr:`Quantity.FORMATTING_SPECS` can also be passed directly.
    """
    fmt_aliases = {"compact": "~P", "normal": "~P", "siunitx": "~Lx"}

    if fmt in fmt_aliases:
        fmt = fmt_aliases[fmt]

    if fmt not in Quantity.FORMATTING_SPECS:
        raise ValueError(
            f'Cannot set default format to "{fmt}", '
            f"fmt must be one of {Quantity.FORMATTING_SPECS} "
            f"or an alias: {', '.join(f'{k}: {v}' for k, v in fmt_aliases.items())}"
        )

    UNIT_REGISTRY.formatter.default_format = fmt


def define_dimensionality(name: str, symbol: str | None = None, if_exists: Literal["raise", "warn"] = "raise") -> None:
    """
    Defines a new dimensionality that can be combined with existing
    dimensionalities. In case the dimensionality is already defined,
    ``DimensionalityRedefinitionError`` will be raised.

    This can be used to define a new dimensionality for an amount
    of some specific substance. For instance, if the dimensionalities
    "air" and "fuel" are defined, the unit ``(kg air) / (kg fuel)`` has
    the simplified dimensionality of ``[air] / [fuel]``.

    .. note::
        Make sure to only define new custom dimensions using this function,
        since the unit needs to be appended to the ``CUSTOM_DIMENSIONS`` list as well.

    Parameters
    ----------
    name : str
        Name of the dimensionality
    symbol : str | None, optional
        Optional (short) symbol, by default None
    if_exists : Literal["raise", "warn"], optional
        What to do when a dimensionality with this name is already defined:
        ``"raise"`` (default) raises ``DimensionalityRedefinitionError``,
        ``"warn"`` logs a warning and keeps the existing definition
    """

    if if_exists not in ("raise", "warn"):
        raise ValueError(f"Invalid value: {if_exists=}")

    if not name.isidentifier():
        raise ValueError(
            f"Dimensionality name must be a valid Python identifier (alphanumeric and underscores, "
            f"cannot start with a digit, no spaces or special characters). Got: {name!r}"
        )

    if name in CUSTOM_DIMENSIONS:
        msg = f"Cannot define new dimensionality with name: {name}, a dimensionality with this name was already defined"
        if if_exists == "raise":
            raise DimensionalityRedefinitionError(msg)
        # the existing definition stays in place: defining it again on the
        # registry (on_redefinition="raise") would raise RedefinitionError
        _LOGGER.warning(msg)
        return

    definition_str = f"{name} = [{name}]"

    if symbol is not None:
        definition_str = f"{definition_str} = {symbol}"

    try:
        UNIT_REGISTRY.define(definition_str)
    except RedefinitionError as e:
        msg = f"Cannot define new dimensionality with name: {name}, a unit with this name was already defined"
        if if_exists == "warn":
            _LOGGER.warning(msg)
            return
        raise DimensionalityRedefinitionError(msg) from e
    CUSTOM_DIMENSIONS.append(name)


class _QuantityMeta(type):
    def __eq__(cls, obj: object) -> bool:
        # override the == operator so that type(val) == Quantity returns True for subclasses
        if obj is Quantity:
            return True

        return super().__eq__(obj)

    def __hash__(cls) -> int:
        # identity hash, which DELIBERATELY deviates from the eq/hash contract for
        # the `subclass == Quantity` convenience above: class-keyed caches
        # (typeguard, pydantic, pint) must keep every subclass a distinct key --
        # a hash shared with Quantity would let those caches alias any subclass
        # to the base class, returning entries for the wrong type
        return id(cls)


class Unit(PlainUnit, Generic[DT]):
    pass


class Quantity(
    NumpyQuantity[Any],
    NonMultiplicativeQuantity[Any],
    MeasurementQuantity[Any],
    Generic[DT, MT],
    metaclass=_QuantityMeta,
):
    """Physical quantity with a magnitude, unit, dimensionality, and magnitude type.

    ``Quantity`` extends Pint's quantity type with runtime dimensionality subclasses
    (for example ``Quantity[Pressure, float]``) and magnitude containers such as
    ``float``, numpy arrays, Polars ``Series`` and Polars ``Expr``. The runtime
    dimensionality subclass is selected from the unit; explicit semantic
    reinterpretation is validated by :meth:`asdim`.

    Absolute temperature and temperature-difference quantities deliberately remain
    separate dimensionality types even though both have Pint dimension
    ``[temperature]``. Use :meth:`to` for unit conversion and :meth:`asdim` only for
    explicit semantic reinterpretation between compatible dimensionality classes.
    """

    # constants
    NORMAL_M3_VARIANTS = ("nm³", "Nm³", "nm3", "Nm3", "nm**3", "Nm**3", "nm^3", "Nm^3")
    TEMPERATURE_DIFFERENCE_UCS = (
        Unit("delta_K")._units,
        Unit("delta_degC")._units,
        Unit("delta_degF")._units,
        Unit("delta_degRe")._units,
    )

    # Tolerance for == / != and the ordering operators, applied identically to every
    # magnitude type: two values are equal when |a - b| <= max(rtol * max(|a|, |b|), atol).
    # This is the math.isclose / polars is_close predicate; see the module-level _is_close.
    # NOTE: no comment line here may begin with "# type:" -- sphinx's source analyzer
    # parses that as a PEP 484 type comment, and a failure there drops the attribute
    # docstrings of the WHOLE module from the rendered API reference.
    rtol: float = 1e-9
    atol: float = 1e-12

    # compact, Latex, HTML, Latex/siunitx formatting
    FORMATTING_SPECS = PINT_FORMATTING_SPECIFIERS
    PINT_PRESENTATION_SPECS: ClassVar[tuple[str, ...]] = ("raw", "Lx", "D", "H", "P", "L", "C")

    _REGISTRY: ClassVar[UnitRegistry[Any]] = UNIT_REGISTRY

    # mapping from dimensionality subclass name to quantity subclass
    # this dict will be populated at runtime
    # use a custom class attribute (not cls.__subclasses__()) for more control
    _subclasses: ClassVar[dict[tuple[DimensionalityTypeName, MagnitudeTypeName | None], type[Quantity[Any, Any]]]] = {}
    _dimension_symbol_map: ClassVar[dict[sp.Basic, Unit]] = {}

    # used to validate dimensionality and magnitude type,
    # if None the dimensionality is not checked
    # subclasses of Quantity have this class attribute set, which
    # will restrict the dimensionality when creating the object
    _dimensionality_type: ClassVar[type[Dimensionality]] = UnknownDimensionality

    _magnitude: MT
    _magnitude_type: type[MT]

    _max_recursion_depth: int = 10

    # NOTE: __repr__ is inherited from pint and shows its generic class name
    # (<Quantity(5.0, 'delta_degree_Celsius')>), while type() and __str__ show the
    # dimensionality subclass. Overriding __repr__ would ripple through doctests,
    # notebook outputs and user logs for no functional gain -- settled, not an oversight.
    def __str__(self) -> str:
        return self.__format__(self._REGISTRY.formatter.default_format)

    def __hash__(self) -> int:
        if not isinstance(self.m, float):
            raise TypeError(f"unhashable type: 'Quantity' (magnitude type: {type(self.m).__name__})")

        # hash on the canonical root-unit representation so the eq/hash contract holds:
        # __eq__ compares across units (Q(1, "m") == Q(100, "cm")), so two quantities that
        # are equal must hash equal -- hashing the raw (m, u) would break that.
        # NOTE: __eq__ is tolerant (rtol, atol), so quantities that are only *approximately*
        # equal may still hash differently -- an inherent limit of tolerant equality, the same
        # way hash(0.1 + 0.2) != hash(0.3); exact equality (the common case) is consistent.
        root = self.to_root_units()

        # __eq__ also answers True against plain numbers (Q(0.5, "") == 0.5), so a
        # dimensionless quantity must hash like its root-unit magnitude (as pint does)
        if root.dimensionless:
            return hash(root.m)

        return hash((root.m, root.u))

    def __getattr__(self, item: str) -> Any:  # noqa: ANN401
        # private and dunder lookups (incl. the numpy __array_* protocol, and
        # _magnitude before it is assigned during construction) are left to
        # pint's NumpyQuantity.__getattr__
        if item.startswith("_"):
            return self._pint_super.__getattr__(item)

        # pl.Series and pl.Expr are the polars world. a pl.Expr is a deferred
        # plan (no data); a pl.Series holds data, but pint's numpy bridge is
        # flaky on it (half the reductions crash on the missing .size, and the
        # rest lose magnitude metadata). for both, the numpy-data surface
        # reached here misfires, so it is disabled -- only unit algebra
        # (arithmetic, comparison, .to, abs) is meaningful, and column/array
        # computation belongs on the underlying polars object, reached via .m
        if isinstance(self._magnitude, (pl.Expr, pl.Series)):
            raise AttributeError(
                f'"{item}" is not supported for Quantity with '
                f"{self._get_magnitude_type_name(type(self._magnitude))} magnitude, "
                'use the ".m" property to operate on the underlying polars object'
            )

        return self._pint_super.__getattr__(item)

    @property
    def _pint_super(self) -> Any:  # noqa: ANN401
        """
        The pint base-class implementation, typed as ``Any``.

        pint does not fully type the magnitude arithmetic and conversion
        methods, so calls that delegate to the base class are routed through
        this property and re-cast to the correct ``Quantity`` type. Using a
        zero-argument ``super()`` here is equivalent to calling ``super()``
        directly in the delegating method.
        """
        return super()

    # __array__ is numpy's coercion hook: np.asarray(qty) / np.array(qty) call
    # it to obtain the raw magnitude as an ndarray (unit stripped), which is how
    # a Quantity interops with numpy, matplotlib, pandas, etc. the override
    # exists because numpy 2.x passes copy=/dtype= kwargs that pint's __array__
    # does not accept. a polars magnitude belongs to the polars world (see
    # __getattr__), so it is not coerced here -- use .m and np.asarray(qty.m)
    def __array__(self, t: Any | None = None, copy: bool = False, dtype: str | None = None) -> np.ndarray:  # noqa: ANN401
        if isinstance(self._magnitude, (pl.Expr, pl.Series)):
            raise TypeError(
                f"Quantity with {self._get_magnitude_type_name(type(self._magnitude))} magnitude "
                'cannot be converted to a numpy array, use the ".m" property to access the polars object'
            )

        if SETTINGS.ignore_ndarray_unit_stripped_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UnitStrippedWarning,
                    message="The unit of the quantity is stripped when downcasting to ndarray.",
                )
                return self._pint_super.__array__(t)

        return self._pint_super.__array__(t)

    @staticmethod
    def validate_magnitude_type(mt: type) -> None:
        """Raise ``TypeError`` unless ``mt`` is a supported magnitude container type."""

        if mt == np.float64:
            raise TypeError(f"Invalid magnitude type: {mt}, expected one of float, np.ndarray, pl.Series, pl.Expr")

        if mt is float:
            return

        if mt is pl.Series:
            return

        if mt is pl.Expr:
            return

        if mt is np.ndarray or get_origin(mt) is np.ndarray:
            return

        raise TypeError(f"Invalid magnitude type: {mt}, expected one of float, np.ndarray, pl.Series, pl.Expr")

    @staticmethod
    def _get_magnitude_type_name(mt: object) -> MagnitudeTypeName:
        origin = get_origin(mt)

        if mt is np.ndarray or origin is np.ndarray:
            return "ndarray"
        elif mt is float:
            return "float"
        elif mt is pl.Series:
            return "pl.Series"
        elif mt is pl.Expr:
            return "pl.Expr"
        else:
            raise TypeError(f"Invalid magnitude type: {mt} (origin {origin})")

    @classmethod
    def _check_comparable_magnitudes(
        cls, m: object, other_m: object, operation: Literal["compare", "combine"] = "compare"
    ) -> None:
        # Mixed containers do not interoperate reliably: numpy-vs-polars raises an opaque
        # ufunc-loop TypeError or dtype error, and pl.Expr silently lifts an ndarray OR a
        # pl.Series into a literal and compares EXACTLY, skipping the (rtol, atol) tolerance
        # every sanctioned path applies -- with a length mismatch surfacing only at collect()
        # time as a ShapeError pointing into the plan. Arithmetic has the same problem and
        # can even change behavior after numpy/pint dispatch has been warmed. None of these
        # combinations is in the typed API, so reject them with an actionable message.
        if cls._is_mixed_container(m, other_m):
            if isinstance(m, pl.Expr) or isinstance(other_m, pl.Expr):
                hint = (
                    "evaluate the pl.Expr side first, or lift the other side into the "
                    "expression explicitly (e.g. with pl.lit(...)) if literal semantics are intended"
                )
            else:
                hint = 'convert one side first, e.g. with .astype("pl.Series") or .astype("ndarray")'

            raise TypeError(
                f"Cannot {operation} a Quantity with {cls._describe_magnitude(m)} magnitude with one with "
                f"{cls._describe_magnitude(other_m)} magnitude; {hint}"
            )

    @classmethod
    def _is_mixed_container(cls, m: object, other_m: object) -> bool:
        container_kinds = ("ndarray", "pl.Series", "pl.Expr")
        m_kind = cls._describe_magnitude(m)
        other_kind = cls._describe_magnitude(other_m)

        return m_kind in container_kinds and other_kind in container_kinds and m_kind != other_kind

    @staticmethod
    def _describe_magnitude(value: object) -> str:
        if isinstance(value, np.ndarray):
            return "ndarray"

        if isinstance(value, pl.Series):
            return "pl.Series"

        if isinstance(value, pl.Expr):
            return "pl.Expr"

        return type(value).__name__

    @staticmethod
    def _floor_magnitude(value: Any) -> Any:  # noqa: ANN401
        if isinstance(value, (pl.Series, pl.Expr)):
            return value.floor()
        if isinstance(value, np.ndarray):
            return np.floor(cast("Numpy1DArray", value))
        return float(np.floor(value))

    @staticmethod
    def _get_magnitude_type_from_name(mt_name: MagnitudeTypeName) -> type:
        match mt_name:
            case "float":
                return float
            case "pl.Expr":
                return pl.Expr
            case "pl.Series":
                return pl.Series
            case "ndarray":
                return Numpy1DArray
            case _:
                assert_never(mt_name)

    @staticmethod
    def get_unknown_dimensionality_subclass() -> type[Quantity[UnknownDimensionality, Any]]:
        """Return the magnitude-agnostic quantity subclass with unknown dimensionality."""

        return Quantity[UnknownDimensionality, Any]

    @classmethod
    def _get_dimensional_subclass(
        cls, dim: type[Dimensionality], mt: type | TypeVar | UnionType | None
    ) -> type[Quantity[DT, MT]]:
        # there are two levels of subclasses to Quantity: DimensionalQuantity and
        # DimensionalMagnitudeQuantity, which is a subclass of DimensionalQuantity
        # this distinction only exists at runtime, the type checker will use the
        # default magnitude type (the default for the MT typevar) in case the magnitude generic is omitted
        dim_name: DimensionalityTypeName = dim.__name__

        if cached_dim_qty := cls._subclasses.get((dim_name, None)):
            DimensionalQuantity = cast("type[Quantity[DT, Any]]", cached_dim_qty)
        else:
            DimensionalQuantity = cast(
                "type[Quantity[DT, Any]]",
                type(
                    f"Quantity[{dim_name}]",
                    (Quantity,),
                    {
                        "_dimensionality_type": dim,
                        "_magnitude_type": None,
                        "__class__": Quantity,
                    },
                ),
            )

            cls._subclasses[dim_name, None] = DimensionalQuantity

        if isinstance(mt, UnionType):
            raise TypeError(
                f"Type unions are not supported for magnitude type MT: {mt}. Use a single magnitude type instead"
            )

        if mt is None or mt is Any or isinstance(mt, TypeVar):
            return DimensionalQuantity

        cls.validate_magnitude_type(mt)
        mt_name = cls._get_magnitude_type_name(mt)

        # check if an existing DimensionalMagnitudeQuantity subclass already has been created
        if cached_dim_magnitude_qty := cls._subclasses.get((dim_name, mt_name)):
            return cast("type[Quantity[DT, MT]]", cached_dim_magnitude_qty)

        DimensionalMagnitudeQuantity = cast(
            "type[Quantity[DT, MT]]",
            type(
                f"Quantity[{dim_name}, {mt_name}]",
                (DimensionalQuantity,),  # ty: ignore[unsupported-dynamic-base]
                {
                    "_magnitude_type": mt,
                    "__class__": DimensionalQuantity,
                },
            ),
        )

        cls._subclasses[dim_name, mt_name] = DimensionalMagnitudeQuantity
        return DimensionalMagnitudeQuantity

    @staticmethod
    def _is_incomplete_dimensionality(dim: type[Dimensionality] | TypeVar) -> bool:
        return dim == UnknownDimensionality or dim is Any or isinstance(dim, TypeVar)

    @staticmethod
    def _units_containers_equal(
        uc1: UnitsContainer, uc2: UnitsContainer, rtol: float = 1e-9, atol: float = 1e-12
    ) -> bool:
        if set(uc1) != set(uc2):
            return False

        for dim_name in uc1:
            _exp1 = uc1[dim_name]
            _exp2 = uc2[dim_name]

            if isinstance(_exp1, complex):
                raise TypeError(f"Exponent for {dim_name=} cannot be complex: {_exp1}")
            if isinstance(_exp2, complex):
                raise TypeError(f"Exponent for {dim_name=} cannot be complex: {_exp2}")

            exp1 = float(_exp1)
            exp2 = float(_exp2)

            # runs on every Quantity construction: keep this pure-Python, a
            # vectorized isclose costs microseconds per exponent on plain floats
            if not math.isclose(exp1, exp2, rel_tol=rtol, abs_tol=atol):
                return False

        return True

    def __class_getitem__(cls, types: type[DT] | tuple[type[DT], type[MT]]) -> type[Quantity[DT, MT]]:
        if isinstance(types, tuple):
            if len(types) != 2:
                raise TypeError(f"Incorrect number of generic type parameters: {len(types)} (expected 1 or 2): {types}")

            dim, mt = types
        else:
            dim = types
            mt = None

        if dim == Dimensionality:
            raise TypeError(f"Generic type parameter to Quantity cannot be the Dimensionality base class: {dim}")

        if cls._is_incomplete_dimensionality(dim):
            return cls._get_dimensional_subclass(UnknownDimensionality, mt)

        if not isinstance(cast(object, dim), type):
            raise TypeError(
                f"Generic type parameter to Quantity must be a type, passed an instance of {type(dim)}: {dim}"
            )

        # check if the attribute dimensions exists instead of using issubclass()
        # issubclass does not work well with autoreloading in Jupyter
        if not hasattr(dim, "dimensions"):
            raise TypeError(f"Generic type parameter to Quantity has no attribute 'dimensions', passed: {dim}")

        dimensions = getattr(dim, "dimensions", None)

        if dimensions is None:
            raise TypeError(
                "Generic type parameter to Quantity is missing "
                f"or has explicitly set attribute 'dimensions' to None: {dim}"
            )

        if not isinstance(dimensions, UnitsContainer):
            raise TypeError(
                "Type parameter to Quantity has incorrect type for attribute dimensions: UnitsContainer, "
                f"passed: {dim} with dimensions: {dimensions} ({type(dimensions)})"
            )

        dim_ = cast(type[Dimensionality], dim)
        subcls = cls._get_dimensional_subclass(dim_, mt)
        return subcls

    @staticmethod
    def _validate_unit(
        unit: object,
    ) -> Unit[DT]:
        if unit is None:
            return Unit("dimensionless")
        elif isinstance(unit, Unit):
            return Unit(unit)
        elif isinstance(unit, Quantity):
            # only reachable through to()/ito(), where "convert to another quantity's
            # unit" is intended -- the constructor rejects a Quantity unit outright
            qty = cast("Quantity[DT, Any]", unit)
            return qty.u
        elif isinstance(unit, dict):
            # compatibility with internal pint API
            return Unit(Quantity._validate_unit(str(UnitsContainer(unit))))
        elif isinstance(unit, UnitsContainer):
            # compatibility with internal pint API
            return Unit(Quantity._validate_unit(str(unit)))
        elif isinstance(unit, str):
            return cast("Unit[DT]", Unit(cast(Any, Quantity._REGISTRY.parse_units(Quantity.correct_unit(unit)))))
        else:
            # e.g. swapped (val, unit) arguments, or a unit object from a foreign
            # pint registry -- fail with the accepted types instead of a bare
            # AttributeError from str-only parsing
            raise TypeError(
                f"unit must be a str, Unit, UnitsContainer, dict or None, got {type(unit).__name__}: {unit!r} "
                "(a Quantity is accepted by .to()/.ito(), not by the Quantity constructor)"
            )

    @staticmethod
    def _validate_magnitude(val: MT | Sequence[float] | str | bytes | None) -> MT:
        if val is None or isinstance(val, (str, bytes)):
            raise ValueError(
                "magnitude must be a numeric scalar, sympy atom, Polars object, "
                "or 1-dimensional sequence of real numbers; strings and None are not supported"
            )

        # bool is an int subclass, so Q(True) would otherwise become 1.0 dimensionless.
        # A bool magnitude is always a mistake (e.g. a swapped argument or a comparison
        # result); the coolprop composition path rejects it for the same reason
        if isinstance(val, bool):
            raise TypeError("magnitude must be a real number, not a bool")

        if isinstance(val, int):
            return cast("MT", float(val))
        elif isinstance(val, float):
            # numpy float64 is a runtime subclass of float, so the type system
            # cannot distinguish it; this normalization to a plain Python float
            # is genuinely needed but unavoidably looks redundant to the checker
            return cast("MT", float(val))
        elif isinstance(val, np.ndarray):
            if len(val.shape) != 1:
                raise ValueError(f"Only 1-dimensional NumPy arrays can be used as magnitude, got shape {val.shape}")
            return cast("MT", Quantity._cast_array_float(val))
        elif isinstance(val, pl.Series):
            if val.dtype == pl.Null:
                return cast("MT", val.cast(pl.Float64))
            if val.dtype.is_integer():
                return cast("MT", val.cast(pl.Float64))
            if not val.dtype.is_float():
                raise TypeError(
                    f"Polars Series magnitude must have a float or integer dtype, got {val.dtype!r}. "
                    "Boolean, non-numeric, nested, and unit-typed Series are not valid magnitudes."
                )
            return cast("MT", val)
        elif isinstance(val, pl.Expr):
            return cast("MT", val)
        elif hasattr(val, "is_Atom"):
            # implicit way of checking if the value is a sympy symbol without having to import SymPy
            # (must come before the numbers.Real check: sympy Float/Integer register as Real,
            # but a sympy magnitude must be kept symbolic)
            return cast("MT", val)
        elif isinstance(val, numbers.Real):
            # remaining real scalars that are not float subclasses: numpy scalars
            # such as np.int64 / np.int32 / np.float32 (e.g. from arr.sum() on an
            # integer array), Fraction, ... -- normalize to a plain float instead
            # of falling through to np.array(), which fails on the 0-d shape
            return cast("MT", float(val))
        else:
            arr = np.array(val)
            if len(arr.shape) == 0:
                raise TypeError(
                    "magnitude must be a numeric scalar, sympy atom, Polars object, "
                    f"or 1-dimensional sequence of real numbers; got {type(val).__name__}"
                )
            if len(arr.shape) != 1:
                raise ValueError(f"Only 1-dimensional sequences can be used as magnitude, got shape {arr.shape}")
            return cast("MT", Quantity._cast_array_float(arr))

    @classmethod
    def get_unit(cls, unit_name: AllUnits | str) -> Unit:
        """Parse ``unit_name`` with the encomp unit registry and return a :class:`Unit`."""

        return Unit(cast(Any, cls._REGISTRY.parse_units(unit_name)))

    def get_subclass(self, dt: type[DT_], mt: type[MT_]) -> type[Quantity[DT_, MT_]]:
        """Return the ``Quantity`` subclass for dimensionality ``dt`` and magnitude type ``mt``."""

        subcls = self._get_dimensional_subclass(dt, mt)
        return cast("type[Quantity[DT_, MT_]]", subcls)

    def _call_subclass(
        self,
        m: MT | MT_,
        unit: Unit[DT] | Unit[DT_] | UnitsContainer,
    ) -> Quantity[DT, MT]:
        u = cast(Unit[DT], unit)
        dt = self.dt

        mt = self._get_magnitude_type_safe(cast(type[MT], type(m)))
        subcls = self.get_subclass(dt, mt)

        return cast("Quantity[Any, Any]", subcls(cast(MT, m), u))

    def __len__(self) -> int:
        # __len__() must return an integer
        # the len() function ensures this at a lower level
        if isinstance(self._magnitude, float | int):
            raise TypeError(f"Quantity with scalar magnitude ({self._magnitude}) has no length")
        elif isinstance(self._magnitude, pl.Expr):
            raise TypeError(f"Cannot determine length of Polars expression: {self._magnitude}")

        return len(cast(Sized, self._magnitude))

    def __copy__(self) -> Quantity[DT, MT]:
        return self._call_subclass(copy.copy(self._magnitude), self._units)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Quantity[DT, MT]:
        if memo is None:
            memo = {}

        return self._call_subclass(copy.deepcopy(self._magnitude, memo), copy.deepcopy(self._units, memo))

    def __reduce__(  # pyright: ignore[reportIncompatibleMethodOverride]  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        self,
    ) -> tuple[object, tuple[object, str, type[Dimensionality] | None]]:
        dim = self.dt if _is_pickle_global(self.dt) else None
        return (_reconstruct_quantity, (self._magnitude, str(self.u._units), dim))

    @staticmethod
    def _cast_array_float(inp: np.ndarray) -> Numpy1DArray:
        # don't fail in case the array contains unsupported objects,
        # cast to float64, matches the Numpy1DArray type definition
        if inp.dtype.kind in {"S", "U"}:
            raise ValueError("magnitude sequences must contain real numbers, not strings")

        if inp.dtype.kind == "O":
            for item in inp:
                if item is None or isinstance(item, (str, bytes)) or not isinstance(item, numbers.Real):
                    raise ValueError(
                        f"magnitude sequences must contain real numbers; got {type(item).__name__}: {item!r}"
                    )

        if inp.dtype == np.float64:
            return cast("Numpy1DArray", inp)

        try:
            return inp.astype(np.float64, casting="unsafe", copy=True)
        except (TypeError, ValueError) as e:
            raise ValueError("magnitude sequences must contain real numbers") from e

    @overload
    def __new__(cls, val: Sequence[float]) -> Quantity[Dimensionless, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: None) -> Quantity[Dimensionless, Numpy1DArray]: ...
    @overload
    def __new__(  # pyright: ignore[reportOverlappingOverload]  # list inputs intentionally yield ndarray
        cls, val: Sequence[float], unit: DimensionlessUnits
    ) -> Quantity[Dimensionless, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: CurrencyUnits) -> Quantity[Currency, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: CurrencyPerEnergyUnits
    ) -> Quantity[CurrencyPerEnergy, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: CurrencyPerVolumeUnits
    ) -> Quantity[CurrencyPerVolume, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: CurrencyPerMassUnits) -> Quantity[CurrencyPerMass, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: CurrencyPerTimeUnits) -> Quantity[CurrencyPerTime, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: LengthUnits) -> Quantity[Length, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: MassUnits) -> Quantity[Mass, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: TimeUnits) -> Quantity[Time, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: FrequencyUnits) -> Quantity[Frequency, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: TemperatureUnits) -> Quantity[Temperature, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: TemperatureDifferenceUnits
    ) -> Quantity[TemperatureDifference, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: SubstanceUnits) -> Quantity[Substance, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: MolarMassUnits) -> Quantity[MolarMass, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: SubstancePerMassUnits) -> Quantity[SubstancePerMass, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: CurrentUnits) -> Quantity[Current, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: LuminosityUnits) -> Quantity[Luminosity, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: AreaUnits) -> Quantity[Area, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: VolumeUnits) -> Quantity[Volume, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: NormalVolumeUnits) -> Quantity[NormalVolume, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: PressureUnits) -> Quantity[Pressure, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: MassFlowUnits) -> Quantity[MassFlow, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: VolumeFlowUnits) -> Quantity[VolumeFlow, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: NormalVolumeFlowUnits) -> Quantity[NormalVolumeFlow, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: NormalVolumePerMassUnits
    ) -> Quantity[NormalVolumePerMass, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: MassPerNormalVolumeUnits
    ) -> Quantity[MassPerNormalVolume, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: DensityUnits) -> Quantity[Density, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: MolarDensityUnits) -> Quantity[MolarDensity, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: SpecificVolumeUnits) -> Quantity[SpecificVolume, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: EnergyUnits) -> Quantity[Energy, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: PowerUnits) -> Quantity[Power, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: PowerPerAreaUnits) -> Quantity[PowerPerArea, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: VelocityUnits) -> Quantity[Velocity, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: ForceUnits) -> Quantity[Force, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: DynamicViscosityUnits) -> Quantity[DynamicViscosity, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: KinematicViscosityUnits
    ) -> Quantity[KinematicViscosity, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: EnergyPerMassUnits) -> Quantity[EnergyPerMass, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: MolarSpecificEnthalpyUnits
    ) -> Quantity[MolarSpecificEnthalpy, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: SpecificHeatCapacityUnits
    ) -> Quantity[SpecificHeatCapacity, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: ThermalConductivityUnits
    ) -> Quantity[ThermalConductivity, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: HeatTransferCoefficientUnits
    ) -> Quantity[HeatTransferCoefficient, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: Unit[DT]) -> Quantity[DT, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: Sequence[float], unit: UnitsContainer | Unit
    ) -> Quantity[UnknownDimensionality, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: str) -> Quantity[UnknownDimensionality, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: MT) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: None) -> Quantity[Dimensionless, MT]: ...
    @overload
    # overlaps the list-input overload above, which intentionally returns an
    # ndarray magnitude (lists are converted to arrays) regardless of MT
    def __new__(cls, val: MT, unit: DimensionlessUnits) -> Quantity[Dimensionless, MT]: ...  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __new__(cls, val: MT, unit: CurrencyUnits) -> Quantity[Currency, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: CurrencyPerEnergyUnits) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: CurrencyPerVolumeUnits) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: CurrencyPerMassUnits) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: CurrencyPerTimeUnits) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: LengthUnits) -> Quantity[Length, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: MassUnits) -> Quantity[Mass, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: TimeUnits) -> Quantity[Time, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: FrequencyUnits) -> Quantity[Frequency, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: TemperatureUnits) -> Quantity[Temperature, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: TemperatureDifferenceUnits) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: SubstanceUnits) -> Quantity[Substance, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: MolarMassUnits) -> Quantity[MolarMass, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: SubstancePerMassUnits) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: CurrentUnits) -> Quantity[Current, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: LuminosityUnits) -> Quantity[Luminosity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: AreaUnits) -> Quantity[Area, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: VolumeUnits) -> Quantity[Volume, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: NormalVolumeUnits) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: PressureUnits) -> Quantity[Pressure, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: MassFlowUnits) -> Quantity[MassFlow, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: VolumeFlowUnits) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: NormalVolumeFlowUnits) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: NormalVolumePerMassUnits) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: MassPerNormalVolumeUnits) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: DensityUnits) -> Quantity[Density, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: MolarDensityUnits) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: SpecificVolumeUnits) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: EnergyUnits) -> Quantity[Energy, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: PowerUnits) -> Quantity[Power, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: PowerPerAreaUnits) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: VelocityUnits) -> Quantity[Velocity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: ForceUnits) -> Quantity[Force, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: DynamicViscosityUnits) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: KinematicViscosityUnits) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: EnergyPerMassUnits) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: MolarSpecificEnthalpyUnits) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: SpecificHeatCapacityUnits) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: ThermalConductivityUnits) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: HeatTransferCoefficientUnits) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __new__(cls, val: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __new__(cls, val: Quantity[DT, MT], unit: Unit[DT]) -> Quantity[DT, MT]: ...
    @overload
    def __new__(cls, val: Quantity[DT, MT], unit: UnitsContainer | Unit) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __new__(cls, val: Quantity[UnknownDimensionality, MT], unit: Unit[DT_]) -> Quantity[DT_, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: Unit[DT]) -> Quantity[DT, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: UnitsContainer | Unit) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: str) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __new__(  # fallback overload: any supported magnitude type not covered above
        cls,
        # str is deliberately absent: string parsing like Q("24 kg") must be rejected
        # statically (it also raises ValueError at runtime) -- Quantity magnitudes and
        # units are always passed separately
        val: float | np.ndarray[Any, Any] | pl.Series | pl.Expr | Sequence[float] | Quantity[Any, Any] | sp.Basic,
        # Quantity is deliberately absent: passing one as the unit drops its magnitude,
        # which is a bug rather than a shorthand. It is rejected here and at runtime;
        # .to() does accept a Quantity, where reusing another quantity's unit is the point
        unit: Unit[Any] | UnitsContainer | dict[str, Any] | str | None = None,
        _depth: int = 0,
    ) -> Quantity[Any, Any]: ...
    def __new__(
        cls,
        val: Any,
        unit: Any = None,
        _depth: int = 0,
    ) -> Quantity[Any, Any]:
        if isinstance(unit, Quantity):
            raise TypeError(
                f"unit must be a str, Unit, UnitsContainer, dict or None, got Quantity: {unit!r}. "
                'Pass the unit itself ("qty.u"), or use "value.to(qty)" to convert to another '
                "quantity's unit"
            )

        unit = cast("Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | None", unit)

        if isinstance(val, Quantity):
            _input_qty = cast("Quantity[DT, MT]", val)
            if unit is not None:
                _input_qty = _input_qty.to(unit)

            val, unit = _input_qty.m, _input_qty.u

        valid_magnitude = cls._validate_magnitude(val)
        valid_unit = cls._validate_unit(unit)

        if issubclass(cls._dimensionality_type, TemperatureDifference) and cls._is_offset_temperature_unit(valid_unit):
            raise DimensionalityTypeError(
                f"Cannot construct TemperatureDifference with offset unit {valid_unit}; use a delta unit instead"
            )

        is_valid_subclass = True

        if cls._is_incomplete_dimensionality(cls._dimensionality_type):
            is_valid_subclass = False
        else:
            # compare dimensionalities with tolerance for float precision
            ucs_equal = cls._units_containers_equal(cls._dimensionality_type.dimensions, valid_unit.dimensionality)

            if not ucs_equal:
                is_valid_subclass = False

        if not is_valid_subclass:
            # NOTE: cannot validate that the subclass has the same dimensionality as the input unit
            # cannot raise error here since this breaks the pint.PlainQuantity methods
            # that use return self.__class__(...)
            # need to be able to change dimensionality (i.e. type) via the __new__ method

            if _depth > cls._max_recursion_depth:
                raise RecursionError(
                    f"RecursionError ({_depth=}) when constructing Quantity class with {val=}, {unit=}"
                )

            # special case for temperature difference
            if cls._is_temperature_difference_unit(valid_unit):
                subcls = cls._get_dimensional_subclass(TemperatureDifference, type(valid_magnitude))  # ty: ignore[invalid-argument-type]
            else:
                dim = Dimensionality.get_dimensionality(valid_unit.dimensionality)
                subcls = cls._get_dimensional_subclass(dim, type(valid_magnitude))  # ty: ignore[invalid-argument-type]

            return subcls(
                valid_magnitude,
                valid_unit,
                _depth=_depth + 1,
            )

        qty = cast("Quantity[DT, MT]", cast(Any, super()).__new__(cls, valid_magnitude, units=valid_unit))

        _m = qty._magnitude
        if isinstance(_m, np.ndarray) and _m.dtype != np.float64:
            qty._magnitude = cast("MT", cls._cast_array_float(_m))

        return qty

    @property
    def m(self) -> MT:
        """Magnitude value stored in this quantity."""

        return self._magnitude

    @m.setter
    def m(self, val: MT) -> None:
        # in-place magnitude replacement: validate like the constructor does, and
        # require the magnitude type to stay unchanged -- switching the type in
        # place would desync the instance from its Quantity[DT, MT] subclass
        validated = self._validate_magnitude(val)

        if type(validated) is not type(self._magnitude):
            raise TypeError(
                f"Cannot set a magnitude of type {type(validated).__name__} on a Quantity with "
                f"{type(self._magnitude).__name__} magnitude, "
                "construct a new Quantity or use .astype() to change the magnitude type"
            )

        if (
            isinstance(self._magnitude, Sized)
            and isinstance(validated, Sized)
            and len(validated) != len(self._magnitude)
        ):
            raise ValueError(f"Cannot replace magnitude of length {len(self._magnitude)} with length {len(validated)}")

        self._magnitude = validated

    @property
    def mt(self) -> type[MT]:
        """Concrete magnitude container type for this quantity."""

        # _magnitude_type is set on the magnitude-specific subclass, but pint
        # builds results via self.__class__(...), which resolves to the
        # magnitude-agnostic dimension-only subclass (where it is None). that is
        # intentional -- the result magnitude type may differ from the source.
        # _get_magnitude_type_safe falls back to the live magnitude's type when
        # given an invalid value (None included), so it recovers the real type
        return self._get_magnitude_type_safe(self._magnitude_type)

    @property
    def mt_name(self) -> MagnitudeTypeName:
        """String name for this quantity's magnitude container type."""

        return self._get_magnitude_type_name(self.mt)

    @property
    def units(self) -> Unit[DT]:
        """Quantity unit as an encomp :class:`Unit` instance."""

        return Unit(super().units)

    @property
    def u(self) -> Unit[DT]:
        """Short alias for :attr:`units`."""

        return self.units

    @property
    def dt(self) -> type[DT]:
        """Dimensionality class associated with this quantity."""

        return cast(type[DT], self._dimensionality_type)

    @property
    def _is_temperature_difference(self) -> bool:
        return issubclass(self.dt, TemperatureDifference)

    @classmethod
    def _is_temperature_difference_unit(cls, unit: Unit[DT]) -> bool:
        return unit._units in cls.TEMPERATURE_DIFFERENCE_UCS

    @classmethod
    def _is_offset_temperature_unit(cls, unit: Unit[Any]) -> bool:
        if unit.dimensionality != Temperature.dimensions:
            return False

        if cls._is_temperature_difference_unit(unit):
            return False

        registry = cast(Any, cls._REGISTRY)  # _is_multiplicative is a stable pint internal
        return not all(registry._is_multiplicative(u) for u in unit._units)

    @classmethod
    def _as_temperature_difference_unit(cls, unit: Unit[Any]) -> Unit[TemperatureDifference]:
        if unit._units == Unit("delta_K")._units:
            return Unit("delta_K")

        if unit._units == Unit("degC")._units:
            return Unit("delta_degC")

        if unit._units == Unit("degF")._units:
            return Unit("delta_degF")

        if unit._units == Unit("degRe")._units:
            return Unit("delta_degRe")

        if cls._is_offset_temperature_unit(unit):
            raise DimensionalityTypeError(f"Cannot reinterpret offset temperature unit {unit} as TemperatureDifference")

        return cast("Unit[TemperatureDifference]", unit)

    def _check_temperature_compatibility(self, unit: Unit[DT]) -> None:
        if self._is_temperature_difference and unit._units not in self.TEMPERATURE_DIFFERENCE_UCS:
            # a temperature difference is a scale-only quantity: any multiplicative
            # [temperature] unit (K, degR, mK, ...) expresses it correctly, and the
            # converted Quantity keeps the TemperatureDifference dimensionality.
            # Offset scales (degC, degF) stay refused -- their zero point would
            # silently reinterpret the difference as an absolute temperature
            registry = cast(Any, self._REGISTRY)  # _is_multiplicative is a stable pint internal
            if all(registry._is_multiplicative(u) for u in unit._units):
                return

            current_name = self.dt.__name__
            new_name = Quantity(1, unit)._dimensionality_type.__name__

            raise DimensionalityTypeError(
                f"Cannot convert {self.units} (dimensionality {current_name}) to {unit} (dimensionality {new_name})"
            )

        # the reverse direction: pint happily scale-converts an absolute temperature
        # to a delta unit (K -> delta_degC), silently reinterpreting an absolute
        # temperature as a difference (and changing the dimensionality type, which
        # to() must preserve). Require the explicit asdim() escape hatch instead.
        if (
            not self._is_temperature_difference
            and self._is_temperature_difference_unit(unit)
            and self.dimensionality == unit.dimensionality
        ):
            raise DimensionalityTypeError(
                f"Cannot convert {self.units} (dimensionality {self.dt.__name__}) to {unit} "
                "(dimensionality TemperatureDifference). Use .asdim(TemperatureDifference) "
                "to explicitly reinterpret an absolute temperature as a difference"
            )

    def _get_magnitude_type_safe(self, mt: type[MT]) -> type[MT]:
        # fall back to the source instance magnitude type in case
        # unsupported magnitudes are used, e.g. sp.Symbol
        # typing does not work at all in this case
        try:
            self.validate_magnitude_type(mt)
        except TypeError:
            mt = type(self.m)

            try:
                self.validate_magnitude_type(mt)
            except TypeError:
                mt = cast(type[MT], float)

        return mt

    def to_reduced_units(self) -> Quantity[DT, MT]:
        """Return a copy with units reduced by canceling common factors."""

        ret = cast("Quantity[DT, MT]", self._pint_super.to_reduced_units())
        return ret

    def to_root_units(self) -> Quantity[DT, MT]:
        """Return a copy converted to Pint root units."""

        ret = cast("Quantity[DT, MT]", super().to_root_units())
        return ret

    def to_base_units(self) -> Quantity[DT, MT]:
        """Return a copy converted to base SI units."""

        self._check_temperature_compatibility(Unit("kelvin"))
        ret = super().to_base_units()
        return cast("Quantity[DT, MT]", ret)

    def _dimensionalities_match(self, unit: Unit[DT_]) -> bool:
        src_dim = cast(dict[str, float], dict(self.dimensionality))
        dst_dim = cast(dict[str, float], dict(unit.dimensionality))

        if set(src_dim.keys()) != set(dst_dim.keys()):
            return False

        return all(
            abs(src_dim.get(key, 0.0) - dst_dim.get(key, 0.0)) < 1e-10
            for key in set(src_dim.keys()) | set(dst_dim.keys())
        )

    def _check_dimensionality(self, dimensions: UnitsContainer) -> bool:
        return self._units_containers_equal(self.dimensionality, dimensions)

    def _to_unit(
        self, unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | Quantity[DT, Any]
    ) -> Unit[DT]:
        return self._validate_unit(unit)

    def to(
        self, unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | Quantity[DT, Any]
    ) -> Quantity[DT, MT]:
        """Return a new quantity converted to ``unit``.

        The returned quantity keeps this quantity's dimensionality type. The target
        may be a unit string, :class:`Unit`, :class:`UnitsContainer`, dict accepted by
        Pint, or another quantity whose unit should be used. Absolute temperatures
        cannot be converted to delta temperature units, and temperature differences
        cannot be converted to offset temperature units; use :meth:`asdim` when that
        reinterpretation is intentional.
        """
        valid_unit = self._to_unit(unit)
        self._check_temperature_compatibility(valid_unit)

        m: MT
        try:
            m = self._pint_super._convert_magnitude_not_inplace(valid_unit)
        except DimensionalityError as e:
            # if direct conversion fails due to complex fractional units,
            # try converting to base units first, then to the target unit
            if self._dimensionalities_match(valid_unit):
                base_quantity = self.to_base_units()
                m = cast(Any, base_quantity)._convert_magnitude_not_inplace(valid_unit)
            else:
                raise e

        if self._is_temperature_difference_unit(valid_unit):
            return cast("Quantity[Any, Any]", Quantity(m, valid_unit))

        converted = self._call_subclass(m, valid_unit)

        return converted

    def m_as(
        self,
        units: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | Quantity[DT, Any],
    ) -> MT:
        """Return the magnitude converted to ``unit``.

        This is the typed shorthand for ``self.to(unit).m`` and applies exactly the
        same dimensionality and temperature-safety checks as :meth:`to`.
        """
        return self.to(units).m

    def ito(self, unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number]) -> None:
        """Convert this quantity in place to ``unit``.

        Like :meth:`to`, this preserves the dimensionality type and applies the same
        temperature-vs-temperature-difference checks. Integer numpy magnitudes may be
        copied to floating point before conversion so unit scaling cannot fail due to
        integer casting rules.
        """
        # NOTE: this method cannot convert the dimensionality type
        # (temperature <-> temperature difference is refused by
        # _check_temperature_compatibility in both directions)
        valid_unit = self._to_unit(unit)
        self._check_temperature_compatibility(valid_unit)

        # it's not safe to convert units as int, the
        # user will have to convert back to int if necessary
        # better to use ":.0f" formatting or round() anyway

        # avoid numpy.core._exceptions.UFuncTypeError (not on all platforms?)
        # convert integer arrays to float(64) (creating a copy)
        _m = self._magnitude
        if isinstance(_m, np.ndarray) and issubclass(_m.dtype.type, numbers.Integral):
            self._magnitude = cast("MT", _m.astype(np.float64))  # ty: ignore[no-matching-overload]

        try:
            self._pint_super.ito(valid_unit)
        except DimensionalityError as e:
            if self._dimensionalities_match(valid_unit):
                base_quantity = self.to_base_units()
                converted_magnitude = cast(Any, base_quantity)._convert_magnitude_not_inplace(valid_unit)
                self._magnitude = cast(MT, converted_magnitude)
                self._units = valid_unit._units
            else:
                raise e

    # check() intentionally accepts a wider set of dimension arguments than
    # pint's PlainQuantity.check, so the override signature is incompatible
    def check(  # pyright: ignore[reportIncompatibleMethodOverride]  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        self,
        dimension: Quantity[Any, Any] | UnitsContainer | Unit[DT_] | Unit | str | Dimensionality | type[Dimensionality],
    ) -> bool:
        """
        Return whether this quantity has the same physical dimensions as ``dimension``.

        This intentionally compares dimensions, not semantic dimensionality classes:
        ``Temperature`` and ``TemperatureDifference`` both have ``[temperature]``,
        and sibling energy-per-mass classes share ``[energy] / [mass]``. Use
        ``isinstance``/``isinstance_types`` or ``check_compatibility`` when that
        semantic distinction matters.
        """
        if isinstance(dimension, Quantity):
            return self._check_dimensionality(dimension.dimensionality)

        if isinstance(dimension, Unit):
            return self._check_dimensionality(dimension.dimensionality)

        if isinstance(dimension, UnitsContainer):
            return self._check_dimensionality(dimension)

        if isinstance(dimension, str):
            if dimension.strip().startswith("["):
                return bool(self._pint_super.check(dimension.strip()))

            return self.check(self._validate_unit(dimension))

        dimension_any = cast(Any, dimension)

        if isinstance(dimension_any, type):
            if not issubclass(dimension_any, Dimensionality):
                raise TypeError(f"Invalid type for dimension: {dimension} ({type(dimension)})")

            return self._check_dimensionality(dimension_any.dimensions)

        if isinstance(dimension_any, Dimensionality):
            return self._check_dimensionality(dimension_any.dimensions)

        if isinstance(dimension_any, PlainQuantity):
            raise TypeError(f"Invalid type for dimension: {dimension} ({type(dimension)})")

        return self._pint_super.check(dimension)

    @classmethod
    def _has_pint_presentation_spec(cls, spec: str) -> bool:
        spec_without_modifiers = spec.replace("~", "").replace("^", "")
        return any(spec_without_modifiers.endswith(pint_spec) for pint_spec in cls.PINT_PRESENTATION_SPECS)

    def __format__(self, spec: str) -> str:
        if spec and not self._has_pint_presentation_spec(spec):
            spec = f"{spec}{self._REGISTRY.formatter.default_format}"

        return super().__format__(spec)

    @staticmethod
    def correct_unit(unit: str) -> str:
        """Normalize supported unit spelling variants before Pint parses them.

        Notably, ``Nm3``/``nm3`` forms are interpreted as normal cubic meters
        (``normal_cubic_meter``), not nanometers cubed. Use ``nanometer**3``
        when a nanoscale volume is intended.
        """

        unit = unit.strip()

        if unit == "-":
            return "dimensionless"

        # normal cubic meter, not nano or Newton
        # there's no consistent way of abbreviating "normal liter",
        # so we'll not even try to parse that, use "nanometer**3" if necessary
        for n in Quantity.NORMAL_M3_VARIANTS:
            if n in unit:
                # the named unit (defined in defs/units.txt) displays as Nm³; being a
                # single token it also composes safely, e.g. "kg/nm3"
                unit = unit.replace(n, "normal_cubic_meter")

        # NOTE: the order of replacements matters here
        replacements = {
            # there are two different delta signs, we want the latter
            "∆": "Δ",
            "°C": "degC",
            "°F": "degF",
            "℃": "degC",
            "℉": "degF",
            # Δ% should maybe be its own unit, but this is not implemented for now
            "Δ%": "%",
            "‰": "permille",
            "r/min": "rpm",
            # ΔK does not really make sense, it's not an offset scale
            "Δ": "delta_",
        }

        for old, new in replacements.items():
            if old in unit:
                unit = unit.replace(old, new)

        percent_basis = r"(?:mol(?:e)?|kg|g|m(?:3|³|\^3|\*\*3)|vol(?:ume)?|mass|wt|weight)"
        unit = re.sub(rf"(?<![A-Za-z0-9_]){percent_basis}\s*-?\s*%", "%", unit, flags=re.IGNORECASE)

        unit = re.sub(r"(?<=[A-Za-z0-9_])%", " percent", unit)
        unit = unit.replace("%", "percent")

        # add ** between letters and numbers if they
        # are right next to each other and if the number is at a word boundary
        unit = re.sub(r"([A-Za-z])(\d+)\b", r"\1**\2", unit)

        return unit

    def _sympy_(self) -> sp.Basic:
        _ensure_sympy()

        if self.dimensionless:
            return cast("sp.Basic", cast(Any, sp).sympify(self.to_base_units().m))

        base_qty = self.to_base_units()

        unit_parts: list[str] = []
        symbols: list[str] = []

        for unit_name, power in base_qty.u._units.items():
            unit_symbol = self._REGISTRY.get_symbol(unit_name)
            unit_parts.append(f"{unit_symbol}**{power}")
            symbols.append(unit_symbol)

        unit_repr = " * ".join(unit_parts)

        if not unit_repr.strip():
            unit_repr = "1"

        # use \text{symbol} to make sure that the unit symbols
        # do not clash with commonly used symbols like "m" or "s"
        expr = cast(
            "sp.Basic",
            cast(Any, sp)
            .sympify(f"{base_qty.m} * {unit_repr}")
            .subs({sp.Symbol(n): self.get_unit_symbol(n) for n in symbols}),
        )

        return expr

    @staticmethod
    def get_unit_symbol(s: str) -> sp.Symbol:
        _ensure_sympy()
        return sp.Symbol("\\text{" + s + "}", nonzero=True, positive=True)

    @classmethod
    def _populate_dimension_symbol_map(cls) -> None:
        # also consider custom dimensions defined
        # with encomp.units.define_dimensionality
        cls._dimension_symbol_map |= {
            cls.get_unit_symbol(n): cls.get_unit(n) for n in list(BASE_SI_UNITS) + CUSTOM_DIMENSIONS
        }

    @classmethod
    def from_expr(cls, expr: sp.Basic) -> Quantity[DT, float]:
        """Create a scalar quantity from a SymPy expression containing SI unit symbols.

        The expression is expected to use the symbols produced by
        :meth:`Quantity._sympy_`. Residual symbols that are not known unit
        symbols raise ``KeyError``.
        """

        # this needs to be populated here to account for custom dimensions
        cls._populate_dimension_symbol_map()

        expr = cast("sp.Basic", cast(Any, expr).simplify())
        args = expr.args

        if not args:
            val = float(cast(Any, expr))
            ret = cls(cast(MT, val), "dimensionless")

            return cast("Quantity[DT, float]", ret)

        try:
            magnitude = float(cast(Any, args[0]))
        except TypeError as e:
            raise ValueError(f"Expression {expr} contains inconsistent units") from e

        dimensions = args[1:]

        unit = cls.get_unit("")

        for d in dimensions:
            unit_i = cls.get_unit("")

            _as_powers_dict = getattr(d, "as_powers_dict", None)
            if _as_powers_dict is None:
                raise TypeError(f"Invalid type: {d=}")

            powers_dict = cast(dict[sp.Basic, float], _as_powers_dict()).items()

            for symbol, power in powers_dict:
                try:
                    s = cls._dimension_symbol_map[symbol]
                except KeyError as e:
                    raise KeyError(f"Expression contains unknown unit symbol: {symbol}") from e
                unit_i = cast("Unit[Any]", unit_i * s**power)

            unit = cast("Unit[Any]", unit * unit_i)

        ret = cls(cast(MT, magnitude), cast(Unit, unit)).to_base_units()
        ret_ = ret._call_subclass(ret.m, ret.u)

        return cast("Quantity[DT, float]", ret_)

    @classmethod
    def __get_pydantic_core_schema__(
        cls: type[Quantity[DT, MT]],
        source_type: Any,  # noqa: ANN401
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        python_schema = core_schema.with_info_plain_validator_function(cls.validate)

        def _serialize(
            qty: Quantity[DT, MT],
            info: core_schema.SerializationInfo,  # noqa: ARG001
        ) -> dict[str, Any]:
            mag = qty.magnitude

            val: float | int | list[Any]

            if isinstance(mag, float | int):
                val = mag
                magnitude_type = "int" if isinstance(mag, int) else "float"
            elif isinstance(mag, np.ndarray):
                val = mag.tolist()
                mag_arr = cast(Any, mag)
                magnitude_type = f"np.ndarray:{mag_arr.dtype.str}:{mag_arr.shape}"
            elif isinstance(mag, pl.Series):
                val = mag.to_list()
                magnitude_type = f"pl.Series:{mag.dtype}"
            elif isinstance(mag, pl.Expr):
                raise ValueError(
                    "Cannot serialize Quantity with pl.Expr magnitude: a Polars expression holds no data; "
                    "materialize it first"
                )
            elif isinstance(mag, list):
                val = [float(x) for x in cast("list[Any]", mag)]  # pyrefly: ignore[redundant-cast]  # cast required by pyright
                magnitude_type = "list"
            else:
                raise ValueError(f"Unknown magnitude type {type(mag)}: {mag}")

            return {
                "unit": str(qty.u._units),
                "value": val,
                "magnitude_type": magnitude_type,
            }

        ser_schema = core_schema.plain_serializer_function_ser_schema(_serialize, info_arg=True, when_used="json")

        return core_schema.json_or_python_schema(
            json_schema=python_schema,
            python_schema=python_schema,
            serialization=ser_schema,
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return {
            "title": cls.__name__,
            "type": "object",
            "properties": {
                "unit": {"type": "string"},
                "value": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "array", "items": {"type": "number"}},
                    ]
                },
                "magnitude_type": {
                    "type": "string",
                    "description": (
                        "describes how to reconstruct the magnitude, "
                        "e.g. 'int', 'float', 'list', "
                        "'np.ndarray:<dtype>:<shape>', or 'pl.Series:<dtype>'"
                    ),
                },
            },
            "required": ["unit", "value", "magnitude_type"],
        }

    @classmethod
    def _pydantic_polars_dtype(cls, dtype_str: str) -> type[pl.DataType]:
        dtype = getattr(pl, dtype_str, None)

        if not isinstance(dtype, type) or not issubclass(dtype, pl.DataType):
            raise ValueError(f"Unknown Polars Series dtype: '{dtype_str}'")

        return dtype

    @classmethod
    def _pydantic_magnitude_from_payload(cls, val: Any, magnitude_type: str) -> Any:  # noqa: ANN401
        if magnitude_type.startswith("np.ndarray"):
            try:
                _, dtype_str, _ = magnitude_type.split(":", 2)
            except ValueError as e:
                raise ValueError(f"Invalid np.ndarray magnitude_type {magnitude_type!r}") from e

            return np.array(val, dtype=np.dtype(dtype_str))

        if magnitude_type.startswith("pl.Series"):
            try:
                _, dtype_str = magnitude_type.split(":", 1)
            except ValueError as e:
                raise ValueError(f"Invalid pl.Series magnitude_type {magnitude_type!r}") from e

            return pl.Series(val, dtype=cls._pydantic_polars_dtype(dtype_str))

        if magnitude_type == "list":
            return val

        if magnitude_type == "int":
            return int(val)

        if magnitude_type == "float":
            return float(val)

        raise ValueError(f"Unknown magnitude_type {magnitude_type!r}")

    @classmethod
    def _pydantic_build_quantity(cls, qty: Any) -> Quantity[Any, Any]:  # noqa: ANN401
        if isinstance(qty, dict) and "value" in qty and "magnitude_type" in qty:
            qty_dict = cast("dict[str, Any]", qty)
            val = qty_dict["value"]
            magnitude_type = cast(str, qty_dict["magnitude_type"])
            magnitude = cls._pydantic_magnitude_from_payload(val, magnitude_type)
            unit = cast(str | None, qty_dict.get("unit"))
            return cast("Quantity[Any, Any]", cls(cast(MT, magnitude), unit=unit))

        return cast("Quantity[Any, Any]", qty if isinstance(qty, Quantity) else cls(cast(Any, qty)))

    @classmethod
    def _pydantic_enforce_dimensionality(cls, qty: Quantity[Any, Any]) -> None:
        if issubclass(cls, cls.get_unknown_dimensionality_subclass()):
            return

        if cls._is_incomplete_dimensionality(cls._dimensionality_type):
            return

        expected = cls._dimensionality_type
        actual = qty._dimensionality_type

        dimensions_match = actual.dimensions == expected.dimensions
        type_matches = issubclass(actual, expected) or issubclass(expected, actual)

        if dimensions_match and type_matches:
            return

        raise PydanticCustomError(
            "quantity_dimensionality",
            "Quantity dimensionality {actual} does not match expected {expected}",
            {"actual": actual.__name__, "expected": expected.__name__},
        )

    @classmethod
    def _pydantic_enforce_magnitude_type(cls, qty: Quantity[Any, Any]) -> None:
        expected_mt = cast(object, getattr(cls, "_magnitude_type", None))

        if expected_mt is None or expected_mt is Any or isinstance(expected_mt, TypeVar):
            return

        expected = cls._get_magnitude_type_name(expected_mt)
        actual = cls._get_magnitude_type_name(cast(object, type(qty.m)))

        if actual == expected:
            return

        raise PydanticCustomError(
            "quantity_magnitude_type",
            "Quantity magnitude type {actual} does not match expected {expected}",
            {"actual": actual, "expected": expected},
        )

    @classmethod
    def validate(
        cls,
        qty: Any,  # noqa: ANN401
        info: Any,  # noqa: ANN401, ARG003
    ) -> Quantity[DT, MT]:
        """Pydantic validator: coerce ``qty`` to this subclass, checking dimensionality and magnitude type.

        Every failure is re-raised as a ``pydantic_core.PydanticCustomError`` with type
        ``quantity_dimensionality``, ``quantity_magnitude_type`` or ``quantity_validation``.
        """

        try:
            ret = cls._pydantic_build_quantity(qty)
            cls._pydantic_enforce_dimensionality(ret)
            cls._pydantic_enforce_magnitude_type(ret)
        except PydanticCustomError:
            raise
        except Exception as e:
            raise PydanticCustomError(
                "quantity_validation",
                "Invalid quantity: {error}",
                {"error": str(e)},
            ) from e

        return cast("Quantity[DT, MT]", ret)

    def check_compatibility(self, other: Quantity[Any, Any] | float) -> None:
        """Raise ``DimensionalityTypeError`` unless ``other`` can be combined with this quantity.

        Compatibility is semantic, not just dimensional: ``Temperature`` and
        ``TemperatureDifference`` share ``[temperature]`` but are not compatible. A plain
        number is compatible only with a dimensionless quantity.
        """

        if not isinstance(other, Quantity):
            if not self.dimensionless:
                raise DimensionalityTypeError(
                    f"Value {other} ({type(other)}) is not compatible with dimensional quantity {self} ({type(self)})"
                )

            return

        dim = self.dt
        other_dim = other._dimensionality_type

        # if the dimensionality of self is a subclass of the
        # dimensionality of other or vice versa
        if issubclass(dim, other_dim) or issubclass(other_dim, dim):
            # verify that the dimensions also match
            # this is also verified in the Dimensionality.__init_subclass__ method
            if dim.dimensions != other_dim.dimensions:
                raise DimensionalityTypeError(
                    f"Quantities with inherited dimensionalities do not match: "
                    f"{type(self)} and {type(other)} with dimensions "
                    f"{dim.dimensions} and {other_dim.dimensions}"
                )

            else:
                return

        # normal case, check that the types of Quantity is the same
        if type(self) is not type(other):
            if self.dt.dimensions == other._dimensionality_type.dimensions:
                raise DimensionalityTypeError(
                    f"Quantities with different dimensionalities are not compatible: "
                    f"{type(self)} and {type(other)}. The dimensions match, "
                    "but the dimensionalities have different types."
                )

            raise DimensionalityTypeError(
                f"Quantities with different dimensionalities are not compatible: {type(self)} and {type(other)}. "
            )

    def is_compatible_with(
        self,
        other: Quantity[Any, Any] | float,
        *contexts: Any,  # noqa: ANN401
        **ctx_kwargs: Any,  # noqa: ANN401
    ) -> bool:
        """Whether ``other`` is convertible to this quantity's unit *and* :meth:`check_compatibility` passes."""

        # add an additional check of the dimensionality types
        is_compatible = super().is_compatible_with(other, *contexts, **ctx_kwargs)

        if not is_compatible:
            return False
        try:
            self.check_compatibility(other)
            return True
        except DimensionalityTypeError:
            return False

    def _temperature_difference_add_sub(
        self,
        other: Quantity[TemperatureDifference, Any] | Quantity[Temperature, Any],
        operator: Literal["add", "sub"],
    ) -> Quantity[Temperature, MT]:
        if self.dt == Temperature:
            assert other._dimensionality_type == TemperatureDifference
            v1 = self.to("degC").m
            v2 = other.to("delta_degC").m

            val = v1 + v2 if operator == "add" else v1 - v2
            temperature_unit = self.u
        else:
            assert self.dt == TemperatureDifference
            assert other._dimensionality_type == Temperature

            v1 = self.to("delta_degC").m
            v2 = other.to("degC").m

            val = v1 + v2 if operator == "add" else v1 - v2
            temperature_unit = other.u

        # the arithmetic runs on the degC scale, but the result is an absolute
        # temperature: express it in the temperature operand's original unit
        # (K + ΔT stays in K) instead of normalizing everything to degC
        result = cast("Quantity[Temperature, MT]", Quantity(val, "degC"))
        return result.to(cast("Unit[Temperature]", temperature_unit))

    def __round__(self, ndigits: int | None = None) -> Quantity[DT, MT]:
        if ndigits is None:
            ndigits = 0

        if isinstance(self.m, float):
            return cast("Quantity[DT, MT]", super().__round__(ndigits))
        elif isinstance(self.m, np.ndarray):
            return cast("Quantity[DT, MT]", self.__class__(np.round(self.m, ndigits), self.u))
        else:
            raise TypeError(f"round() is not supported for magnitude type {type(self.m)}")

    @property
    def is_scalar(self) -> bool:
        """Whether the magnitude is a scalar (``float``) rather than a container."""

        return isinstance(self.m, float)

    @property
    def ndim(self) -> int:
        """Number of magnitude dimensions: 0 for a scalar, 1 for a vector magnitude."""

        if isinstance(self.m, (float, int)):
            return 0

        return getattr(self.m, "ndim", 0)

    def asdim(self, other: type[DT_] | Quantity[DT_, MT]) -> Quantity[DT_, MT]:
        """Return this quantity reinterpreted as another dimensionality class.

        ``other`` can be a dimensionality class or another quantity whose
        dimensionality should be used. This is not a unit conversion: it succeeds only
        when the source and target have the same physical dimensions. For
        ``TemperatureDifference``, offset temperature units such as ``degC`` are
        rewritten to their delta units (``delta_degC``) so the result cannot be
        confused with an absolute temperature.
        """
        if isinstance(other, Quantity):
            dim = other._dimensionality_type
            assert dim is not None
        else:
            dim = other

        if dim == self.dt:
            return cast("Quantity[DT_, MT]", self)

        if dim == UnknownDimensionality:
            return cast("Quantity[DT_, MT]", self)

        if dim == Dimensionality:
            raise TypeError(f"Cannot convert {self} to base dimensionality {dim}")

        if str(self.dt.dimensions) != str(dim.dimensions):
            raise ExpectedDimensionalityError(
                f"Cannot convert {self} to dimensionality {dim}, "
                f"the dimensions do not match: "
                f"{self.dt.dimensions} != "
                f"{dim.dimensions}"
            )

        unit: Unit[Any] = self.u
        if dim == TemperatureDifference:
            unit = self._as_temperature_difference_unit(unit)

        subcls = self._get_dimensional_subclass(dim, type(self.m))  # ty: ignore[invalid-argument-type]
        return cast("Quantity[DT_, MT]", subcls(self.m, unit))

    def unknown(self) -> Quantity[UnknownDimensionality, MT]:
        """Return this quantity with its dimensionality erased, i.e. ``asdim(UnknownDimensionality)``."""

        return self.asdim(UnknownDimensionality)

    @overload
    def astype(self, magnitude_type: Literal["float"]) -> Quantity[DT, float]: ...

    @overload
    def astype(self, magnitude_type: Literal["ndarray"]) -> Quantity[DT, Numpy1DArray]: ...

    @overload
    def astype(self, magnitude_type: Literal["pl.Expr"]) -> Quantity[DT, pl.Expr]: ...

    @overload
    def astype(self, magnitude_type: Literal["pl.Series"]) -> Quantity[DT, pl.Series]: ...

    @overload
    def astype(self, magnitude_type: type[MT_] | MagnitudeTypeName) -> Quantity[DT, MT_]: ...

    def astype(self, magnitude_type: type[Any] | MagnitudeTypeName) -> Quantity[Any, Any]:
        """Return this quantity with its magnitude converted to another container type.

        ``magnitude_type`` accepts ``float``, numpy array, Polars ``Series``,
        Polars ``Expr``, or the corresponding string names. Units and dimensionality
        are unchanged. Converting to ``pl.Expr`` is only defined for scalar quantities,
        which become a literal expression.
        """
        if isinstance(magnitude_type, str):
            magnitude_type = self._get_magnitude_type_from_name(magnitude_type)

        magnitude_type_origin = get_origin(magnitude_type)
        m, u = self.m, self.u

        dt = self.dt

        if type(m) is magnitude_type or type(m) is magnitude_type_origin:
            return cast("Quantity[DT, Any]", self)
        elif magnitude_type is pl.Expr:
            if isinstance(m, float):
                return cast("Quantity[DT, Any]", self.get_subclass(dt, pl.Expr)(pl.lit(m), u))

            raise TypeError(
                f"Cannot convert magnitude with type {type(m)} to Polars expression, "
                "only scalar (float) quantities can be converted to pl.Expr"
            )
        elif magnitude_type is float:
            if isinstance(m, float):
                return cast("Quantity[DT, Any]", self.get_subclass(dt, float)(float(cast(Any, m)), u))

            raise TypeError(
                f"Cannot convert magnitude with type {type(m)} to float; "
                "only scalar (float) quantities can be converted to float"
            )
        elif magnitude_type is np.ndarray or magnitude_type_origin is np.ndarray:
            _m = [m] if not isinstance(m, Iterable) else m
            vals = np.array(_m)
            return cast("Quantity[DT, Any]", self.get_subclass(dt, np.ndarray)(vals, u))
        elif magnitude_type is pl.Series:
            if isinstance(m, pl.Expr):
                raise TypeError(
                    "Cannot convert a pl.Expr magnitude to pl.Series; evaluate the expression in a Polars frame first"
                )
            _m = [m] if not isinstance(m, Iterable) else m
            vals = pl.Series(values=_m)
            return cast("Quantity[DT, Any]", self.get_subclass(dt, pl.Series)(vals, u))
        else:
            raise TypeError(f"Cannot convert magnitude from type {type(m)} to {magnitude_type}")

    @overload
    def __pow__(self: Quantity[Length, MT], other: Literal[2]) -> Quantity[Area, MT]: ...
    @overload
    def __pow__(self: Quantity[Length, MT], other: Literal[3]) -> Quantity[Volume, MT]: ...
    @overload
    def __pow__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]: ...
    @overload
    # raising to the literal power 1 preserves the dimensionality; this
    # intentionally overlaps the general float/int overload above
    def __pow__(self, other: Literal[1]) -> Quantity[DT, MT]: ...  # pyright: ignore[reportOverlappingOverload]
    @overload
    def __pow__(self, other: float) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __pow__(self, other: Quantity[Dimensionless, MT]) -> Quantity[UnknownDimensionality, MT]: ...
    def __pow__(self, other: Quantity[Dimensionless, Any] | float) -> Quantity[Any, Any]:
        if isinstance(other, Quantity):
            self._check_comparable_magnitudes(self.m, other.m, "combine")  # ty: ignore[invalid-argument-type]
        ret = cast("Quantity[DT, MT]", self._pint_super.__pow__(other))
        return self._call_subclass(ret.m, ret.u)

    def __ipow__(self, other: Any) -> Any:  # noqa: ANN401
        return cast("Quantity[Any, Any]", cast(Any, self).__pow__(other))

    @overload
    def __add__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __add__(
        self: Quantity[Temperature, float], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[Temperature, MT_]: ...
    @overload
    def __add__(
        self: Quantity[Temperature, MT], other: Quantity[TemperatureDifference, float]
    ) -> Quantity[Temperature, MT]: ...
    @overload
    def __add__(
        self: Quantity[Temperature, MT], other: Quantity[TemperatureDifference, MT]
    ) -> Quantity[Temperature, MT]: ...
    @overload
    def __add__(
        self: Quantity[TemperatureDifference, float], other: Quantity[Temperature, MT_]
    ) -> Quantity[Temperature, MT_]: ...
    @overload
    def __add__(
        self: Quantity[TemperatureDifference, MT], other: Quantity[Temperature, float]
    ) -> Quantity[Temperature, MT]: ...
    @overload
    def __add__(
        self: Quantity[TemperatureDifference, MT], other: Quantity[Temperature, MT]
    ) -> Quantity[Temperature, MT]: ...
    @overload
    def __add__(self, other: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __add__(self, other: Quantity[DT, float]) -> Quantity[DT, MT]: ...
    @overload
    def __add__(self: Quantity[DT, float], other: Quantity[DT, MT_]) -> Quantity[DT, MT_]: ...
    def __add__(self, other: Quantity[Any, Any] | float) -> Quantity[Any, Any]:
        if isinstance(other, Quantity):
            self._check_comparable_magnitudes(self.m, other.m, "combine")  # ty: ignore[invalid-argument-type]
        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            if not isinstance(other, Quantity):
                raise e

            self_is_temp_or_diff_temp = self.dt in (Temperature, TemperatureDifference)
            other_is_temp_or_diff_temp = other._dimensionality_type in (Temperature, TemperatureDifference)

            if self_is_temp_or_diff_temp and other_is_temp_or_diff_temp:
                return self._temperature_difference_add_sub(other, "add")  # ty: ignore[invalid-argument-type]

            raise e

        ret = cast("Quantity[DT, MT]", self._pint_super.__add__(other))

        return self._call_subclass(ret.m, ret.u)

    def __iadd__(self, other: Any) -> Any:  # noqa: ANN401
        return cast("Quantity[Any, Any]", cast(Any, self).__add__(other))

    @overload
    def __radd__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __radd__(self, other: float) -> Quantity[Any, Any]: ...
    def __radd__(self, other: float) -> Quantity[Any, Any]:
        ret = cast("Quantity[DT, MT]", self._pint_super.__radd__(other))

        return self._call_subclass(ret.m, ret.u)

    @overload
    def __sub__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __sub__(
        self: Quantity[Temperature, float], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[Temperature, MT_]: ...
    @overload
    def __sub__(
        self: Quantity[Temperature, MT], other: Quantity[TemperatureDifference, float]
    ) -> Quantity[Temperature, MT]: ...
    @overload
    def __sub__(
        self: Quantity[Temperature, MT], other: Quantity[TemperatureDifference, MT]
    ) -> Quantity[Temperature, MT]: ...
    # NOTE: TemperatureDifference - Temperature is intentionally NOT defined
    # (physically not a temperature; pint refuses it too) -- only T ± ΔT, ΔT + T
    # and T - T are meaningful
    @overload
    def __sub__(
        self: Quantity[Temperature, MT], other: Quantity[Temperature, MT]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __sub__(self, other: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __sub__(self, other: Quantity[DT, float]) -> Quantity[DT, MT]: ...
    @overload
    def __sub__(self: Quantity[DT, float], other: Quantity[DT, MT_]) -> Quantity[DT, MT_]: ...
    def __sub__(self, other: Quantity[Any, Any] | float) -> Quantity[Any, Any]:
        if isinstance(other, Quantity):
            self._check_comparable_magnitudes(self.m, other.m, "combine")  # ty: ignore[invalid-argument-type]
        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            if not isinstance(other, Quantity):
                raise e

            # only Temperature - TemperatureDifference is meaningful here; the
            # reverse (ΔT - T) is not a temperature and stays an error
            if self.dt == Temperature and other._dimensionality_type == TemperatureDifference:
                return self._temperature_difference_add_sub(other, "sub")  # ty: ignore[invalid-argument-type]

            raise e

        ret = cast("Quantity[DT, MT]", self._pint_super.__sub__(other))

        if isinstance(other, Quantity) and self.dt == Temperature and other._dimensionality_type == Temperature:
            _mt = type(ret.m)
            subcls = self._get_dimensional_subclass(TemperatureDifference, _mt)  # ty: ignore[invalid-argument-type]
            return subcls(ret.m, ret.u)

        return self._call_subclass(ret.m, ret.u)

    def __isub__(self, other: Any) -> Any:  # noqa: ANN401
        return cast("Quantity[Any, Any]", cast(Any, self).__sub__(other))

    @overload
    def __rsub__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __rsub__(self, other: float) -> Quantity[Any, Any]: ...
    def __rsub__(self, other: float) -> Quantity[Any, Any]:
        ret = cast("Quantity[DT, MT]", self._pint_super.__rsub__(other))

        return self._call_subclass(ret.m, ret.u)

    @overload
    def __eq__(self: Quantity[Dimensionless, float], other: float) -> bool: ...
    @overload
    def __eq__(self: Quantity[Dimensionless, Numpy1DArray], other: float) -> Numpy1DBoolArray: ...
    @overload
    def __eq__(self: Quantity[Dimensionless, pl.Series], other: float) -> pl.Series: ...
    @overload
    def __eq__(self: Quantity[Dimensionless, pl.Expr], other: float) -> pl.Expr: ...
    @overload
    def __eq__(self: Quantity[DT, float], other: Quantity[UnknownDimensionality, float]) -> bool: ...
    @overload
    def __eq__(
        self: Quantity[DT, Numpy1DArray], other: Quantity[UnknownDimensionality, Numpy1DArray]
    ) -> Numpy1DBoolArray: ...
    @overload
    def __eq__(self: Quantity[DT, Numpy1DArray], other: Quantity[UnknownDimensionality, float]) -> Numpy1DBoolArray: ...
    @overload
    def __eq__(self: Quantity[DT, float], other: Quantity[UnknownDimensionality, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __eq__(self: Quantity[DT, pl.Series], other: Quantity[UnknownDimensionality, pl.Series]) -> pl.Series: ...
    @overload
    def __eq__(self: Quantity[DT, pl.Series], other: Quantity[UnknownDimensionality, float]) -> pl.Series: ...
    @overload
    def __eq__(self: Quantity[DT, pl.Expr], other: Quantity[UnknownDimensionality, pl.Expr]) -> pl.Expr: ...
    @overload
    def __eq__(self: Quantity[DT, pl.Expr], other: Quantity[UnknownDimensionality, float]) -> pl.Expr: ...
    @overload
    def __eq__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    @overload
    def __eq__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __eq__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, float]) -> Numpy1DBoolArray: ...
    @overload
    def __eq__(self: Quantity[DT, float], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __eq__(self: Quantity[DT, pl.Series], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __eq__(self: Quantity[DT, pl.Series], other: Quantity[DT, float]) -> pl.Series: ...
    @overload
    def __eq__(self: Quantity[DT, pl.Expr], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    @overload
    def __eq__(self: Quantity[DT, pl.Expr], other: Quantity[DT, float]) -> pl.Expr: ...
    # for vector magnitudes __eq__ returns an array/Series/Expr of element-wise
    # results, intentionally widening object.__eq__'s bool return (as numpy does)
    def __eq__(self, other: object) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]
        if not isinstance(other, (Quantity, float, int)):
            return bool(self._pint_super.__eq__(other))

        try:
            self.check_compatibility(cast(Any, other))
        except DimensionalityTypeError:
            # Python convention: == across incomparable operands answers False rather
            # than raising (the ordering comparisons DO raise -- no answer is correct
            # there). hash() is keyed on root units, so the eq/hash contract holds.
            return False

        if isinstance(other, (float, int)):
            # float() so a bool operand still compares (Quantity rejects bool magnitudes)
            other = Quantity(float(other), "dimensionless")

        m = self.m
        other_m = cast(float | Numpy1DArray | pl.Series | pl.Expr, cast(Any, other).to(self.u).m)
        self._check_comparable_magnitudes(m, other_m)  # ty: ignore[invalid-argument-type]

        if isinstance(m, (float, int, np.ndarray)) and isinstance(other_m, (float, int, np.ndarray)):
            ret = _is_close(cast(Any, m), cast(Any, other_m), self.rtol, self.atol)

            if isinstance(ret, np.bool):
                return bool(cast(Any, ret))
            else:
                return ret

        if isinstance(m, pl.Series) and isinstance(other_m, (float, int, pl.Series)):
            return m.is_close(other_m, rel_tol=self.rtol, abs_tol=self.atol)

        if isinstance(other_m, pl.Series) and isinstance(m, (float, int)):
            return other_m.is_close(m, rel_tol=self.rtol, abs_tol=self.atol)

        if isinstance(m, pl.Expr) and isinstance(other_m, (float, int, pl.Expr)):
            return m.is_close(other_m, rel_tol=self.rtol, abs_tol=self.atol)

        if isinstance(other_m, pl.Expr) and isinstance(m, (float, int)):
            return other_m.is_close(m, rel_tol=self.rtol, abs_tol=self.atol)

        ret = m == other_m

        return ret

    @overload
    def __ne__(self: Quantity[Dimensionless, float], other: float) -> bool: ...
    @overload
    def __ne__(self: Quantity[Dimensionless, Numpy1DArray], other: float) -> Numpy1DBoolArray: ...
    @overload
    def __ne__(self: Quantity[Dimensionless, pl.Series], other: float) -> pl.Series: ...
    @overload
    def __ne__(self: Quantity[Dimensionless, pl.Expr], other: float) -> pl.Expr: ...
    @overload
    def __ne__(self: Quantity[DT, float], other: Quantity[UnknownDimensionality, float]) -> bool: ...
    @overload
    def __ne__(
        self: Quantity[DT, Numpy1DArray], other: Quantity[UnknownDimensionality, Numpy1DArray]
    ) -> Numpy1DBoolArray: ...
    @overload
    def __ne__(self: Quantity[DT, Numpy1DArray], other: Quantity[UnknownDimensionality, float]) -> Numpy1DBoolArray: ...
    @overload
    def __ne__(self: Quantity[DT, float], other: Quantity[UnknownDimensionality, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __ne__(self: Quantity[DT, pl.Series], other: Quantity[UnknownDimensionality, pl.Series]) -> pl.Series: ...
    @overload
    def __ne__(self: Quantity[DT, pl.Series], other: Quantity[UnknownDimensionality, float]) -> pl.Series: ...
    @overload
    def __ne__(self: Quantity[DT, pl.Expr], other: Quantity[UnknownDimensionality, pl.Expr]) -> pl.Expr: ...
    @overload
    def __ne__(self: Quantity[DT, pl.Expr], other: Quantity[UnknownDimensionality, float]) -> pl.Expr: ...
    @overload
    def __ne__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    @overload
    def __ne__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __ne__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, float]) -> Numpy1DBoolArray: ...
    @overload
    def __ne__(self: Quantity[DT, float], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __ne__(self: Quantity[DT, pl.Series], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __ne__(self: Quantity[DT, pl.Series], other: Quantity[DT, float]) -> pl.Series: ...
    @overload
    def __ne__(self: Quantity[DT, pl.Expr], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    @overload
    def __ne__(self: Quantity[DT, pl.Expr], other: Quantity[DT, float]) -> pl.Expr: ...
    # for vector magnitudes __ne__ returns an array/Series/Expr of element-wise
    # results, intentionally widening object.__ne__'s bool return (as numpy does)
    def __ne__(self, other: object) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:  # pyright: ignore[reportIncompatibleMethodOverride]  # ty: ignore[invalid-method-override]
        if not isinstance(other, (Quantity, float, int)):
            return bool(self._pint_super.__ne__(other))

        try:
            self.check_compatibility(cast(Any, other))
        except DimensionalityTypeError:
            # mirror __eq__: incomparable operands are simply not equal
            return True

        if isinstance(other, (float, int)):
            # float() so a bool operand still compares (Quantity rejects bool magnitudes)
            other = Quantity(float(other), "dimensionless")

        m = self.m
        other_m = cast(float | Numpy1DArray | pl.Series | pl.Expr, cast(Any, other).to(self.u).m)
        self._check_comparable_magnitudes(m, other_m)  # ty: ignore[invalid-argument-type]

        if isinstance(m, (float, int, np.ndarray)) and isinstance(other_m, (float, int, np.ndarray)):
            ret = ~_is_close(cast(Any, m), cast(Any, other_m), self.rtol, self.atol)

            if isinstance(ret, np.bool):
                return bool(cast(Any, ret))
            else:
                return ret

        if isinstance(m, pl.Series) and isinstance(other_m, (float, int, pl.Series)):
            return m.is_close(other_m, rel_tol=self.rtol, abs_tol=self.atol).not_()

        if isinstance(other_m, pl.Series) and isinstance(m, (float, int)):
            return other_m.is_close(m, rel_tol=self.rtol, abs_tol=self.atol).not_()

        if isinstance(m, pl.Expr) and isinstance(other_m, (float, int, pl.Expr)):
            return m.is_close(other_m, rel_tol=self.rtol, abs_tol=self.atol).not_()

        if isinstance(other_m, pl.Expr) and isinstance(m, (float, int)):
            return other_m.is_close(m, rel_tol=self.rtol, abs_tol=self.atol).not_()

        ret = m != other_m

        return ret

    def _ordering_comparison(
        self,
        other: Quantity[Any, Any] | float,
        op: Literal["__gt__", "__ge__", "__lt__", "__le__"],
    ) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:
        # Tolerant __eq__ is not optional: unit conversion is lossy, so Q(1, "L") and
        # Q(1000, "cm³") differ in the last bit and an exact __eq__ would call them unequal.
        # Given that, closeness is not transitive -- a == b and b == c does not imply a == c --
        # and only two of these three can hold at once:
        #
        #   (1) __eq__ is tolerant
        #   (2) the operators agree pairwise: a == b implies not (a > b), and a > b implies
        #       not (a <= b)
        #   (3) __lt__ is a strict WEAK order, which sorted()/min()/max() need to be exact
        #       on every input
        #
        # encomp keeps (1) and (2). Quantities within (rtol, atol) are ties, so __lt__ is a
        # strict PARTIAL order: irreflexive, asymmetric and transitive, but ties do not chain.
        # sorted()/min() are therefore exact unless the data contains a tolerance *chain* --
        # each adjacent pair within tolerance, the endpoints not -- which spans less than the
        # width at which the library already declares the values equal. Sort on the raw
        # magnitudes (`key=lambda q: q.to(unit).m`) when a strict total order is required.
        #
        # ordering across dimensionality types has no correct answer (e.g. Temperature
        # vs TemperatureDifference share [temperature] but must not order), so it
        # raises -- unlike __eq__, which can answer False for incompatible operands
        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            raise DimensionalityComparisonError(f"Cannot compare {self} with {other}") from e

        if isinstance(other, Quantity):
            self._check_comparable_magnitudes(self.m, other.m)  # ty: ignore[invalid-argument-type]

        # only the STRICT pint operator is ever evaluated; a non-strict result is derived
        # from it below. Using pint's raw >= / <= would poison the derivation for NaN:
        # polars' total order calls NaN equal to NaN, so its raw `nan >= nan` is True while
        # encomp's tolerant __eq__ says False -- and `ge == (gt or eq)` would not hold.
        strict_op: str = {"__ge__": "__gt__", "__le__": "__lt__"}.get(op, op)

        try:
            ret = getattr(self._pint_super, strict_op)(other)
        except (ValueError, DimensionalityError) as e:
            raise DimensionalityComparisonError(str(e)) from e

        # __eq__ is tolerant (rtol, atol), so every ordering operator must agree with it or the
        # five relations do not form an ordering. pint compares exactly, so fold the tolerance
        # in here: the strict operators exclude equality, and the non-strict ones are exactly
        # `strict or equal`. This keeps `a > b` implies `not (a <= b)` (which sorted()/bisect
        # assume), and makes `ge == (gt or eq)` / `le == (lt or eq)` hold for EVERY input --
        # including NaN operands on polars magnitudes -- in all four magnitude containers.
        equal = cast(Any, self).__eq__(other)
        not_equal = (not equal) if isinstance(equal, bool) else ~equal

        ret = ret & not_equal

        if op in ("__ge__", "__le__"):
            ret = ret | equal

        if isinstance(ret, np.bool):
            return bool(cast(Any, ret))

        return cast("bool | Numpy1DBoolArray | pl.Series | pl.Expr", ret)

    @overload
    def __gt__(self: Quantity[Dimensionless, float], other: float) -> bool: ...
    @overload
    def __gt__(self: Quantity[Dimensionless, Numpy1DArray], other: float) -> Numpy1DBoolArray: ...
    @overload
    def __gt__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    @overload
    def __gt__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __gt__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, float]) -> Numpy1DBoolArray: ...
    @overload
    def __gt__(self: Quantity[DT, float], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __gt__(self: Quantity[DT, pl.Series], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __gt__(self: Quantity[DT, pl.Series], other: Quantity[DT, float]) -> pl.Series: ...
    @overload
    def __gt__(self: Quantity[DT, float], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __gt__(self: Quantity[DT, pl.Expr], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    @overload
    def __gt__(self: Quantity[DT, pl.Expr], other: Quantity[DT, float]) -> pl.Expr: ...
    @overload
    def __gt__(self: Quantity[DT, float], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    def __gt__(self, other: Quantity[DT, Any] | float) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:  # ty: ignore[invalid-method-override]
        return self._ordering_comparison(other, "__gt__")

    @overload
    def __ge__(self: Quantity[Dimensionless, float], other: float) -> bool: ...
    @overload
    def __ge__(self: Quantity[Dimensionless, Numpy1DArray], other: float) -> Numpy1DBoolArray: ...
    @overload
    def __ge__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    @overload
    def __ge__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __ge__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, float]) -> Numpy1DBoolArray: ...
    @overload
    def __ge__(self: Quantity[DT, float], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __ge__(self: Quantity[DT, pl.Series], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __ge__(self: Quantity[DT, pl.Series], other: Quantity[DT, float]) -> pl.Series: ...
    @overload
    def __ge__(self: Quantity[DT, float], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __ge__(self: Quantity[DT, pl.Expr], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    @overload
    def __ge__(self: Quantity[DT, pl.Expr], other: Quantity[DT, float]) -> pl.Expr: ...
    @overload
    def __ge__(self: Quantity[DT, float], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    def __ge__(self, other: Quantity[DT, Any] | float) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:  # ty: ignore[invalid-method-override]
        return self._ordering_comparison(other, "__ge__")

    @overload
    def __lt__(self: Quantity[Dimensionless, float], other: float) -> bool: ...
    @overload
    def __lt__(self: Quantity[Dimensionless, Numpy1DArray], other: float) -> Numpy1DBoolArray: ...
    @overload
    def __lt__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    @overload
    def __lt__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __lt__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, float]) -> Numpy1DBoolArray: ...
    @overload
    def __lt__(self: Quantity[DT, float], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __lt__(self: Quantity[DT, pl.Series], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __lt__(self: Quantity[DT, pl.Series], other: Quantity[DT, float]) -> pl.Series: ...
    @overload
    def __lt__(self: Quantity[DT, float], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __lt__(self: Quantity[DT, pl.Expr], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    @overload
    def __lt__(self: Quantity[DT, pl.Expr], other: Quantity[DT, float]) -> pl.Expr: ...
    @overload
    def __lt__(self: Quantity[DT, float], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    def __lt__(self, other: Quantity[DT, Any] | float) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:  # ty: ignore[invalid-method-override]
        return self._ordering_comparison(other, "__lt__")

    @overload
    def __le__(self: Quantity[Dimensionless, float], other: float) -> bool: ...
    @overload
    def __le__(self: Quantity[Dimensionless, Numpy1DArray], other: float) -> Numpy1DBoolArray: ...
    @overload
    def __le__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    @overload
    def __le__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __le__(self: Quantity[DT, Numpy1DArray], other: Quantity[DT, float]) -> Numpy1DBoolArray: ...
    @overload
    def __le__(self: Quantity[DT, float], other: Quantity[DT, Numpy1DArray]) -> Numpy1DBoolArray: ...
    @overload
    def __le__(self: Quantity[DT, pl.Series], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __le__(self: Quantity[DT, pl.Series], other: Quantity[DT, float]) -> pl.Series: ...
    @overload
    def __le__(self: Quantity[DT, float], other: Quantity[DT, pl.Series]) -> pl.Series: ...
    @overload
    def __le__(self: Quantity[DT, pl.Expr], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    @overload
    def __le__(self: Quantity[DT, pl.Expr], other: Quantity[DT, float]) -> pl.Expr: ...
    @overload
    def __le__(self: Quantity[DT, float], other: Quantity[DT, pl.Expr]) -> pl.Expr: ...
    def __le__(self, other: Quantity[DT, Any] | float) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:  # ty: ignore[invalid-method-override]
        return self._ordering_comparison(other, "__le__")

    @overload
    def __mul__(self: Quantity[Dimensionless, float], other: Quantity[DT_, MT_]) -> Quantity[DT_, MT_]: ...
    @overload
    def __mul__(
        self: Quantity[Dimensionless, MT], other: Quantity[Dimensionless, MT]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(self: Quantity[Dimensionless, MT], other: Quantity[DT_, float]) -> Quantity[DT_, MT]: ...
    @overload
    def __mul__(self: Quantity[Dimensionless, MT], other: Quantity[DT_, MT]) -> Quantity[DT_, MT]: ...
    @overload
    def __mul__(self: Quantity[DT, MT], other: Quantity[Dimensionless, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self: Quantity[DT, MT], other: Quantity[Dimensionless, float]) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self: Quantity[DT, float], other: Quantity[Dimensionless, MT_]) -> Quantity[DT, MT_]: ...

    # MassFlow * Time = Mass
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[Time, MT]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[Time, float]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, float], other: Quantity[Time, MT_]) -> Quantity[Mass, MT_]: ...

    # Time * MassFlow = Mass
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[MassFlow, MT]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[MassFlow, float]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, float], other: Quantity[MassFlow, MT_]) -> Quantity[Mass, MT_]: ...

    # VolumeFlow * Time = Volume
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Time, MT]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Time, float]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, float], other: Quantity[Time, MT_]) -> Quantity[Volume, MT_]: ...

    # Time * VolumeFlow = Volume
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[VolumeFlow, MT]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[VolumeFlow, float]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, float], other: Quantity[VolumeFlow, MT_]) -> Quantity[Volume, MT_]: ...

    # Power * Time = Energy
    @overload
    def __mul__(self: Quantity[Power, MT], other: Quantity[Time, MT]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Power, MT], other: Quantity[Time, float]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Power, float], other: Quantity[Time, MT_]) -> Quantity[Energy, MT_]: ...

    # Time * Power = Energy
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[Power, MT]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[Power, float]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, float], other: Quantity[Power, MT_]) -> Quantity[Energy, MT_]: ...

    # Velocity * Time = Length
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Time, MT]) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Time, float]) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, float], other: Quantity[Time, MT_]) -> Quantity[Length, MT_]: ...

    # Time * Velocity = Length
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[Velocity, MT]) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[Velocity, float]) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, float], other: Quantity[Velocity, MT_]) -> Quantity[Length, MT_]: ...

    # Density * Volume = Mass
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[Volume, MT]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[Volume, float]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, float], other: Quantity[Volume, MT_]) -> Quantity[Mass, MT_]: ...

    # Volume * Density = Mass
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[Density, MT]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[Density, float]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, float], other: Quantity[Density, MT_]) -> Quantity[Mass, MT_]: ...

    # Length * Length = Area
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Length, MT]) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Length, float]) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, float], other: Quantity[Length, MT_]) -> Quantity[Area, MT_]: ...

    # Length * Area = Volume
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Area, MT]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Area, float]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, float], other: Quantity[Area, MT_]) -> Quantity[Volume, MT_]: ...

    # Area * Length = Volume
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[Length, MT]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[Length, float]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, float], other: Quantity[Length, MT_]) -> Quantity[Volume, MT_]: ...

    # MassFlow * EnergyPerMass = Power
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[EnergyPerMass, MT]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[EnergyPerMass, float]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, float], other: Quantity[EnergyPerMass, MT_]) -> Quantity[Power, MT_]: ...

    # EnergyPerMass * MassFlow = Power
    @overload
    def __mul__(self: Quantity[EnergyPerMass, MT], other: Quantity[MassFlow, MT]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[EnergyPerMass, MT], other: Quantity[MassFlow, float]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[EnergyPerMass, float], other: Quantity[MassFlow, MT_]) -> Quantity[Power, MT_]: ...

    # Mass * EnergyPerMass = Energy
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[EnergyPerMass, MT]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[EnergyPerMass, float]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, float], other: Quantity[EnergyPerMass, MT_]) -> Quantity[Energy, MT_]: ...

    # EnergyPerMass * Mass = Energy
    @overload
    def __mul__(self: Quantity[EnergyPerMass, MT], other: Quantity[Mass, MT]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[EnergyPerMass, MT], other: Quantity[Mass, float]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[EnergyPerMass, float], other: Quantity[Mass, MT_]) -> Quantity[Energy, MT_]: ...

    # Density * VolumeFlow = MassFlow
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[VolumeFlow, MT]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[VolumeFlow, float]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, float], other: Quantity[VolumeFlow, MT_]) -> Quantity[MassFlow, MT_]: ...

    # VolumeFlow * Density = MassFlow
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Density, MT]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Density, float]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, float], other: Quantity[Density, MT_]) -> Quantity[MassFlow, MT_]: ...

    # Area * Velocity = VolumeFlow
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[Velocity, MT]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[Velocity, float]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, float], other: Quantity[Velocity, MT_]) -> Quantity[VolumeFlow, MT_]: ...

    # Velocity * Area = VolumeFlow
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Area, MT]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Area, float]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, float], other: Quantity[Area, MT_]) -> Quantity[VolumeFlow, MT_]: ...

    # MassFlow * SpecificVolume = VolumeFlow
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[SpecificVolume, MT]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[SpecificVolume, float]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, float], other: Quantity[SpecificVolume, MT_]) -> Quantity[VolumeFlow, MT_]: ...

    # SpecificVolume * MassFlow = VolumeFlow
    @overload
    def __mul__(self: Quantity[SpecificVolume, MT], other: Quantity[MassFlow, MT]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[SpecificVolume, MT], other: Quantity[MassFlow, float]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[SpecificVolume, float], other: Quantity[MassFlow, MT_]) -> Quantity[VolumeFlow, MT_]: ...

    # Pressure * Volume = Energy
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[Volume, MT]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[Volume, float]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, float], other: Quantity[Volume, MT_]) -> Quantity[Energy, MT_]: ...

    # Volume * Pressure = Energy
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[Pressure, MT]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[Pressure, float]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, float], other: Quantity[Pressure, MT_]) -> Quantity[Energy, MT_]: ...

    # Pressure * VolumeFlow = Power
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[VolumeFlow, MT]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[VolumeFlow, float]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, float], other: Quantity[VolumeFlow, MT_]) -> Quantity[Power, MT_]: ...

    # VolumeFlow * Pressure = Power
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Pressure, MT]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Pressure, float]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, float], other: Quantity[Pressure, MT_]) -> Quantity[Power, MT_]: ...

    # Mass * SpecificVolume = Volume
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[SpecificVolume, MT]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[SpecificVolume, float]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, float], other: Quantity[SpecificVolume, MT_]) -> Quantity[Volume, MT_]: ...

    # SpecificVolume * Mass = Volume
    @overload
    def __mul__(self: Quantity[SpecificVolume, MT], other: Quantity[Mass, MT]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[SpecificVolume, MT], other: Quantity[Mass, float]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[SpecificVolume, float], other: Quantity[Mass, MT_]) -> Quantity[Volume, MT_]: ...

    # SpecificHeatCapacity * TemperatureDifference = EnergyPerMass
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[TemperatureDifference, MT]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[TemperatureDifference, float]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, float], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[EnergyPerMass, MT_]: ...

    # TemperatureDifference * SpecificHeatCapacity = EnergyPerMass
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT], other: Quantity[SpecificHeatCapacity, MT]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT], other: Quantity[SpecificHeatCapacity, float]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, float], other: Quantity[SpecificHeatCapacity, MT_]
    ) -> Quantity[EnergyPerMass, MT_]: ...

    # Pressure * Area = Force
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[Area, MT]) -> Quantity[Force, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[Area, float]) -> Quantity[Force, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, float], other: Quantity[Area, MT_]) -> Quantity[Force, MT_]: ...

    # Area * Pressure = Force
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[Pressure, MT]) -> Quantity[Force, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[Pressure, float]) -> Quantity[Force, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, float], other: Quantity[Pressure, MT_]) -> Quantity[Force, MT_]: ...

    # Force * Length = Energy
    @overload
    def __mul__(self: Quantity[Force, MT], other: Quantity[Length, MT]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Force, MT], other: Quantity[Length, float]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Force, float], other: Quantity[Length, MT_]) -> Quantity[Energy, MT_]: ...

    # Length * Force = Energy
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Force, MT]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Force, float]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, float], other: Quantity[Force, MT_]) -> Quantity[Energy, MT_]: ...

    # Force * Velocity = Power
    @overload
    def __mul__(self: Quantity[Force, MT], other: Quantity[Velocity, MT]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Force, MT], other: Quantity[Velocity, float]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Force, float], other: Quantity[Velocity, MT_]) -> Quantity[Power, MT_]: ...

    # Velocity * Force = Power
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Force, MT]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Force, float]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, float], other: Quantity[Force, MT_]) -> Quantity[Power, MT_]: ...

    # Substance * MolarMass = Mass
    @overload
    def __mul__(self: Quantity[Substance, MT], other: Quantity[MolarMass, MT]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Substance, MT], other: Quantity[MolarMass, float]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Substance, float], other: Quantity[MolarMass, MT_]) -> Quantity[Mass, MT_]: ...

    # MolarMass * Substance = Mass
    @overload
    def __mul__(self: Quantity[MolarMass, MT], other: Quantity[Substance, MT]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[MolarMass, MT], other: Quantity[Substance, float]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[MolarMass, float], other: Quantity[Substance, MT_]) -> Quantity[Mass, MT_]: ...

    # MolarDensity * Volume = Substance
    @overload
    def __mul__(self: Quantity[MolarDensity, MT], other: Quantity[Volume, MT]) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(self: Quantity[MolarDensity, MT], other: Quantity[Volume, float]) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(self: Quantity[MolarDensity, float], other: Quantity[Volume, MT_]) -> Quantity[Substance, MT_]: ...

    # Volume * MolarDensity = Substance
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[MolarDensity, MT]) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[MolarDensity, float]) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, float], other: Quantity[MolarDensity, MT_]) -> Quantity[Substance, MT_]: ...

    # HeatTransferCoefficient * TemperatureDifference = PowerPerArea
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[TemperatureDifference, MT]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[TemperatureDifference, float]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, float], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[PowerPerArea, MT_]: ...

    # TemperatureDifference * HeatTransferCoefficient = PowerPerArea
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT], other: Quantity[HeatTransferCoefficient, MT]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT], other: Quantity[HeatTransferCoefficient, float]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, float], other: Quantity[HeatTransferCoefficient, MT_]
    ) -> Quantity[PowerPerArea, MT_]: ...

    # HeatTransferCoefficient * Length = ThermalConductivity
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[Length, MT]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[Length, float]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, float], other: Quantity[Length, MT_]
    ) -> Quantity[ThermalConductivity, MT_]: ...

    # Length * HeatTransferCoefficient = ThermalConductivity
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[HeatTransferCoefficient, MT]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[HeatTransferCoefficient, float]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, float], other: Quantity[HeatTransferCoefficient, MT_]
    ) -> Quantity[ThermalConductivity, MT_]: ...

    # Unknown * Unknown = Unknown
    @overload
    def __mul__(
        self: Quantity[UnknownDimensionality, MT], other: Quantity[UnknownDimensionality, MT]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __mul__(
        self: Quantity[UnknownDimensionality, MT], other: Quantity[UnknownDimensionality, float]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __mul__(
        self: Quantity[UnknownDimensionality, float], other: Quantity[UnknownDimensionality, MT_]
    ) -> Quantity[UnknownDimensionality, MT_]: ...

    @overload
    def __mul__(
        self: Quantity[DT, float], other: Quantity[Any, Numpy1DArray]
    ) -> Quantity[UnknownDimensionality, Numpy1DArray]: ...
    @overload
    def __mul__(
        self: Quantity[DT, float], other: Quantity[Any, pl.Series]
    ) -> Quantity[UnknownDimensionality, pl.Series]: ...
    @overload
    def __mul__(
        self: Quantity[DT, float], other: Quantity[Any, pl.Expr]
    ) -> Quantity[UnknownDimensionality, pl.Expr]: ...
    @overload
    def __mul__(self: Quantity[DT, float], other: Quantity[Any, float]) -> Quantity[UnknownDimensionality, float]: ...
    @overload
    def __mul__(self, other: float) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self, other: Quantity[DT_, float]) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __mul__(self, other: Quantity[DT_, MT]) -> Quantity[UnknownDimensionality, MT]: ...
    def __mul__(self, other: Quantity[Any, Any] | float) -> Quantity[Any, Any]:
        if isinstance(other, Quantity):
            self._check_comparable_magnitudes(self.m, other.m, "combine")  # ty: ignore[invalid-argument-type]
        ret = cast("Quantity[DT, MT]", self._pint_super.__mul__(other))

        # preserve the dimensionality for other
        # it might be a distinct subclass with identical units as another dimensionality
        if self.dimensionless and isinstance(other, Quantity):
            subcls = self.get_subclass(other._dimensionality_type, type(ret.m))
            return subcls(ret)

        return self._call_subclass(ret.m, ret.u)

    def __imul__(self, other: Any) -> Any:  # noqa: ANN401
        return cast("Quantity[Any, Any]", cast(Any, self).__mul__(other))

    def __rmul__(self, other: float) -> Quantity[DT, MT]:
        ret = cast("Quantity[DT, MT]", self._pint_super.__rmul__(other))
        return self._call_subclass(ret.m, ret.u)

    @overload
    def __truediv__(
        self: Quantity[Dimensionless, float], other: Quantity[Dimensionless, Numpy1DArray]
    ) -> Quantity[Dimensionless, Numpy1DArray]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, float], other: Quantity[Dimensionless, pl.Series]
    ) -> Quantity[Dimensionless, pl.Series]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, float], other: Quantity[Dimensionless, pl.Expr]
    ) -> Quantity[Dimensionless, pl.Expr]: ...

    # Unknown / Unknown = Unknown
    @overload
    def __truediv__(
        self: Quantity[UnknownDimensionality, MT], other: Quantity[UnknownDimensionality, MT]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[UnknownDimensionality, MT], other: Quantity[UnknownDimensionality, float]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[UnknownDimensionality, float], other: Quantity[UnknownDimensionality, MT_]
    ) -> Quantity[UnknownDimensionality, MT_]: ...

    @overload
    def __truediv__(self: Quantity[DT, MT], other: Quantity[Dimensionless, float]) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self: Quantity[DT, MT], other: Quantity[Dimensionless, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self: Quantity[DT, float], other: Quantity[Dimensionless, MT_]) -> Quantity[DT, MT_]: ...
    @overload
    def __truediv__(self: Quantity[DT, MT], other: Quantity[DT, MT]) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __truediv__(self: Quantity[DT, MT], other: Quantity[DT, float]) -> Quantity[Dimensionless, MT]: ...

    # Mass / Time = MassFlow
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Time, MT]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Time, float]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, float], other: Quantity[Time, MT_]) -> Quantity[MassFlow, MT_]: ...

    # Volume / Time = VolumeFlow
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Time, MT]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Time, float]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, float], other: Quantity[Time, MT_]) -> Quantity[VolumeFlow, MT_]: ...

    # Energy / Time = Power
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Time, MT]) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Time, float]) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, float], other: Quantity[Time, MT_]) -> Quantity[Power, MT_]: ...

    # Length / Time = Velocity
    @overload
    def __truediv__(self: Quantity[Length, MT], other: Quantity[Time, MT]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Length, MT], other: Quantity[Time, float]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Length, float], other: Quantity[Time, MT_]) -> Quantity[Velocity, MT_]: ...

    # Energy / Mass = EnergyPerMass
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Mass, MT]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Mass, float]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, float], other: Quantity[Mass, MT_]) -> Quantity[EnergyPerMass, MT_]: ...

    # Mass / Volume = Density
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Volume, MT]) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Volume, float]) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, float], other: Quantity[Volume, MT_]) -> Quantity[Density, MT_]: ...

    # Power / MassFlow = EnergyPerMass
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[MassFlow, MT]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[MassFlow, float]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, float], other: Quantity[MassFlow, MT_]) -> Quantity[EnergyPerMass, MT_]: ...

    # Power / EnergyPerMass = MassFlow
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[EnergyPerMass, MT]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[EnergyPerMass, float]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, float], other: Quantity[EnergyPerMass, MT_]) -> Quantity[MassFlow, MT_]: ...

    # Mass / MassFlow = Time
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[MassFlow, MT]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[MassFlow, float]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, float], other: Quantity[MassFlow, MT_]) -> Quantity[Time, MT_]: ...

    # Volume / VolumeFlow = Time
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[VolumeFlow, MT]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[VolumeFlow, float]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, float], other: Quantity[VolumeFlow, MT_]) -> Quantity[Time, MT_]: ...

    # Energy / Power = Time
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Power, MT]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Power, float]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, float], other: Quantity[Power, MT_]) -> Quantity[Time, MT_]: ...

    # Length / Velocity = Time
    @overload
    def __truediv__(self: Quantity[Length, MT], other: Quantity[Velocity, MT]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Length, MT], other: Quantity[Velocity, float]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Length, float], other: Quantity[Velocity, MT_]) -> Quantity[Time, MT_]: ...

    # Mass / Density = Volume
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Density, MT]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Density, float]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, float], other: Quantity[Density, MT_]) -> Quantity[Volume, MT_]: ...

    # Energy / EnergyPerMass = Mass
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[EnergyPerMass, MT]) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[EnergyPerMass, float]) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, float], other: Quantity[EnergyPerMass, MT_]) -> Quantity[Mass, MT_]: ...

    # Area / Length = Length
    @overload
    def __truediv__(self: Quantity[Area, MT], other: Quantity[Length, MT]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[Area, MT], other: Quantity[Length, float]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[Area, float], other: Quantity[Length, MT_]) -> Quantity[Length, MT_]: ...

    # Volume / Length = Area
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Length, MT]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Length, float]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, float], other: Quantity[Length, MT_]) -> Quantity[Area, MT_]: ...

    # Volume / Area = Length
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Area, MT]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Area, float]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, float], other: Quantity[Area, MT_]) -> Quantity[Length, MT_]: ...

    # MassFlow / Density = VolumeFlow
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[Density, MT]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[Density, float]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, float], other: Quantity[Density, MT_]) -> Quantity[VolumeFlow, MT_]: ...

    # MassFlow / VolumeFlow = Density
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[VolumeFlow, MT]) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[VolumeFlow, float]) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, float], other: Quantity[VolumeFlow, MT_]) -> Quantity[Density, MT_]: ...

    # VolumeFlow / Area = Velocity
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[Area, MT]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[Area, float]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, float], other: Quantity[Area, MT_]) -> Quantity[Velocity, MT_]: ...

    # VolumeFlow / Velocity = Area
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[Velocity, MT]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[Velocity, float]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, float], other: Quantity[Velocity, MT_]) -> Quantity[Area, MT_]: ...

    # VolumeFlow / MassFlow = SpecificVolume
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[MassFlow, MT]) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[MassFlow, float]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, float], other: Quantity[MassFlow, MT_]
    ) -> Quantity[SpecificVolume, MT_]: ...

    # VolumeFlow / SpecificVolume = MassFlow
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[SpecificVolume, MT]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[SpecificVolume, float]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, float], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[MassFlow, MT_]: ...

    # Energy / Pressure = Volume
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Pressure, MT]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Pressure, float]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, float], other: Quantity[Pressure, MT_]) -> Quantity[Volume, MT_]: ...

    # Energy / Volume = Pressure
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Volume, MT]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Volume, float]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, float], other: Quantity[Volume, MT_]) -> Quantity[Pressure, MT_]: ...

    # Power / Pressure = VolumeFlow
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Pressure, MT]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Pressure, float]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, float], other: Quantity[Pressure, MT_]) -> Quantity[VolumeFlow, MT_]: ...

    # Power / VolumeFlow = Pressure
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[VolumeFlow, MT]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[VolumeFlow, float]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, float], other: Quantity[VolumeFlow, MT_]) -> Quantity[Pressure, MT_]: ...

    # Volume / Mass = SpecificVolume
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Mass, MT]) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Mass, float]) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, float], other: Quantity[Mass, MT_]) -> Quantity[SpecificVolume, MT_]: ...

    # Volume / SpecificVolume = Mass
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[SpecificVolume, MT]) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[SpecificVolume, float]) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, float], other: Quantity[SpecificVolume, MT_]) -> Quantity[Mass, MT_]: ...

    # EnergyPerMass / TemperatureDifference = SpecificHeatCapacity
    # (the reverse, EnergyPerMass / SpecificHeatCapacity, is intentionally omitted: it
    # resolves to Temperature rather than TemperatureDifference since the two share the
    # [temperature] dimension, so it is left to fall through to UnknownDimensionality)
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[TemperatureDifference, MT]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[TemperatureDifference, float]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, float], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT_]: ...

    # Force / Area = Pressure
    @overload
    def __truediv__(self: Quantity[Force, MT], other: Quantity[Area, MT]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[Force, MT], other: Quantity[Area, float]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[Force, float], other: Quantity[Area, MT_]) -> Quantity[Pressure, MT_]: ...

    # Force / Pressure = Area
    @overload
    def __truediv__(self: Quantity[Force, MT], other: Quantity[Pressure, MT]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(self: Quantity[Force, MT], other: Quantity[Pressure, float]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(self: Quantity[Force, float], other: Quantity[Pressure, MT_]) -> Quantity[Area, MT_]: ...

    # Energy / Length = Force
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Length, MT]) -> Quantity[Force, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Length, float]) -> Quantity[Force, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, float], other: Quantity[Length, MT_]) -> Quantity[Force, MT_]: ...

    # Energy / Force = Length
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Force, MT]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Force, float]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, float], other: Quantity[Force, MT_]) -> Quantity[Length, MT_]: ...

    # Power / Force = Velocity
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Force, MT]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Force, float]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, float], other: Quantity[Force, MT_]) -> Quantity[Velocity, MT_]: ...

    # Power / Velocity = Force
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Velocity, MT]) -> Quantity[Force, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Velocity, float]) -> Quantity[Force, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, float], other: Quantity[Velocity, MT_]) -> Quantity[Force, MT_]: ...

    # Mass / MolarMass = Substance
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[MolarMass, MT]) -> Quantity[Substance, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[MolarMass, float]) -> Quantity[Substance, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, float], other: Quantity[MolarMass, MT_]) -> Quantity[Substance, MT_]: ...

    # Mass / Substance = MolarMass
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Substance, MT]) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Substance, float]) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, float], other: Quantity[Substance, MT_]) -> Quantity[MolarMass, MT_]: ...

    # Substance / Volume = MolarDensity
    @overload
    def __truediv__(self: Quantity[Substance, MT], other: Quantity[Volume, MT]) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Substance, MT], other: Quantity[Volume, float]) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Substance, float], other: Quantity[Volume, MT_]) -> Quantity[MolarDensity, MT_]: ...

    # Substance / MolarDensity = Volume
    @overload
    def __truediv__(self: Quantity[Substance, MT], other: Quantity[MolarDensity, MT]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Substance, MT], other: Quantity[MolarDensity, float]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Substance, float], other: Quantity[MolarDensity, MT_]) -> Quantity[Volume, MT_]: ...

    # PowerPerArea / TemperatureDifference = HeatTransferCoefficient
    # (the reverse, PowerPerArea / HeatTransferCoefficient, is intentionally omitted for
    # the same reason as EnergyPerMass / SpecificHeatCapacity above: it resolves to
    # Temperature rather than TemperatureDifference)
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[TemperatureDifference, MT]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[TemperatureDifference, float]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, float], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[HeatTransferCoefficient, MT_]: ...

    # ThermalConductivity / Length = HeatTransferCoefficient
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[Length, MT]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[Length, float]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, float], other: Quantity[Length, MT_]
    ) -> Quantity[HeatTransferCoefficient, MT_]: ...

    # ThermalConductivity / HeatTransferCoefficient = Length
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[HeatTransferCoefficient, MT]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[HeatTransferCoefficient, float]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, float], other: Quantity[HeatTransferCoefficient, MT_]
    ) -> Quantity[Length, MT_]: ...

    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[DT_, float]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[DT_, MT]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DT, float], other: Quantity[Any, Numpy1DArray]
    ) -> Quantity[UnknownDimensionality, Numpy1DArray]: ...
    @overload
    def __truediv__(
        self: Quantity[DT, float], other: Quantity[Any, pl.Series]
    ) -> Quantity[UnknownDimensionality, pl.Series]: ...
    @overload
    def __truediv__(
        self: Quantity[DT, float], other: Quantity[Any, pl.Expr]
    ) -> Quantity[UnknownDimensionality, pl.Expr]: ...
    @overload
    def __truediv__(
        self: Quantity[DT, float], other: Quantity[Any, float]
    ) -> Quantity[UnknownDimensionality, float]: ...
    @overload
    def __truediv__(self, other: float) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[DT_, float]) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[DT_, MT]) -> Quantity[UnknownDimensionality, MT]: ...
    def __truediv__(self, other: Quantity[Any, Any] | float) -> Quantity[Any, Any]:
        if isinstance(other, Quantity):
            self._check_comparable_magnitudes(self.m, other.m, "combine")  # ty: ignore[invalid-argument-type]
        ret = cast("Quantity[DT, MT]", self._pint_super.__truediv__(other))

        # preserve the dimensionality for other
        # it might be a distinct subclass with identical units as another dimensionality
        if self.dimensionless and isinstance(other, Quantity):
            subcls = self.get_subclass(other._dimensionality_type, type(ret.m))
            return subcls(ret)

        return self._call_subclass(ret.m, ret.u)

    def __itruediv__(self, other: Any) -> Any:  # noqa: ANN401
        return cast("Quantity[Any, Any]", cast(Any, self).__truediv__(other))

    @overload
    def __rtruediv__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]: ...

    # reciprocals: a scalar divided by a quantity (e.g. 1 / density = specific volume)
    @overload
    def __rtruediv__(self: Quantity[Density, MT], other: float) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[SpecificVolume, MT], other: float) -> Quantity[Density, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[Time, MT], other: float) -> Quantity[Frequency, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[Frequency, MT], other: float) -> Quantity[Time, MT]: ...

    @overload
    def __rtruediv__(self, other: float) -> Quantity[UnknownDimensionality, MT]: ...

    def __rtruediv__(self, other: Any) -> Quantity[Any, Any]:
        ret = self._pint_super.__rtruediv__(other)
        return cast("Quantity[UnknownDimensionality, MT]", self._call_subclass(ret.m, ret.u))

    @overload
    def __floordiv__(
        self: Quantity[DT, Numpy1DArray], other: Quantity[DT, float]
    ) -> Quantity[Dimensionless, Numpy1DArray]: ...
    @overload
    def __floordiv__(
        self: Quantity[DT, pl.Series], other: Quantity[DT, float]
    ) -> Quantity[Dimensionless, pl.Series]: ...
    @overload
    def __floordiv__(self: Quantity[DT, pl.Expr], other: Quantity[DT, float]) -> Quantity[Dimensionless, pl.Expr]: ...
    @overload
    def __floordiv__(self: Quantity[DT, MT], other: Quantity[DT, MT]) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __floordiv__(self: Quantity[DT, MT], other: Quantity[Dimensionless, float]) -> Quantity[DT, MT]: ...
    @overload
    def __floordiv__(self: Quantity[DT, MT], other: Quantity[Dimensionless, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __floordiv__(self: Quantity[DT, MT], other: float) -> Quantity[DT, MT]: ...
    def __floordiv__(self, other: Quantity[Any, Any] | float) -> Quantity[Any, Any]:
        if isinstance(other, Quantity):
            self._check_comparable_magnitudes(self.m, other.m, "combine")  # ty: ignore[invalid-argument-type]

        if isinstance(other, (float, int)):
            return self._call_subclass(cast("MT", self._floor_magnitude(self.m / other)), self.u)

        if other.dimensionless:
            magnitude = self._floor_magnitude(self.m / other.to_base_units().m)
            return self._call_subclass(magnitude, self.u)

        ret = self._pint_super.__floordiv__(other)
        if self.is_compatible_with(other):
            divisor = other.to(self.u).m
        else:
            return self._call_subclass(ret.m, ret.u)

        # Floor the quotient rather than using each container's `//` kernel: Python
        # and numpy implement floating floor-division with subtly different rounding
        # from Polars (e.g. 1 m // 1 cm used to be 99 vs 100).
        magnitude = self._floor_magnitude(self.m / divisor)
        return self._call_subclass(magnitude, ret.u)

    def __ifloordiv__(self, other: Any) -> Any:  # noqa: ANN401
        return cast("Quantity[Any, Any]", cast(Any, self).__floordiv__(other))

    def __abs__(self) -> Quantity[DT, MT]:
        ret = cast("Quantity[DT, MT]", super().__abs__())
        return self._call_subclass(ret.m, ret.u)

    def __pos__(self) -> Quantity[DT, MT]:
        ret = cast("Quantity[DT, MT]", super().__pos__())
        return self._call_subclass(ret.m, ret.u)

    def __neg__(self) -> Quantity[DT, MT]:
        ret = cast("Quantity[DT, MT]", super().__neg__())
        return self._call_subclass(ret.m, ret.u)

    # only an ndarray magnitude (the numpy world) is iterable -- it yields
    # scalar Quantity elements. float/pl.Expr/pl.Series are not: the generic
    # overload returns NoReturn because __iter__ always raises for them (see
    # the body), which lets the type checker flag iterating such a Quantity
    # (e.g. passing one into pl.select) instead of silently allowing it.
    @overload
    def __iter__(self: Quantity[DT, Numpy1DArray]) -> Iterator[Quantity[DT, float]]: ...
    @overload
    def __iter__(self: Quantity[DT, MT]) -> NoReturn: ...
    def __iter__(self) -> Iterator[Any]:
        mag = self._magnitude
        # float and pl.Expr are not iterable, and pl.Series (the polars world)
        # routes data access through .m like pl.Expr. the usual way to land here
        # is passing a Quantity into a polars context (pl.select / with_columns /
        # filter), which iterates any Iterable as a collection of expressions. A
        # Quantity is not an expression and converting one drops its unit, so
        # refuse and point at the explicit boundary (.m).
        if isinstance(mag, (float, pl.Expr, pl.Series)):
            raise TypeError(
                f"Quantity with {self._get_magnitude_type_name(type(mag))} magnitude is not "
                'iterable; if using with Polars: materialize the magnitude in an explicit unit with ".to(<unit>).m".'
            )
        return (self._call_subclass(n.m, n.u) for n in super().__iter__())

    @overload
    def __getitem__(self: Quantity[DT, pl.Series], index: int) -> Quantity[DT, float]: ...  # pyrefly: ignore[bad-override]
    @overload
    def __getitem__(self: Quantity[DT, pl.Series], index: slice) -> Quantity[DT, pl.Series]: ...
    @overload
    def __getitem__(self: Quantity[DT, Numpy1DArray], index: int) -> Quantity[DT, float]: ...
    @overload
    def __getitem__(self: Quantity[DT, Numpy1DArray], index: slice) -> Quantity[DT, Numpy1DArray]: ...
    # numpy fancy indexing: a boolean mask (the type the comparison operators return, so
    # q[q > threshold] type-checks) or an integer index array/sequence. Not offered for a
    # pl.Series magnitude: polars refuses a boolean-mask __getitem__ (use .m.filter(...))
    @overload
    def __getitem__(
        self: Quantity[DT, Numpy1DArray], index: Numpy1DBoolArray | Numpy1DIntArray | Sequence[int]
    ) -> Quantity[DT, Numpy1DArray]: ...
    def __getitem__(  # ty: ignore[invalid-method-override]
        self, index: int | slice | Numpy1DBoolArray | Numpy1DIntArray | Sequence[int]
    ) -> Quantity[Any, Any]:
        ret = cast("Quantity[DT, Any]", self._pint_super.__getitem__(index))

        magnitude_type = cast("type[Any]", type(ret.m))  # ty: ignore[redundant-cast]
        subcls = self._get_dimensional_subclass(self.dt, self._get_magnitude_type_safe(magnitude_type))  # ty: ignore[invalid-argument-type]
        instance = cast(Any, subcls)(ret.m, ret.u)

        return cast("Quantity[Any, Any]", instance)


def _is_pickle_global(dim: type[Dimensionality]) -> bool:
    module = sys.modules.get(dim.__module__)
    return module is not None and getattr(module, dim.__name__, None) is dim


def _reconstruct_quantity(magnitude: object, unit: str, dim: type[Dimensionality] | None) -> Quantity[Any, Any]:
    qty = Quantity(cast(Any, magnitude), unit)
    if dim is None:
        return qty
    return qty.asdim(dim)


def _quantity_typeguard_checker(
    value: Any,  # noqa: ANN401
    origin_type: Any,  # noqa: ANN401
    args: tuple[Any, ...],
    memo: TypeCheckMemo,  # noqa: ARG001
) -> None:
    # imported here rather than at module scope: encomp.misc resolves encomp.units lazily,
    # and this keeps the dependency one-directional
    from .misc import isinstance_types

    annotation = origin_type[args] if args else origin_type

    if not isinstance_types(value, annotation):
        raise TypeCheckError(f"is not an instance of {annotation.__module__}.{annotation.__qualname__}")


def _quantity_typeguard_lookup(
    origin_type: Any,  # noqa: ANN401
    args: tuple[Any, ...],  # noqa: ARG001
    extras: tuple[Any, ...],  # noqa: ARG001
) -> Any:  # noqa: ANN401
    if isclass(origin_type) and issubclass(origin_type, Quantity):
        return _quantity_typeguard_checker

    return None


# Teach typeguard how to check a Quantity, so `@typechecked` and the nested type forms that
# isinstance_types hands to check_type both compare dimensionality AND magnitude type instead of
# falling back to a plain isinstance. isinstance is not enough: Quantity[UnknownDimensionality] --
# the runtime class behind Q[Any, Any] -- is a *sibling* of every concrete dimensionality rather
# than a base, so `isinstance(Q(1, "bar"), Q[Any, Any])` is False.
checker_lookup_functions.insert(0, _quantity_typeguard_lookup)

# register the Quantity and Unit implementations on the registry, so every object
# created through it (results of pint-internal operations, parse_units output) is
# the encomp type. The .u / .units properties and _validate_unit additionally
# re-wrap at the encomp boundaries, covering values that bypassed the registry
setattr(UNIT_REGISTRY, "Quantity", Quantity)  # noqa: B010
setattr(UNIT_REGISTRY, "Unit", Unit)  # noqa: B010
