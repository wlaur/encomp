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
import numbers
import re
import warnings
from collections.abc import Iterable, Iterator, Sequence, Sized
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
from pint.errors import DimensionalityError
from pint.facets.measurement.objects import MeasurementQuantity
from pint.facets.nonmultiplicative.objects import NonMultiplicativeQuantity
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain.quantity import PlainQuantity
from pint.facets.plain.unit import PlainUnit
from pint.registry import LazyRegistry, UnitRegistry
from pint.util import UnitsContainer
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from .misc import isinstance_types
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
    MassUnits,
    MolarMass,
    MolarMassUnits,
    NormalVolume,
    NormalVolumeFlow,
    NormalVolumeFlowUnits,
    NormalVolumeUnits,
    Numpy1DArray,
    Numpy1DBoolArray,
    Power,
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

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = None


def _ensure_sympy() -> None:
    global sp
    if sp is None:
        import sympy as sp


_LOGGER = logging.getLogger(__name__)

DimensionalityTypeName = Annotated[str, "Dimensionality name"]
MagnitudeTypeName = Literal[
    "float",
    "ndarray",
    "pl.Series",
    "pl.Expr",
]

if SETTINGS.ignore_ndarray_unit_stripped_warning:
    warnings.filterwarnings(
        "ignore",
        message="The unit of the quantity is stripped when downcasting to ndarray.",
    )


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
    pass


class DimensionalityTypeError(_DimensionalityError):
    pass


class DimensionalityComparisonError(_DimensionalityError):
    pass


class DimensionalityRedefinitionError(ValueError):
    pass


# keep track of user-created dimensions
# NOTE: make sure to list all that are defined in defs/units.txt ("# custom dimensions")
CUSTOM_DIMENSIONS: list[str] = [
    "currency",
    "normal",
]


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
    def __setattr__(self, key: str, value: Any) -> None:  # noqa: ANN401
        # ensure that static options cannot be overridden
        if key in _REGISTRY_STATIC_OPTIONS and value != _REGISTRY_STATIC_OPTIONS[key]:
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


UNIT_REGISTRY = cast(UnitRegistry[Any], _LazyRegistry())

for k, v in _REGISTRY_STATIC_OPTIONS.items():
    setattr(UNIT_REGISTRY, k, v)

# make sure that UNIT_REGISTRY is the only registry that can be used
setattr(pint, "_DEFAULT_REGISTRY", UNIT_REGISTRY)  # noqa: B010
cast(Any, pint.application_registry).set(UNIT_REGISTRY)

# the default format must be set after Quantity and Unit are registered
UNIT_REGISTRY.formatter.default_format = SETTINGS.default_unit_format


def set_quantity_format(fmt: str = "compact") -> None:
    fmt_aliases = {"normal": "~P", "siunitx": "~Lx"}

    if fmt in fmt_aliases:
        fmt = fmt_aliases[fmt]

    if fmt not in Quantity.FORMATTING_SPECS:
        raise ValueError(
            f'Cannot set default format to "{fmt}", '
            f"fmt is one of {Quantity.FORMATTING_SPECS} "
            "or alias siunitx: ~L, compact: ~P"
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
    """

    if not name.isidentifier():
        raise ValueError(
            f"Dimensionality name must be a valid Python identifier (alphanumeric and underscores, "
            f"cannot start with a digit, no spaces or special characters). Got: {name!r}"
        )

    if name in CUSTOM_DIMENSIONS:
        msg = f"Cannot define new dimensionality with name: {name}, a dimensionality with this name was already defined"
        if if_exists == "raise":
            raise DimensionalityRedefinitionError(msg)
        elif if_exists == "warn":
            _LOGGER.warning(msg)
        else:
            raise ValueError(f"Invalid value: {if_exists=}")

    definition_str = f"{name} = [{name}]"

    if symbol is not None:
        definition_str = f"{definition_str} = {symbol}"

    UNIT_REGISTRY.define(definition_str)
    CUSTOM_DIMENSIONS.append(name)


class _QuantityMeta(type):
    def __eq__(cls, obj: object) -> bool:
        # override the == operator so that type(val) == Quantity returns True for subclasses
        if obj is Quantity:
            return True

        return super().__eq__(obj)

    def __hash__(cls) -> int:
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
    # constants
    NORMAL_M3_VARIANTS = ("nm³", "Nm³", "nm3", "Nm3", "nm**3", "Nm**3", "nm^3", "Nm^3")
    TEMPERATURE_DIFFERENCE_UCS = (Unit("delta_degC")._units, Unit("delta_degF")._units)

    # used for float and numpy array comparison
    rtol: float = 1e-9
    atol: float = 1e-12

    # compact, Latex, HTML, Latex/siunitx formatting
    FORMATTING_SPECS = PINT_FORMATTING_SPECIFIERS

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

    def __str__(self) -> str:
        return self.__format__(self._REGISTRY.formatter.default_format)

    def __hash__(self) -> int:
        if not isinstance(self.m, float):
            raise TypeError(f"unhashable type: 'Quantity' (magnitude type: {type(self.m).__name__})")

        # hash on the canonical root-unit representation so the eq/hash contract holds:
        # __eq__ compares across units (Q(1, "m") == Q(100, "cm")), so two quantities that
        # are equal must hash equal -- hashing the raw (m, u) would break that. to_root_units
        # is used (not to_base_units) because it does not raise for a TemperatureDifference.
        # NOTE: __eq__ is tolerant (np.isclose), so quantities that are only *approximately*
        # equal may still hash differently -- an inherent limit of tolerant equality, the same
        # way hash(0.1 + 0.2) != hash(0.3); exact equality (the common case) is now consistent.
        root = self.to_root_units()
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

        return self._pint_super.__array__(t)

    @staticmethod
    def validate_magnitude_type(mt: type) -> None:
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
    def _get_magnitude_type_name(mt: type) -> MagnitudeTypeName:
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
                (DimensionalQuantity,),
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

            if not np.isclose(exp1, exp2, rtol=rtol, atol=atol):
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
        unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | Quantity[DT, Any] | None,
    ) -> Unit[DT]:
        if unit is None:
            return Unit("dimensionless")
        elif isinstance(unit, Unit):
            return Unit(unit)
        elif isinstance(unit, Quantity):
            return unit.u
        elif isinstance(unit, dict):
            # compatibility with internal pint API
            return Unit(Quantity._validate_unit(str(UnitsContainer(unit))))
        elif isinstance(unit, UnitsContainer):
            # compatibility with internal pint API
            return Unit(Quantity._validate_unit(str(unit)))
        else:
            return cast("Unit[DT]", Unit(cast(Any, Quantity._REGISTRY.parse_units(Quantity.correct_unit(unit)))))

    @staticmethod
    def _validate_magnitude(val: MT | Sequence[float]) -> MT:
        if isinstance(val, int):
            return cast("MT", float(val))
        elif isinstance(val, float):
            # numpy float64 is a runtime subclass of float, so the type system
            # cannot distinguish it; this normalization to a plain Python float
            # is genuinely needed but unavoidably looks redundant to the checker
            return cast("MT", float(val))
        elif isinstance(val, np.ndarray):
            if len(val.shape) != 1:
                raise ValueError(f"Only 1-dimensional Numpy arrays can be used as magnitude, got shape {val.shape}")
            return val
        elif isinstance(val, (pl.Series, pl.Expr)):
            return val
        elif hasattr(val, "is_Atom"):
            # implicit way of checking if the value is a sympy symbol without having to import Sympy
            return cast("MT", val)
        else:
            arr = cast(MT, np.array(val).astype(np.float64))
            return arr

    @classmethod
    def get_unit(cls, unit_name: AllUnits | str) -> Unit:
        return Unit(cast(Any, cls._REGISTRY.parse_units(unit_name)))

    def get_subclass(self, dt: type[DT_], mt: type[MT_]) -> type[Quantity[DT_, MT_]]:
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

    @staticmethod
    def _cast_array_float(inp: np.ndarray) -> Numpy1DArray:
        # don't fail in case the array contains unsupported objects,
        # cast to float64, matches the Numpy1DArray type definition
        try:
            return inp.astype(np.float64, casting="unsafe", copy=True)
        except TypeError:
            return inp

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
    def __new__(cls, val: Sequence[float], unit: DensityUnits) -> Quantity[Density, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: SpecificVolumeUnits) -> Quantity[SpecificVolume, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: EnergyUnits) -> Quantity[Energy, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: Sequence[float], unit: PowerUnits) -> Quantity[Power, Numpy1DArray]: ...
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
    def __new__(cls, val: MT, unit: DensityUnits) -> Quantity[Density, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: SpecificVolumeUnits) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: EnergyUnits) -> Quantity[Energy, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: PowerUnits) -> Quantity[Power, MT]: ...
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
    def __new__(  # fallback overload: matches anything not covered above
        cls,
        val: Any,  # noqa: ANN401
        unit: Any = None,  # noqa: ANN401
        _depth: int = 0,
    ) -> Quantity[Any, Any]: ...
    def __new__(
        cls,
        val: Any,
        unit: Any = None,
        _depth: int = 0,
    ) -> Quantity[Any, Any]:
        unit = cast("Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | None", unit)

        if isinstance(val, Quantity):
            _input_qty = cast("Quantity[DT, MT]", val)
            if unit is not None:
                _input_qty = _input_qty.to(unit)

            val, unit = _input_qty.m, _input_qty.u

        valid_magnitude = cls._validate_magnitude(val)
        valid_unit = cls._validate_unit(unit)

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
                subcls = cls._get_dimensional_subclass(TemperatureDifference, type(valid_magnitude))
            else:
                dim = Dimensionality.get_dimensionality(valid_unit.dimensionality)
                subcls = cls._get_dimensional_subclass(dim, type(valid_magnitude))

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
        return self._magnitude

    @m.setter
    def m(self, val: MT) -> None:
        self._magnitude = val

    @property
    def mt(self) -> type[MT]:
        # _magnitude_type is set on the magnitude-specific subclass, but pint
        # builds results via self.__class__(...), which resolves to the
        # magnitude-agnostic dimension-only subclass (where it is None). that is
        # intentional -- the result magnitude type may differ from the source.
        # _get_magnitude_type_safe falls back to the live magnitude's type when
        # given an invalid value (None included), so it recovers the real type
        return self._get_magnitude_type_safe(self._magnitude_type)

    @property
    def mt_name(self) -> MagnitudeTypeName:
        return self._get_magnitude_type_name(self.mt)

    @property
    def units(self) -> Unit[DT]:
        return Unit(super().units)

    @property
    def u(self) -> Unit[DT]:
        return self.units

    @property
    def dt(self) -> type[DT]:
        return cast(type[DT], self._dimensionality_type)

    @property
    def _is_temperature_difference(self) -> bool:
        return self.dt == TemperatureDifference

    @classmethod
    def _is_temperature_difference_unit(cls, unit: Unit[DT]) -> bool:
        return unit._units in cls.TEMPERATURE_DIFFERENCE_UCS

    def _check_temperature_compatibility(self, unit: Unit[DT]) -> None:
        if self._is_temperature_difference and unit._units not in self.TEMPERATURE_DIFFERENCE_UCS:
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
        ret = cast("Quantity[DT, MT]", self._pint_super.to_reduced_units())
        return ret

    def to_root_units(self) -> Quantity[DT, MT]:
        ret = cast("Quantity[DT, MT]", super().to_root_units())
        return ret

    def to_base_units(self) -> Quantity[DT, MT]:
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

    def _to_unit(
        self, unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | Quantity[DT, Any]
    ) -> Unit[DT]:
        return self._validate_unit(unit)

    def to(
        self, unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | Quantity[DT, Any]
    ) -> Quantity[DT, MT]:
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

    def ito(self, unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number]) -> None:
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
            self._magnitude = cast("MT", _m.astype(np.float64))

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
    def check(  # pyright: ignore[reportIncompatibleMethodOverride]  # pyrefly: ignore[bad-override]
        self,
        dimension: Quantity[Any, Any] | UnitsContainer | Unit[DT_] | Unit | str | Dimensionality | type[Dimensionality],
    ) -> bool:
        if isinstance(dimension, type) and dimension == TemperatureDifference:
            return self.check("delta_degC")

        if isinstance(dimension, type) and dimension == Temperature:
            return self.check("degC")

        if isinstance(dimension, Quantity):
            return self.dt == dimension._dimensionality_type

        if isinstance(dimension, str):
            return self.check(self._validate_unit(dimension))

        if isinstance(dimension, Unit):
            # it's not possible to know if an instance of Unit is Temperature or TemperatureDifference
            # until it is used to construct a Quantity
            unit_qty = Quantity(1.0, dimension)

            if isinstance_types(unit_qty, Quantity[TemperatureDifference]):
                unit_qty = Quantity(1.0, dimension).asdim(TemperatureDifference)

            return self.check(unit_qty)

        if hasattr(dimension, "dimensions"):
            _dims = getattr(dimension, "dimensions", None)
            if _dims is None:
                raise TypeError(f"Attribute 'dimensions' is missing or None: {dimension}")
            return self._pint_super.check(cast(UnitsContainer, _dims))

        if isinstance(dimension, (Dimensionality, type, PlainQuantity)):
            raise TypeError(f"Invalid type for dimension: {dimension} ({type(dimension)})")

        return self._pint_super.check(dimension)

    def __format__(self, spec: str) -> str:
        if not spec.endswith(Quantity.FORMATTING_SPECS):
            spec = f"{spec}{self._REGISTRY.formatter.default_format}"

        return super().__format__(spec)

    @staticmethod
    def correct_unit(unit: str) -> str:
        unit = unit.strip()

        if unit == "-":
            return "dimensionless"

        # normal cubic meter, not nano or Newton
        # there's no consistent way of abbreviating "normal liter",
        # so we'll not even try to parse that, use "nanometer**3" if necessary
        for n in Quantity.NORMAL_M3_VARIANTS:
            if n in unit:
                # include brackets, otherwise "kg/nm3" is incorrectly converted to "kg/normal*m3"
                unit = unit.replace(n, "(normal * m³)")

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
            "%": "percent",
            "‰": "permille",
            "r/min": "rpm",
            # ΔK does not really make sense, it's not an offset scale
            "ΔK": "K",
            "Δ": "delta_",
        }

        for old, new in replacements.items():
            if old in unit:
                unit = unit.replace(old, new)

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
                s = cls._dimension_symbol_map[symbol]
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

            val: float | int | list[float]

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
            elif isinstance(mag, list):
                val = [float(x) for x in cast("list[Any]", mag)]  # pyrefly: ignore[redundant-cast]  # cast required by pyright
                magnitude_type = "list"
            else:
                raise ValueError(f"Unknown magnitude type {type(mag)}: {mag}")

            return {
                "unit": str(qty.u),
                "value": val,
                "magnitude_type": magnitude_type,
            }

        ser_schema = core_schema.plain_serializer_function_ser_schema(_serialize, info_arg=True)

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
                        {"type": "null"},
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
    def validate(
        cls,
        qty: Any,  # noqa: ANN401
        info: Any,  # noqa: ANN401, ARG003
    ) -> Quantity[DT, MT]:
        if isinstance(qty, dict) and "value" in qty and "magnitude_type" in qty:
            val = cast(Any, qty["value"])
            magnitude_type = cast(Literal["int", "float", "list"] | str, qty["magnitude_type"])

            if magnitude_type.startswith("np.ndarray"):
                _, dtype_str, _ = magnitude_type.split(":", 2)
                arr = np.array(val, dtype=np.dtype(dtype_str))
                magnitude = arr
            elif magnitude_type.startswith("pl.Series"):
                _, dtype_str = magnitude_type.split(":", 1)

                dtype: type[pl.DataType]

                match dtype_str:
                    case "Float32":
                        dtype = pl.Float32
                    case "Float64":
                        dtype = pl.Float64
                    case "Int64":
                        dtype = pl.Int64
                    case "Int32":
                        dtype = pl.Int32
                    case "Int16":
                        dtype = pl.Int16
                    case _:
                        raise ValueError(f"Unknown Polars Series dtype: '{dtype_str}'")

                magnitude = pl.Series(val, dtype=dtype)
            elif magnitude_type == "list":
                magnitude = val
            elif magnitude_type == "int":
                magnitude = int(val)
            elif magnitude_type == "float":
                magnitude = float(val)
            else:
                raise TypeError(f"Unknown magnitude_type {magnitude_type!r}")

            unit = cast(str | None, cast("dict[str, Any]", qty).get("unit"))
            ret = cls(cast(MT, magnitude), unit=unit)
        else:
            ret = cast("Quantity[DT, MT]", qty if isinstance(qty, Quantity) else cls(cast(Any, qty)))

        if isinstance(ret, cls):
            return ret

        if issubclass(cls, cls.get_unknown_dimensionality_subclass()):
            return ret

        if cls._is_incomplete_dimensionality(cls._dimensionality_type):
            return ret

        raise ExpectedDimensionalityError(
            f"Value {ret} ({type(ret).__name__}) does not match expected dimensionality {cls.__name__}"
        )

    def check_compatibility(self, other: Quantity[Any, Any] | float) -> None:
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
        if self.check(Temperature):
            assert other.check(TemperatureDifference)
            v1 = self.to("degC").m
            v2 = other.to("delta_degC").m

            val = v1 + v2 if operator == "add" else v1 - v2
        else:
            assert self.check(TemperatureDifference)
            assert other.check(Temperature)

            v1 = self.to("delta_degC").m
            v2 = other.to("degC").m

            val = v1 + v2 if operator == "add" else v1 - v2

        return cast("Quantity[Temperature, MT]", Quantity(val, "degC"))

    def __round__(self, ndigits: int | None = None) -> Quantity[DT, MT]:
        if ndigits is None:
            ndigits = 0

        if isinstance(self.m, float):
            return cast("Quantity[DT, MT]", super().__round__(ndigits))
        elif isinstance(self.m, np.ndarray):
            return cast("Quantity[DT, MT]", self.__class__(np.round(self.m, ndigits), self.u))
        else:
            raise NotImplementedError(f"__round__ is not implemented for magnitude type {type(self.m)}")

    @property
    def is_scalar(self) -> bool:
        return isinstance(self.m, float)

    @property
    def ndim(self) -> int:
        if isinstance(self.m, (float, int)):
            return 0

        return getattr(self.m, "ndim", 0)

    def asdim(self, other: type[DT_] | Quantity[DT_, MT]) -> Quantity[DT_, MT]:
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

        subcls = self._get_dimensional_subclass(dim, type(self.m))
        return cast("Quantity[DT_, MT]", subcls(self.m, self.u))

    def unknown(self) -> Quantity[UnknownDimensionality, MT]:
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
            if isinstance(m, Iterable):
                return cast("Quantity[DT, Any]", self.get_subclass(dt, np.ndarray)([float(n) for n in m], u))
            else:
                return cast("Quantity[DT, Any]", self.get_subclass(dt, float)(float(cast(Any, m)), u))
        elif magnitude_type is np.ndarray or magnitude_type_origin is np.ndarray:
            _m = [m] if not isinstance(m, Iterable) else m
            vals = np.array(_m)
            return cast("Quantity[DT, Any]", self.get_subclass(dt, np.ndarray)(vals, u))
        elif magnitude_type is pl.Series:
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
        ret = cast("Quantity[DT, MT]", self._pint_super.__pow__(other))
        return ret

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
        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            if not isinstance(other, Quantity):
                raise e

            self_is_temp_or_diff_temp = self.check(Temperature) or self.check(TemperatureDifference)
            other_is_temp_or_diff_temp = other.check(Temperature) or other.check(TemperatureDifference)

            if self_is_temp_or_diff_temp and other_is_temp_or_diff_temp:
                return self._temperature_difference_add_sub(other, "add")

            raise e

        ret = cast("Quantity[DT, MT]", self._pint_super.__add__(other))

        return self._call_subclass(ret.m, ret.u)

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
        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            if not isinstance(other, Quantity):
                raise e

            # only Temperature - TemperatureDifference is meaningful here; the
            # reverse (ΔT - T) is not a temperature and stays an error
            if self.check(Temperature) and other.check(TemperatureDifference):
                return self._temperature_difference_add_sub(other, "sub")

            raise e

        ret = cast("Quantity[DT, MT]", self._pint_super.__sub__(other))

        if isinstance(other, Quantity) and self.dt == Temperature and other._dimensionality_type == Temperature:
            _mt = type(ret.m)
            subcls = self._get_dimensional_subclass(TemperatureDifference, _mt)
            return subcls(ret.m, ret.u)

        return self._call_subclass(ret.m, ret.u)

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
    def __eq__(self, other: object) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:  # pyright: ignore[reportIncompatibleMethodOverride]
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
            other = Quantity(other, "dimensionless")

        m = self.m
        other_m = cast(float | Numpy1DArray | pl.Series | pl.Expr, cast(Any, other).to(self.u).m)

        if isinstance(m, (float, int, np.ndarray)) and isinstance(other_m, (float, int, np.ndarray)):
            ret = np.isclose(cast(Any, m), cast(Any, other_m), self.rtol, self.atol)

            if isinstance(ret, np.bool):
                return bool(cast(Any, ret))
            else:
                return ret

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
    def __ne__(self, other: object) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not isinstance(other, (Quantity, float, int)):
            return bool(self._pint_super.__ne__(other))

        try:
            self.check_compatibility(cast(Any, other))
        except DimensionalityTypeError:
            # mirror __eq__: incomparable operands are simply not equal
            return True

        if isinstance(other, (float, int)):
            other = Quantity(other, "dimensionless")

        m = self.m
        other_m = cast(float | Numpy1DArray | pl.Series | pl.Expr, cast(Any, other).to(self.u).m)

        if isinstance(m, (float, int, np.ndarray)) and isinstance(other_m, (float, int, np.ndarray)):
            ret = ~np.isclose(cast(Any, m), cast(Any, other_m), self.rtol, self.atol)

            if isinstance(ret, np.bool):
                return bool(cast(Any, ret))
            else:
                return ret

        ret = m != other_m

        return ret

    def _ordering_comparison(
        self,
        other: Quantity[Any, Any] | float,
        op: Literal["__gt__", "__ge__", "__lt__", "__le__"],
    ) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:
        # ordering across dimensionality types has no correct answer (e.g. Temperature
        # vs TemperatureDifference share [temperature] but must not order), so it
        # raises -- unlike __eq__, which can answer False for incompatible operands
        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            raise DimensionalityComparisonError(f"Cannot compare {self} with {other}") from e

        try:
            return getattr(self._pint_super, op)(other)
        except (ValueError, DimensionalityError) as e:
            raise DimensionalityComparisonError(str(e)) from e

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
    def __gt__(self, other: Quantity[DT, Any] | float) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:
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
    def __ge__(self, other: Quantity[DT, Any] | float) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:
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
    def __lt__(self, other: Quantity[DT, Any] | float) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:
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
    def __le__(self, other: Quantity[DT, Any] | float) -> bool | Numpy1DBoolArray | pl.Series | pl.Expr:
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
        ret = cast("Quantity[DT, MT]", self._pint_super.__mul__(other))

        # preserve the dimensionality for other
        # it might be a distinct subclass with identical units as another dimensionality
        if self.dimensionless and isinstance(other, Quantity):
            subcls = self.get_subclass(other._dimensionality_type, type(ret.m))
            return subcls(ret)

        return self._call_subclass(ret.m, ret.u)

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
        ret = cast("Quantity[DT, MT]", self._pint_super.__truediv__(other))

        # preserve the dimensionality for other
        # it might be a distinct subclass with identical units as another dimensionality
        if self.dimensionless and isinstance(other, Quantity):
            subcls = self.get_subclass(other._dimensionality_type, type(ret.m))
            return subcls(ret)

        return self._call_subclass(ret.m, ret.u)

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
        if isinstance(other, (float, int)):
            return self._call_subclass(cast("MT", self.m // other), self.u)
        elif other.dimensionless:
            return self._call_subclass(self.m // other.to_base_units().m, self.u)

        ret = self._pint_super.__floordiv__(other)

        return self._call_subclass(ret.m, ret.u)

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
    def __getitem__(self: Quantity[DT, Numpy1DArray], index: int) -> Quantity[DT, float]: ...
    def __getitem__(self, index: int) -> Quantity[Any, Any]:
        ret = cast("Quantity[DT, float]", self._pint_super.__getitem__(index))

        subcls = self._get_dimensional_subclass(self.dt, type(ret.m))
        instance = cast(Any, subcls)(ret.m, ret.u)

        return cast("Quantity[DT, float]", instance)


# override the implementations for the Quantity
# and Unit classes for the current registry
# this ensures that all Quantity objects created with this registry are the correct type
setattr(UNIT_REGISTRY, "Quantity", Quantity)  # noqa: B010
setattr(UNIT_REGISTRY, "Unit", Unit)  # noqa: B010
