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
from collections.abc import Iterable, Sized
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    assert_never,
    cast,
    get_args,
    get_origin,
    overload,
)

import numpy as np
import pandas as pd
import pint
import polars as pl
from pint._typing import QuantityOrUnitLike
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
    MAGNITUDE_TYPES,
    MT,
    MT_,
    UNSET_DIMENSIONALITY,
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
    Frequency,
    HeatingValue,
    HeatTransferCoefficient,
    HigherHeatingValue,
    KinematicViscosity,
    KinematicViscosityUnits,
    Length,
    LengthUnits,
    LowerHeatingValue,
    Luminosity,
    LuminosityUnits,
    Mass,
    MassFlow,
    MassFlowUnits,
    MassPerEnergy,
    MassPerNormalVolume,
    MassUnits,
    MolarDensity,
    MolarMass,
    MolarMassUnits,
    MolarSpecificEnthalpy,
    MolarSpecificEntropy,
    Normal,
    NormalTemperature,
    NormalVolume,
    NormalVolumeFlow,
    NormalVolumeFlowUnits,
    NormalVolumePerMass,
    NormalVolumeUnits,
    Numpy1DArray,
    Numpy1DBoolArray,
    Power,
    PowerPerArea,
    PowerPerLength,
    PowerPerTemperature,
    PowerPerVolume,
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
    Time,
    TimeUnits,
    Velocity,
    VelocityUnits,
    Volume,
    VolumeFlow,
    VolumeFlowUnits,
    VolumeUnits,
)

if TYPE_CHECKING:
    import sympy as sp  # type: ignore[import-untyped]
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
    "pd.Series",
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


def _units_containers_equal(uc1: UnitsContainer, uc2: UnitsContainer, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    if set(uc1._d.keys()) != set(uc2._d.keys()):
        return False

    for dim_name in uc1._d:
        _exp1 = uc1._d[dim_name]
        _exp2 = uc2._d[dim_name]

        if isinstance(_exp1, complex):
            raise TypeError(f"Exponent for {dim_name=} cannot be complex: {_exp1}")
        if isinstance(_exp2, complex):
            raise TypeError(f"Exponent for {dim_name=} cannot be complex: {_exp2}")

        exp1 = float(_exp1)
        exp2 = float(_exp2)

        if not np.isclose(exp1, exp2, rtol=rtol, atol=atol):
            return False

    return True


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


class _UnitRegistry(UnitRegistry):
    def __setattr__(self, key: str, value: Any) -> None:  # noqa: ANN401
        # ensure that static options cannot be overridden
        if key in _REGISTRY_STATIC_OPTIONS and value != _REGISTRY_STATIC_OPTIONS[key]:
            return

        return super().__setattr__(key, value)


class _LazyRegistry(LazyRegistry):
    def __init(self) -> None:
        args, kwargs = self.__dict__["params"]
        kwargs["on_redefinition"] = "raise"

        # override the filename
        kwargs["filename"] = str(SETTINGS.units.resolve().absolute())

        setattr(self, "__class__", _UnitRegistry)  # noqa: B010
        self.__init__(*args, **kwargs)
        assert self._after_init != "raise"
        self._after_init()


UNIT_REGISTRY = cast(UnitRegistry, _LazyRegistry())

for k, v in _REGISTRY_STATIC_OPTIONS.items():
    setattr(UNIT_REGISTRY, k, v)

# make sure that UNIT_REGISTRY is the only registry that can be used
setattr(pint, "_DEFAULT_REGISTRY", UNIT_REGISTRY)  # noqa: B010
pint.application_registry.set(UNIT_REGISTRY)

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
    NumpyQuantity,
    NonMultiplicativeQuantity,
    MeasurementQuantity,
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

    _REGISTRY: ClassVar[UnitRegistry] = UNIT_REGISTRY

    # mapping from dimensionality subclass name to quantity subclass
    # this dict will be populated at runtime
    # use a custom class attribute (not cls.__subclasses__()) for more control
    _subclasses: ClassVar[dict[tuple[DimensionalityTypeName, MagnitudeTypeName | None], type[Quantity[Any, Any]]]] = {}
    _dimension_symbol_map: ClassVar[dict[sp.Basic, Unit]] = {}

    # used to validate dimensionality and magnitude type,
    # if None the dimensionality is not checked
    # subclasses of Quantity have this class attribute set, which
    # will restrict the dimensionality when creating the object
    _dimensionality_type: ClassVar[type[Dimensionality]] = UNSET_DIMENSIONALITY

    # instance attributes
    _magnitude: MT
    _magnitude_type: type[MT]
    _original_magnitude_type: type[MT]
    _original_magnitude_kwargs: dict[str, Any]

    _max_recursion_depth = 10

    def __str__(self) -> str:
        return self.__format__(self._REGISTRY.formatter.default_format)

    def __hash__(self) -> int:
        if not isinstance(self.m, float):
            raise TypeError(f"unhashable type: 'Quantity' (magnitude type: {type(self.m).__name__})")

        return hash((self.m, self.u))

    # NOTE: pint NumpyQuantity does not have copy and dtype as kwargs for __array__
    def __array__(self, t: Any | None = None, copy: bool = False, dtype: str | None = None) -> np.ndarray:  # noqa: ANN401
        return super().__array__(t)

    @staticmethod
    def validate_magnitude_type(mt: type) -> None:
        if mt == np.ndarray:
            return

        if mt == np.float64:
            raise TypeError(
                f"Invalid magnitude type: {mt}, expected one of float, np.ndarray, pd.Series, pl.Series, pl.Expression"
            )

        for n in MAGNITUDE_TYPES:
            if mt is n:
                return
            if isinstance(n, type) and isinstance(mt, type) and issubclass(mt, n):
                return
            if mt == n:
                return

        raise TypeError(
            f"Invalid magnitude type: {mt}, expected one of float, np.ndarray, pd.Series, pl.Series, pl.Expression"
        )

    @staticmethod
    def _get_magnitude_type_name(mt: type) -> MagnitudeTypeName:
        origin = get_origin(mt)

        if mt is np.ndarray or origin is np.ndarray:
            return "ndarray"
        elif mt is float:
            return "float"
        elif mt is pl.Series or origin is pl.Series:
            return "pl.Series"
        elif mt is pl.Expr:
            return "pl.Expr"
        if mt is pd.Series or origin is pd.Series:
            return "pd.Series"
        else:
            raise TypeError(f"Invalid magnitude type: {mt} (origin {origin})")

    @classmethod
    def _get_dimensional_subclass(cls, dim: type[Dimensionality], mt: type | None) -> type[Quantity[DT, MT]]:
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

        if mt is None or mt is Any or isinstance(mt, TypeVar):
            return DimensionalQuantity

        if isinstance(mt, UnionType):
            mt = next(n for n in get_args(mt) if n is not None)
            if mt is None:
                raise RuntimeError("No non-null magnitude types were detected")

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
    def _is_incomplete_dimensionality(dim: type[Dimensionality]) -> bool:
        return dim is UNSET_DIMENSIONALITY or dim is Any or isinstance(dim, TypeVar)

    def __class_getitem__(cls, types: type[DT] | tuple[type[DT], type[MT]]) -> type[Quantity[DT, MT]]:
        if isinstance(types, tuple):
            if len(types) != 2:
                raise TypeError(f"Incorrect number of generic type parameters: {len(types)} (expected 1 or 2): {types}")

            dim, mt = types
        else:
            dim = types
            mt = None

        if cls._is_incomplete_dimensionality(dim):
            return cls._get_dimensional_subclass(UNSET_DIMENSIONALITY, mt)

        if not isinstance(dim, type):
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
        unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | QuantityOrUnitLike | None,
    ) -> Unit[DT]:
        if unit is None:
            return Unit("dimensionless")

        if isinstance(unit, Unit):
            return Unit(unit)

        # compatibility with internal pint API
        if isinstance(unit, dict):
            return Unit(Quantity._validate_unit(str(UnitsContainer(unit))))

        # compatibility with internal pint API
        if isinstance(unit, UnitsContainer):
            return Unit(Quantity._validate_unit(str(unit)))

        if isinstance(unit, str):
            return Unit(Quantity._REGISTRY.parse_units(Quantity.correct_unit(unit)))

        if isinstance(unit, PlainUnit):
            return Unit(unit)

        if isinstance(unit, PlainQuantity):
            return Unit(unit.u)

        assert_never(unit)

    @staticmethod
    def _validate_magnitude(val: MT | list[float] | list[int]) -> MT:
        if isinstance(val, int):
            return float(val)
        elif isinstance(val, float):
            # also convert np.float64 to Python float
            return float(val)
        elif isinstance(val, np.ndarray):
            if len(val.shape) != 1:
                raise ValueError(f"Only 1-dimensional Numpy arrays can be used as magnitude, got shape {val.shape}")
            return val
        elif isinstance(val, (pd.Series, pl.Series, pl.Expr)):
            return val
        elif isinstance(val, (list, tuple)):
            arr = cast(MT, np.array(val).astype(np.float64))
            return arr
        # implicit way of checking if the value is a sympy symbol without having to import Sympy
        elif hasattr(val, "is_Atom"):
            return val
        else:
            raise TypeError(f"Input magnitude has incorrect type {type(val)}: {val}")

    @classmethod
    def get_unit(cls, unit_name: AllUnits | str) -> Unit:
        return Unit(cls._REGISTRY.parse_units(unit_name))

    @property
    def subclass(self) -> type[Quantity[DT, MT]]:
        return self._get_dimensional_subclass(self._dimensionality_type, self._magnitude_type)

    def _set_original_magnitude_attributes(self, mt_orig: type[MT], mt_orig_kwargs: dict[str, Any]) -> None:
        self._original_magnitude_type = mt_orig
        self._original_magnitude_kwargs = mt_orig_kwargs

    def __len__(self) -> int:
        # __len__() must return an integer
        # the len() function ensures this at a lower level
        if isinstance(self._magnitude, float | int):
            raise TypeError(
                f"Quantity with scalar magnitude ({self._magnitude}, type {type(self._magnitude)}) has no length"
            )
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

    # region: overload __new__
    @overload
    def __new__(cls, val: list[float] | list[int]) -> Quantity[Dimensionless, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: None) -> Quantity[Dimensionless, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: DimensionlessUnits
    ) -> Quantity[Dimensionless, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: CurrencyUnits) -> Quantity[Currency, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: CurrencyPerEnergyUnits
    ) -> Quantity[CurrencyPerEnergy, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: CurrencyPerVolumeUnits
    ) -> Quantity[CurrencyPerVolume, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: CurrencyPerMassUnits
    ) -> Quantity[CurrencyPerMass, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: CurrencyPerTimeUnits
    ) -> Quantity[CurrencyPerTime, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: LengthUnits) -> Quantity[Length, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: MassUnits) -> Quantity[Mass, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: TimeUnits) -> Quantity[Time, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: TemperatureUnits) -> Quantity[Temperature, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: TemperatureDifferenceUnits
    ) -> Quantity[TemperatureDifference, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: SubstanceUnits) -> Quantity[Substance, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: MolarMassUnits) -> Quantity[MolarMass, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: SubstancePerMassUnits
    ) -> Quantity[SubstancePerMass, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: CurrentUnits) -> Quantity[Current, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: LuminosityUnits) -> Quantity[Luminosity, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: AreaUnits) -> Quantity[Area, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: VolumeUnits) -> Quantity[Volume, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: NormalVolumeUnits) -> Quantity[NormalVolume, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: PressureUnits) -> Quantity[Pressure, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: MassFlowUnits) -> Quantity[MassFlow, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: VolumeFlowUnits) -> Quantity[VolumeFlow, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: NormalVolumeFlowUnits
    ) -> Quantity[NormalVolumeFlow, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: DensityUnits) -> Quantity[Density, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: SpecificVolumeUnits
    ) -> Quantity[SpecificVolume, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: EnergyUnits) -> Quantity[Energy, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: PowerUnits) -> Quantity[Power, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: VelocityUnits) -> Quantity[Velocity, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: DynamicViscosityUnits
    ) -> Quantity[DynamicViscosity, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: KinematicViscosityUnits
    ) -> Quantity[KinematicViscosity, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: EnergyPerMassUnits
    ) -> Quantity[EnergyPerMass, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: SpecificHeatCapacityUnits
    ) -> Quantity[SpecificHeatCapacity, Numpy1DArray]: ...
    @overload
    def __new__(cls, val: list[float] | list[int], unit: Unit[DT]) -> Quantity[DT, Numpy1DArray]: ...
    @overload
    def __new__(
        cls, val: list[float] | list[int], unit: str | UnitsContainer | Unit
    ) -> Quantity[Any, Numpy1DArray]: ...

    @overload
    def __new__(cls, val: MT) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: None) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: DimensionlessUnits) -> Quantity[Dimensionless, MT]: ...
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
    def __new__(cls, val: MT, unit: DynamicViscosityUnits) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: KinematicViscosityUnits) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: EnergyPerMassUnits) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: SpecificHeatCapacityUnits) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: Unit[DT]) -> Quantity[DT, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: str | UnitsContainer | Unit) -> Quantity[Any, MT]: ...
    @overload
    def __new__(cls, val: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __new__(
        cls,
        val: MT | list[float] | Quantity[DT, MT],
        unit: Unit[DT] | Unit | UnitsContainer | str | dict[str, numbers.Number] | None = None,
        _mt_orig: type[MT] | None = None,
        _mt_orig_kwargs: dict[str, Any] | None = None,
        _depth: int = 0,
    ) -> Quantity[Any, Any]: ...

    # endregion

    def __new__(
        cls,
        val: MT | list[float] | list[int] | Quantity[DT, MT],
        unit: Unit[DT] | Unit | UnitsContainer | str | dict[str, numbers.Number] | None = None,
        _mt_orig: type[MT] | None = None,
        _mt_orig_kwargs: dict[str, Any] | None = None,
        _depth: int = 0,
    ) -> Quantity[Any, Any]:
        if isinstance(val, Quantity):
            _input_qty = val
            if unit is not None:
                _input_qty = _input_qty.to(unit)

            val, unit = _input_qty.m, _input_qty.u

        valid_magnitude = cls._validate_magnitude(val)
        valid_unit = cls._validate_unit(unit)

        # NOTE: "original" in this case does not necessarily refer to the type
        # that is actually passed to the constructor
        # for example list[int] is converted to np.ndarray before this (in _validate_magnitude)
        _original_magnitude_type = _mt_orig if _mt_orig is not None else type(valid_magnitude)
        _original_magnitude_kwargs = _mt_orig_kwargs if _mt_orig_kwargs is not None else {}

        # special case for pd.Series, must convert to np.ndarray to avoid
        # TypeError: PlainQuantity cannot wrap upcast type
        # NOTE: pl.Series and pl.Expr don't require this type of workaround
        if isinstance(valid_magnitude, pd.Series):
            # preserve pd.Series metadata
            # (not dtype, this would cause issues with float/int)
            _original_magnitude_kwargs |= {
                "index": valid_magnitude.index,
                "name": valid_magnitude.name,
            }

            valid_magnitude = cast(MT, cls._cast_array_float(valid_magnitude.to_numpy()))

        is_complete_subclass = True

        if cls._is_incomplete_dimensionality(cls._dimensionality_type):
            is_complete_subclass = False
        else:
            # compare dimensionalities with tolerance for float precision
            if not _units_containers_equal(cls._dimensionality_type.dimensions, valid_unit.dimensionality):
                is_complete_subclass = False

        if not is_complete_subclass:
            # TODO: how to validate that the subclass has the same dimensionality as the input unit?
            # cannot raise error here since this breaks the pint.PlainQuantity methods
            # that use return self.__class__(...)

            if _depth > cls._max_recursion_depth:
                raise RecursionError(
                    f"RecursionError ({_depth=}) when constructing Quantity class with {val=}, {unit=}"
                )

            # special case for temperature difference
            if valid_unit._units in cls.TEMPERATURE_DIFFERENCE_UCS:
                subcls = cls._get_dimensional_subclass(TemperatureDifference, _original_magnitude_type)
            else:
                dim = Dimensionality.get_dimensionality(valid_unit.dimensionality)
                subcls = cls._get_dimensional_subclass(dim, _original_magnitude_type)

            return subcls(
                valid_magnitude,
                valid_unit,
                _mt_orig=_original_magnitude_type,
                _mt_orig_kwargs=_original_magnitude_kwargs,
                _depth=_depth + 1,
            )

        _qty = super().__new__(cls, valid_magnitude, units=valid_unit)
        qty = cast("Quantity[DT, MT]", _qty)

        # ensure that pint did not change the dtype of numpy arrays
        if isinstance(qty._magnitude, np.ndarray):
            qty._magnitude = cls._cast_array_float(qty._magnitude)

        # propagate the original magnitude type and constructor kwargs
        qty._set_original_magnitude_attributes(_original_magnitude_type, _original_magnitude_kwargs)
        return qty

    @property
    def m(self) -> MT:
        if self._original_magnitude_type is pd.Series or get_origin(self._original_magnitude_type) is pd.Series:
            assert isinstance(self._magnitude, np.ndarray)
            return cast(MT, pd.Series(self._magnitude, **self._original_magnitude_kwargs))

        return self._magnitude

    @property
    def units(self) -> Unit[DT]:
        return Unit(super().units)

    @property
    def u(self) -> Unit[DT]:
        return self.units

    @property
    def _is_temperature_difference(self) -> bool:
        return self._dimensionality_type is TemperatureDifference

    def _is_temperature_difference_unit(self, unit: Unit) -> bool:
        return unit._units in self.TEMPERATURE_DIFFERENCE_UCS

    def _check_temperature_compatibility(self, unit: Unit[DT]) -> None:
        if self._is_temperature_difference and unit._units not in self.TEMPERATURE_DIFFERENCE_UCS:
            current_name = self._dimensionality_type.__name__
            new_name = Quantity(1, unit)._dimensionality_type.__name__

            raise DimensionalityTypeError(
                f"Cannot convert {self.units} (dimensionality {current_name}) to {unit} (dimensionality {new_name})"
            )

    def _call_subclass(self, *args: Any) -> Quantity[DT, MT]:  # noqa: ANN401
        # handle the edge case where a 1-element pd.Series is
        # multiplied or divided by an N-element pd.Series

        if len(args) == 1:
            qty = args[0]
            if not isinstance(qty, Quantity):
                raise TypeError(f"Invalid single input: {args}")

            m, u = qty.m, qty.u
        elif len(args) == 2:
            m, u = args
        else:
            raise ValueError(f"Invalid input: {args}")

        index = self._original_magnitude_kwargs.get("index")
        if index is not None:
            magnitude_length = len(m) if isinstance(m, Sized) else None

            # strip the index in case it's not compatible
            if magnitude_length is not None and magnitude_length != len(index):
                del self._original_magnitude_kwargs["index"]

        return self.subclass(
            cast(MT, m),
            cast(Unit[DT], u),
            _mt_orig=self._original_magnitude_type,
            _mt_orig_kwargs=self._original_magnitude_kwargs,
        )

    def to_reduced_units(self) -> Quantity[DT, MT]:
        ret = super().to_reduced_units()
        return self._call_subclass(ret)

    def to_root_units(self) -> Quantity[DT, MT]:
        ret = super().to_root_units()
        return self._call_subclass(ret)

    def to_base_units(self) -> Quantity[DT, MT]:
        self._check_temperature_compatibility(Unit("kelvin"))

        ret = super().to_base_units()
        return self._call_subclass(ret)

    def _dimensionalities_match(self, unit: Unit[DT_]) -> bool:
        src_dim = cast(dict[str, float], dict(self.dimensionality))
        dst_dim = cast(dict[str, float], dict(unit.dimensionality))

        if set(src_dim.keys()) != set(dst_dim.keys()):
            return False

        return all(
            abs(float(src_dim.get(key, 0)) - float(dst_dim.get(key, 0))) < 1e-10
            for key in set(src_dim.keys()) | set(dst_dim.keys())
        )

    def _to_unit(
        self, unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | QuantityOrUnitLike
    ) -> Unit[DT]:
        return self._validate_unit(unit)

    def to(
        self, unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | QuantityOrUnitLike
    ) -> Quantity[DT, MT]:
        valid_unit = self._to_unit(unit)
        self._check_temperature_compatibility(valid_unit)

        m: MT
        try:
            m = self._convert_magnitude_not_inplace(valid_unit)
        except DimensionalityError as e:
            # if direct conversion fails due to complex fractional units,
            # try converting to base units first, then to the target unit
            if self._dimensionalities_match(valid_unit):
                base_quantity = self.to_base_units()
                m = base_quantity._convert_magnitude_not_inplace(valid_unit)
            else:
                raise e

        if self._is_temperature_difference_unit(valid_unit):
            ret = Quantity(
                m,
                valid_unit,
                _mt_orig=self._original_magnitude_type,
                _mt_orig_kwargs=self._original_magnitude_kwargs,
            )
            return ret

        converted = self._call_subclass(m, unit)

        return converted

    def ito(
        self, unit: AllUnits | Unit[DT] | UnitsContainer | str | dict[str, numbers.Number] | QuantityOrUnitLike
    ) -> None:
        # NOTE: this method cannot convert the dimensionality type
        valid_unit = self._to_unit(unit)
        self._check_temperature_compatibility(valid_unit)

        if self._is_temperature_difference_unit(valid_unit) and not self._is_temperature_difference:
            raise ValueError(
                f"Cannot convert {self} ({type(self)}) to {valid_unit} inplace, use qty_converted = qty.to(...) instead"
            )

        # it's not safe to convert units as int, the
        # user will have to convert back to int if necessary
        # better to use ":.0f" formatting or round() anyway

        # avoid numpy.core._exceptions.UFuncTypeError (not on all platforms?)
        # convert integer arrays to float(64) (creating a copy)
        if isinstance(self._magnitude, np.ndarray) and issubclass(self._magnitude.dtype.type, numbers.Integral):
            self._magnitude = self._magnitude.astype(np.float64)

        try:
            super().ito(valid_unit)
        except DimensionalityError as e:
            if self._dimensionalities_match(valid_unit):
                base_quantity = self.to_base_units()
                converted_magnitude = base_quantity._convert_magnitude_not_inplace(valid_unit)
                self._magnitude = cast(MT, converted_magnitude)
                self._units = valid_unit._units
            else:
                raise e

    def check(
        self,
        dimension: Quantity[Any, Any]
        | UnitsContainer
        | Unit
        | str
        | Dimensionality
        | type[Dimensionality]
        | QuantityOrUnitLike,
    ) -> bool:
        if isinstance(dimension, Quantity):
            return self._dimensionality_type == dimension._dimensionality_type

        if isinstance(dimension, Unit):
            # it's not possible to know if an instance of Unit is Temperature or TemperatureDifference
            # until it is used to construct a Quantity
            unit_qty = Quantity(1.0, dimension)

            if isinstance_types(unit_qty, Quantity[TemperatureDifference]):
                unit_qty = Quantity[TemperatureDifference, float](1.0, dimension)

            return self.check(unit_qty)

        if hasattr(dimension, "dimensions"):
            _dims = getattr(dimension, "dimensions", None)
            if _dims is None:
                raise TypeError(f"Attribute 'dimensions' is missing or None: {dimension}")
            return super().check(cast(UnitsContainer, _dims))

        if isinstance(dimension, (Dimensionality, type, PlainQuantity)):
            raise TypeError(f"Invalid type for dimension: {dimension} ({type(dimension)})")

        return super().check(dimension)

    def __format__(self, format_type: str) -> str:
        """
        Overrides the ``__format__`` method for Quantity:
        Ensure that the default formatting spec is used for fixed
        precision formatting (":.2f", ":.2g") when no explicit
        format is specified.

        Parameters
        ----------
        format_type : str
            Value format specifier

        Returns
        -------
        str
            Formatted output
        """

        if not format_type.endswith(Quantity.FORMATTING_SPECS):
            format_type = f"{format_type}{self._REGISTRY.formatter.default_format}"

        return super().__format__(format_type)

    @staticmethod
    def correct_unit(unit: str) -> str:
        """
        Corrects the unit name to make it compatible with ``pint``.

        * Adds ``**`` between the unit and the exponent if it's missing,
          for example ``kg/m3 → kg/m**3``.
        * Parses "Nm³" (and variations of this) as "normal * m³"
          (use explicit "nanometer³" to get this unit)
        * Converts % and ‰ to percent and permille
        * Changes the ``Δ`` character to ``delta_``, for example ``Δ°C`` to ``delta_°C``
        * Interprets ``-`` as ``dimensionless``
        * Converts the single-character symbols ``℃`` and ``℉`` to ``degC`` and ``degF``
        * Converts the compound units ``Δ°C`` and ``Δ°F``
          to ``delta_degC`` and ``delta_degF``

        Parameters
        ----------
        unit : str
            A (potentially inconsistently specified) unit name

        Returns
        -------
        str
            The corrected unit name
        """

        unit = str(unit).strip()

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
            "°C": "degC",
            "°F": "degF",
            "℃": "degC",
            "℉": "degF",
            "%": "percent",
            "‰": "permille",
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
        """
        Compatibility with ``sympy``.
        Converts the unit dimensions to symbols and multiplies with the magnitude.
        Need to use base units, compound units will not cancel out otherwise.
        """
        _ensure_sympy()

        if self.dimensionless:
            return sp.sympify(self.to_base_units().m)

        base_qty = self.to_base_units()

        unit_parts = []
        symbols = []

        for unit_name, power in base_qty.u._units.items():
            unit_symbol = self._REGISTRY.get_symbol(unit_name)
            unit_parts.append(f"{unit_symbol}**{power}")
            symbols.append(unit_symbol)

        unit_repr = " * ".join(unit_parts)

        if not unit_repr.strip():
            unit_repr = "1"

        # use \text{symbol} to make sure that the unit symbols
        # do not clash with commonly used symbols like "m" or "s"
        expr = sp.sympify(f"{base_qty.m} * {unit_repr}").subs({sp.Symbol(n): self.get_unit_symbol(n) for n in symbols})

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
        """
        Converts a Sympy expression with unit symbols
        into a Quantity. This only works in case expression *only* contains base SI unit symbols.
        """

        # this needs to be populated here to account for custom dimensions
        cls._populate_dimension_symbol_map()

        expr = expr.simplify()
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
                unit_i *= cls._dimension_symbol_map[symbol] ** power

            unit *= unit_i

        ret = cls(cast(MT, magnitude), cast(Unit, unit))
        return cast("Quantity[DT, float]", ret.to_base_units())

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

            val: int | float | list

            if isinstance(mag, int | float):
                val = mag
                mtype = "int" if isinstance(mag, int) else "float"
            elif isinstance(mag, np.ndarray):
                val = mag.tolist()
                mtype = f"np.ndarray:{mag.dtype.str}:{mag.shape}"
            elif isinstance(mag, pl.Series):
                val = mag.to_list()
                mtype = f"pl.Series:{mag.dtype}"
            elif isinstance(mag, list):
                val = [float(x) for x in mag]
                mtype = "list"
            else:
                raise ValueError(f"Unknown magnitude type {type(mag)}: {mag}")

            return {
                "unit": str(qty.u),
                "value": val,
                "magnitude_type": mtype,
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
            val = qty["value"]
            magnitude_type = qty["magnitude_type"]

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
                raise ExpectedDimensionalityError(f"Unknown magnitude_type {magnitude_type!r}")
            unit = qty.get("unit")
            ret = cls(cast(MT, magnitude), unit=unit)
        else:
            ret = qty if isinstance(qty, Quantity) else cls(cast(Any, qty))

        if isinstance(ret, cls):
            return ret

        raise ExpectedDimensionalityError(
            f"Value {ret} ({type(ret).__name__}) does not match expected dimensionality {cls.__name__}"
        )

    def check_compatibility(self, other: Quantity[Any, Any] | float | int) -> None:
        if not isinstance(other, Quantity):
            if not self.dimensionless:
                raise DimensionalityTypeError(
                    f"Value {other} ({type(other)}) is not compatible with dimensional quantity {self} ({type(self)})"
                )

            return

        dim = self._dimensionality_type
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
            if self._dimensionality_type.dimensions == other._dimensionality_type.dimensions:
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
        other: Quantity[Any, Any] | float | int,
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

    @overload
    def __eq__(self: Quantity[Any, float], other: Quantity[Any, float] | float | int) -> bool: ...

    @overload
    def __eq__(
        self: Quantity[Any, pl.Expr], other: Quantity[Any, pl.Expr] | Quantity[Any, float] | float | int
    ) -> pl.Expr: ...

    @overload
    def __eq__(
        self: Quantity[Any, Numpy1DArray],
        other: Quantity[Any, Numpy1DArray] | Quantity[Any, float] | Numpy1DArray | float | int,
    ) -> Numpy1DBoolArray: ...

    @overload
    def __eq__(self: Quantity[Any, Any], other: Quantity[Any, Numpy1DArray]) -> Numpy1DBoolArray: ...

    @overload
    def __eq__(self: Quantity[Any, pd.Series], other: Quantity[Any, pd.Series] | float | int) -> pd.Series[bool]: ...

    @overload
    def __eq__(self: Quantity[Any, pl.Series], other: Quantity[Any, pl.Series] | float | int) -> pl.Series: ...

    @overload
    def __eq__(
        self: Quantity[Any, Any], other: object
    ) -> bool | Numpy1DBoolArray | pd.Series | pl.Series | pl.Expr: ...

    def __eq__(self, other: object) -> bool | Numpy1DBoolArray | pd.Series | pl.Series | pl.Expr:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not isinstance(other, (Quantity, float, int)):
            return bool(super().__eq__(other))

        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            raise DimensionalityComparisonError(f"Cannot compare {self} with {other}") from e

        if isinstance(other, (float, int)):
            other = Quantity(other, "dimensionless")

        m = self.m
        other_m = cast(float | Numpy1DArray | pd.Series | pl.Series | pl.Expr, other.to(self.u).m)

        if isinstance(m, (float, int, np.ndarray)) and isinstance(other_m, (float, int, np.ndarray)):
            ret = np.isclose(m, other_m, self.rtol, self.atol)

            if isinstance(ret, np.bool):
                return bool(ret)
            else:
                return ret

        ret = m == other_m

        return ret

    # region: overload __floordiv__, __pow__, __add__, __sub__, __gt__, __ge__, __lt__, __le__

    @overload
    def __floordiv__(self: Quantity[Dimensionless, MT], other: float | int) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __floordiv__(self, other: Quantity[DT, Any]) -> Quantity[Dimensionless, Any]: ...
    def __floordiv__(self, other: Quantity[Any, Any] | float | int) -> Quantity[Any, Any]:
        return super().__floordiv__(other)

    @overload
    def __pow__(self, other: Literal[1]) -> Quantity[DT, MT]: ...
    @overload
    def __pow__(self: Quantity[Length, MT], other: Literal[2]) -> Quantity[Area, MT]: ...
    @overload
    def __pow__(self: Quantity[Length, MT], other: Literal[3]) -> Quantity[Volume, MT]: ...
    @overload
    def __pow__(self: Quantity[Dimensionless, Any], other: float | int) -> Quantity[Dimensionless, Any]: ...
    @overload
    def __pow__(self, other: Quantity[Dimensionless, Any]) -> Quantity[Any, Any]: ...
    @overload
    def __pow__(self, other: float | int) -> Quantity[Any, Any]: ...
    def __pow__(self, other: Quantity[Dimensionless, Any] | float | int) -> Quantity[Any, Any]:
        return cast("Quantity[Any, Any]", super().__pow__(other))

    @overload
    def __add__(self: Quantity[Dimensionless, Any], other: float | int) -> Quantity[Dimensionless, Any]: ...
    @overload
    def __add__(
        self: Quantity[Temperature, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[Temperature, Any]: ...
    @overload
    def __add__(self, other: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __add__(self, other: Quantity[DT, float]) -> Quantity[DT, MT]: ...
    @overload
    def __add__(self: Quantity[DT, float], other: Quantity[DT, MT_]) -> Quantity[DT, MT_]: ...
    @overload
    def __add__(self, other: Quantity[DT, Any]) -> Quantity[DT, Any]: ...
    @overload
    def __add__(self: Quantity[Any, Any], other: Quantity[Any, Any]) -> Quantity[Any, Any]: ...
    def __add__(self, other: Quantity[Any, Any] | float | int) -> Quantity[Any, Any]:
        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            if not isinstance(other, Quantity):
                raise e

            self_is_temp_or_diff_temp = isinstance_types(self, Quantity[Temperature, Any]) or isinstance_types(
                self, Quantity[TemperatureDifference, Any]
            )

            other_is_temp_or_diff_temp = isinstance_types(other, Quantity[Temperature, Any]) or isinstance_types(
                other, Quantity[TemperatureDifference, Any]
            )

            if self_is_temp_or_diff_temp and other_is_temp_or_diff_temp:
                if self._dimensionality_type is TemperatureDifference:
                    raise e

                return self._temperature_difference_add_sub(other, "add")

            raise e

        ret = super().__add__(other)

        return self._call_subclass(ret)

    @overload
    def __sub__(self: Quantity[Dimensionless, MT], other: float | int) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __sub__(
        self: Quantity[Temperature, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[Temperature, Any]: ...
    @overload
    def __sub__(
        self: Quantity[Temperature, MT], other: Quantity[Temperature, Any]
    ) -> Quantity[TemperatureDifference, Any]: ...
    @overload
    def __sub__(self, other: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __sub__(self, other: Quantity[DT, float]) -> Quantity[DT, MT]: ...
    @overload
    def __sub__(self: Quantity[DT, float], other: Quantity[DT, MT_]) -> Quantity[DT, MT_]: ...
    @overload
    def __sub__(self, other: Quantity[DT, Any]) -> Quantity[DT, Any]: ...
    @overload
    def __sub__(self: Quantity[Any, Any], other: Quantity[Any, Any]) -> Quantity[Any, Any]: ...
    def __sub__(self, other: Quantity[Any, Any] | float | int) -> Quantity[Any, Any]:
        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            if not isinstance(other, Quantity):
                raise e

            self_is_temp_or_diff_temp = isinstance_types(self, Quantity[Temperature, Any]) or isinstance_types(
                self, Quantity[TemperatureDifference, Any]
            )

            other_is_temp_or_diff_temp = isinstance_types(other, Quantity[Temperature, Any]) or isinstance_types(
                other, Quantity[TemperatureDifference, Any]
            )

            if self_is_temp_or_diff_temp and other_is_temp_or_diff_temp:
                if self._dimensionality_type is TemperatureDifference:
                    raise e

                return self._temperature_difference_add_sub(other, "sub")

            raise e

        ret = super().__sub__(other)

        if (
            isinstance(other, Quantity)
            and self._dimensionality_type is Temperature
            and other._dimensionality_type is Temperature
        ):
            _mt = type(ret.m)
            subcls = self._get_dimensional_subclass(TemperatureDifference, _mt)
            return subcls(ret.m, ret.u)

        return self._call_subclass(ret)

    @overload
    def __gt__(self: Quantity[Dimensionless, float], other: float | int) -> bool: ...
    @overload
    def __gt__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    def __gt__(self, other: Quantity[DT, Any] | float | int) -> bool:
        try:
            return super().__gt__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    @overload
    def __ge__(self: Quantity[Dimensionless, float], other: float | int) -> bool: ...
    @overload
    def __ge__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    def __ge__(self, other: Quantity[DT, Any] | float | int) -> bool:
        try:
            return super().__ge__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    @overload
    def __lt__(self: Quantity[Dimensionless, float], other: float | int) -> bool: ...
    @overload
    def __lt__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    def __lt__(self, other: Quantity[DT, Any] | float | int) -> bool:
        try:
            return super().__lt__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    @overload
    def __le__(self: Quantity[Dimensionless, float], other: float | int) -> bool: ...
    @overload
    def __le__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    def __le__(self, other: Quantity[DT, Any] | float | int) -> bool:
        try:
            return super().__le__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    # endregion

    # region: overload __mul__

    # region: autogenerated __mul__

    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[EnergyPerMass, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[EnergyPerMass, MT], other: Quantity[Mass, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[HeatingValue, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[HeatingValue, MT], other: Quantity[Mass, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[LowerHeatingValue, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[LowerHeatingValue, MT], other: Quantity[Mass, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[HigherHeatingValue, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[HigherHeatingValue, MT], other: Quantity[Mass, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[NormalTemperature, MT]: ...
    @overload
    def __mul__(self: Quantity[Normal, MT], other: Quantity[Volume, MT_]) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(self: Quantity[Normal, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[Normal, MT], other: Quantity[MassPerNormalVolume, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Length, MT_]) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Area, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Velocity, MT_]) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[DynamicViscosity, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[KinematicViscosity, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[Frequency, MT_]) -> Quantity[Velocity, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[PowerPerLength, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[PowerPerArea, MT_]) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(self: Quantity[Length, MT], other: Quantity[PowerPerVolume, MT_]) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[ThermalConductivity, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[HeatTransferCoefficient, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[SpecificVolume, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[Frequency, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[SubstancePerMass, MT_]) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[CurrencyPerMass, MT_]) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(self: Quantity[Mass, MT], other: Quantity[NormalVolumePerMass, MT_]) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[Pressure, MT_]) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[MassFlow, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[NormalVolumeFlow, MT_]) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[Power, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[Velocity, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[KinematicViscosity, MT_]) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[Frequency, MT_]) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[CurrencyPerTime, MT_]) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[PowerPerVolume, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(self: Quantity[Time, MT], other: Quantity[EnergyPerMass, MT_]) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[NormalTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[PowerPerTemperature, MT_],
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[ThermalConductivity, MT_],
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[HeatTransferCoefficient, MT_],
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[MolarSpecificEntropy, MT_],
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[SpecificHeatCapacity, MT_],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[Substance, MT], other: Quantity[MolarMass, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Substance, MT], other: Quantity[MolarSpecificEnthalpy, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[Length, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[Velocity, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[Frequency, MT_]) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[PowerPerArea, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Area, MT], other: Quantity[PowerPerVolume, MT_]) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[HeatTransferCoefficient, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[Normal, MT_]) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[Pressure, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[Density, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[Frequency, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[MolarDensity, MT_]) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[CurrencyPerVolume, MT_]) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(self: Quantity[Volume, MT], other: Quantity[PowerPerVolume, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolume, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[NormalVolume, MT], other: Quantity[MassPerNormalVolume, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[Time, MT_]) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[Volume, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[SpecificVolume, MT_]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[Velocity, MT_]) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[Frequency, MT_]) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(self: Quantity[Pressure, MT], other: Quantity[MassPerEnergy, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[Time, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[SpecificVolume, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[KinematicViscosity, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[MassFlow, MT], other: Quantity[EnergyPerMass, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[SpecificHeatCapacity, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Normal, MT_]) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Time, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Pressure, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[Density, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[VolumeFlow, MT], other: Quantity[DynamicViscosity, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(self: Quantity[NormalVolumeFlow, MT], other: Quantity[Time, MT_]) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[Volume, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[SpecificVolume, MT_]) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[SubstancePerMass, MT_]) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[NormalVolumePerMass, MT_]) -> Quantity[Normal, MT]: ...
    @overload
    def __mul__(self: Quantity[Density, MT], other: Quantity[EnergyPerMass, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[SpecificVolume, MT], other: Quantity[Mass, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(self: Quantity[SpecificVolume, MT], other: Quantity[Pressure, MT_]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[SpecificVolume, MT], other: Quantity[MassFlow, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[SpecificVolume, MT], other: Quantity[Density, MT_]) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[Energy, MT], other: Quantity[Frequency, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Energy, MT], other: Quantity[CurrencyPerEnergy, MT_]) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(self: Quantity[Energy, MT], other: Quantity[MassPerEnergy, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[Power, MT], other: Quantity[Time, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Power, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(self: Quantity[Power, MT], other: Quantity[MassPerEnergy, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Length, MT_]) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Time, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Area, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Pressure, MT_]) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(self: Quantity[Velocity, MT], other: Quantity[Velocity, MT_]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[DynamicViscosity, MT], other: Quantity[Length, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[DynamicViscosity, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[DynamicViscosity, MT], other: Quantity[Frequency, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[SpecificHeatCapacity, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(self: Quantity[KinematicViscosity, MT], other: Quantity[Length, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[KinematicViscosity, MT], other: Quantity[Time, MT_]) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(self: Quantity[KinematicViscosity, MT], other: Quantity[MassFlow, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Density, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[KinematicViscosity, MT], other: Quantity[MassPerEnergy, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __mul__(self: Quantity[Frequency, MT], other: Quantity[Length, MT_]) -> Quantity[Velocity, MT]: ...
    @overload
    def __mul__(self: Quantity[Frequency, MT], other: Quantity[Mass, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Frequency, MT], other: Quantity[Time, MT_]) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(self: Quantity[Frequency, MT], other: Quantity[Area, MT_]) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[Frequency, MT], other: Quantity[Volume, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[NormalVolume, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[Frequency, MT], other: Quantity[Pressure, MT_]) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __mul__(self: Quantity[Frequency, MT], other: Quantity[Energy, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[Frequency, MT], other: Quantity[DynamicViscosity, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[Frequency, MT], other: Quantity[Currency, MT_]) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(self: Quantity[MolarMass, MT], other: Quantity[Substance, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(self: Quantity[MolarMass, MT], other: Quantity[MolarDensity, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[SpecificHeatCapacity, MT_]
    ) -> Quantity[MolarSpecificEntropy, MT]: ...
    @overload
    def __mul__(self: Quantity[SubstancePerMass, MT], other: Quantity[Mass, MT_]) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(self: Quantity[SubstancePerMass, MT], other: Quantity[Density, MT_]) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[MolarSpecificEntropy, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT],
        other: Quantity[MolarSpecificEnthalpy, MT_],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[MolarDensity, MT], other: Quantity[Volume, MT_]) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[MolarDensity, MT], other: Quantity[MolarMass, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(self: Quantity[Currency, MT], other: Quantity[Frequency, MT_]) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(self: Quantity[CurrencyPerEnergy, MT], other: Quantity[Energy, MT_]) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[Power, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[CurrencyPerMass, MT], other: Quantity[Mass, MT_]) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[Density, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __mul__(self: Quantity[CurrencyPerVolume, MT], other: Quantity[Volume, MT_]) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(self: Quantity[CurrencyPerTime, MT], other: Quantity[Time, MT_]) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(self: Quantity[PowerPerLength, MT], other: Quantity[Length, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerLength, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[PowerPerArea, MT], other: Quantity[Length, MT_]) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(self: Quantity[PowerPerArea, MT], other: Quantity[Area, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[PowerPerVolume, MT], other: Quantity[Length, MT_]) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(self: Quantity[PowerPerVolume, MT], other: Quantity[Time, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(self: Quantity[PowerPerVolume, MT], other: Quantity[Area, MT_]) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(self: Quantity[PowerPerVolume, MT], other: Quantity[Volume, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[Length, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[ThermalConductivity, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[Length, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[Area, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(self: Quantity[MassPerNormalVolume, MT], other: Quantity[Normal, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(self: Quantity[MassPerNormalVolume, MT], other: Quantity[NormalVolume, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT], other: Quantity[NormalVolumeFlow, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT],
        other: Quantity[NormalVolumePerMass, MT_],
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(self: Quantity[MassPerEnergy, MT], other: Quantity[Pressure, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(self: Quantity[MassPerEnergy, MT], other: Quantity[Energy, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(self: Quantity[MassPerEnergy, MT], other: Quantity[Power, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[MassPerEnergy, MT], other: Quantity[KinematicViscosity, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[PowerPerLength, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEntropy, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEntropy, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __mul__(self: Quantity[NormalVolumePerMass, MT], other: Quantity[Mass, MT_]) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(self: Quantity[NormalVolumePerMass, MT], other: Quantity[Density, MT_]) -> Quantity[Normal, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT],
        other: Quantity[MassPerNormalVolume, MT_],
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(self: Quantity[EnergyPerMass, MT], other: Quantity[Mass, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(self: Quantity[EnergyPerMass, MT], other: Quantity[Time, MT_]) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(self: Quantity[EnergyPerMass, MT], other: Quantity[MassFlow, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(self: Quantity[EnergyPerMass, MT], other: Quantity[Density, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[Substance, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT],
        other: Quantity[SubstancePerMass, MT_],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[MolarSpecificEntropy, MT]: ...

    # endregion

    @overload
    def __mul__(self: Quantity[Dimensionless, MT], other: Quantity[DT_, MT_]) -> Quantity[DT_, MT]: ...
    @overload
    def __mul__(self, other: Quantity[Dimensionless, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self, other: float | int) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self, other: Quantity[Any, MT_]) -> Quantity[Any, MT_]: ...
    @overload
    def __mul__(self, other: Quantity[Any, MT]) -> Quantity[Any, MT]: ...

    # endregion

    def __mul__(self, other: Quantity[Any, Any] | float | int) -> Quantity[Any, Any]:
        ret = super().__mul__(other)
        if self.dimensionless and isinstance(other, Quantity):
            return other.subclass(ret)

        return self._call_subclass(ret)

    def __rmul__(self, other: Quantity[Any, Any] | float | int) -> Quantity[Any, Any]:
        ret = super().__rmul__(other)

        return self._call_subclass(ret)

    # region: overloads __truediv__

    @overload
    def __truediv__(self, other: Quantity[DT, MT_]) -> Quantity[Dimensionless, MT]: ...

    # region: autogenerated __truediv__

    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Mass, MT_]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Dimensionless, MT], other: Quantity[Time, MT_]) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[Density, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(self: Quantity[Dimensionless, MT], other: Quantity[Frequency, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(self: Quantity[Normal, MT], other: Quantity[Density, MT_]) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Normal, MT], other: Quantity[NormalVolumePerMass, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(self: Quantity[Length, MT], other: Quantity[Time, MT_]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Length, MT], other: Quantity[Velocity, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Time, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Substance, MT_]) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Volume, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[NormalVolume, MT_]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[MassFlow, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Density, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[Energy, MT_]) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[MolarMass, MT_]) -> Quantity[Substance, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Mass, MT], other: Quantity[MassPerEnergy, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Time, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Time, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Substance, MT], other: Quantity[Mass, MT_]) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Substance, MT], other: Quantity[Volume, MT_]) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Substance, MT], other: Quantity[SubstancePerMass, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Substance, MT], other: Quantity[MolarDensity, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Area, MT], other: Quantity[Length, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[Area, MT], other: Quantity[Time, MT_]) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Area, MT], other: Quantity[KinematicViscosity, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Length, MT_]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Mass, MT_]) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Time, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[Area, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Volume, MT], other: Quantity[SpecificVolume, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(self: Quantity[NormalVolume, MT], other: Quantity[Normal, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[NormalVolume, MT], other: Quantity[Time, MT_]) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[NormalVolume, MT], other: Quantity[Volume, MT_]) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(self: Quantity[NormalVolume, MT], other: Quantity[NormalVolumeFlow, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Pressure, MT], other: Quantity[Time, MT_]) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Pressure, MT], other: Quantity[Density, MT_]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(self: Quantity[Pressure, MT], other: Quantity[PowerPerVolume, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Pressure, MT], other: Quantity[EnergyPerMass, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[Length, MT_]) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[Mass, MT_]) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[NormalVolumeFlow, MT_]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[Density, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[Power, MT_]) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[DynamicViscosity, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[Frequency, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[MassFlow, MT], other: Quantity[MassPerEnergy, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Length, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[Area, MT_]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[Volume, MT_]) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[MassFlow, MT_]) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[SpecificVolume, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[Velocity, MT_]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[VolumeFlow, MT], other: Quantity[Frequency, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[NormalVolumeFlow, MT], other: Quantity[Normal, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[NormalVolume, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[NormalVolumeFlow, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Density, MT], other: Quantity[Normal, MT_]) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Density, MT], other: Quantity[Pressure, MT_]) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(self: Quantity[Density, MT], other: Quantity[MolarMass, MT_]) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Density, MT], other: Quantity[MolarDensity, MT_]) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Density, MT], other: Quantity[MassPerNormalVolume, MT_]) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(self: Quantity[Density, MT], other: Quantity[MassPerEnergy, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Time, MT_]) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Substance, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Volume, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Pressure, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[MassFlow, MT_]) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[Power, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[DynamicViscosity, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[KinematicViscosity, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Energy, MT], other: Quantity[EnergyPerMass, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Length, MT_]) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Area, MT_]) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Volume, MT_]) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Pressure, MT_]) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[MassFlow, MT_]) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[VolumeFlow, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Energy, MT_]) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[Frequency, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[PowerPerLength, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[PowerPerArea, MT_]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[PowerPerVolume, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[PowerPerTemperature, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(self: Quantity[Power, MT], other: Quantity[EnergyPerMass, MT_]) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[Velocity, MT], other: Quantity[Length, MT_]) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(self: Quantity[Velocity, MT], other: Quantity[Frequency, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[DynamicViscosity, MT], other: Quantity[Time, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[DynamicViscosity, MT], other: Quantity[Pressure, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Density, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[PowerPerLength, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __truediv__(self: Quantity[KinematicViscosity, MT], other: Quantity[Length, MT_]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Time, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[KinematicViscosity, MT], other: Quantity[Area, MT_]) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(self: Quantity[KinematicViscosity, MT], other: Quantity[Velocity, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[KinematicViscosity, MT], other: Quantity[Frequency, MT_]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarMass, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarMass, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarDensity, MT], other: Quantity[Density, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarDensity, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(self: Quantity[Currency, MT], other: Quantity[Mass, MT_]) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Currency, MT], other: Quantity[Time, MT_]) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __truediv__(self: Quantity[Currency, MT], other: Quantity[Volume, MT_]) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Currency, MT], other: Quantity[Energy, MT_]) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(self: Quantity[Currency, MT], other: Quantity[CurrencyPerEnergy, MT_]) -> Quantity[Energy, MT]: ...
    @overload
    def __truediv__(self: Quantity[Currency, MT], other: Quantity[CurrencyPerMass, MT_]) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(self: Quantity[Currency, MT], other: Quantity[CurrencyPerVolume, MT_]) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(self: Quantity[Currency, MT], other: Quantity[CurrencyPerTime, MT_]) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[Density, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[Power, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(self: Quantity[CurrencyPerTime, MT], other: Quantity[Frequency, MT_]) -> Quantity[Currency, MT]: ...
    @overload
    def __truediv__(self: Quantity[CurrencyPerTime, MT], other: Quantity[Currency, MT_]) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerLength, MT], other: Quantity[Length, MT_]) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerLength, MT], other: Quantity[Area, MT_]) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerLength, MT], other: Quantity[PowerPerArea, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerLength, MT], other: Quantity[PowerPerVolume, MT_]) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[ThermalConductivity, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerArea, MT], other: Quantity[Length, MT_]) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerArea, MT], other: Quantity[Pressure, MT_]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerArea, MT], other: Quantity[Velocity, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerArea, MT], other: Quantity[PowerPerVolume, MT_]) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[HeatTransferCoefficient, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerVolume, MT], other: Quantity[Pressure, MT_]) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(self: Quantity[PowerPerVolume, MT], other: Quantity[Frequency, MT_]) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT], other: Quantity[Length, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT], other: Quantity[Area, MT_]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[ThermalConductivity, MT_],
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[HeatTransferCoefficient, MT_],
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[SpecificHeatCapacity, MT_],
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[Length, MT_]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT],
        other: Quantity[HeatTransferCoefficient, MT_],
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT],
        other: Quantity[SpecificHeatCapacity, MT_],
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEntropy, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEntropy, MT],
        other: Quantity[SpecificHeatCapacity, MT_],
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalTemperature, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalTemperature, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(self: Quantity[EnergyPerMass, MT], other: Quantity[Velocity, MT_]) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[SpecificHeatCapacity, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[MolarSpecificEntropy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT],
        other: Quantity[MolarSpecificEntropy, MT_],
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[MolarSpecificEntropy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SpecificHeatCapacity, MT],
        other: Quantity[MolarSpecificEntropy, MT_],
    ) -> Quantity[SubstancePerMass, MT]: ...

    # endregion

    @overload
    def __truediv__(self, other: Quantity[Dimensionless, MT_]) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self, other: float | int) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[DT_, MT_]) -> Quantity[Any, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[Any, MT_]) -> Quantity[Any, MT]: ...
    @overload
    def __truediv__(self: Quantity[Any, MT], other: Quantity[Any, MT_]) -> Quantity[Any, MT]: ...

    def __truediv__(self, other: Quantity[Any, Any] | float | int) -> Quantity[Any, MT]:
        ret = super().__truediv__(other)
        return self._call_subclass(ret)

    # region: overload  __rtruediv__

    # region: autogenerated __rtruediv__

    @overload
    def __rtruediv__(self: Quantity[Time, MT], other: float | int) -> Quantity[Frequency, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[Density, MT], other: float | int) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[SpecificVolume, MT], other: float | int) -> Quantity[Density, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[Frequency, MT], other: float | int) -> Quantity[Time, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[MolarMass, MT], other: float | int) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[SubstancePerMass, MT], other: float | int) -> Quantity[MolarMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[MassPerNormalVolume, MT], other: float | int
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[MassPerEnergy, MT], other: float | int) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[NormalVolumePerMass, MT], other: float | int
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __rtruediv__(self: Quantity[EnergyPerMass, MT], other: float | int) -> Quantity[MassPerEnergy, MT]: ...

    # endregion

    @overload
    def __rtruediv__(self, other: float | int) -> Quantity[Any, MT]: ...

    # endregion

    def __rtruediv__(self, other: Quantity[Any, Any] | float | int) -> Quantity[Any, Any]:
        ret = super().__rtruediv__(other)
        return self._call_subclass(ret)

    def _temperature_difference_add_sub(
        self,
        other: Quantity[TemperatureDifference, Any],
        operator: Literal["add", "sub"],
    ) -> Quantity[Temperature, MT]:
        v1 = self.to("degC").m
        v2 = other.to("delta_degC").m

        val = v1 + v2 if operator == "add" else v1 - v2

        return Quantity[Temperature, MT](val, "degC")

    def __round__(self, ndigits: int | None = None) -> Quantity[DT, MT]:
        if isinstance(self.m, float):
            return cast("Quantity[DT, MT]", super().__round__(ndigits))

        if not isinstance(self.m, np.ndarray):
            raise NotImplementedError(f"__round__ is not implemented for magnitude type {type(self.m)}")

        return self.__class__(np.round(self.m, ndigits or 0), self.u)

    # region: overload __getitem__

    @overload
    def __getitem__(self: Quantity[DT, pd.Series], index: int) -> Quantity[DT, float]: ...
    @overload
    def __getitem__(self: Quantity[DT, pl.Series], index: int) -> Quantity[DT, float]: ...
    @overload
    def __getitem__(self: Quantity[DT, pl.Expr], index: int) -> Quantity[DT, pl.Expr]: ...
    @overload
    def __getitem__(self: Quantity[DT, Numpy1DArray], index: int) -> Quantity[DT, float]: ...

    # endregion

    def __getitem__(self, index: int) -> Quantity[DT, Any]:
        ret = super().__getitem__(index)
        subcls = self._get_dimensional_subclass(
            self._dimensionality_type, type(ret.m) if isinstance(ret, Quantity) else type(ret)
        )
        return subcls(ret.m, ret.u)

    @property
    def is_scalar(self) -> bool:
        return isinstance(self.m, float)

    @property
    def ndim(self) -> int:
        # compatibility with pandas broadcasting
        # if ndim == 0, pandas considers the object
        # a scalar and will fill array when assigning columns
        # this is similar to setting UNIT_REGISTRY.force_ndarray_like
        # except that it still allows for scalar magnitudes

        if isinstance(self.m, float | int):
            return 0

        return getattr(self.m, "ndim", 0)

    def asdim(self, other: type[DT_] | Quantity[DT_, MT]) -> Quantity[DT_, MT]:
        if isinstance(other, Quantity):
            dim = other._dimensionality_type
            assert dim is not None
        else:
            dim = other

        if dim is UNSET_DIMENSIONALITY:
            raise TypeError(f"Cannot convert {self} to unset dimensionality {dim}")

        if str(self._dimensionality_type.dimensions) != str(dim.dimensions):
            raise TypeError(
                f"Cannot convert {self} to dimensionality {dim}, "
                f"the dimensions do not match: "
                f"{self._dimensionality_type.dimensions} != "
                f"{dim.dimensions}"
            )

        subcls = self._get_dimensional_subclass(dim, self._magnitude_type)
        return cast("Quantity[DT_, MT]", subcls(self.m, self.u))

    def astype(
        self,
        magnitude_type: type[MT_],
        **kwargs: Any,  # noqa: ANN401
    ) -> Quantity[DT, MT_]:
        m, u = self.m, self.u

        if magnitude_type is pl.Expr:
            if isinstance(m, float):
                return cast("Quantity[DT, MT_]", self.subclass(cast(MT, pl.lit(m)), u))

            raise TypeError(
                f"Cannot convert magnitude with type {type(m)} to Polars expression, "
                "only scalar (float) quantities can be converted to pl.Expr"
            )

        # astype for np.ndarray should be called directly except for some special cases
        if isinstance(m, np.ndarray) and magnitude_type not in (pd.Series, pl.Series, list):
            return cast("Quantity[DT, MT_]", self.subclass(m.astype(magnitude_type), u))

        if magnitude_type is float:
            if isinstance(m, Iterable):
                return cast("Quantity[DT, MT_]", self.subclass([float(n) for n in m], u))
            else:
                return cast("Quantity[DT, MT_]", self.subclass(cast(MT, float(cast(Any, m))), u))

        if magnitude_type is np.ndarray or get_origin(magnitude_type) is np.ndarray:
            _m = [m] if not isinstance(m, Iterable) else m
            return cast("Quantity[DT, MT_]", self.subclass(cast(MT, np.array(_m)), u))

        if magnitude_type is pd.Series or get_origin(magnitude_type) is pd.Series:
            _m = [m] if not isinstance(m, Iterable) else m
            return cast("Quantity[DT, MT_]", self.subclass(cast(MT, pd.Series(cast(Any, _m), **kwargs)), u))

        if magnitude_type is pl.Series or get_origin(magnitude_type) is pl.Series:
            _m = [m] if not isinstance(m, Iterable) else m
            kwargs["values"] = _m
            return cast("Quantity[DT, MT_]", self.subclass(cast(MT, pl.Series(**kwargs)), u))

        # ensure that this method returns a new instance
        return cast("Quantity[DT, MT_]", self.__copy__())


# override the implementations for the Quantity
# and Unit classes for the current registry
# this ensures that all Quantity objects created with this registry are the correct type
setattr(UNIT_REGISTRY, "Quantity", Quantity)  # noqa: B010
setattr(UNIT_REGISTRY, "Unit", Unit)  # noqa: B010
