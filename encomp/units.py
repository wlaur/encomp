"""
Imports and extends the ``pint`` library for physical units.
Always import this module when working with ``encomp`` (most other modules
will import this one).

Implements a type-aware system on top of ``pint`` that verifies
that the dimensionality of the unit is correct.
"""


from __future__ import annotations

import copy
import numbers
import re
import warnings
from types import UnionType
from typing import Any, ClassVar, Generic, Iterable, Literal, TypeVar

import numpy as np
import pandas as pd
import pint
import polars as pl
import sympy as sp
from pint.errors import DimensionalityError
from pint.facets.formatting.objects import FormattingQuantity, FormattingUnit
from pint.facets.measurement.objects import MeasurementQuantity
from pint.facets.nonmultiplicative.objects import NonMultiplicativeQuantity
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.numpy.unit import NumpyUnit
from pint.registry import LazyRegistry, UnitRegistry
from pint.util import UnitsContainer
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from .misc import isinstance_types
from .settings import SETTINGS
from .utypes import (
    _BASE_SI_UNITS,
    DT,
    DT_,
    MT,
    MT_,
    Dimensionality,
    Temperature,
    TemperatureDifference,
    Unknown,
    Unset,
    Variable,
)


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

    def __init__(self, msg: str = ""):
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
# NOTE: make sure to list all that are defined in data/units.txt ("# custom dimensions")
CUSTOM_DIMENSIONS: list[str] = ["currency", "normal"]


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
    def __setattr__(self, key, value):
        # ensure that static options cannot be overridden
        if key in _REGISTRY_STATIC_OPTIONS:
            if value != _REGISTRY_STATIC_OPTIONS[key]:
                return

        return super().__setattr__(key, value)


class _LazyRegistry(LazyRegistry):
    def __init(self):
        args, kwargs = self.__dict__["params"]
        kwargs["on_redefinition"] = "raise"

        # override the filename
        kwargs["filename"] = str(SETTINGS.units.resolve().absolute())

        self.__class__ = _UnitRegistry
        self.__init__(*args, **kwargs)
        self._after_init()  # type: ignore


ureg: UnitRegistry = _LazyRegistry()  # type: ignore
for k, v in _REGISTRY_STATIC_OPTIONS.items():
    setattr(ureg, k, v)

# make sure that ureg is the only registry that can be used
pint._DEFAULT_REGISTRY = ureg  # type: ignore
pint.application_registry.set(ureg)


def define_dimensionality(name: str, symbol: str | None = None) -> None:
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

    if name in CUSTOM_DIMENSIONS:
        raise DimensionalityRedefinitionError(
            "Cannot define new dimensionality with "
            f"name: {name}, a dimensionality with this name was already defined"
        )

    definition_str = f"{name} = [{name}]"

    if symbol is not None:
        definition_str = f"{definition_str} = {symbol}"

    ureg.define(definition_str)

    CUSTOM_DIMENSIONS.append(name)


class QuantityMeta(type):
    def __eq__(mcls, obj: object) -> bool:
        # override the == operator so that
        # type(val) == Quantity returns True for subclasses
        if obj is Quantity:
            return True

        return super().__eq__(obj)

    def __hash__(mcls):
        return id(mcls)

    def __call__(mcls, *args, **kwargs):
        # TODO: determine if this is called from the underlying pint
        # API or directly and validate the subclass and unit dimensionality
        # based on this

        return super().__call__(*args, **kwargs)


class Unit(NumpyUnit, FormattingUnit, Generic[DT]):
    pass


class Quantity(
    NonMultiplicativeQuantity,
    MeasurementQuantity,
    NumpyQuantity,
    FormattingQuantity,
    Generic[DT, MT],
    metaclass=QuantityMeta,
):
    """
    Subclass of pint's ``Quantity`` with additional type hints,  functionality
    and integration with other libraries.

    Encodes the output dimensionalities of some common operations,
    for example ``Length**2 -> Area``. This is implemented by overloading the
    ``__mul__, __truediv__, __rtruediv__, __pow__`` methods.

    .. note::
        The overload signatures are defined in a separate file (``units.pyi``)

    """

    # constants
    NORMAL_M3_VARIANTS = ("nm³", "Nm³", "nm3", "Nm3", "nm**3", "Nm**3", "nm^3", "Nm^3")
    TEMPERATURE_DIFFERENCE_UCS = (Unit("delta_degC")._units, Unit("delta_degF")._units)
    # compact, Latex, HTML, Latex/siunitx formatting
    FORMATTING_SPECS = ("~P", "~L", "~H", "~Lx")

    # class attributes
    _REGISTRY: ClassVar[UnitRegistry] = ureg

    # mapping from dimensionality subclass name to quantity subclass
    # this dict will be populated at runtime
    # use a custom class attribute (not cls.__subclasses__()) for more control
    _subclasses: ClassVar[
        dict[tuple[str, str | None], type[Quantity[Variable, Any]]]
    ] = {}
    _dimension_symbol_map: ClassVar[dict[sp.Basic, Unit]] = {}

    # used to validate dimensionality and magnitude type,
    # if None the dimensionality is not checked
    # subclasses of Quantity have this class attribute set, which
    # will restrict the dimensionality when creating the object
    _dimensionality_type: ClassVar[type[Dimensionality]] = Unset

    # instance attributes
    _magnitude: MT
    _magnitude_type: type[MT]
    _original_magnitude_type: type[MT]
    _original_magnitude_kwargs: dict[str, Any]

    def __hash__(self) -> int:
        return super().__hash__()

    def __str__(self) -> str:
        return self.__format__(self._REGISTRY.default_format)

    @classmethod
    def _construct_dimensional_magnitude_quantity(
        cls,
        dimensional_cls: type[Quantity[DT, Any]],
        dim_name: str,
        mt: type[MT],
        mt_name: str,
    ) -> type[Quantity[DT, MT]]:
        subcls = type(
            f"Quantity[{dim_name}, {mt_name}]",
            (dimensional_cls,),
            {
                "_magnitude_type": mt,
                "__class__": dimensional_cls,
            },
        )

        # store the subclass for future lookups
        cls._subclasses[dim_name, mt_name] = subcls

        return subcls

    @classmethod
    def _get_dimensional_subclass(
        cls, dim: type[DT], mt: type[MT] | None
    ) -> type[Quantity[DT, MT]]:
        # there are two levels of subclasses to Quantity: DimensionalQuantity and
        # DimensionalMagnitudeQuantity, which is a subclass of DimensionalQuantity
        # this distinction only exists at runtime, the type checker will use the
        # default magnitude type (np.ndarray) in case the magnitude generic is omitted

        dim_name = dim.__name__

        # check if an existing DimensionalQuantity subclass already has been created
        if cached_dim_qty := cls._subclasses.get((dim_name, None)):
            DimensionalQuantity = cached_dim_qty

        else:
            DimensionalQuantity = type(
                f"Quantity[{dim_name}]",
                (Quantity,),
                {
                    "_dimensionality_type": dim,
                    "_magnitude_type": None,
                    "__class__": Quantity,
                },
            )

            cls._subclasses[dim_name, None] = DimensionalQuantity

        if mt is None:
            return DimensionalQuantity

        # TODO: properly support union magnitude types, e.g. float | np.ndarray
        # this will just use the first potential type
        # this is related to co-variant etc., figure out how this works
        if isinstance(mt, UnionType):
            mt = mt.__args__[0]

        mt_name = mt.__name__

        # check if an existing DimensionalMagnitudeQuantity subclass already has been created
        if cached_dim_magnitude_qty := cls._subclasses.get((dim_name, mt_name)):
            return cached_dim_magnitude_qty
        else:
            return cls._construct_dimensional_magnitude_quantity(
                DimensionalQuantity, dim_name, mt, mt_name
            )

    def __class_getitem__(
        cls, types: type[DT] | tuple[type[DT], type[MT]]
    ) -> type[Quantity[DT, MT]]:
        # default magnitude type is np.ndarray, this is hard-coded in several places

        try:
            dim, mt = types
        except Exception:
            dim, mt = types, None

        # avoid runtime errors when evaluating type hints
        if (
            isinstance(dim, TypeVar)
            or dim is Unknown
            or dim is Variable
            or dim is Unset
        ):
            return cls._get_dimensional_subclass(Variable, mt)

        if not isinstance(dim, type):
            raise TypeError(
                "Generic type parameter to Quantity must be a type object, "
                f"passed an instance of {type(dim)}: {dim}"
            )

        # check if the attribute dimensions exists instead of using issubclass()
        # issubclass does not work well with autoreloading in Jupyter for example
        if not hasattr(dim, "dimensions"):
            raise TypeError(
                'Generic type parameter to Quantity has no attribute "dimensions", '
                f"passed: {dim}"
            )

        if not isinstance(dim.dimensions, UnitsContainer):
            raise TypeError(
                "Type parameter to Quantity has incorrect type for "
                "attribute dimensions: UnitsContainer, "
                f"passed: {dim} with dimensions: {dim.dimensions} ({type(dim.dimensions)})"
            )

        subcls = cls._get_dimensional_subclass(dim, mt)

        return subcls

    @staticmethod
    def _validate_unit(
        unit: Unit[DT_] | UnitsContainer | str | dict[str, numbers.Number] | None
    ) -> Unit[DT_]:
        if isinstance(unit, Quantity):
            raise TypeError(
                f"Input unit has incorrect type Quantity: {unit}. "
                "Do not create nested Quantity objects, convert "
                "to separate magnitude and unit objects first."
            )

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

        raise ValueError(
            f"Incorrect input for unit: {unit} ({type(unit)}), "
            "expected Unit, UnitsContainer, str, dict[str, Number], Quantity or None"
        )

    @staticmethod
    def _validate_magnitude(val: MT) -> MT:
        # NOTE: container types (e.g. list[int]) are tested using only the first element

        if isinstance(val, str):
            raise TypeError(f'Input magnitude has incorrect type str: "{val}"')

        # bool is always converted to float, the type hints don't consider bool at all
        if isinstance(val, bool):
            val = float(val)
        if isinstance_types(val, list[bool]):
            val = [float(n) for n in val]

        # TODO: list of int is converted to np.ndarray, type hints cannot represent this properly
        # do not input list[int] in code that should be type checked
        # however, this input is still allowed for convenience, for example Q([1, 2, 3], 'kg')
        if isinstance_types(val, list[int]):
            val = np.array(val)

        # list[float] is not converted
        if isinstance_types(val, list[float]):
            pass

        # int is considered equivalent to float by type checkers
        if isinstance(val, int):
            val = float(val)

        return val

    @staticmethod
    def _get_magnitude_type(val: MT) -> type[MT]:
        if isinstance(val, list):
            if not len(val):
                return list

            # NOTE: this does not work for heterogenous lists,
            # don't use this type of input to avoid issues
            return list[type(val[0])]

        return type(val)

    @classmethod
    def get_unit(cls, unit_name: str) -> Unit:
        return Unit(cls._REGISTRY.parse_units(unit_name))

    @property
    def subclass(self) -> type[Quantity[DT, MT]]:
        return self._get_dimensional_subclass(
            self._dimensionality_type, self._magnitude_type
        )

    def _set_original_magnitude_attributes(
        self, mt_orig: MT, mt_orig_kwargs: dict[str, Any]
    ) -> None:
        self._original_magnitude_type = mt_orig
        self._original_magnitude_kwargs = mt_orig_kwargs

    def __len__(self) -> int:
        # __len__() must return an integer
        # the len() function ensures this at a lower level
        if isinstance(self._magnitude, (float, int)):
            raise TypeError(
                f"Quantity with scalar magnitude ({self._magnitude}) has no length"
            )

        elif isinstance(self._magnitude, pl.Expr):
            raise TypeError(
                f"Cannot determine length of Polars expression: {self._magnitude}"
            )

        return len(self._magnitude)

    def __copy__(self) -> Quantity[DT, MT]:
        ret = self._call_subclass(copy.copy(self._magnitude), self._units)

        return ret

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Quantity[DT, MT]:
        if memo is None:
            memo = {}

        ret = self._call_subclass(
            copy.deepcopy(self._magnitude, memo), copy.deepcopy(self._units, memo)
        )

        return ret

    @classmethod
    def _validate_datetime_magnitude(
        cls,
        magnitude: MT,
        unit: Unit[DT] | UnitsContainer | str | None,
        valid_unit: Unit[DT],
    ) -> None:
        _val_datetime = isinstance(
            magnitude, (pd.DatetimeIndex, pl.Datetime, pd.Timestamp)
        )

        if _val_datetime:
            if (
                not valid_unit.dimensionless
                or Quantity(1, valid_unit).to_base_units().m != 1
            ):
                raise ValueError(
                    f"Passing datetime magnitude(s) ({magnitude}) "
                    "is only valid for dimensionless Quantities with no scaling factor, "
                    f"passed unit {unit} ({valid_unit.dimensionality})"
                )

        _magnitude_type_datetime = cls._magnitude_type in (
            pd.Timestamp,
            pd.DatetimeIndex,
            pl.Datetime,
        )

        if _magnitude_type_datetime:
            if (
                not valid_unit.dimensionless
                or Quantity(1, valid_unit).to_base_units().m != 1
            ):
                raise ValueError(
                    f"Setting a datetime magnitude type ({cls._magnitude_type.__name__}) "
                    "is only valid for dimensionless Quantities with no scaling factor, "
                    f"passed unit {unit} ({valid_unit.dimensionality})"
                )

    @staticmethod
    def _cast_array_float(inp: np.ndarray) -> np.ndarray:
        # don't fail in case the array contains unsupported objects,
        # for example uncertainties.ufloat
        try:
            return inp.astype(np.float64, casting="unsafe", copy=True)
        except TypeError:
            return inp

    def __new__(  # type: ignore
        cls,
        val: MT | Quantity[DT, MT],
        unit: Unit[DT] | UnitsContainer | str | None = None,
        # # this is a hack to force the type checker to default to Unknown
        # # in case the generic type is not specified at all
        # # do not pass the _dt parameter directly, always use square brackets to
        # # specify the dimensionality type
        # TODO: why is this required when the TypeVar has default=Unknown?
        _dt: type[DT] = Unknown,  # type: ignore
        _mt_orig: type[MT] | None = None,
        _mt_orig_kwargs: dict[str, Any] | None = None,
    ) -> Quantity[DT]:
        if _dt is not Unknown:
            raise ValueError(
                f'Cannot pass a value for private parameter "_dt", passed {_dt} ({type(_dt)})'
            )

        if isinstance(val, Quantity):
            if unit is not None:
                raise ValueError(
                    f"Cannot pass unit: {unit} when "
                    f"input val is a Quantity: {val}. "
                    "The unit must be None when passing a Quantity as input"
                )

            val, unit = val.m, val.u

        valid_unit = cls._validate_unit(unit)
        valid_magnitude = cls._validate_magnitude(val)

        # NOTE: "original" in this case does not necessarily refer to the type
        # of the input magnitude, for example list[int] is converted to np.ndarray before
        # this magnitude type is checked
        _original_magnitude_type = (
            _mt_orig
            if _mt_orig is not None
            else cls._get_magnitude_type(valid_magnitude)
        )

        _original_magnitude_kwargs = (
            _mt_orig_kwargs if _mt_orig_kwargs is not None else {}
        )

        # special case for pd.Series, must convert to np.ndarray to avoid
        # TypeError: PlainQuantity cannot wrap upcast type
        # NOTE: pl.Series and pl.Expr don't require this type of workaround
        if isinstance(valid_magnitude, pd.Series):
            # preserve pd.Series metadata (not dtype, this would cause issues with float/int)
            _original_magnitude_kwargs |= {
                "index": valid_magnitude.index,
                "name": valid_magnitude.name,
            }

            valid_magnitude = valid_magnitude.to_numpy()

        is_incomplete_subclass = cls._dimensionality_type is Unset or str(
            cls._dimensionality_type.dimensions
        ) != str(valid_unit.dimensionality)

        if is_incomplete_subclass:
            # TODO: how to validate that the subclass has the same dimensionality
            # as the input unit?
            # cannot raise error here since this breaks the pint.PlainQuantity methods
            # that use return self.__class__(...)

            # special case for temperature difference
            if valid_unit._units in cls.TEMPERATURE_DIFFERENCE_UCS:
                subcls = cls._get_dimensional_subclass(
                    TemperatureDifference, type(valid_magnitude)
                )
            else:
                dim = Dimensionality.get_dimensionality(valid_unit.dimensionality)
                subcls = cls._get_dimensional_subclass(dim, type(valid_magnitude))

            return subcls(
                valid_magnitude,
                valid_unit,
                _mt_orig=_original_magnitude_type,
                _mt_orig_kwargs=_original_magnitude_kwargs,
            )

        cls._validate_datetime_magnitude(valid_magnitude, unit, valid_unit)

        qty: Quantity[DT, MT] = super().__new__(  # type: ignore
            cls, valid_magnitude, units=valid_unit
        )

        # ensure that pint did not change the dtype of numpy arrays
        if isinstance(qty._magnitude, np.ndarray):
            qty._magnitude = cls._cast_array_float(qty._magnitude)

        # propagate the original magnitude type and constructor kwargs
        qty._set_original_magnitude_attributes(
            _original_magnitude_type, _original_magnitude_kwargs
        )

        return qty

    @property
    def m(self) -> MT:
        if self._original_magnitude_type == list[int]:
            # NOTE: this is a workaround to match the type checker output (np.ndarray)
            # this is not consistent with the behavior for list[float]
            return self._magnitude

        elif self._original_magnitude_type == list[float]:
            return [float(n) for n in self._magnitude]

        elif self._original_magnitude_type == pd.Timestamp:
            return pd.Timestamp(self._magnitude, **self._original_magnitude_kwargs)

        elif self._original_magnitude_type == pd.Series:
            return pd.Series(self._magnitude, **self._original_magnitude_kwargs)

        else:
            return self._magnitude

    @property
    def u(self) -> Unit[DT]:
        return Unit(super().u)

    @property
    def units(self) -> Unit[DT]:
        return Unit(super().units)

    @property
    def _is_temperature_difference(self):
        return self._dimensionality_type is TemperatureDifference

    def _check_temperature_compatibility(self, unit: Unit[DT]) -> None:
        if self._is_temperature_difference:
            if unit._units not in self.TEMPERATURE_DIFFERENCE_UCS:
                current_name = self._dimensionality_type.__name__
                new_name = Quantity(1, unit)._dimensionality_type.__name__

                raise DimensionalityTypeError(
                    f"Cannot convert {self.units} (dimensionality {current_name}) "
                    f"to {unit} (dimensionality {new_name})"
                )

    def _call_subclass(self, *args) -> Quantity[DT, MT]:
        # handle the edge case where a 1-element pd.Series is
        # multiplied or divided by an N-element pd.Series

        if "index" in self._original_magnitude_kwargs:
            index = self._original_magnitude_kwargs["index"]

            try:
                magnitude_length = len(args[0])
            except Exception:
                magnitude_length = None

            # strip the index in case it's not compatible
            if magnitude_length is not None and magnitude_length != len(index):
                del self._original_magnitude_kwargs["index"]

        return self.subclass(
            *args,
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

    def _to_unit(self, unit: Unit[DT_] | UnitsContainer | str | dict) -> Unit[DT_]:
        return self._validate_unit(unit)

    def to(self, unit: Unit[DT] | UnitsContainer | str | dict) -> Quantity[DT, MT]:
        unit = self._to_unit(unit)

        self._check_temperature_compatibility(unit)
        m = self._convert_magnitude_not_inplace(unit)

        if unit._units in self.TEMPERATURE_DIFFERENCE_UCS:
            return Quantity[TemperatureDifference](
                m,
                unit,
                _mt_orig=self._original_magnitude_type,
                _mt_orig_kwargs=self._original_magnitude_kwargs,
            )

        converted = self._call_subclass(m, unit)

        return converted

    def ito(self, unit: Unit[DT] | UnitsContainer | str) -> None:
        # NOTE: this method cannot convert the dimensionality type

        unit = self._to_unit(unit)

        self._check_temperature_compatibility(unit)

        # it's not safe to convert units as int, the
        # user will have to convert back to int if necessary
        # better to use ":.0f" formatting or round() anyway

        # avoid numpy.core._exceptions.UFuncTypeError (not on all platforms?)
        # convert integer arrays to float(64) (creating a copy)
        if isinstance(self._magnitude, np.ndarray) and issubclass(
            self._magnitude.dtype.type, numbers.Integral
        ):
            self._magnitude = self._magnitude.astype(np.float64)

        super().ito(unit)

    def check(
        self,
        unit: Quantity
        | UnitsContainer
        | Unit
        | str
        | Dimensionality
        | type[Dimensionality],
    ) -> bool:
        # TODO: fix typing for this method, remove type: ignore

        if isinstance(unit, Quantity):
            return self._dimensionality_type == unit._dimensionality_type

        if isinstance(unit, Unit):
            # it's not possible to know if an instance of Unit is Temperature or TemperatureDifference
            # until it is used to construct a Quantity

            unit_qty = Quantity(1, unit)

            if isinstance(unit_qty, Quantity[TemperatureDifference]):
                unit_qty = Quantity[TemperatureDifference](1.0, unit)

            return self.check(unit_qty)

        if hasattr(unit, "dimensions"):
            unit = unit.dimensions  # type: ignore

        return super().check(unit)  # type: ignore

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
            format_type = f"{format_type}{self._REGISTRY.default_format}"

        return super().__format__(format_type)

    @staticmethod
    def correct_unit(unit: str) -> str:
        """
        Corrects the unit name to make it compatible with ``pint``.

        * Adds ``**`` between the unit and the exponent if it's missing, for example ``kg/m3 → kg/m**3``.
        * Parses "Nm³" (and variations of this) as "normal * m³" (use explicit "nanometer³" to get this unit)
        * Converts % and ‰ to percent and permille
        * Changes the ``Δ`` character to ``delta_``, for example ``Δ°C`` to ``delta_°C``
        * Interprets ``-`` as ``dimensionless``
        * Converts the single-character symbols ``℃`` and ``℉`` to ``degC`` and ``degF``
        * Converts the compound units ``Δ°C`` and ``Δ°F`` to ``delta_degC`` and ``delta_degF``

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
        # so we'll not even try to parse that
        # use "nanometer**3" if necessary
        for n in Quantity.NORMAL_M3_VARIANTS:
            if n in unit:
                # include brackets, otherwise "kg/nm3" is
                # incorrectly converted to "kg/normal*m3"
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

    def _sympy_(self):
        """
        Compatibility with ``sympy``.
        Converts the unit dimensions to symbols and multiplies with the magnitude.
        Need to use base units, compound units will not cancel out otherwise.
        """

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
        expr = sp.sympify(f"{base_qty.m} * {unit_repr}").subs(
            {sp.Symbol(n): self.get_unit_symbol(n) for n in symbols}
        )

        return expr

    @staticmethod
    def get_unit_symbol(s: str) -> sp.Symbol:
        return sp.Symbol("\\text{" + s + "}", nonzero=True, positive=True)

    @classmethod
    def _populate_dimension_symbol_map(cls) -> None:
        # also consider custom dimensions defined with encomp.units.define_dimensionality
        cls._dimension_symbol_map |= {
            cls.get_unit_symbol(n): cls.get_unit(n)
            for n in list(_BASE_SI_UNITS) + CUSTOM_DIMENSIONS
        }

    @classmethod
    def from_expr(cls, expr: sp.Basic) -> Quantity[Unknown, float]:
        """
        Converts a Sympy expression with unit symbols
        into a Quantity. This only works in case expression
        *only* contains base SI unit symbols.
        """

        # this needs to be populated here to account for custom dimensions
        cls._populate_dimension_symbol_map()

        expr = expr.simplify()
        args = expr.args

        if not args:
            return cls(float(expr), "dimensionless")  # type: ignore

        try:
            magnitude = float(args[0])  # type: ignore
        except TypeError as e:
            raise ValueError(f"Expression {expr} contains inconsistent units") from e

        dimensions = args[1:]

        unit = cls.get_unit("")

        for d in dimensions:
            unit_i = cls.get_unit("")

            for symbol, power in d.as_powers_dict().items():  # type: ignore
                unit_i *= cls._dimension_symbol_map[symbol] ** power

            unit *= unit_i

        return cls(magnitude, unit).to_base_units()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, qty, info) -> Quantity[DT, MT]:
        # convert non-Quantity inputs to Quantity before checking subclass
        ret = cls(qty)

        if isinstance(ret, cls):
            return ret

        raise ExpectedDimensionalityError(
            f"Value {ret} ({type(ret).__name__}) does not "
            f"match the expected dimensionality {cls.__name__}"
        )

    def check_compatibility(self, other: Quantity | float | int) -> None:
        """
        Checks compatibility for addition and subtraction.
        """

        if not isinstance(other, Quantity):
            if not self.dimensionless:
                raise DimensionalityTypeError(
                    f"Cannot add or subtract {other} ({type(other)}) to "
                    f"quantity {self} ({type(self)})"
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
            if (
                self._dimensionality_type.dimensions
                == other._dimensionality_type.dimensions
            ):
                raise DimensionalityTypeError(
                    f"Quantities with different dimensionalities are not compatible: "
                    f"{type(self)} and {type(other)}. The dimensions match, "
                    "but the dimensionalities have different types."
                )

            raise DimensionalityTypeError(
                f"Quantities with different dimensionalities are not compatible: "
                f"{type(self)} and {type(other)}. "
            )

    def is_compatible_with(
        self, other: Quantity | float | int, *contexts, **ctx_kwargs
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

    def __eq__(self, other):
        # this returns an array of one or both inputs have array magnitudes
        is_equal = super().__eq__(other)

        if isinstance(is_equal, np.ndarray):
            if self.is_compatible_with(other):
                return is_equal.astype(bool)

            return np.zeros_like(is_equal).astype(bool)

        return is_equal and self.is_compatible_with(other)

    def __mul__(self, other):
        ret = super().__mul__(other)
        if self.dimensionless and isinstance(other, Quantity):
            return other.subclass(ret)

        return self._call_subclass(ret)

    def __rmul__(self, other):
        ret = super().__rmul__(other)

        return self._call_subclass(ret)

    def __truediv__(self, other):
        ret = super().__truediv__(other)
        return self._call_subclass(ret)

    def __rtruediv__(self, other):
        ret = super().__rtruediv__(other)
        return self._call_subclass(ret)

    def _temperature_difference_add_sub(
        self: Quantity[Temperature, MT],
        other: Quantity[TemperatureDifference, MT],
        operator: Literal["add", "sub"],
    ) -> Quantity[Temperature, MT]:
        v1 = self.to("degC").m
        v2 = other.to("delta_degC").m

        if operator == "add":
            val = v1 + v2
        else:
            val = v1 - v2

        result = Quantity(val, "degC")

        return result

    def __add__(self, other):
        try:
            self.check_compatibility(other)

        except DimensionalityTypeError as e:
            if isinstance_types(
                self, Quantity[Temperature] | Quantity[TemperatureDifference]
            ) and isinstance_types(
                other, Quantity[Temperature] | Quantity[TemperatureDifference]
            ):
                if self._dimensionality_type is TemperatureDifference:
                    raise e

                return self._temperature_difference_add_sub(other, "add")

            raise e

        ret = super().__add__(other)

        return self._call_subclass(ret)

    def __sub__(self, other):
        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:
            if isinstance_types(
                self, Quantity[Temperature] | Quantity[TemperatureDifference]
            ) and isinstance_types(
                other, Quantity[Temperature] | Quantity[TemperatureDifference]
            ):
                if self._dimensionality_type is TemperatureDifference:
                    raise e

                return self._temperature_difference_add_sub(other, "sub")

            raise e

        ret = super().__sub__(other)

        if (
            self._dimensionality_type is Temperature
            and other._dimensionality_type is Temperature
        ):
            return Quantity[TemperatureDifference, type(ret.m)](ret.m, ret.u)

        return self._call_subclass(ret)

    def __gt__(self, other):
        try:
            return super().__gt__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    def __ge__(self, other):
        try:
            return super().__ge__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    def __lt__(self, other):
        try:
            return super().__lt__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    def __le__(self, other):
        try:
            return super().__le__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    def __round__(self, ndigits: int | None = None) -> Quantity[DT]:
        if isinstance(self.m, float):
            return super().__round__(ndigits)

        return self.__class__(np.round(self.m, ndigits or 0), self.u)

    def __getitem__(self, index: int) -> Quantity[DT]:
        ret = super().__getitem__(index)

        if isinstance(ret._magnitude, pd.Timestamp):
            scalar_type = pd.Timestamp
        else:
            scalar_type = float

        # NOTE: this is a special case, scalar magnitudes cannot have kwargs

        subcls = self._get_dimensional_subclass(self._dimensionality_type, scalar_type)
        ret._original_magnitude_type = scalar_type

        return subcls(
            ret.m,
            ret.u,
        )

    @property
    def is_scalar(self) -> bool:
        return isinstance(self.m, (float, int))

    @property
    def ndim(self) -> int:
        # compatibility with pandas broadcasting
        # if ndim == 0, pandas considers the object
        # a scalar and will fill array when assigning columns
        # this is similar to setting ureg.force_ndarray_like
        # except that it still allows for scalar magnitudes

        if isinstance(self.m, (float, int)):
            return 0

        return self.m.ndim

    def asdim(self, other: type[DT_] | Quantity[DT_, MT]) -> Quantity[DT_, MT]:
        if isinstance(other, Quantity):
            dim = other._dimensionality_type
            assert dim is not None
        else:
            dim = other

        if str(self._dimensionality_type.dimensions) != str(dim.dimensions):
            raise TypeError(
                f"Cannot convert {self} to dimensionality {dim}, "
                f"the dimensions do not match: {self._dimensionality_type.dimensions} != "
                f"{dim.dimensions}"
            )

        subcls = self._get_dimensional_subclass(dim, self._magnitude_type)
        return subcls(self)

    def astype(self, magnitude_type: type[MT_], **kwargs: Any) -> Quantity[DT, MT_]:
        m, u = self.m, self.u

        _is_iterable = isinstance(m, Iterable)

        # astype for np.ndarray should be called directly except for some special cases
        custom_conversion = [pd.Series, pl.Series, list[float]]

        if isinstance(m, np.ndarray) and magnitude_type not in custom_conversion:
            return self.subclass(m.astype(magnitude_type), u)

        if magnitude_type in (pd.DatetimeIndex, pd.Timestamp):
            raise ValueError(
                f"Cannot convert to datetime magnitude type: {magnitude_type}"
            )

        if magnitude_type == pl.Expr:
            raise ValueError("Cannot convert magnitude to Polars expression")

        if magnitude_type == list[int]:
            raise NotImplementedError(
                "Cannot convert magnitude to list[int], use list[float] instead"
            )

        if magnitude_type in (float, int):
            if _is_iterable:
                return self.subclass([float(n) for n in m], u)
            else:
                return self.subclass(float(m), u)

        if magnitude_type == list[float]:
            if not _is_iterable:
                m = [m]
            return self.subclass([float(n) for n in m], u)

        if magnitude_type == np.ndarray:
            if not _is_iterable:
                m = [m]
            return self.subclass(np.array(m), u)

        if magnitude_type == pd.Series:
            if not _is_iterable:
                m = [m]
            return self.subclass(pd.Series(m, **kwargs), u)

        if magnitude_type == pl.Series:
            if not _is_iterable:
                m = [m]
            kwargs["values"] = m
            return self.subclass(pl.Series(**kwargs), u)

        # ensure that this method returns a new instance
        return self.__copy__()


# override the implementations for the Quantity and Unit classes for the current registry
# this ensures that all Quantity objects created with this registry are the correct type
ureg.Quantity = Quantity
ureg.Unit = Unit

# the default format must be set after Quantity and Unit are registered
ureg.default_format = SETTINGS.default_unit_format


def set_quantity_format(fmt: str = "compact") -> None:
    """
    Sets the ``default_format`` attribute for the currently
    active pint unit registry.

    Parameters
    ----------
    fmt : str
        Unit format string: one of ``'~P', '~L', '~H', '~Lx'``.
        Also accepts aliases: ``'compact': '~P'`` and ``'siunitx': '~Lx'``.

    Raises
    ------
    ValueError
        In case ``fmt`` is not among the valid format strings.
    """

    fmt_aliases = {"normal": "~P", "siunitx": "~Lx"}

    if fmt in fmt_aliases:
        fmt = fmt_aliases[fmt]

    if fmt not in Quantity.FORMATTING_SPECS:
        raise ValueError(
            f'Cannot set default format to "{fmt}", '
            f"fmt is one of {Quantity.FORMATTING_SPECS} "
            "or alias siunitx: ~L, compact: ~P"
        )

    ureg.default_format = fmt
