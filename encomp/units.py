"""
Imports and extends the ``pint`` library for physical units.
Always import this module when working with ``encomp`` (most other modules
will import this one).

Implements a type-aware system on top of ``pint`` that verifies
that the dimensionality of the unit is correct.
"""


from __future__ import annotations

import re
import warnings
import numbers
from typing import Union, Optional, Generic, Any, Literal, overload
import sympy as sp
import numpy as np
import pandas as pd


import pint
from pint.unit import UnitsContainer, Unit
from pint.registry import UnitRegistry
from pint.errors import DimensionalityError

from encomp.settings import SETTINGS

from encomp.utypes import (_BASE_SI_UNITS,
                           DimensionlessUnits,
                           CurrencyUnits,
                           CurrencyPerEnergyUnits,
                           CurrencyPerMassUnits,
                           CurrencyPerVolumeUnits,
                           CurrencyPerTimeUnits,
                           LengthUnits,
                           MassUnits,
                           TimeUnits,
                           TemperatureUnits,
                           SubstanceUnits,
                           CurrentUnits,
                           LuminosityUnits,
                           AreaUnits,
                           VolumeUnits,
                           NormalVolumeUnits,
                           PressureUnits,
                           MassFlowUnits,
                           VolumeFlowUnits,
                           NormalVolumeFlowUnits,
                           DensityUnits,
                           SpecificVolumeUnits,
                           EnergyUnits,
                           PowerUnits,
                           VelocityUnits,
                           DynamicViscosityUnits,
                           KinematicViscosityUnits)

from encomp.utypes import (MagnitudeInput,
                           MagnitudeScalar,
                           Magnitude,
                           DT,
                           DT_,
                           Dimensionality,
                           Unknown,
                           Impossible,
                           Dimensionless,
                           Currency,
                           CurrencyPerEnergy,
                           CurrencyPerMass,
                           CurrencyPerVolume,
                           CurrencyPerTime,
                           Substance,
                           Density,
                           Energy,
                           Power,
                           Time,
                           Temperature,
                           Length,
                           Area,
                           Volume,
                           Mass,
                           MassFlow,
                           VolumeFlow,
                           NormalVolume,
                           NormalVolumeFlow,
                           SpecificVolume,
                           Current,
                           Luminosity,
                           Pressure,
                           Velocity,
                           DynamicViscosity,
                           KinematicViscosity
                           )

if SETTINGS.ignore_ndarray_unit_stripped_warning:
    warnings.filterwarnings(
        'ignore',
        message='The unit of the quantity is stripped when downcasting to ndarray.')


# custom errors inherit from pint.errors.DimensionalityError
# this makes it possible to use
# try:
#     ...
# except DimensionalityError:
#     ...
# to catch all unit/dimensionality-related errors
class _DimensionalityError(DimensionalityError):

    def __init__(self, msg=''):

        self.msg = msg
        super().__init__(None, None, dim1=None, dim2=None, extra_msg=msg)

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
_CUSTOM_DIMENSIONS: list[str] = []


# there should only be one registry at a time, pint raises ValueError
# in case quantities from different registries interact

class CustomRegistry:
    """
    Custom registry implementation, based on
    ``LazyRegistry``.
    """

    def __init__(self, args=None, kwargs=None):
        self.__dict__['params'] = args or (), kwargs or {}

    def __init(self):
        args, kwargs = self.__dict__['params']
        kwargs['on_redefinition'] = 'raise'
        kwargs['filename'] = str(SETTINGS.units.resolve().absolute())
        self.__class__ = UnitRegistry
        self.__init__(*args, **kwargs)
        self._after_init()

    def __getattr__(self, item):
        if item == '_on_redefinition':
            return 'raise'
        self.__init()
        return getattr(self, item)

    def __setattr__(self, key, value):
        if key == '__class__':
            super().__setattr__(key, value)
        else:
            self.__init()
            setattr(self, key, value)

    def __getitem__(self, item):
        self.__init()
        return self[item]

    def __call__(self, *args, **kwargs):
        self.__init()
        return self(*args, **kwargs)


ureg = CustomRegistry()


# if False, degC must be explicitly converted to K when multiplying
# this is False by default, there's no reason to set this to True
ureg.autoconvert_offset_to_baseunit = SETTINGS.autoconvert_offset_to_baseunit

# enable support for matplotlib axis ticklabels etc...
try:
    ureg.setup_matplotlib()
except ImportError:
    pass

ureg.default_format = SETTINGS.default_unit_format

try:
    ureg._registry.default_format = SETTINGS.default_unit_format
except Exception:
    pass


# shorthand for the @wraps(ret, args, strict=True|False) decorator
wraps = ureg.wraps
check = ureg.check


def define_dimensionality(name: str, symbol: str = None) -> None:
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
        since the unit needs to be appended to the ``_CUSTOM_DIMENSIONS`` list as well.

    Parameters
    ----------
    name : str
        Name of the dimensionality
    symbol : str, optional
        Optional (short) symbol, by default None
    """

    if name in _CUSTOM_DIMENSIONS:
        raise DimensionalityRedefinitionError(
            'Cannot define new dimensionality with '
            f'name: {name}, a dimensionality with this name was already defined')

    definition_str = f'{name} = [{name}]'

    if symbol is not None:
        definition_str = f'{definition_str} = {symbol}'

    ureg.define(definition_str)

    _CUSTOM_DIMENSIONS.append(name)


# define commonly used media as dimensionalities
# "normal" is used to signify normal volume, e.g. "Nm³/s"
for dimensionality_name in (
    'normal',
    'air',
    'water',
    'fuel'
):

    try:
        define_dimensionality(dimensionality_name)
    except pint.errors.DefinitionSyntaxError as e:
        pass


# "currency" is used to represent some unspecified currency
# note that it's not possible to convert different currencies using this system
# the default currencies will have an approximate scaling factor
# NOTE: actual currency operations should use decimal.Decimal or similar
# to account for rounding etc.
# Quantity supports Decimal magnitudes, but it's not included in the type hints
# unit conversions do not work with Decimal magnitudes (scaling factors are floats)

_currency_definition = """
currency = [currency]
SEK = 1 * currency
EUR = 10 * currency
USD = 10 * currency
"""

for n in _currency_definition.split('\n'):
    if not n.strip():
        continue
    ureg.define(n)

_CUSTOM_DIMENSIONS.append('currency')


def _load_additional_units() -> None:

    with open(SETTINGS.additional_units, 'r', encoding='utf-8') as f:

        lines = f.read().split('\n')
        for line in lines:

            if line.startswith('#'):
                continue

            if line.strip():

                try:
                    ureg.define(line)
                except pint.errors.RedefinitionError:
                    pass


if (
    SETTINGS.additional_units is not None and
    SETTINGS.additional_units.is_file()
):
    _load_additional_units()


class Quantity(pint.quantity.Quantity, Generic[DT]):
    """
    Subclass of ``pint.quantity.Quantity`` with additional functionality
    and integration with other libraries.

    Encodes the output dimensionalities of some common operations,
    for example ``Length**2 -> Area``. This is implemented by overloading the
    ``__mul__, __truediv__, __pow__`` methods.


    .. todo::

        Maybe the overload methods should be moved to a separate stub file?

    """

    # override the _MagnitudeType typevar since this cannot be a Generic
    # only one generic type variable can be used in this class
    _magnitude: Magnitude

    _REGISTRY: CustomRegistry = ureg  # type: ignore

    _DIMENSIONAL_SUBCLASSES: dict[type[Dimensionality], type[Quantity]] = {}

    # used to validate dimensionality using type checking,
    # if None the dimensionality is not checked
    # subclasses of Quantity have this class attribute set, which
    # will restrict the dimensionality when creating the object
    _dimensionality_type: Optional[type[Dimensionality]] = None

    _dimension_symbol_map: Optional[dict[sp.Basic, Unit]] = None

    # compact, Latex, HTML, Latex/siunitx formatting
    FORMATTING_SPECS = ('~P', '~L', '~H', '~Lx')

    # common unit names not supported by pint, also some misspellings
    # TODO: this does not work with compound units, need to use ureg.define
    # to override units at the pint parsing stage
    UNIT_CORRECTIONS = {
        'kpa': 'kPa',
        'mpa': 'MPa',
        'pa': 'Pa',
        'F': 'degF',
        'C': 'degC',
        '°C': 'degC',
        '°F': 'degF',
        'delta_C': 'delta_degC',
        'delta_°C': 'delta_degC',
        'delta_F': 'delta_degF',
        'delta_°F': 'delta_degF',
        'kmh': 'km/hour',
        'mh2o': 'meter_H2O',
        'mH20': 'meter_H2O',
        'mH2O': 'meter_H2O',
        'm H2O': 'meter_H2O',
        'm h2o': 'meter_H2O',
        'meter water': 'meter_H2O',
        'm water': 'meter_H2O',
        'm_water': 'meter_H2O',
        'meter_water': 'meter_H2O',
        'feet_water': 'feet_H2O',
        'foot_water': 'feet_H2O',
        'ft_H2O': 'feet_H2O',
        'ft_water': 'feet_H2O'
    }

    def __hash__(self) -> int:
        return super().__hash__()

    @classmethod
    def _get_dimensional_subclass(cls, dim: type[Dimensionality]) -> type[Quantity[DT]]:

        if dim in cls._DIMENSIONAL_SUBCLASSES:
            return cls._DIMENSIONAL_SUBCLASSES[dim]

        quantity_name = f'Quantity[{dim.__name__}]'

        DimensionalQuantity = type(
            quantity_name,
            (Quantity,),
            {
                '_dimensionality_type': dim,
                '__class__': Quantity,

            }
        )

        cls._DIMENSIONAL_SUBCLASSES[dim] = DimensionalQuantity

        return DimensionalQuantity

    def __class_getitem__(cls, dim: type[DT]) -> type[Quantity[DT]]:

        if not isinstance(dim, type):
            raise TypeError(
                'Generic type parameter to Quantity must be an '
                'encomp.utypes.Dimensionality subtype, '
                f'passed an instance of {type(dim)}: {dim}'
            )

        if not issubclass(dim, Dimensionality):
            raise TypeError(
                'Generic type parameter to Quantity must be an '
                'encomp.utypes.Dimensionality subtype, '
                f'passed: {dim}'
            )

        return cls._get_dimensional_subclass(dim)

    @staticmethod
    def _validate_unit(unit: Union[Unit, UnitsContainer, str, Quantity, dict]) -> Unit:

        if isinstance(unit, Unit):
            return unit

        if isinstance(unit, Quantity):
            return unit.u

        # compatibility with internal pint API
        if isinstance(unit, dict):
            return Quantity._validate_unit(str(UnitsContainer(unit)))

        # compatibility with internal pint API
        if isinstance(unit, UnitsContainer):
            return Quantity._validate_unit(str(unit))

        if isinstance(unit, str):
            return Quantity._REGISTRY.parse_units(Quantity.correct_unit(unit))

        raise ValueError(
            f'Incorrect input for unit: {unit} ({type(unit)}), '
            'expected Unit, UnitsContainer, str or Quantity'
        )

    @classmethod
    def get_unit(cls, unit_name: str) -> Unit:
        return cls._REGISTRY.parse_units(unit_name)

    def __len__(self) -> int:

        if isinstance(self._magnitude, (float, int)):

            raise TypeError(
                f'Quantity with scalar magnitude ({self._magnitude}) has no len(). '
                'In case this error occurs when assigning to a pd.DataFrame, '
                'try assigning the magnitude instead of the quantity '
                '(df["column"] = qty.m instead of df["column"] = qty)'
            )

        return len(self._magnitude)

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str]  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: DimensionlessUnits
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyUnits
                ) -> Quantity[Currency]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyPerEnergyUnits
                ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyPerVolumeUnits
                ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyPerMassUnits
                ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyPerTimeUnits
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: LengthUnits) -> Quantity[Length]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: MassUnits) -> Quantity[Mass]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: TimeUnits) -> Quantity[Time]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: TemperatureUnits) -> Quantity[Temperature]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: SubstanceUnits) -> Quantity[Substance]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrentUnits) -> Quantity[Current]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: LuminosityUnits) -> Quantity[Luminosity]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: AreaUnits) -> Quantity[Area]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: VolumeUnits) -> Quantity[Volume]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: NormalVolumeUnits) -> Quantity[NormalVolume]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: PressureUnits) -> Quantity[Pressure]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: MassFlowUnits) -> Quantity[MassFlow]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: VolumeFlowUnits) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: NormalVolumeFlowUnits) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: DensityUnits) -> Quantity[Density]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: SpecificVolumeUnits) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: EnergyUnits) -> Quantity[Energy]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: PowerUnits) -> Quantity[Power]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: VelocityUnits) -> Quantity[Velocity]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: DynamicViscosityUnits) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: KinematicViscosityUnits) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __new__(
        cls,
        val: Union[MagnitudeInput, Quantity[DT], str],
        unit: Union[Unit, UnitsContainer,
                    str, Quantity[DT], None] = None,

        # this is a hack to force the type checker to default to Unknown
        # in case the generic type is not specified at all
        _dt: type[DT] = Unknown  # type: ignore
    ) -> Quantity[DT]:
        ...

    def __new__(  # type: ignore
        cls,
        val: Union[MagnitudeInput, Quantity[DT], str],
        unit: Union[Unit, UnitsContainer, str, Quantity[DT], None] = None,

        # this is a hack to force the type checker to default to Unknown
        # in case the generic type is not specified at all
        _dt: type[DT] = Unknown  # type: ignore
    ) -> Quantity[DT]:

        if unit is None:

            # support passing only string input, this is not recommended
            delimiter = ' '

            if isinstance(val, str) and delimiter in val:

                m_str, u = val.split(delimiter, 1)
                m = float(m_str.strip())
                u = u.strip()
                val = Quantity(m, u)  # type: ignore

            # this allows us to create new dimensionless quantities
            # by omitting the unit
            unit = getattr(val, 'u', None) or ''

        if isinstance(val, Quantity):

            # don't return val.to(unit) directly, since we want to make this
            # the correct dimensional subclass as well
            val = val._convert_magnitude_not_inplace(unit)  # type: ignore

        if isinstance(val, pd.Series):

            # support passing pd.Series directly
            val = val.values

        if isinstance(val, str):
            val = float(val)

        valid_unit = cls._validate_unit(unit)

        if cls._dimensionality_type is None:

            # in case this Quantity was initialized without specifying
            # the dimensionality, check the dimensionality and return the
            # subclass with correct dimensionality
            # the name of the dimensionality class will be the first one
            # defined with this UnitsContainer, i.e. custom dimensions
            # will not override the defaults when creating Quantities with
            # a dynamically determined subclass
            dim = Dimensionality.get_dimensionality(
                valid_unit.dimensionality
            )

            # __new__ will return an instance of this subclass
            # NOTE: the type checker will not be able to infer the class in this case,
            # since it is created dynamically

            subcls = cls._get_dimensional_subclass(dim)
            return subcls(val, valid_unit)

        expected_dimensionality = cls._dimensionality_type

        if valid_unit.dimensionality != expected_dimensionality.dimensions:
            raise ExpectedDimensionalityError(
                f'Quantity with unit "{valid_unit}" has incorrect '
                f'dimensionality {valid_unit.dimensionality}, '
                f'expected {expected_dimensionality.dimensions}'
            )

        # numpy array magnitudes must be copied
        # list input to pint.Quantity.__new__ will be converted to
        # np.ndarray, so there's no danger of modifying lists that are input to Quantity
        if isinstance(val, np.ndarray):
            val = val.copy()

        # at this point the value and dimensionality are verified to be correct
        # pass the inputs to pint to actually construct the Quantity

        # pint.quantity.Quantity uses Generic[_MagnitudeType] in addition to Generic[DT],
        # ignore this type mismatch since the Generic will be overridden
        qty: Quantity[DT] = super().__new__(  # type: ignore
            cls,
            val,
            units=valid_unit
        )

        # avoid casting issues with numpy, use float64 instead of int32
        # it's always possible for the user to change the dtype of the _magnitude attribute
        # in case int32 or similar is necessary
        if isinstance(qty._magnitude, np.ndarray) and qty._magnitude.dtype == np.int32:
            qty._magnitude = qty._magnitude.astype(np.float64)

        return qty

    def _to_unit(self, unit: Union[Unit, UnitsContainer, str, Quantity[DT], dict]) -> Unit:

        return self._validate_unit(unit)

    @property
    def m(self) -> Magnitude:
        return self._magnitude

    def to_reduced_units(self) -> Quantity[DT]:
        return super().to_reduced_units()  # type: ignore

    def to_base_units(self) -> Quantity[DT]:
        return super().to_base_units()  # type: ignore

    def to(self,  # type: ignore[override]
           unit: Union[Unit, UnitsContainer, str, Quantity[DT], dict]) -> Quantity[DT]:

        unit = self._to_unit(unit)
        m = self._convert_magnitude_not_inplace(unit)

        return self.__class__(m, unit)

    def ito(self,  # type: ignore[override]
            unit: Union[Unit, UnitsContainer, str, Quantity[DT]]) -> None:

        unit = self._to_unit(unit)

        # it's not safe to convert units as int, the
        # user will have to convert back to int if necessary
        # better to use ":.0f" formatting or round() anyway

        # avoid numpy.core._exceptions.UFuncTypeError (not on all platforms?)
        # convert integer arrays to float (creating a copy)
        if (
            isinstance(self._magnitude, np.ndarray) and
            issubclass(self._magnitude.dtype.type, numbers.Integral)
        ):

            self._magnitude = self._magnitude.astype(float)

        return super().ito(unit)

    def check(self,
              unit: Union[Quantity, UnitsContainer, Unit,
                          str, Dimensionality, type[Dimensionality]]
              ) -> bool:

        # TODO: fix typing for this method, remove type: ignore

        if isinstance(unit, Quantity):
            unit = unit.dimensionality

        if hasattr(unit, 'dimensions'):
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
            format_type = f'{format_type}{Quantity.default_format}'

        return super().__format__(format_type)

    @staticmethod
    def correct_unit(unit: str) -> str:
        """
        Corrects the unit name to make it compatible with pint.

        * Fixes common misspellings and case-errors (kpa vs. kPa)
        * Adds ``**`` between the unit and the exponent if it's missing, for example ``kg/m3 → kg/m**3``.
        * Parses "Nm³" (and variations of this) as "normal * m³"
        * Converts % and ‰ to percent and permille

        Parameters
        ----------
        unit : str
            A (potentially incorrect) unit name

        Returns
        -------
        str
            The corrected unit name
        """

        unit = str(unit).strip()

        # normal cubic meter, not nano or Newton
        # there's no consistent way of writing normal liter,
        # so we'll not even try to parse that
        for n in ['nm³', 'Nm³', 'nm3', 'Nm3', 'nm**3', 'Nm**3', 'nm^3', 'Nm^3']:
            if n in unit:
                # include brackets, otherwise "kg/nm3" is
                # incorrectly converted to "kg/normal*m3"
                unit = unit.replace(n, '(normal * m³)')

        # replace unicode Δ°C or Δ°F with delta_degC or delta_degF
        unit = re.sub(r'\bΔ\s*°(C|F)\b', r'delta_deg\g<1>', unit)

        # percent & permille sign (pint cannot parse "%" and "‰" characters)
        unit = unit.replace('%', 'percent')
        unit = unit.replace('‰', 'permille')

        # add ** between letters and numbers if they
        # are right next to each other and if the number is at a word boundary
        unit = re.sub(r'([A-Za-z])(\d+)\b', r'\1**\2', unit)

        if unit in Quantity.UNIT_CORRECTIONS:
            unit = Quantity.UNIT_CORRECTIONS[unit]

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
            unit_parts.append(f'{unit_symbol}**{power}')
            symbols.append(unit_symbol)

        unit_repr = ' * '.join(unit_parts)

        if not unit_repr.strip():
            unit_repr = '1'

        # use \text{symbol} to make sure that the unit symbols
        # do not clash with commonly used symbols like "m" or "s"
        expr = sp.sympify(f'{base_qty.m} * {unit_repr}').subs(
            {n: self.get_unit_symbol(n)
             for n in symbols})

        return expr

    @staticmethod
    def get_unit_symbol(s: str) -> sp.Symbol:
        return sp.Symbol('\\text{' + s + '}', nonzero=True, positive=True)

    @classmethod
    def get_dimension_symbol_map(cls) -> dict[sp.Basic, Unit]:

        if cls._dimension_symbol_map is not None:
            return cls._dimension_symbol_map

        # also consider custom dimensions defined with encomp.units.define_dimensionality
        cls._dimension_symbol_map = {
            cls.get_unit_symbol(n): cls.get_unit(n)
            for n in list(_BASE_SI_UNITS) + _CUSTOM_DIMENSIONS
        }

        return cls._dimension_symbol_map

    @classmethod
    def from_expr(cls, expr: sp.Basic) -> Quantity:
        """
        Converts a Sympy expression with unit symbols
        into a Quantity. This only works in case expression
        *only* contains base SI unit symbols.
        """

        _dimension_symbol_map = cls.get_dimension_symbol_map()

        expr = expr.simplify()
        args = expr.args

        if not args:
            return cls(float(expr), 'dimensionless')  # type: ignore

        try:
            magnitude = float(args[0])  # type: ignore
        except TypeError as e:
            raise ValueError(
                f'Expression {expr} contains inconsistent units') from e

        dimensions = args[1:]

        unit = cls.get_unit('')

        for d in dimensions:

            unit_i = cls.get_unit('')

            for symbol, power in d.as_powers_dict().items():  # type: ignore
                unit_i *= _dimension_symbol_map[symbol]**power

            unit *= unit_i

        return cls(magnitude, unit).to_base_units()

    @property
    def dim(self) -> UnitsContainer:
        return self.dimensionality

    @property
    def dimensionality_name(self) -> str:
        return str(self.dimensionality)

    @property
    def dim_name(self) -> str:
        return self.dimensionality_name

    @classmethod
    def __get_validators__(cls):

        # used by pydantic.BaseModel to validate fields
        yield cls.validate

    @classmethod
    def validate(cls, qty: Quantity[DT]) -> Quantity[DT]:

        if not isinstance(qty, Quantity):
            raise TypeError(
                'Expexted instance of Quantity, '
                f'got {qty} ({type(qty)})'
            )

        return cls(qty.m, qty.u)  # type: ignore

    def check_compatibility(self, other: Union[Quantity, MagnitudeScalar]) -> None:

        if not isinstance(other, Quantity):

            if not self.dimensionless:
                raise DimensionalityTypeError(
                    f'Cannot add {other} ({type(other)}) to dimensional '
                    f'quantity {self} ({type(self)})'
                )

            return

        dim = self._dimensionality_type
        other_dim = other._dimensionality_type

        # this never happens at runtime
        if dim is None or other_dim is None:
            raise DimensionalityTypeError(
                f'One or both dimensionalities are None: {type(self)} and {type(other)}'
            )

        # if the dimensionality of self is a subclass of the
        # dimensionality of other or vice versa
        if issubclass(dim, other_dim) or issubclass(other_dim, dim):

            # verify that the actual dimensions also match
            # this is also verified in the Dimensionality.__init_subclass__ method
            if dim.dimensions != other_dim.dimensions:
                raise DimensionalityTypeError(
                    f'Quantities with inherited dimensionalities do not match: '
                    f'{type(self)} and {type(other)} with dimensions '
                    f'{dim.dimensions} and {other_dim.dimensions}'
                )

            else:
                return

        # normal case, check that the types of Quantity is the same
        if not type(self) is type(other):

            # different error message if the actual dimensions match, but not the type
            if self._dimensionality_type is None or other._dimensionality_type is None:
                raise DimensionalityTypeError(
                    f'One or both quantities are missing dimensionalities: '
                    f'{type(self)} and {type(other)}. '
                )

            if self._dimensionality_type.dimensions == other._dimensionality_type.dimensions:
                raise DimensionalityTypeError(
                    f'Quantities with different dimensionalities are not compatible: '
                    f'{type(self)} and {type(other)}. The actual dimensions match, '
                    'but the dimensionalities have different types.'
                )

            raise DimensionalityTypeError(
                f'Quantities with different dimensionalities are not compatible: '
                f'{type(self)} and {type(other)}. '
            )

    @overload
    def __mul__(self: Quantity[Unknown], other) -> Quantity[Unknown]:
        ...

    @overload
    def __mul__(self, other: Quantity[Unknown]) -> Quantity[Unknown]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[Length]  # type: ignore
                ) -> Quantity[Area]:
        ...

    @overload
    def __mul__(self: Quantity[Area], other: Quantity[Length]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[Area]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[MassFlow]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[MassFlow], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[VolumeFlow]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[VolumeFlow], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[NormalVolumeFlow]  # type: ignore
                ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __mul__(self: Quantity[NormalVolumeFlow], other: Quantity[Time]  # type: ignore
                ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __mul__(self: Quantity[Power], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[Power]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Density], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerEnergy], other: Quantity[Energy]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerMass], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerVolume], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerTime], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Energy], other: Quantity[CurrencyPerEnergy]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[CurrencyPerMass]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Volume], other: Quantity[CurrencyPerVolume]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[CurrencyPerTime]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Dimensionless], other: Quantity[DT_]  # type: ignore
                ) -> Quantity[DT_]:
        ...

    @overload
    # type: ignore
    def __mul__(self, other: Quantity[Dimensionless]) -> Quantity[DT]:
        ...

    @overload
    def __mul__(self, other: MagnitudeScalar) -> Quantity[DT]:
        ...

    @overload
    def __mul__(self, other: Quantity[DT_]) -> Quantity[Unknown]:
        ...

    def __mul__(self, other):
        return super().__mul__(other)

    def __rmul__(self, other: MagnitudeScalar  # type: ignore[override]
                 ) -> Quantity[DT]:
        return super().__rmul__(other)

    @overload
    def __truediv__(self: Quantity[Unknown], other: Quantity[Unknown]  # type: ignore
                    ) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self, other: Quantity[DT]  # type: ignore
                    ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __truediv__(self: Quantity[Unknown], other) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self, other: Quantity[Unknown]) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[Area]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Area]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[Area], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[MassFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolume], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[Density]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Mass]  # type: ignore
                    ) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[Power]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Energy]  # type: ignore
                    ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Mass]  # type: ignore
                    ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __truediv__(self: Quantity[Dimensionless], other: Quantity[DT_]  # type: ignore
                    ) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self, other: Quantity[Dimensionless]  # type: ignore
                    ) -> Quantity[DT]:
        ...

    @overload
    def __truediv__(self, other: MagnitudeScalar) -> Quantity[DT]:
        ...

    @overload
    def __truediv__(self, other: Quantity[DT_]) -> Quantity[Unknown]:
        ...

    def __truediv__(self, other):
        return super().__truediv__(other)

    def __rtruediv__(self, other: MagnitudeScalar) -> Quantity[Unknown]:
        return super().__rtruediv__(other)

    @overload
    def __floordiv__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
        ...

    @overload
    def __floordiv__(self, other: Quantity[DT]) -> Quantity[Dimensionless]:
        ...

    def __floordiv__(self, other):
        return super().__floordiv__(other)

    def __rfloordiv__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
        return super().__rfloordiv__(other)

    @overload
    def __pow__(self, other: Literal[1]  # type: ignore
                ) -> Quantity[DT]:
        ...

    @overload
    def __pow__(self: Quantity[Length], other: Literal[2]  # type: ignore
                ) -> Quantity[Area]:
        ...

    @overload
    def __pow__(self: Quantity[Length], other: Literal[3]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __pow__(self: Quantity[Unknown], other: MagnitudeScalar  # type: ignore
                ) -> Quantity[Unknown]:
        ...

    @overload
    def __pow__(self: Quantity[Dimensionless], other: MagnitudeScalar  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __pow__(self, other: Quantity[Dimensionless]) -> Quantity[Unknown]:
        ...

    @overload
    def __pow__(self, other: MagnitudeScalar) -> Quantity[Unknown]:
        ...

    def __pow__(self, other):
        return super().__pow__(other)

    def __rpow__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> MagnitudeScalar:
        return super().__rpow__(other)

    @overload
    def __add__(self: Quantity[Unknown], other) -> Quantity[Impossible]:
        ...

    @overload
    def __add__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
        ...

    @overload
    def __add__(self, other: Quantity[DT]) -> Quantity[DT]:
        ...

    def __add__(self, other):

        self.check_compatibility(other)

        return super().__add__(other)

    def __radd__(self: Quantity[Dimensionless],  # type: ignore[override]
                 other: MagnitudeScalar) -> Quantity[Dimensionless]:

        return super().__radd__(other)

    @overload
    def __sub__(self: Quantity[Unknown], other) -> Quantity[Impossible]:
        ...

    @overload
    def __sub__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
        ...

    @overload
    def __sub__(self, other: Quantity[DT]) -> Quantity[DT]:
        ...

    def __sub__(self, other):

        self.check_compatibility(other)

        return super().__sub__(other)

    def __rsub__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
        return super().__rsub__(other)

    @overload  # type: ignore[override]
    def __gt__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> bool:
        ...

    @overload
    def __gt__(self, other: Quantity[DT]) -> bool:
        ...

    def __gt__(self, other):

        try:
            return super().__gt__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    @overload  # type: ignore[override]
    def __ge__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> bool:
        ...

    @overload
    def __ge__(self, other: Quantity[DT]) -> bool:
        ...

    def __ge__(self, other):

        try:
            return super().__ge__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    @overload  # type: ignore[override]
    def __lt__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> bool:
        ...

    @overload
    def __lt__(self, other: Quantity[DT]) -> bool:
        ...

    def __lt__(self, other):

        try:
            return super().__lt__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    @overload  # type: ignore[override]
    def __le__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> bool:
        ...

    @overload
    def __le__(self, other: Quantity[DT]) -> bool:
        ...

    def __le__(self, other):

        try:
            return super().__le__(other)
        except ValueError as e:
            raise DimensionalityComparisonError(str(e)) from e

    def __eq__(self, other: Any) -> bool:  # type: ignore[override]
        return super().__eq__(other)


# NOTE: to use a shorter name for Quantity, import the class using this name,
# for example: from encomp.units import Quantity as Q

# override the implementation of the Quantity class for the current registry
# this ensures that all Quantity objects created with this registry
# uses the subclass encomp.units.Quantity instead of pint.Quantity
# pint uses an "ApplicationRegistry" wrapper class since v. 0.18,
# account for this by setting the attribute on the "_registry" member
ureg.Quantity = Quantity

try:
    ureg._registry.Quantity = Quantity
except Exception:
    pass


def set_quantity_format(fmt: str = 'compact') -> None:
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

    fmt_aliases = {
        'normal': '~P',
        'siunitx': '~Lx'
    }

    if fmt in fmt_aliases:
        fmt = fmt_aliases[fmt]

    if fmt not in Quantity.FORMATTING_SPECS:
        raise ValueError(f'Cannot set default format to "{fmt}", '
                         f'fmt is one of {Quantity.FORMATTING_SPECS} '
                         'or alias siunitx: ~L, compact: ~P')

    ureg.default_format = fmt

    try:
        ureg._registry.default_format = fmt
    except Exception:
        pass


@overload
def convert_volume_mass(inp: Quantity[Mass],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[Volume]:
    ...


@overload
def convert_volume_mass(inp: Quantity[MassFlow],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[VolumeFlow]:
    ...


@overload
def convert_volume_mass(inp: Quantity[Volume],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[Mass]:
    ...


@overload
def convert_volume_mass(inp: Quantity[VolumeFlow],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[MassFlow]:
    ...


@overload
def convert_volume_mass(inp: Quantity,
                        rho: Optional[Quantity[Density]] = None
                        ) -> Union[Quantity[Mass],
                                   Quantity[MassFlow],
                                   Quantity[Volume],
                                   Quantity[VolumeFlow]
                                   ]:
    ...


def convert_volume_mass(inp, rho=None):
    """
    Converts mass to volume or vice versa.

    Parameters
    ----------
    inp : Union[M, V]
        Input mass or volume (or flow)
    rho : Quantity[Density], optional
        Density, by default 997 kg/m³

    Returns
    -------
    Union[V, M]
        Calculated volume or mass (or flow)
    """

    if rho is None:
        rho = Quantity[Density](997, 'kg/m³')

    if not isinstance(rho, Quantity[Density]):  # type: ignore
        raise TypeError(
            f'Incorrect type for rho: {rho}'
        )

    if isinstance(inp, (Quantity[Mass], Quantity[MassFlow])):  # type: ignore
        return (inp / rho).to_reduced_units()

    elif isinstance(inp, (Quantity[Volume], Quantity[VolumeFlow])):  # type: ignore
        return (inp * rho).to_reduced_units()

    else:
        raise TypeError(
            f'Incorrect input: {inp}'
        )
