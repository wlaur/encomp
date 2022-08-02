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
import copy
import numbers
from typing import Union, Optional, Generic, Union, Any, TypeVar, Literal

import sympy as sp
import numpy as np
import pandas as pd


import pint
from pint.unit import UnitsContainer, Unit
from pint.registry import UnitRegistry, LazyRegistry
from pint.errors import DimensionalityError

from encomp.settings import SETTINGS
from encomp.misc import isinstance_types
from encomp.utypes import (_BASE_SI_UNITS,
                           MagnitudeInput,
                           MagnitudeScalar,
                           Magnitude,
                           DT,
                           Dimensionality,
                           Dimensionless,
                           Temperature,
                           TemperatureDifference,
                           TemperatureDifferenceUnits,
                           Unknown)

if SETTINGS.ignore_ndarray_unit_stripped_warning:
    warnings.filterwarnings(
        'ignore',
        message='The unit of the quantity is stripped when downcasting to ndarray.'
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

    def __init__(self, msg: str = ''):

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
CUSTOM_DIMENSIONS: list[str] = []


_REGISTRY_STATIC_OPTIONS = {

    # if False, degC must be explicitly converted to K when multiplying
    # this is False by default, there's no reason to set this to True
    'autoconvert_offset_to_baseunit': SETTINGS.autoconvert_offset_to_baseunit,

    # if this is True, scalar magnitude inputs will
    # be converted to 1-element arrays
    # tests are written with the assumption that this is False
    'force_ndarray_like': False,
    'force_ndarray': False,

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
        args, kwargs = self.__dict__['params']
        kwargs['on_redefinition'] = 'raise'

        # override the filename
        kwargs['filename'] = str(SETTINGS.units.resolve().absolute())

        self.__class__ = _UnitRegistry
        self.__init__(*args, **kwargs)
        self._after_init()


ureg: UnitRegistry = _LazyRegistry()  # type: ignore

for k, v in _REGISTRY_STATIC_OPTIONS.items():
    setattr(ureg, k, v)

# make sure that ureg is the only registry that can be used
pint._DEFAULT_REGISTRY = ureg  # type: ignore
pint.application_registry.set(ureg)


# enable support for matplotlib axis ticklabels etc...
try:
    ureg.setup_matplotlib()
except ImportError:
    pass

# this option should be changeable
ureg.default_format = SETTINGS.default_unit_format

try:
    ureg._registry.default_format = SETTINGS.default_unit_format
except Exception:
    pass


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
        since the unit needs to be appended to the ``CUSTOM_DIMENSIONS`` list as well.

    Parameters
    ----------
    name : str
        Name of the dimensionality
    symbol : str, optional
        Optional (short) symbol, by default None
    """

    if name in CUSTOM_DIMENSIONS:
        raise DimensionalityRedefinitionError(
            'Cannot define new dimensionality with '
            f'name: {name}, a dimensionality with this name was already defined')

    definition_str = f'{name} = [{name}]'

    if symbol is not None:
        definition_str = f'{definition_str} = {symbol}'

    ureg.define(definition_str)

    CUSTOM_DIMENSIONS.append(name)


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

CUSTOM_DIMENSIONS.append('currency')


def _load_additional_units() -> None:

    with open(SETTINGS.additional_units, 'r', encoding='utf-8') as f:

        for line in f.read().split('\n'):

            if line.startswith('#') or not line.strip():
                continue

            try:
                ureg.define(line)
            except pint.errors.RedefinitionError:
                pass


if (
    SETTINGS.additional_units is not None and
    SETTINGS.additional_units.is_file()
):
    _load_additional_units()


class QuantityMeta(type):

    def __eq__(mcls, obj: object) -> bool:

        # override the == operator so that
        # type(val) == Quantity returns True for subclasses
        if obj is Quantity:
            return True

        return super().__eq__(obj)

    def __hash__(mcls):
        return id(mcls)


class Quantity(pint.quantity.Quantity, Generic[DT], metaclass=QuantityMeta):
    """
    Subclass of ``pint.quantity.Quantity`` with additional functionality
    and integration with other libraries.

    Encodes the output dimensionalities of some common operations,
    for example ``Length**2 -> Area``. This is implemented by overloading the
    ``__mul__, __truediv__, __rtruediv__, __pow__`` methods.


    .. note::

        The overload signatures are defined in a separate file (``units.pyi``)

    """

    # override the _MagnitudeType typevar since this cannot be a Generic
    # only one generic type variable can be used in this class
    _magnitude: Magnitude

    # override ClassVar
    _REGISTRY: UnitRegistry = ureg  # type: ignore

    # mapping from dimensionality subclass name to quantity subclass
    # this dict will be populated at runtime
    # use a custom class attribute (not cls.__subclasses__()) for more control
    _subclasses: dict[str, type[Quantity]] = {}

    # used to validate dimensionality using type checking,
    # if None the dimensionality is not checked
    # subclasses of Quantity have this class attribute set, which
    # will restrict the dimensionality when creating the object
    dimensionality_type: Optional[type[Dimensionality]] = None

    _dimension_symbol_map: Optional[dict[sp.Basic, Unit]] = None

    # compact, Latex, HTML, Latex/siunitx formatting
    FORMATTING_SPECS = ('~P', '~L', '~H', '~Lx')

    # common unit names not supported by pint, also some misspellings
    # TODO: this does not work with compound units, need to use ureg.define
    # to override units at the pint parsing stage
    UNIT_CORRECTIONS = {
        '-': 'dimensionless',
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

    NORMAL_M3_VARIANTS = ('nm³', 'Nm³', 'nm3', 'Nm3',
                          'nm**3', 'Nm**3', 'nm^3', 'Nm^3')

    TEMPERATURE_DIFFERENCE_UCS = (Unit('delta_degC')._units,
                                  Unit('delta_degF')._units)

    # attributes used internally by pint
    __used: bool

    def __hash__(self) -> int:
        return super().__hash__()

    @classmethod
    def _get_dimensional_subclass(cls, dim: Optional[type[Dimensionality]]) -> type[Quantity[DT]]:

        # this never happens at runtime
        if dim is None:
            dim = Dimensionless

        if dim.__name__ in cls._subclasses:
            return cls._subclasses[dim.__name__]

        quantity_name = f'Quantity[{dim.__name__}]'

        DimensionalQuantity = type(
            quantity_name,
            (Quantity,),
            {
                'dimensionality_type': dim,
                '__class__': Quantity,

            }
        )

        cls._subclasses[dim.__name__] = DimensionalQuantity

        return DimensionalQuantity

    def __class_getitem__(cls, dim: type[DT]) -> type[Quantity[DT]]:

        # NOTE: this method is called when defining type hints
        # on Python <3.10 (unless from __future__ import annotations is active)

        # use Unknown as a proxy for type variables
        if isinstance(dim, TypeVar):
            dim = Unknown

        if not isinstance(dim, type):
            raise TypeError(
                'Generic type parameter to Quantity must be a type object, '
                f'passed an instance of {type(dim)}: {dim}'
            )

        # check if the attribute dimensions exists instead of using issubclass()
        if not hasattr(dim, 'dimensions'):
            raise TypeError(
                'Generic type parameter to Quantity has no attribute "dimensions", '
                f'passed: {dim}'
            )

        if not isinstance(dim.dimensions, (UnitsContainer, type(None))):
            raise TypeError(
                'Generic type parameter to Quantity has incorrect type for attribute "dimensions", '
                f'passed: {dim} with dimensions: {dim.dimensions} ({type(dim.dimensions)})'
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

    @property
    def subclass(self) -> type[Quantity[DT]]:
        return self._get_dimensional_subclass(self.dimensionality_type)

    def __len__(self) -> int:

        # __len__() must return an integer
        # the len() function ensures this at a lower level
        if isinstance_types(self._magnitude, MagnitudeScalar):

            raise TypeError(
                f'Quantity with scalar magnitude ({self._magnitude}) has no length'
            )

        elif isinstance(self._magnitude, np.ndarray):
            return len(self._magnitude)

        raise TypeError(
            f'Cannot determine length of {self._magnitude} '
            f'({type(self._magnitude)})'
        )

    def __copy__(self) -> Quantity[DT]:

        ret = self.subclass(
            copy.copy(self._magnitude),
            self._units
        )

        ret.__used = self.__used

        return ret

    def __deepcopy__(self, memo: Optional[dict[int, Any]] = None) -> Quantity[DT]:

        if memo is None:
            memo = {}

        ret = self.subclass(
            copy.deepcopy(self._magnitude, memo),
            copy.deepcopy(self._units, memo)
        )

        ret.__used = self.__used

        return ret

    def __new__(  # type: ignore
        cls,
        val: Union[MagnitudeInput, Quantity[DT], str],
        unit: Union[Unit, UnitsContainer, str, Quantity[DT], None] = None,

        # this is a hack to force the type checker to default to Unknown
        # in case the generic type is not specified at all
        # do not pass the _dt parameter directly, always use square brackets to
        # specify the dimensionality type
        _dt: type[DT] = Unknown  # type: ignore
    ) -> Quantity[DT]:

        if unit is None:

            # nested calls, for example Q(Q(1, 'kg')) are handled
            # separately to preserve the dimensionality type
            # TODO: this creates unnecessary copies of the magnitude
            # this is an issue for large array magnitudes
            if isinstance(val, Quantity) and cls.dimensionality_type is None:
                return val.__deepcopy__()

            # support passing only string input, this is not recommended
            delimiter = ' '

            if isinstance(val, str) and delimiter in val:

                m_str, u = val.split(delimiter, 1)
                m = float(m_str.strip())
                u = u.strip()
                val = Quantity(m, u)  # type: ignore

            unit = getattr(val, 'u', None) or ''

        if isinstance(val, Quantity):

            # don't return val.to(unit) directly, since we want to make this
            # the correct dimensional subclass as well
            val = val._convert_magnitude_not_inplace(unit)  # type: ignore

        if isinstance(val, pd.Series):

            # support passing pd.Series directly
            val = val.values  # type: ignore

        if isinstance(val, str):
            val = float(val)

        valid_unit = cls._validate_unit(unit)

        if cls.dimensionality_type is Unknown:
            raise TypeError(
                'Cannot initialize Quantity with Unknown (or variable) dimensionality, '
                'this dimensionality type can only be used in type hints'
            )

        if cls.dimensionality_type is None:

            # special case for temperature difference

            if valid_unit._units in cls.TEMPERATURE_DIFFERENCE_UCS:
                subcls = cls._get_dimensional_subclass(TemperatureDifference)
                return subcls(val, valid_unit)

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

            # the __new__ method will be called again, as part of the subclass
            subcls = cls._get_dimensional_subclass(dim)

            return subcls(val, valid_unit)

        expected_dimensionality = cls.dimensionality_type

        if expected_dimensionality.dimensions is None:
            raise DimensionalityTypeError(
                'Cannot initialize Quantity with dimensionality '
                f'{expected_dimensionality.__name__}, the "dimensions" attribute is None'
            )

        # compare string representation to avoid issues with float accuracy and hashing
        if str(valid_unit.dimensionality) != str(expected_dimensionality.dimensions):
            raise ExpectedDimensionalityError(
                f'Quantity with unit "{str(valid_unit) or "dimensionless"}" has incorrect '
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

    @property
    def _is_temperature_difference(self):
        return self.dimensionality_type is TemperatureDifference

    def _check_temperature_compatibility(self, unit: Unit) -> None:

        if self._is_temperature_difference:

            if unit._units not in self.TEMPERATURE_DIFFERENCE_UCS:

                raise DimensionalityTypeError(
                    f'Cannot convert {self.units} (dimensionality '
                    f'{self.dimensionality_type.__name__}) '  # type: ignore
                    f'to {unit} (dimensionality '
                    f'{Quantity(1, unit).dimensionality_type.__name__})'
                )

    def to_reduced_units(self) -> Quantity[DT]:

        ret = super().to_reduced_units()
        return self.subclass(ret)  # type: ignore

    def to_base_units(self) -> Quantity[DT]:

        self._check_temperature_compatibility(Unit('kelvin'))

        ret = super().to_base_units()
        return self.subclass(ret)  # type: ignore

    def to(self,  # type: ignore[override]
           unit: Union[Unit, UnitsContainer, str, Quantity[DT], dict]) -> Quantity[DT]:

        unit = self._to_unit(unit)

        self._check_temperature_compatibility(unit)

        m = self._convert_magnitude_not_inplace(unit)

        if unit._units in self.TEMPERATURE_DIFFERENCE_UCS:
            return Quantity[TemperatureDifference](m, unit)  # type: ignore

        return self.subclass(m, unit)

    def ito(self,  # type: ignore[override]
            unit: Union[Unit, UnitsContainer, str, Quantity[DT]]) -> None:

        # NOTE: this method cannot convert the dimensionality type

        unit = self._to_unit(unit)

        self._check_temperature_compatibility(unit)

        # it's not safe to convert units as int, the
        # user will have to convert back to int if necessary
        # better to use ":.0f" formatting or round() anyway

        # avoid numpy.core._exceptions.UFuncTypeError (not on all platforms?)
        # convert integer arrays to float(64) (creating a copy)
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
        * Parses "Nm³" (and variations of this) as "normal * m³" (use explicit "nanometer³" to get this unit)
        * Converts % and ‰ to percent and permille
        * Changes the ``Δ`` character to ``delta_``, for example ``Δ°C`` to ``delta_degC``

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
        # there's no consistent way of abbreviating "normal liter",
        # so we'll not even try to parse that
        # use "nanometer**3" if necessary
        for n in Quantity.NORMAL_M3_VARIANTS:

            if n in unit:

                # include brackets, otherwise "kg/nm3" is
                # incorrectly converted to "kg/normal*m3"
                unit = unit.replace(n, '(normal * m³)')

        # replace unicode Δ°C or Δ°F with delta_degC or delta_degF
        unit = re.sub(r'\bΔ\s*°(C|F)\b', r'delta_deg\g<1>', unit)
        # the ° character is optional
        unit = re.sub(r'\bΔ\s*(C|F)\b', r'delta_deg\g<1>', unit)

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
            for n in list(_BASE_SI_UNITS) + CUSTOM_DIMENSIONS
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

    @classmethod
    def __get_validators__(cls):

        # used by pydantic.BaseModel to validate fields
        yield cls.validate

    @classmethod
    def validate(cls, qty: Union[str, MagnitudeInput, Quantity[DT]]) -> Quantity[DT]:

        if not isinstance_types(qty, Union[str, MagnitudeInput, Quantity]):
            raise TypeError(
                'Expected instance of str, MagnitudeInput or Quantity, '
                f'got {qty} ({type(qty)})'
            )

        return cls(qty)

    def check_compatibility(self, other: Union[Quantity, MagnitudeScalar]) -> None:
        """
        Checks compatibility for addition and subtraction.
        """

        if not isinstance(other, Quantity):

            if not self.dimensionless:
                raise DimensionalityTypeError(
                    f'Cannot add {other} ({type(other)}) to dimensional '
                    f'quantity {self} ({type(self)})'
                )

            return

        dim = self.dimensionality_type
        other_dim = other.dimensionality_type

        # this never happens at runtime
        if dim is None or other_dim is None:
            raise DimensionalityTypeError(
                f'One or both dimensionalities are None: {type(self)} and {type(other)}'
            )

        # if the dimensionality of self is a subclass of the
        # dimensionality of other or vice versa
        if issubclass(dim, other_dim) or issubclass(other_dim, dim):

            # verify that the dimensions also match
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

            # different error message if the dimensions match, but not the type
            if self.dimensionality_type is None or other.dimensionality_type is None:
                raise DimensionalityTypeError(
                    f'One or both quantities are missing dimensionalities: '
                    f'{type(self)} and {type(other)}. '
                )

            if self.dimensionality_type.dimensions == other.dimensionality_type.dimensions:
                raise DimensionalityTypeError(
                    f'Quantities with different dimensionalities are not compatible: '
                    f'{type(self)} and {type(other)}. The dimensions match, '
                    'but the dimensionalities have different types.'
                )

            raise DimensionalityTypeError(
                f'Quantities with different dimensionalities are not compatible: '
                f'{type(self)} and {type(other)}. '
            )

    def is_compatible_with(self, other, *contexts, **ctx_kwargs):

        # add an additional check of the dimensionality types
        is_compatible = super().is_compatible_with(
            other, *contexts, **ctx_kwargs
        )

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
                return is_equal

            return np.zeros_like(is_equal).astype(bool)

        return is_equal and self.is_compatible_with(other)

    def __mul__(self, other):

        ret = super().__mul__(other)

        if isinstance(other, (Quantity, Unit)):
            return ret

        return self.subclass(ret)

    def __rmul__(self, other):

        ret = super().__rmul__(other)

        return self.subclass(ret)

    def __truediv__(self, other):

        ret = super().__truediv__(other)

        if isinstance(other, (Quantity, Unit)):
            return ret

        return self.subclass(ret)

    def _temperature_difference_add_sub(self, other, operator: Literal['add', 'sub']):

        v1 = self.to('degC').m
        v2 = other.to('delta_degC').m

        if operator == 'add':
            val = v1 + v2
        else:
            val = v1 - v2

        return Quantity[Temperature](val, 'degC')

    def __add__(self, other):

        try:
            self.check_compatibility(other)

        except DimensionalityTypeError as e:

            if not isinstance_types(
                    [self, other],
                    list[Union[Quantity[Temperature],
                               Quantity[TemperatureDifference]]]
            ):
                raise e

            if self.dimensionality_type is TemperatureDifference:
                raise e

            return self._temperature_difference_add_sub(other, 'add')

        ret = super().__add__(other)

        return self.subclass(ret)

    def __sub__(self, other):

        try:
            self.check_compatibility(other)
        except DimensionalityTypeError as e:

            if not isinstance_types(
                    [self, other],
                    list[Union[Quantity[Temperature],
                               Quantity[TemperatureDifference]]]
            ):
                raise e

            if self.dimensionality_type is TemperatureDifference:
                raise e

            return self._temperature_difference_add_sub(other, 'sub')

        ret = super().__sub__(other)

        if (
            self.dimensionality_type is Temperature and
            other.dimensionality_type is Temperature
        ):
            return Quantity[TemperatureDifference](ret)

        return self.subclass(ret)

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

    def __round__(self, ndigits: Optional[int] = None) -> Quantity[DT]:

        if isinstance(self.m, np.ndarray):
            return self.__class__(np.round(self.m, ndigits or 0), self.u)

        return super().__round__(ndigits)  # type: ignore

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

    def asdim(self, other):

        if isinstance(other, Quantity):
            dim = other.dimensionality_type
        else:
            dim = other

        return Quantity[dim](self)


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
