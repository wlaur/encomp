"""
Imports and extends the ``pint`` library for physical units.
Always import this module when working with ``encomp`` (most other modules
will import this one).

Implements a type-aware system on top of ``pint`` that verifies
that the dimensionality of the unit is correct.

.. todo::
    When/if ``pint`` implements a typing system (there is an open PR for this),
    this module will have to be rewritten to be compatible with that.

.. note::
    This module will modify the default ``pint`` registry -- do not import
    in case the default behavior of the unit registry is expected.
"""

import re
import warnings
import numbers
from typing import Union, Type, Optional
from functools import lru_cache
import sympy as sp
import numpy as np
import pandas as pd

import pint
from pint.unit import UnitsContainer, Unit, UnitDefinition
from pint.converters import ScaleConverter

from encomp.settings import SETTINGS
from encomp.utypes import (Magnitude,
                           _DIMENSIONALITIES_REV,
                           _BASE_SI_UNITS,
                           Density,
                           Mass,
                           MassFlow,
                           Volume,
                           VolumeFlow,
                           get_dimensionality_name)

if SETTINGS.ignore_ndarray_unit_stripped_warning:
    warnings.filterwarnings('ignore',
                            message='The unit of the quantity is stripped when downcasting to ndarray.')


class DimensionalityError(ValueError):
    pass


class DimensionalityRedefinitionError(ValueError):
    pass


# always use pint.get_application_registry() to get the UnitRegistry instance
# there should only be one registry at a time, pint raises ValueError
# in case quantities from different registries interact
ureg = pint.get_application_registry()


# keep track of user-created dimensions
_CUSTOM_DIMENSIONS = []

# TODO: move the preprocessing from the Quantity class here instead
ureg.preprocessors = [
    lambda x: x.replace('%', 'percent')
]

# if False, degC must be explicitly converted to K when multiplying
ureg.autoconvert_offset_to_baseunit = SETTINGS.autoconvert_offset_to_baseunit

# enable support for matplotlib axis ticklabels etc...
ureg.setup_matplotlib()


try:
    # define percent as 1 / 100
    ureg.define(UnitDefinition('percent', '%',
                (), ScaleConverter(1 / 100.0)))
except pint.errors.RedefinitionError:
    pass


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

ureg.default_format = SETTINGS.default_unit_format


class Quantity(pint.quantity.Quantity):
    """
    Subclass of ``pint.quantity.Quantity`` with additional functionality
    and integration with other libraries.
    """

    _REGISTRY = ureg
    default_format = _REGISTRY.default_format

    # used to validate dimensionality using type checking,
    # if None the dimensionality is not checked
    # subclasses of Quantity have this class attribute set, which
    # will restrict the dimensionality when creating the object
    _expected_dimensionality = None

    # compact, Latex, HTML, Latex/siunitx formatting
    FORMATTING_SPECS = ('~P', '~L', '~H', '~Lx')

    # common unit names not supported by pint, also some misspellings
    UNIT_CORRECTIONS = {
        'kpa': 'kPa',
        'mpa': 'MPa',
        'pa': 'Pa',
        'F': 'degF',
        'C': 'degC',
        '°C': 'degC',
        '°F': 'degF',
        'h': 'hour',
        'km/h': 'km/hour',
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
        'ft_water': 'feet_H2O'}

    @classmethod
    def get_unit(cls, unit_name: str) -> Unit:
        return cls._REGISTRY.parse_units(unit_name)

    @lru_cache()
    def _get_subclass_with_dimensions(dim: UnitsContainer) -> Type['Quantity']:

        if dim is not None:
            dim_name = get_dimensionality_name(dim)
        else:
            dim_name = 'None'

        # return a new subclass definition that restricts the dimensionality
        # the parent class is Quantity (this class does not restrict dimensionality)
        # also override the __class__ attribute so that the internal pint API (__mul__, __div___, etc...)
        # returns a Quantity object that does not restrict the dimensionality (since it will change
        # when dividing or multiplying quantities with each other)
        # the created Quantity will later be converted to the correct dimensional subclass
        # when __new__() is evaluated
        DimensionalQuantity = type(f'Quantity[{dim_name}]',
                                   (Quantity,),
                                   {'_expected_dimensionality': dim,
                                    '__class__': Quantity})

        return DimensionalQuantity

    def __class_getitem__(cls, dim: Union[UnitsContainer, Unit, str, 'Quantity']) -> Type['Quantity']:

        # use same dimensionality as another Quantity or Unit
        if isinstance(dim, (Quantity, Unit)):
            dim = dim.dimensionality

        if isinstance(dim, str):

            # pint also uses empty string to represent dimensionless
            if dim == '':
                dim = 'Dimensionless'

            # this is case-sensitive
            if dim in _DIMENSIONALITIES_REV:
                dim = _DIMENSIONALITIES_REV[dim]
            else:
                raise ValueError(f'Dimension {dim} is not defined')

        if not isinstance(dim, UnitsContainer):
            raise ValueError('Quantity type annotation must be a dimensionality, '
                             f'passed "{dim}" ({type(dim)})')

        return cls._get_subclass_with_dimensions(dim)

    def __new__(cls,
                val: Union[Magnitude, 'Quantity'],
                unit: Union[Unit, UnitsContainer, str, 'Quantity']) -> 'Quantity':

        if isinstance(val, Quantity):

            # don't return val.to(unit) directly, since we want to make this
            # the correct dimensional subclass as well
            val = val._convert_magnitude_not_inplace(unit)

        if isinstance(val, pd.Series):

            # support passing pd.Series directly
            val = val.values

        # pint.Quantity.to_root_units calls __class__(magnitude, other)
        # where other is a UnitsContainer
        if isinstance(unit, UnitsContainer):
            unit = cls._REGISTRY.parse_units(str(unit))

        if isinstance(unit, Quantity):
            unit = unit.u

        if isinstance(unit, str):
            unit = cls._REGISTRY.parse_units(Quantity.correct_unit(unit))

        if cls._expected_dimensionality is None:

            # in case this Quantity was initialized without specifying
            # the dimensionality, check the dimensionality and return the
            # subclass with correct dimensionality
            DimensionalQuantity = cls._get_subclass_with_dimensions(
                unit.dimensionality)

            # __new__ will return an instance of this subclass
            return DimensionalQuantity(val, unit)

        expected_dimensionality = cls._expected_dimensionality

        if unit.dimensionality != expected_dimensionality:

            dim_name = get_dimensionality_name(unit.dimensionality)
            expected_name = get_dimensionality_name(expected_dimensionality)

            raise DimensionalityError(f'Quantity with unit "{unit}" has incorrect '
                                      f'dimensionality {dim_name}, '
                                      f'expected {expected_name}')

        # numpy array magnitudes must be copied, otherwise they will
        # be changed for the original object as well
        # list input to pint.Quantity.__new__ will be convert to
        # np.ndarray, so there's no danger of modifying lists that are input to Quantity
        if isinstance(val, np.ndarray):
            val = val.copy()

        # at this point the value and dimensionality are verified to be correct
        # pass the inputs to pint to actually construct the Quantity
        qty = super().__new__(cls, val, units=unit)

        # avoid casting issues with numpy, use float64 instead of int32
        # it's always possible for the user to change the dtype of the _magnitude attribute
        # in case int32 or similar is necessary (unlikely, fast calculations should not use pint at all)
        if isinstance(qty._magnitude, np.ndarray) and qty._magnitude.dtype == np.int32:
            qty._magnitude = qty._magnitude.astype(np.float64)

        return qty

    def _to_unit(self, unit: Union[Unit, UnitsContainer, str, 'Quantity']) -> Unit:

        # compatibility with internal pint API
        if isinstance(unit, UnitsContainer):
            unit = self._REGISTRY.parse_units(str(unit))

        if isinstance(unit, str):
            unit = self._REGISTRY.parse_units(Quantity.correct_unit(unit))

        if isinstance(unit, Quantity):
            unit = unit.u

        return unit

    def to(self, unit: Union[Unit, UnitsContainer, str, 'Quantity']) -> 'Quantity':

        unit = self._to_unit(unit)
        m = self._convert_magnitude_not_inplace(unit)

        return self.__class__(m, unit)

    def ito(self, unit: Union[Unit, UnitsContainer, str, 'Quantity']) -> None:

        unit = self._to_unit(unit)

        # it's not safe to convert units as int, the
        # user will have to convert back to int if necessary
        # better to use ":.0f" formatting or round() anyway

        # avoid numpy.core._exceptions.UFuncTypeError (not on all platforms?)
        # convert integer arrays to float (creating a copy)
        if (isinstance(self._magnitude, np.ndarray) and
                issubclass(self._magnitude.dtype.type, numbers.Integral)):

            self._magnitude = self._magnitude.astype(float)

        return super().ito(unit)

    def __format__(self, format_type: str) -> str:
        """
        Overloads the ``__format__`` method for Quantity:
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
        * Replaces ``h`` with ``hr`` (hour), since ``pint`` interprets ``h`` as the Planck constant.
          Use ``planck_constant`` to get this value if necessary.

        Parameters
        ----------
        unit : str
            A (potentially incorrect) unit name

        Returns
        -------
        str
            The corrected unit name, compatible with ``pint``
        """

        unit = str(unit).strip()

        # replace h with hr, pint interprets h as Planck constant
        # h is the SI symbol for hour, should be supported
        # use "planck_constant" to get the value for this constant
        unit = re.sub(r'(\bh\b)', 'hr', unit)

        # replace unicode Δ°C or Δ°F with delta_degC or delta_degF
        unit = re.sub(r'\bΔ\s*°(C|F)\b', r'delta_deg\g<1>', unit)

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
            return sp.sympify(self.m)

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

        if hasattr(cls, '_dimension_symbol_map') and cls._dimension_symbol_map is not None:
            return cls._dimension_symbol_map

        # also consider custom dimensions defined with encomp.units.define_dimensionality
        cls._dimension_symbol_map = {cls.get_unit_symbol(n): cls.get_unit(n)
                                     for n in list(_BASE_SI_UNITS) + _CUSTOM_DIMENSIONS}

        return cls._dimension_symbol_map

    @classmethod
    def from_expr(cls, expr: sp.Basic) -> 'Quantity':
        """
        Converts a Sympy expression with unit symbols
        into a Quantity. This only works in case expression
        *only* contains base SI unit symbols.
        """

        _dimension_symbol_map = cls.get_dimension_symbol_map()

        expr = expr.simplify()
        args = expr.args

        if not args:
            return cls(float(expr), 'dimensionless')

        try:
            magnitude = float(args[0])
        except TypeError as e:
            raise ValueError(
                f'Expression {expr} contains inconsistent units') from e

        dimensions = args[1:]

        unit = cls.get_unit('')

        for d in dimensions:

            unit_i = cls.get_unit('')

            for symbol, power in d.as_powers_dict().items():
                unit_i *= _dimension_symbol_map[symbol]**power

            unit *= unit_i

        return cls(magnitude, unit).to_base_units()

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, qty: 'Quantity') -> 'Quantity':
        return cls(qty.m, qty.u)


# override the implementation of the Quantity class for the current registry
# this ensures that all Quantity objects created with this registry
# uses the subclass encomp.units.Quantity instead of pint.Quantity
ureg.Quantity = Quantity

# shorthand for the Quantity class
Q = Quantity

# shorthand for the @wraps(ret, args, strict=True|False) decorator
wraps = ureg.wraps


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


def convert_volume_mass(inp: Union[Quantity[Mass],
                                   Quantity[MassFlow],
                                   Quantity[Volume],
                                   Quantity[VolumeFlow]],
                        rho: Optional[Quantity[Density]] = None) -> Union[Quantity[Mass],
                                                                          Quantity[MassFlow],
                                                                          Quantity[Volume],
                                                                          Quantity[VolumeFlow]]:
    """
    Converts mass to volume or vice versa.

    Parameters
    ----------
    inp : Union[Quantity[Mass], Quantity[MassFlow], Quantity[Volume], Quantity[VolumeFlow]]
        Input mass or volume
    rho : Quantity[Density], optional
        Density, by default 997 kg/m³ (or ``encomp.constants.CONTSTANTS.default_density``)

    Returns
    -------
    Union[Quantity[Mass], Quantity[MassFlow], Quantity[Volume], Quantity[VolumeFlow]]
        The input converted to mass or volume
    """

    if rho is None:

        # TODO: possible to avoid issues with circular imports?
        from encomp.constants import CONSTANTS
        rho = CONSTANTS.default_density

    if inp.dimensionality in (Mass, MassFlow):
        return (inp / rho).to_reduced_units()

    else:
        return (inp * rho).to_reduced_units()
