"""
Imports and extends the pint library for physical units.
Always import this module when working with ``encomp`` (most other modules
will import this one).

Implements a type-aware system on top of pint that verifies
that the dimensionality of the unit is correct.

.. todo::
    When/if pint implements a typing system (there is an open PR for this),
    this module will have to be rewritten to be compatible with that.

.. note::
    This module will modify the default pint registry -- do not import
    in case the default behavior of the unit registry is expected.
"""

import re
from typing import Union, Type, Tuple, Optional
from contextlib import contextmanager
from functools import lru_cache

import pint
from pint.unit import UnitsContainer, Unit

from encomp.settings import SETTINGS
from encomp.utypes import (Magnitude,
                           _DIMENSIONALITIES_REV,
                           get_dimensionality_name)


class DimensionalityError(ValueError):
    pass


# always use pint.get_application_registry() to get the UnitRegistry instance
# there should only be one registry at a time, pint raises ValueError
# in case quantities from different registries interact
ureg = pint.get_application_registry()

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

ureg.default_format = '~P'  # compact format


class Quantity(pint.quantity.Quantity):

    _REGISTRY = ureg
    default_format = _REGISTRY.default_format

    # used to validate dimensionality using type checking,
    # if None the dimensionality is not checked
    # subclasses of Quantity have this class attribute set, which
    # will restrict the dimensionality when creating the object
    _expected_dimensionality = None

    # valid range for this Quantity, this can be passed with the Annotated type
    _interval = (None, None)

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

    @lru_cache
    def _get_subclass_with_dimensions(dim: UnitsContainer,
                                      interval: Tuple[Optional['Quantity'],
                                                      Optional['Quantity']]) -> Type['Quantity']:

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

    def __class_getitem__(cls,
                          dim: Union[UnitsContainer,
                                     Unit, str, 'Quantity', tuple]) -> Type['Quantity']:

        # possible to restrict the values to an interval with
        # Q[Dim, lower] or
        # Q[Dim, lower, upper]
        # TODO: this does not work with type checking, since it creates a new subclass
        # at the moment it can only be used to check inputs when constructing Quantities
        # might have to use Annotated somehow to get typeguard to check this correctly
        if isinstance(dim, tuple):

            dim, *interval = dim

            if len(interval) == 1:
                interval = (interval[0], None)

            else:
                interval = tuple(interval[:2])

                if interval[1] < interval[0]:
                    raise ValueError(f'Upper limit {interval[1]} cannot be below '
                                     f'lower limit {interval[0]}')

            raise NotImplementedError(
                'TODO: implement intervals for Quantity classes')

        else:
            interval = (None, None)

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

        return cls._get_subclass_with_dimensions(dim, interval)

    def __new__(cls,
                val: Union[Magnitude, 'Quantity'],
                unit: Union[Unit, UnitsContainer, str, 'Quantity']) -> 'Quantity':

        if isinstance(val, Quantity):
            # don't return val.to(unit) directly, since we want to make this
            # the correct dimensional subclass as well
            val = val._convert_magnitude(unit)

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
            DimensionalQuantity = cls._get_subclass_with_dimensions(unit.dimensionality,
                                                                    cls._interval)

            # __new__ will return an instance of this subclass
            return DimensionalQuantity(val, unit)

        expected_dimensionality = cls._expected_dimensionality

        if unit.dimensionality != expected_dimensionality:

            dim_name = get_dimensionality_name(unit.dimensionality)
            expected_name = get_dimensionality_name(expected_dimensionality)

            raise DimensionalityError(f'Quantity with unit "{unit}" has incorrect '
                                      f'dimensionality {dim_name}, '
                                      f'expected {expected_name}')

        # at this point the value and dimensionality are verified to be correct
        # pass the inputs to pint to actually construct the Quantity
        qty = super().__new__(cls, val, units=unit)

        # in case the interval was specified, check it after constructing the Quantity
        # the limits must be specified with the correct units
        lower, upper = cls._interval

        if lower is not None:
            if qty < lower:
                raise ValueError(
                    f'{qty} is below the specified lower limit {lower}')

        if upper is not None:
            if qty > upper:
                raise ValueError(
                    f'{qty} is above the specified upper limit {upper}')

        return qty

    def _to_unit(self, unit: Union[Unit, UnitsContainer, str, 'Quantity']) -> Unit:

        # compatibility with internal pint API, this should maybe be avoided
        if isinstance(unit, UnitsContainer):
            unit = self._REGISTRY.parse_units(str(unit))

        if isinstance(unit, str):
            unit = self._REGISTRY.parse_units(Quantity.correct_unit(unit))

        if isinstance(unit, Quantity):
            unit = unit.u

        return unit

    def to(self, unit: Union[Unit, UnitsContainer, str, 'Quantity']) -> 'Quantity':

        unit = self._to_unit(unit)
        m = self._convert_magnitude(unit)

        return self.__class__(m, unit)

    def ito(self, unit: Union[Unit, UnitsContainer, str, 'Quantity']) -> None:

        unit = self._to_unit(unit)

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


# override the implementation of the Quantity class for the current registry
# this ensures that all Quantity objects created with this registry
# uses the subclass encomp.units.Quantity instead of pint.Quantity
ureg.Quantity = Quantity

# shorthand for the Quantity class
Q = Quantity


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


@contextmanager
def quantity_format(fmt: str = 'compact'):
    """
    Context manager version of :py:func:`encomp.units.set_quantity_format`
    that resets to the previous value afterwards.

    Parameters
    ----------
    fmt : str
        Unit format string: one of ``'~P', '~L', '~H', '~Lx'``.
        Also accepts aliases: ``'compact': '~P'`` and ``'siunitx': '~Lx'``.
    """

    old_fmt = ureg.default_format

    set_quantity_format(fmt)

    try:
        yield
    finally:
        set_quantity_format(old_fmt)
