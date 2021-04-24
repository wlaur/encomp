"""
Imports and extends the pint library for physical units.
Always import this module when working with ``encomp`` (most other modules
will import this one).

Implements a type-aware system on top of pint that verifies
that the dimensionality of the unit is correct.

.. todo::
    When/if pint implements a typing system (there is an open PR for this),
    this module will have to be rewritten.

.. note::
    This module will modify the default pint registry -- do not import
    in case the default behavior of the unit registry is expected.
"""

import re
from typing import Union, Any
from contextlib import contextmanager
from functools import lru_cache

import pint
from pint.unit import UnitsContainer

from encomp.settings import SETTINGS
from encomp.utypes import *
from encomp.utypes import (_DIMENSIONALITIES,
                           _DIMENSIONALITIES_REV,
                           get_dimensionality_name)

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


@lru_cache
def _quantity_factory(dim=None):

    if dim is not None:
        dim_name = get_dimensionality_name(dim)
    else:
        dim_name = 'None'

    class Quantity(pint.quantity.Quantity, TypedQuantity):

        _REGISTRY = ureg
        default_format = _REGISTRY.default_format

        # this name is used when calling type(), also in the typeguard TypeError messages
        __qualname__ = f'Quantity[{dim_name}]'

        # used to validate dimensionality using type checking,
        # if None the dimensionality is not checked
        _expected_dimensionality = dim

        # compact, Latex, HTML, Latex/siunitx formatting
        FORMATTING_SPECS = ('~P', '~L', '~H', '~Lx')

        # common unit names not supported by pint,
        # also some misspellings
        UNIT_CORRECTIONS = {
            'kpa': 'kPa',
            'mpa': 'MPa',
            'pa': 'Pa',
            'bar(a)': 'bar',
            'bara': 'bar',
            'bar-a': 'bar',
            'kPa(a)': 'kPa',
            'kPaa': 'kPa',
            'kPa-a': 'kPa',
            'psi(a)': 'psi',
            'psia': 'psi',
            'psi-a': 'psi',
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
            'ft_water': 'feet_H2O',
            'F': 'degF',
            'C': 'degC',
            '°C': 'degC',
            '°F': 'degF'}

        def __class_getitem__(cls, dim: Union[UnitsContainer, str, 'Quantity', TypeVar, Any]):

            # use same dimensionality as another Quantity
            if hasattr(dim, 'dimensionality'):
                dim = dim.dimensionality

            if isinstance(dim, str):

                dim = dim.replace('[', '').replace(']', '')

                if dim.title() in _DIMENSIONALITIES_REV:
                    dim = _DIMENSIONALITIES_REV[dim.title()]

            if not isinstance(dim, UnitsContainer):
                raise ValueError('Quantity type annotation must be a dimensionality, '
                                 f'passed "{dim}" ({type(dim)})')

            # return a new class definition that restricts the dimensionality
            # these are changed, the class for each dimensionality is only created once
            return _quantity_factory(dim=dim)

        def __new__(cls,
                    val: Union[Magnitude, 'Quantity', Any],
                    unit: Union[Unit, 'Quantity', Any]):  # return type is not known

            if isinstance(val, Quantity):
                return val.to(unit)

            if isinstance(unit, Quantity):
                unit = unit.u

            if isinstance(val, str):
                val = float(val.strip())

            if isinstance(unit, str):
                unit = Quantity.correct_unit(unit)

            qty = super().__new__(cls, val, units=unit)

            # check the dimensionality in case it is specified
            if cls._expected_dimensionality is not None:

                expected_dimensionality = cls._expected_dimensionality

                if qty.dimensionality != expected_dimensionality:

                    dim_name = get_dimensionality_name(qty.dimensionality)
                    expected_dim_name = get_dimensionality_name(
                        expected_dimensionality)

                    raise ValueError(f'Quantity {qty} has incorrect '
                                     f'dimensionality {dim_name}, '
                                     f'expected {expected_dim_name}')

                return qty

            # in case this Quantity was initialized without specifying
            # the dimensionality, check the dimensionality and return an
            # instance of the class with correct dimensionality
            return _quantity_factory(dim=qty.dimensionality)(val, unit)

        def to(self, unit: Unit):

            if isinstance(unit, str):
                unit = Quantity.correct_unit(unit)

            return super().to(unit)

        def ito(self, unit: Unit) -> None:

            if isinstance(unit, str):
                unit = Quantity.correct_unit(unit)

            return super().ito(unit)

        def __format__(self, format_type: str) -> str:
            """
            Overloads the ``__format__`` method for Quantities:
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

        def to_json(self) -> list:
            """
            JSON serialization for a Quantity: 2-element list
            with magnitude and unit (as str).
            The first list element might be a sequence (list or ``np.array``).

            Returns
            -------
            dict
                JSON representation, 2-element list with ``[val, unit]``
            """

            return [self.m, str(self.u)]

        @staticmethod
        def correct_unit(unit: str) -> str:
            """
            Corrects the unit name to make it compatible with pint.

            * Fixes some common misspellings
            * Removes "(abs)" or "(a)" and similar suffixes from pressure units
            *  Adds ``**`` between the unit and the exponent if it's missing, for example ``kg/m3 → kg/m**3``.
            * Replaces ``h`` with ``hr`` (hour), since ``pint`` interprets ``h`` as the Planck constant. Use ``planck_constant`` to get this value if necessary.

            Parameters
            ----------
            unit : str
                The input unit name, potentially not correct for use with ``pint``

            Returns
            -------
            str
                The corrected unit name, used by `pint`
            """

            unit = str(unit).strip()

            # replace h with hr, pint interprets h as Planck constant
            # h is the SI symbol for hour, should be supported
            # use planck_constant to get the value for this constant
            unit = re.sub(r'(\bh\b)', 'hr', unit)

            # replace unicode Δ°C or Δ°F with delta_degC or delta_degF
            unit = re.sub(r'\bΔ\s*°(C|F)\b', r'delta_deg\g<1>', unit)

            # add ** between letters and numbers if they
            # are right next to each other and if the number is at a word boundary
            unit = re.sub(r'([A-Za-z])(\d+)\b', r'\1**\2', unit)

            if unit in Quantity.UNIT_CORRECTIONS:
                unit = Quantity.UNIT_CORRECTIONS[unit]

            return unit

    return Quantity


# this is a Quantity class that does not restrict the dimensionality
Quantity = _quantity_factory()

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

    if fmt not in Quantity.formatting_specs:
        raise ValueError(f'Cannot set default format to {fmt}, '
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
