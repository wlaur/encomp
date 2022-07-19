"""
Contains type definitions for :py:class:`encomp.units.Quantity` objects.

The dimensionalities defined in this module can be combined with ``*`` and ``/``.
Some commonly used derived dimensionalities (like density) are defined for convenience.

This module can be star-imported, variables prefixed with ``_`` will not
be imported in this case.
"""

from __future__ import annotations

from typing import TypeVar, Union, Optional
from typing import Literal as L
from typing import _LiteralGenericAlias  # type: ignore

from abc import ABC


import numpy as np
import pandas as pd
from pint.unit import UnitsContainer


MagnitudeScalar = Union[float, int]

MagnitudeInput = Union[
    MagnitudeScalar,
    list[MagnitudeScalar],
    tuple[MagnitudeScalar, ...],
    np.ndarray,
    pd.Series
]

Magnitude = Union[MagnitudeScalar, np.ndarray]

_BASE_SI_UNITS = ('m', 'kg', 's', 'K', 'mol', 'A', 'cd')

# these string literals are used to infer the dimensionality of commonly created quantities
# they are only used by type checkers and ignored at runtime
DimensionlessUnits = L['', '%', '-', 'dimensionless', None]

CurrencyUnits = L[
    'SEK', 'EUR', 'USD',
    'kSEK', 'kEUR', 'kUSD',
    'MSEK', 'MEUR', 'MUSD'
]

CurrencyPerEnergyUnits = L[
    'SEK/MWh', 'EUR/MWh',
    'SEK/kWh', 'EUR/kWh',
    'SEK/GWh', 'EUR/GWh',
    'SEK/TWh', 'EUR/TWh'
]

CurrencyPerMassUnits = L[
    'SEK/kg', 'EUR/kg',
    'SEK/t', 'EUR/t',
    'SEK/ton', 'EUR/ton',
    'SEK/g', 'EUR/g',
    'SEK/mg', 'EUR/mg',
    'SEK/ug', 'EUR/ug'
]

CurrencyPerVolumeUnits = L[
    'SEK/L', 'EUR/L',
    'SEK/l', 'EUR/l',
    'SEK/liter', 'EUR/liter',
    'SEK/m3', 'EUR/m3',
    'SEK/m^3', 'EUR/m^3',
    'SEK/m**3', 'EUR/m**3',
    'SEK/m³', 'EUR/m³'
]

CurrencyPerTimeUnits = L[
    'SEK/h', 'EUR/h',  'SEK/hr', 'EUR/hr',
    'SEK/hour', 'EUR/hour', 'SEK/d', 'EUR/d',
    'SEK/day', 'EUR/day', 'SEK/w', 'EUR/w',
    'SEK/week', 'EUR/week', 'SEK/y', 'EUR/y',
    'SEK/yr', 'EUR/yr', 'SEK/year', 'EUR/year',
    'SEK/a', 'EUR/a'
]


LengthUnits = L['m', 'meter', 'km', 'cm', 'mm', 'um']

MassUnits = L['kg', 'g', 'ton', 'tonne', 't', 'mg', 'ug']

TimeUnits = L[
    's', 'second', 'min',
    'minute', 'h', 'hr',
    'hour', 'd', 'day',
    'w', 'week', 'y', 'yr',
    'a', 'year', 'ms', 'us'
]

TemperatureUnits = L[
    'C', 'degC', '°C', 'K',
    'F', 'degF', '°F',
    'delta_C', 'delta_degC', 'Δ°C', 'ΔC',
    'delta_F', 'delta_degF', 'Δ°F', 'ΔF'
]

SubstanceUnits = L['mol', 'kmol']

CurrentUnits = L['A', 'mA']

LuminosityUnits = L['lm']

AreaUnits = L[
    'm2', 'm^2', 'm**2', 'm²',
    'cm2', 'cm^2', 'cm**2', 'cm²'
]

VolumeUnits = L[
    'L', 'l', 'liter',
    'm3', 'm^3', 'm³', 'm**3',
    'dm3', 'dm^3', 'dm³', 'dm**3',
    'cm3', 'cm^3', 'cm³', 'cm**3'
]

NormalVolumeUnits = L[
    'normal liter', 'Nm3',
    'nm3', 'Nm^3', 'nm^3', 'Nm³',
    'nm³', 'Nm**3', 'nm**3'
]

PressureUnits = L['bar', 'kPa', 'Pa', 'MPa', 'mbar', 'mmHg']

MassFlowUnits = L[
    'kg/s', 'kg/h', 'kg/hr',
    'g/s', 'g/h', 'g/hr',
    'ton/h', 't/h', 'ton/hr',
    't/hr', 't/d', 'ton/day',
    't/w', 'ton/week', 't/y',
    't/a', 't/year', 'ton/y',
    'ton/a', 'ton/year'
]

VolumeFlowUnits = L[
    'm3/s', 'm3/h', 'm3/hr',
    'm**3/s', 'm**3/h', 'm**3/hr',
    'm^3/s', 'm^3/h', 'm^3/hr',
    'm³/s', 'm³/h', 'm³/hr',
    'liter/second', 'l/s', 'L/s',
    'liter/hour', 'l/h', 'L/h',
    'L/hr', 'l/hr'
]

NormalVolumeFlowUnits = L[
    'Nm3/s', 'Nm3/h', 'Nm3/hr',
    'nm3/s', 'nm3/h', 'nm3/hr',
    'Nm^3/s', 'Nm^3/h', 'Nm^3/hr',
    'nm^3/s', 'nm^3/h', 'nm^3/hr',
    'Nm³/s', 'Nm³/h', 'Nm³/hr',
    'nm³/s', 'nm³/h', 'nm³/hr',
    'Nm**3/s', 'Nm**3/h', 'Nm**3/hr',
    'nm**3/s', 'nm**3/h', 'nm**3/hr'
]

DensityUnits = L[
    'kg/m3', 'kg/m**3',
    'kg/m^3', 'kg/m³',
    'kg/liter', 'g/l',
    'g/L', 'gram/liter'
]

SpecificVolumeUnits = L[
    'm3/kg', 'm^3/kg', 'm³/kg',
    'l/g', 'L/g'
]

EnergyUnits = L[
    'J', 'kJ', 'MJ',
    'GJ', 'TJ', 'PJ',
    'kWh', 'MWh', 'Wh',
    'GWh', 'TWh'
]

PowerUnits = L[
    'W', 'kW', 'MW', 'GW', 'TW', 'mW',
    'kWh/d', 'kWh/w', 'kWh/y', 'kWh/yr', 'kWh/year',
    'MWh/d', 'MWh/w', 'MWh/y', 'MWh/yr', 'MWh/year',
    'GWh/d', 'GWh/w', 'GWh/y', 'GWh/yr', 'GWh/year',
    'TWh/d', 'TWh/w', 'TWh/y', 'TWh/yr', 'TWh/year'
]

VelocityUnits = L[
    'm/s', 'km/s', 'm/min',
    'cm/s', 'cm/min',
    'km/h', 'kmh', 'kph'
]

DynamicViscosityUnits = L['Pa*s', 'Pa s', 'cP']

KinematicViscosityUnits = L[
    'm2/s', 'm**2/s', 'm^2/s',
    'm²/s', 'cSt', 'cm2/s',
    'cm**2/s', 'cm^2/s', 'cm²/s'
]

HeatingValueUnits = L[
    'MJ/kg', 'MWh/kg', 'kJ/kg', 'kWh/kg',
    'MJ/t', 'MWh/t', 'kJ/t', 'kWh/t',
    'MJ/ton', 'MWh/ton', 'kJ/ton', 'kWh/ton'
]


def get_registered_units() -> dict[str, tuple[str, ...]]:

    ret = {}

    for k, v in globals().items():
        if isinstance(v, _LiteralGenericAlias):
            if k.endswith('Units'):
                ret[k.removesuffix('Units')] = v.__args__

    return ret


class Dimensionality(ABC):
    r"""
    Represents the *dimensionality* of a unit, i.e.
    a combination (product) of the base dimensions (with optional rational exponents).

    A dimension ca be expressed as

    .. math::

        \Pi \, d^n_d, d \in \{T, L, M ,I, \Theta, N, J, \ldots\}, n_d \in \mathbb{Q}

    where $\{T, L, M, ...\}$ are the base dimensions (time, length, mass, ...) and $n_d$ is a rational number.

    Subclasses of this abstract base class are used
    as type parameters when creating instances of
    :py:class:`encomp.units.Quantity`.

    The ``dimensions`` class attribute defines the dimensions
    of the dimensionality using an instance of
    ``pint.unit.UnitsContainer``.
    """

    # set _distinct to False for dimensionalities that are not distinct
    # purely based on the dimensions. For example, [energy] / [mass]
    # might mean LowerHeatingValue, UpperHeatingValue, ...
    # and the dimensions alone are not sufficient to determine what
    # the dimensionality should be
    _distinct: Optional[bool] = None

    dimensions: Optional[UnitsContainer] = None

    # keeps track of all the dimensionalities that have been
    # used in the current process
    # use the class definition as key, since multiple dimensionalities
    # might have the same UnitsContainer
    _registry: dict[type[Dimensionality], UnitsContainer] = {}

    # also store a reversed map, this might not contain all items in _registry
    # dimensionalities where is_distinct() returns True will have precedence
    _registry_reversed: dict[UnitsContainer, type[Dimensionality]] = {}

    def __init_subclass__(cls) -> None:

        if not hasattr(cls, 'dimensions'):
            raise AttributeError(
                'Subtypes of Dimensionality must define the '
                'attribute "dimensions" (an instance of pint.unit.UnitsContainer)'
            )

        # the Unknown dimensionality subclass has dimensions=None
        # it will never be used at runtime, only during type checking
        if cls.dimensions is None:
            return

        if not isinstance(cls.dimensions, UnitsContainer):
            raise TypeError(
                'The "dimensions" attribute of the Dimensionality type '
                'must be an instance of pint.unit.UnitsContainer, '
                f'passed {cls} with dimensions: {cls.dimensions} ({type(cls.dimensions)})'
            )

        # make sure a subclass of an existing Dimensionality has the same dimensions
        # the first element in __mro__ is the class that is being created, the
        # second is the direct parent class
        # parent must be either Dimensionality or a subclass
        parent: type[Dimensionality] = cls.__mro__[1]

        if parent.dimensions is not None:
            if parent.dimensions != cls.dimensions:
                raise TypeError(
                    f'Cannot create subclass of {parent} where '
                    'the dimensions do not match. Tried to '
                    f'create subclass with {cls.dimensions} but '
                    f'the parent has dimensions {parent.dimensions}'
                )

        # NOTE: dict keys are class definitions, not class names
        # for example, re-running a notebook cell will
        # create a new class each time, which will have its own entry in
        # the dimensionality registry
        if cls in cls._registry:
            return

        cls._registry[cls] = cls.dimensions

        # unless specifically overridden with _distinct,
        # this will be True only for the first subtype with specific dimensions
        if cls.is_distinct():
            cls._registry_reversed[cls.dimensions] = cls

    @classmethod
    def get_dimensionality(cls, dimensions: UnitsContainer) -> type[Dimensionality]:

        if dimensions in cls._registry_reversed:
            return cls._registry_reversed[dimensions]

        # create a new, custom Dimensionality
        # not possible to generate a proper name for this,
        # so it will just contain the literal dimensions
        # this will call __init_subclass__ to register the type
        _Dimensionality = type(
            f'Dimensionality[{dimensions}]',
            (Dimensionality,),
            {
                'dimensions': dimensions
            }
        )

        return _Dimensionality

    @classmethod
    def is_distinct(cls) -> bool:

        if cls._distinct is None:

            if cls.dimensions is None:
                return True

            ucs = list(cls._registry.values())

            # NOTE: the output of this classmethod
            # might change when the registry is updated
            return ucs.count(cls.dimensions) == 1

        return cls._distinct


# type variables that represent a certain dimensionality
# the DT_ type is used to signify a different dimensionality than DT
# NOTE: the DT/DT_ type variables will represent an instance of DT when used
# as type hints (e.g. def func(...) -> DT)
# a function that returns a class definition/type should be annotated as
# func(...) -> type[DT]
# however, when used as type parameters, they do not represent instances
# e.g. Quantity[DT] means a subclass of Quantity with dimensionality type DT,
# the dimensionality is not an instance of DT
DT = TypeVar('DT', bound=Dimensionality)
DT_ = TypeVar('DT_', bound=Dimensionality)

_DimensionlessUC = UnitsContainer({})
_CurrencyUC = UnitsContainer({'[currency]': 1})
_NormalUC = UnitsContainer({'[normal]': 1})
_LengthUC = UnitsContainer({'[length]': 1})
_MassUC = UnitsContainer({'[mass]': 1})
_TimeUC = UnitsContainer({'[time]': 1})
_TemperatureUC = UnitsContainer({'[temperature]': 1})
_SubstanceUC = UnitsContainer({'[substance]': 1})
_CurrentUC = UnitsContainer({'[current]': 1})
_LuminosityUC = UnitsContainer({'[luminosity]': 1})


# NOTE: each subclass definition will create an entry in Dimensionality._registry
# reloading (re-importing) this module will clear and reset the registry

class Unknown(Dimensionality):
    dimensions = None


class Dimensionless(Dimensionality):
    dimensions = _DimensionlessUC


class Normal(Dimensionality):
    dimensions = _NormalUC


class Length(Dimensionality):
    dimensions = _LengthUC


class Mass(Dimensionality):
    dimensions = _MassUC


class Time(Dimensionality):
    dimensions = _TimeUC


class Temperature(Dimensionality):
    dimensions = _TemperatureUC


class Substance(Dimensionality):
    dimensions = _SubstanceUC


class Current(Dimensionality):
    dimensions = _CurrentUC


class Luminosity(Dimensionality):
    dimensions = _LuminosityUC


# derived dimensionalities
_AreaUC = _LengthUC**2
_VolumeUC = _LengthUC**3
_NormalVolumeUC = _VolumeUC * _NormalUC
_PressureUC = _MassUC / _LengthUC / _TimeUC**2
_MassFlowUC = _MassUC / _TimeUC
_VolumeFlowUC = _VolumeUC / _TimeUC
_NormalVolumeFlowUC = _NormalVolumeUC / _TimeUC
_DensityUC = _MassUC / _VolumeUC
_SpecificVolumeUC = 1 / _DensityUC
_EnergyUC = _MassUC * _LengthUC**2 / _TimeUC**2
_PowerUC = _EnergyUC / _TimeUC
_VelocityUC = _LengthUC / _TimeUC
_DynamicViscosityUC = _MassUC / _LengthUC / _TimeUC
_KinematicViscosityUC = _LengthUC**2 / _TimeUC
_FrequencyUC = 1 / _TimeUC
_MolarMassUC = _MassUC / _SubstanceUC
_MolarDensityUC = _SubstanceUC / _VolumeUC
_CurrencyPerEnergyUC = _CurrencyUC / _EnergyUC
_CurrencyPerMassUC = _CurrencyUC / _MassUC
_CurrencyPerVolumeUC = _CurrencyUC / _VolumeUC
_CurrencyPerTimeUC = _CurrencyUC / _TimeUC
_PowerPerLengthUC = _PowerUC / _LengthUC
_PowerPerAreaUC = _PowerUC / _AreaUC
_PowerPerVolumeUC = _PowerUC / _VolumeUC
_PowerPerTemperatureC = _PowerUC / _TemperatureUC
_ThermalConductivityUC = _PowerUC / _LengthUC / _TemperatureUC
_HeatTransferCoefficientUC = _PowerUC / _AreaUC / _TemperatureUC
_MassPerNormalVolumeUC = _MassUC / _NormalVolumeUC
_MassPerEnergyUC = _MassUC / _EnergyUC
_MolarSpecificEntropyUC = _EnergyUC / _MassUC / _SubstanceUC


class Area(Dimensionality):
    dimensions = _AreaUC


class Volume(Dimensionality):
    dimensions = _VolumeUC


class NormalVolume(Dimensionality):
    dimensions = _NormalVolumeUC


class Pressure(Dimensionality):
    dimensions = _PressureUC


class MassFlow(Dimensionality):
    dimensions = _MassFlowUC


class VolumeFlow(Dimensionality):
    dimensions = _VolumeFlowUC


class NormalVolumeFlow(Dimensionality):
    dimensions = _NormalVolumeFlowUC


class Density(Dimensionality):
    dimensions = _DensityUC


class SpecificVolume(Dimensionality):
    dimensions = _SpecificVolumeUC


class Energy(Dimensionality):
    dimensions = _EnergyUC


class Power(Dimensionality):
    dimensions = _PowerUC


class Velocity(Dimensionality):
    dimensions = _VelocityUC


class DynamicViscosity(Dimensionality):
    dimensions = _DynamicViscosityUC


class KinematicViscosity(Dimensionality):
    dimensions = _KinematicViscosityUC


class Frequency(Dimensionality):
    dimensions = _FrequencyUC


class MolarMass(Dimensionality):
    dimensions = _MolarMassUC


class MolarDensity(Dimensionality):
    dimensions = _MolarDensityUC


class Currency(Dimensionality):
    dimensions = _CurrencyUC


class CurrencyPerEnergy(Dimensionality):
    dimensions = _CurrencyPerEnergyUC


class CurrencyPerMass(Dimensionality):
    dimensions = _CurrencyPerMassUC


class CurrencyPerVolume(Dimensionality):
    dimensions = _CurrencyPerVolumeUC


class CurrencyPerTime(Dimensionality):
    dimensions = _CurrencyPerTimeUC


class PowerPerLength(Dimensionality):
    dimensions = _PowerPerLengthUC


class PowerPerArea(Dimensionality):
    dimensions = _PowerPerAreaUC


class PowerPerVolume(Dimensionality):
    dimensions = _PowerPerVolumeUC


class PowerPerTemperature(Dimensionality):
    dimensions = _PowerPerTemperatureC


class ThermalConductivity(Dimensionality):
    dimensions = _ThermalConductivityUC


class HeatTransferCoefficient(Dimensionality):
    dimensions = _HeatTransferCoefficientUC


class MassPerNormalVolume(Dimensionality):
    dimensions = _MassPerNormalVolumeUC


class MassPerEnergy(Dimensionality):
    dimensions = _MassPerEnergyUC


class MolarSpecificEntropy(Dimensionality):
    dimensions = _MolarSpecificEntropyUC


# these dimensionalities might have different names depending on the context
# they are not included as inputs for the autogenerated type hint
# (unless the _distinct class attribute is set to True)


class HeatingValue(Dimensionality):
    _distinct = True
    dimensions = _EnergyUC / _MassUC


class LowerHeatingValue(Dimensionality):
    dimensions = _EnergyUC / _MassUC


class HigherHeatingValue(Dimensionality):
    dimensions = _EnergyUC / _MassUC


class SpecificEnthalpy(Dimensionality):
    dimensions = _EnergyUC / _MassUC


class SpecificInternalEnergy(Dimensionality):
    dimensions = _EnergyUC / _MassUC


class MolarSpecificEnthalpy(Dimensionality):
    _distinct = True
    dimensions = _EnergyUC / _SubstanceUC


class MolarSpecificInternalEnergy(Dimensionality):
    dimensions = _EnergyUC / _SubstanceUC


class SpecificHeatCapacity(Dimensionality):
    _distinct = True
    dimensions = _EnergyUC / _MassUC / _TemperatureUC


class SpecificEntropy(Dimensionality):
    dimensions = _EnergyUC / _MassUC / _TemperatureUC


# related to CoolProp humid air
# these dimensionalities are not distinct, the same
# combination of dimensions can be mean multiple things

class IndistinctDimensionality(Dimensionality):
    _distinct = False


class SpecificHeatPerDryAir(IndistinctDimensionality):
    dimensions = _EnergyUC / _MassUC / _TemperatureUC


class SpecificHeatPerHumidAir(IndistinctDimensionality):
    dimensions = _EnergyUC / _MassUC / _TemperatureUC


class MixtureEnthalpyPerDryAir(IndistinctDimensionality):
    dimensions = _EnergyUC / _MassUC


class MixtureEnthalpyPerHumidAir(IndistinctDimensionality):
    dimensions = _EnergyUC / _MassUC


class MixtureEntropyPerDryAir(IndistinctDimensionality):
    dimensions = _EnergyUC / _MassUC / _TemperatureUC


class MixtureEntropyPerHumidAir(IndistinctDimensionality):
    dimensions = _EnergyUC / _MassUC / _TemperatureUC


class MixtureVolumePerDryAir(IndistinctDimensionality):
    dimensions = _VolumeUC / _MassUC


class MixtureVolumePerHumidAir(IndistinctDimensionality):
    dimensions = _VolumeUC / _MassUC
