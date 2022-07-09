"""
Contains type definitions for :py:class:`encomp.units.Quantity` objects.
If ``encomp.settings.SETTINGS.type_checking`` is ``True``,
these types will be enforced everywhere.

The dimensionalities defined in this module can be combined with ``*`` and ``/``.
Some commonly used derived dimensionalities (like density) are defined for convenience.
"""

from __future__ import annotations

from typing import TypeVar, Union, Optional, Literal
from typing import _LiteralGenericAlias  # type: ignore
from typing import Union
from abc import ABC


import numpy as np
import pandas as pd
from pint.unit import UnitsContainer


MagnitudeScalar = Union[float, int]

MagnitudeInput = Union[MagnitudeScalar,
                       list[MagnitudeScalar],
                       tuple[MagnitudeScalar, ...],
                       np.ndarray,
                       pd.Series]

# the actual Quantity._magnitude attribute is scalar or np.ndarray
# list, tuple, Series will be converted
Magnitude = Union[MagnitudeScalar, np.ndarray]

_BASE_SI_UNITS: tuple[str, ...] = ('m', 'kg', 's', 'K', 'mol', 'A', 'cd')

# these string literals are used to infer the dimensionality of commonly created quantities
DimensionlessUnits = Literal['', '%']
CurrencyUnits = Literal['SEK', 'EUR', 'USD',
                        'kSEK', 'kEUR', 'kUSD',
                        'MSEK', 'MEUR', 'MUSD']

CurrencyPerEnergyUnits = Literal['SEK/MWh', 'EUR/MWh',
                                 'SEK/kWh', 'EUR/kWh',
                                 'SEK/GWh', 'EUR/GWh',
                                 'SEK/TWh', 'EUR/TWh']

CurrencyPerMassUnits = Literal['SEK/kg', 'EUR/kg',
                               'SEK/t', 'EUR/t',
                               'SEK/ton', 'EUR/ton',
                               'SEK/g', 'EUR/g',
                               'SEK/mg', 'EUR/mg',
                               'SEK/ug', 'EUR/ug']

CurrencyPerVolumeUnits = Literal['SEK/L', 'EUR/L',
                                 'SEK/l', 'EUR/l',
                                 'SEK/liter', 'EUR/liter',
                                 'SEK/m3', 'EUR/m3'
                                 'SEK/m^3', 'EUR/m^3'
                                 'SEK/m**3', 'EUR/m**3'
                                 'SEK/m³', 'EUR/m³']

CurrencyPerTimeUnits = Literal['SEK/h', 'EUR/h',  'SEK/hr', 'EUR/hr',
                               'SEK/hour', 'EUR/hour', 'SEK/d', 'EUR/d',
                               'SEK/day', 'EUR/day', 'SEK/w', 'EUR/w',
                               'SEK/week', 'EUR/week', 'SEK/y', 'EUR/y',
                               'SEK/yr', 'EUR/yr', 'SEK/year', 'EUR/year'
                               'SEK/a', 'EUR/a']


LengthUnits = Literal['m', 'meter', 'km', 'cm', 'mm', 'um']
MassUnits = Literal['kg', 'g', 'ton', 'tonne', 't', 'mg', 'ug']
TimeUnits = Literal['s', 'second', 'min', 'minute', 'h', 'hr',
                    'hour', 'd', 'day', 'w', 'week', 'y', 'yr',
                    'a', 'year', 'ms', 'us']

TemperatureUnits = Literal['C', 'degC', '°C', 'K', 'F', 'degF', '°F',
                           'delta_C', 'delta_degC', 'Δ°C',
                           'delta_F', 'delta_degF', 'Δ°F']

SubstanceUnits = Literal['mol', 'kmol']
CurrentUnits = Literal['A', 'mA']
LuminosityUnits = Literal['lm']

AreaUnits = Literal['m2', 'm^2', 'm**2', 'm²', 'cm2', 'cm^2', 'cm**2', 'cm²']
VolumeUnits = Literal['L', 'l', 'liter', 'm3', 'm^3', 'm³', 'm**3',
                      'dm3', 'dm^3', 'dm³', 'dm**3',
                      'cm3', 'cm^3', 'cm³', 'cm**3']

NormalVolumeUnits = Literal['normal liter', 'Nm3',
                            'nm3', 'Nm^3', 'nm^3', 'Nm³',
                            'nm³', 'Nm**3', 'nm**3']

PressureUnits = Literal['bar', 'kPa', 'Pa', 'MPa', 'mbar', 'mmHg']

MassFlowUnits = Literal['kg/s', 'kg/h', 'kg/hr', 'g/s', 'g/h', 'g/hr',
                        'ton/h', 't/h', 'ton/hr',
                        't/hr', 't/d', 'ton/day', 't/w', 'ton/week', 't/y',
                        't/a', 't/year', 'ton/y', 'ton/a', 'ton/year']

VolumeFlowUnits = Literal['m3/s', 'm3/h', 'm3/hr',
                          'm**3/s', 'm**3/h', 'm**3/hr',
                          'm^3/s', 'm^3/h', 'm^3/hr',
                          'm³/s', 'm³/h', 'm³/hr',
                          'liter/second', 'l/s', 'L/s',
                          'liter/hour', 'l/h', 'L/h', 'L/hr', 'l/hr']

NormalVolumeFlowUnits = Literal['Nm3/s', 'Nm3/h', 'Nm3/hr',
                                'nm3/s', 'nm3/h', 'nm3/hr',
                                'Nm^3/s', 'Nm^3/h', 'Nm^3/hr',
                                'nm^3/s', 'nm^3/h', 'nm^3/hr',
                                'Nm³/s', 'Nm³/h', 'Nm³/hr',
                                'nm³/s', 'nm³/h', 'nm³/hr',
                                'Nm**3/s', 'Nm**3/h', 'Nm**3/hr'
                                'nm**3/s', 'nm**3/h', 'nm**3/hr']

DensityUnits = Literal['kg/m3', 'kg/m**3',
                       'kg/m^3', 'kg/m³', 'g/l', 'g/L', 'gram/liter']
SpecificVolumeUnits = Literal['m3/kg', 'm^3/kg', 'm³/kg', 'l/g', 'L/g']

EnergyUnits = Literal['J', 'kJ', 'MJ', 'GJ', 'TJ', 'PJ',
                      'kWh', 'MWh', 'Wh', 'GWh', 'TWh']

PowerUnits = Literal['W', 'kW', 'MW', 'GW', 'TW', 'mW',
                     'kWh/d', 'kWh/w', 'kWh/y', 'kWh/yr', 'kWh/year',
                     'MWh/d', 'MWh/w', 'MWh/y', 'MWh/yr', 'MWh/year',
                     'GWh/d', 'GWh/w', 'GWh/y', 'GWh/yr', 'GWh/year',
                     'TWh/d', 'TWh/w', 'TWh/y', 'TWh/yr', 'TWh/year']

VelocityUnits = Literal['m/s', 'km/s', 'm/min',
                        'cm/s', 'cm/min', 'km/h', 'kmh', 'kph']
DynamicViscosityUnits = Literal['Pa*s', 'Pa s', 'cP']

KinematicViscosityUnits = Literal['m2/s', 'm**2/s', 'm^2/s',
                                  'm²/s', 'cSt', 'cm2/s',
                                  'cm**2/s', 'cm^2/s', 'cm²/s']


def get_registered_units() -> dict[str, tuple[str, ...]]:

    ret = {}

    for k, v in globals().items():
        if isinstance(v, _LiteralGenericAlias):
            if k.endswith('Units'):
                ret[k.removesuffix('Units')] = v.__args__

    return ret


class Dimensionality(ABC):

    dimensions: Optional[UnitsContainer] = None

    # keeps track of all the dimensionalities that have been
    # used in the current process
    # use the class definition as key, since multiple dimensionalities
    # might have the same UnitsContainer
    _registry: dict[type[Dimensionality], UnitsContainer] = {}

    # also store a reversed map, this might not contain all items in _registry
    _registry_reversed: dict[UnitsContainer, type[Dimensionality]] = {}

    def __init_subclass__(cls) -> None:

        if not hasattr(cls, 'dimensions'):
            raise AttributeError(
                'Subtypes of Dimensionality must define the '
                'attribute "dimensions" (an instance of pint.unit.UnitsContainer)'
            )

        # the Unknown and Impossible dimensionalities subclass has dimensions=None
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

        # in case there are multiple class definitions with the same UnitRegistry,
        # do not overwrite previously defined ones
        if cls.dimensions not in cls._registry_reversed:
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


# type variables that represent a certain dimensionality
# the DT_ type is used to signify a different dimensionality than DT
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


class Impossible(Dimensionality):
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


# # these dimensionalities might have different names depending on the context
_HeatingValueUC = _EnergyUC / _MassUC
_LowerHeatingValueUC = _EnergyUC / _MassUC
_HigherHeatingValueUC = _EnergyUC / _MassUC
_SpecificHeatCapacityUC = _EnergyUC / _MassUC / _TemperatureUC

_SpecificEnthalpyUC = _EnergyUC / _MassUC
_MolarSpecificEnthalpyUC = _EnergyUC / _SubstanceUC
_SpecificEntropyUC = _SpecificEnthalpyUC / _TemperatureUC
_MolarSpecificEntropyUC = _SpecificEnthalpyUC / _SubstanceUC
_SpecificInternalEnergyUC = _EnergyUC / _MassUC
_MolarSpecificInternalEnergyUC = _EnergyUC / _SubstanceUC

_HeatCapacityUC = _EnergyUC / _MassUC / _TemperatureUC
_ThermalConductivityUC = _PowerUC / _LengthUC / _TemperatureUC
_HeatTransferCoefficientUC = _PowerUC / _AreaUC / _TemperatureUC
_MassPerNormalVolumeUC = _MassUC / _NormalVolumeUC
_MassPerEnergyUC = _MassUC / _EnergyUC
_CurrencyPerEnergyUC = _CurrencyUC / _EnergyUC
_CurrencyPerMassUC = _CurrencyUC / _MassUC
_CurrencyPerVolumeUC = _CurrencyUC / _VolumeUC
_CurrencyPerTimeUC = _CurrencyUC / _TimeUC

# related to CoolProp humid air
_SpecificHeatPerDryAirUC = _EnergyUC / _MassUC / _TemperatureUC
_SpecificHeatPerHumidAirUC = _EnergyUC / _MassUC / _TemperatureUC
_MixtureEnthalpyPerDryAirUC = _EnergyUC / _MassUC
_MixtureEnthalpyPerHumidAirUC = _EnergyUC / _MassUC
_MixtureEntropyPerDryAirUC = _EnergyUC / _MassUC / _TemperatureUC
_MixtureEntropyPerHumidAirUC = _EnergyUC / _MassUC / _TemperatureUC
_MixtureVolumePerDryAirUC = _VolumeUC / _MassUC
_MixtureVolumePerHumidAirUC = _VolumeUC / _MassUC


class HeatingValue(Dimensionality):
    dimensions = _HeatingValueUC


class LowerHeatingValue(Dimensionality):
    dimensions = _LowerHeatingValueUC


class HigherHeatingValue(Dimensionality):
    dimensions = _HigherHeatingValueUC


class SpecificHeatCapacity(Dimensionality):
    dimensions = _SpecificHeatCapacityUC


class SpecificEnthalpy(Dimensionality):
    dimensions = _SpecificEnthalpyUC


class MolarSpecificEnthalpy(Dimensionality):
    dimensions = _MolarSpecificEnthalpyUC


class SpecificEntropy(Dimensionality):
    dimensions = _SpecificEntropyUC


class MolarSpecificEntropy(Dimensionality):
    dimensions = _MolarSpecificEntropyUC


class SpecificInternalEnergy(Dimensionality):
    dimensions = _SpecificInternalEnergyUC


class MolarSpecificInternalEnergy(Dimensionality):
    dimensions = _MolarSpecificInternalEnergyUC


class HeatCapacity(Dimensionality):
    dimensions = _HeatCapacityUC


class ThermalConductivity(Dimensionality):
    dimensions = _ThermalConductivityUC


class HeatTransferCoefficient(Dimensionality):
    dimensions = _HeatTransferCoefficientUC


class MassPerNormalVolume(Dimensionality):
    dimensions = _MassPerNormalVolumeUC


class MassPerEnergy(Dimensionality):
    dimensions = _MassPerEnergyUC


class Currency(Dimensionality):
    dimensions = _CurrencyUC


class CurrencyPerEnergy(Dimensionality):
    dimensions = _CurrencyPerEnergyUC


class CurrencyPerMass(Dimensionality):
    dimensions = _CurrencyPerMassUC


class CurrencyPerVolume(Dimensionality):
    dimensions = _CurrencyPerVolumeUC


class CurrencyPerTime(Dimensionality):
    dimensions = _CurrencyPerVolumeUC


class SpecificHeatPerDryAir(Dimensionality):
    dimensions = _SpecificHeatPerDryAirUC


class SpecificHeatPerHumidAir(Dimensionality):
    dimensions = _SpecificHeatPerHumidAirUC


class MixtureEnthalpyPerDryAir(Dimensionality):
    dimensions = _MixtureEnthalpyPerDryAirUC


class MixtureEnthalpyPerHumidAir(Dimensionality):
    dimensions = _MixtureEnthalpyPerHumidAirUC


class MixtureEntropyPerDryAir(Dimensionality):
    dimensions = _MixtureEntropyPerDryAirUC


class MixtureEntropyPerHumidAir(Dimensionality):
    dimensions = _MixtureEntropyPerHumidAirUC


class MixtureVolumePerDryAir(Dimensionality):
    dimensions = _MixtureVolumePerDryAirUC


class MixtureVolumePerHumidAir(Dimensionality):
    dimensions = _MixtureVolumePerHumidAirUC
