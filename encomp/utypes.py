"""
Contains type definitions for :py:class:`encomp.units.Quantity` objects.
If ``encomp.settings.SETTINGS.type_checking`` is ``True``,
these types will be enforced everywhere.

The dimensionalities defined in this module can be combined with ``*`` and ``/``.
Some commonly used derived dimensionalities (like density) are defined for convenience.
"""

from __future__ import annotations

from typing import TypeVar, Union, Any, Annotated
from typing import Union
from abc import ABC


import numpy as np
import pandas as pd
from pint.unit import UnitsContainer

DimensionalityName = Annotated[str, 'Dimensionality name']


# type alias for the magnitude input to Quantity
MagnitudeValue = Union[float, int]

Magnitude = Union[MagnitudeValue,
                  list[MagnitudeValue],
                  tuple[MagnitudeValue, ...],
                  np.ndarray[Any, Any],
                  pd.Series]


_BASE_SI_UNITS: tuple[str, ...] = ('m', 'kg', 's', 'K', 'mol', 'A', 'cd')


class Dimensionality(ABC):

    dimensions: UnitsContainer

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

        if not isinstance(cls.dimensions, UnitsContainer):
            raise TypeError(
                'The "dimensions" attribute of the Dimensionality type '
                'must be an instance of pint.unit.UnitsContainer, '
                f'passed {cls} with dimensions: {cls.dimensions} ({type(cls.dimensions)})'
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
# the DT_ type is used to signify different dimensionalities than DT,
# and DT__  signifies different than DT and DT__
DT = TypeVar('DT', bound=Dimensionality)
DT_ = TypeVar('DT_', bound=Dimensionality)
DT__ = TypeVar('DT__', bound=Dimensionality)

_DimensionlessUC = UnitsContainer({})
_NormalUC = UnitsContainer({'[normal]': 1})
_LengthUC = UnitsContainer({'[length]': 1})
_MassUC = UnitsContainer({'[mass]': 1})
_TimeUC = UnitsContainer({'[time]': 1})
_TemperatureUC = UnitsContainer({'[temperature]': 1})
_SubstanceUC = UnitsContainer({'[substance]': 1})
_CurrentUC = UnitsContainer({'[current]': 1})
_LuminosityUC = UnitsContainer({'[luminosity]': 1})


# NOTE: each subclass defintion will create an entry in Dimensionality._registry
# reloading (re-importing) this module will clear and reset the registry

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


# # these dimensionalities might have different names depending on the context
_HeatingValueUC = _EnergyUC / _MassUC
_LowerHeatingValueUC = _EnergyUC / _MassUC
_HigherHeatingValueUC = _EnergyUC / _MassUC
_SpecificEnthalpyUC = _EnergyUC / _MassUC
_HeatCapacityUC = _EnergyUC / _MassUC / _TemperatureUC
_ThermalConductivityUC = _PowerUC / _LengthUC / _TemperatureUC
_HeatTransferCoefficientUC = _PowerUC / _AreaUC / _TemperatureUC
_MassPerNormalVolumeUC = _MassUC / _NormalVolumeUC
_MassPerEnergyUC = _MassUC / _EnergyUC
_CurrencyUC = _DimensionlessUC
_CurrencyPerEnergyUC = _CurrencyUC / _EnergyUC

class HeatingValue(Dimensionality):
    dimensions = _HeatingValueUC


class LowerHeatingValue(Dimensionality):
    dimensions = _LowerHeatingValueUC


class HigherHeatingValue(Dimensionality):
    dimensions = _HigherHeatingValueUC


class SpecificEnthalpy(Dimensionality):
    dimensions = _SpecificEnthalpyUC


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
