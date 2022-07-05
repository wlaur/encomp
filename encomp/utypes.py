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


class Dimensionality:

    uc: UnitsContainer

    _existing: dict[UnitsContainer, type[Dimensionality]] = {}

    @classmethod
    def get_cls(cls, uc: UnitsContainer) -> type[Dimensionality]:

        if uc in cls._existing:
            return cls._existing[uc]

        name = f'DimType[{uc}]'

        _Dimensionality = type(
            name,
            (Dimensionality,),
            {
                'uc': uc
            }
        )

        cls._existing[uc] = _Dimensionality

        return _Dimensionality


DT = TypeVar('DT', bound=Dimensionality)
DT_ = TypeVar('DT_', bound=Dimensionality)
DT__ = TypeVar('DT__', bound=Dimensionality)


_NormalUC = UnitsContainer({'[normal]': 1})
_LengthUC = UnitsContainer({'[length]': 1})
_MassUC = UnitsContainer({'[mass]': 1})
_TimeUC = UnitsContainer({'[time]': 1})
_TemperatureUC = UnitsContainer({'[temperature]': 1})
_SubstanceUC = UnitsContainer({'[substance]': 1})
_CurrentUC = UnitsContainer({'[current]': 1})
_LuminosityUC = UnitsContainer({'[luminosity]': 1})


class Dimensionless(Dimensionality):
    uc = UnitsContainer({})


class Normal(Dimensionality):
    uc = _NormalUC


class Length(Dimensionality):
    uc = _LengthUC


class Mass(Dimensionality):
    uc = _MassUC


class Time(Dimensionality):
    uc = _TimeUC


class Temperature(Dimensionality):
    uc = _TemperatureUC


class Substance(Dimensionality):
    uc = _SubstanceUC


class Current(Dimensionality):
    uc = _CurrentUC


class Luminosity(Dimensionality):
    uc = _LuminosityUC


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


class Area(Dimensionality):
    uc = _AreaUC


class Volume(Dimensionality):
    uc = _VolumeUC


class NormalVolume(Dimensionality):
    uc = _NormalVolumeUC


class Pressure(Dimensionality):
    uc = _PressureUC


class MassFlow(Dimensionality):
    uc = _MassFlowUC


class VolumeFlow(Dimensionality):
    uc = _VolumeFlowUC


class NormalVolumeFlow(Dimensionality):
    uc = _NormalVolumeFlowUC


class Density(Dimensionality):
    uc = _DensityUC


class SpecificVolume(Dimensionality):
    uc = _SpecificVolumeUC


class Energy(Dimensionality):
    uc = _EnergyUC


class Power(Dimensionality):
    uc = _PowerUC


class Velocity(Dimensionality):
    uc = _VelocityUC


class DynamicViscosity(Dimensionality):
    uc = _DynamicViscosityUC


class KinematicViscosity(Dimensionality):
    uc = _KinematicViscosityUC


class Frequency(Dimensionality):
    uc = _FrequencyUC


class MolarMass(Dimensionality):
    uc = _MolarMassUC


class HeatingValue(Dimensionality):
    uc = _HeatingValueUC


class LowerHeatingValue(Dimensionality):
    uc = _LowerHeatingValueUC


class HigherHeatingValue(Dimensionality):
    uc = _HigherHeatingValueUC


class SpecificEnthalpy(Dimensionality):
    uc = _SpecificEnthalpyUC


class HeatCapacity(Dimensionality):
    uc = _HeatCapacityUC


class ThermalConductivity(Dimensionality):
    uc = _ThermalConductivityUC


class HeatTransferCoefficient(Dimensionality):
    uc = _HeatTransferCoefficientUC


class MassPerNormalVolume(Dimensionality):
    uc = _MassPerNormalVolumeUC


class MassPerEnergy(Dimensionality):
    uc = _MassPerEnergyUC


_instances: list[type[Dimensionality]] = [
    Dimensionless,
    Normal,
    Length,
    Mass,
    Time,
    Temperature,
    Substance,
    Current,
    Luminosity,
    Area,
    Volume,
    NormalVolume,
    Pressure,
    MassFlow,
    VolumeFlow,
    NormalVolumeFlow,
    Density,
    SpecificVolume,
    Energy,
    Power,
    Velocity,
    DynamicViscosity,
    KinematicViscosity,
    Frequency,
    MolarMass,
    HeatingValue,
    LowerHeatingValue,
    HigherHeatingValue,
    SpecificEnthalpy,
    HeatCapacity,
    ThermalConductivity,
    HeatTransferCoefficient,
    MassPerNormalVolume,
    MassPerEnergy,
]

Dimensionality._existing.update({n.uc: n for n in _instances})
