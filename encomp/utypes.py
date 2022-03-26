"""
Contains type definitions for :py:class:`encomp.units.Quantity` objects.
If ``encomp.settings.SETTINGS.type_checking`` is ``True``,
these types will be enforced everywhere.

The dimensionalities defined in this module can be combined with ``*`` and ``/``.
Some commonly used derived dimensionalities (like density) are defined for convenience.
"""

from typing import Generic, TypeVar
from typing import Union


import numpy as np
import pandas as pd
from pint.unit import UnitsContainer


R = TypeVar('R')


# mypy compatibility
class Dimensionality(UnitsContainer, Generic[R]):
    pass


# type alias for the magnitude input to Quantity
MagnitudeValue = Union[float, int]
Magnitude = Union[MagnitudeValue,
                  list[MagnitudeValue],
                  tuple[MagnitudeValue, ...],
                  np.ndarray,
                  pd.Series]

# base dimensionalities: the 7 base dimensions in the SI system and dimensionless
# NOTE: these must be defined as Dimensionality(...) * Dimensionless to avoid issues with mypy
_Dimensionless: Dimensionality = Dimensionality()


def base_dimension(name: str) -> Dimensionality:
    return Dimensionality({f'[{name}]': 1}) * _Dimensionless


Dimensionless = Dimensionality() * _Dimensionless
Normal = base_dimension('normal')

Length = base_dimension('length')
Mass = base_dimension('mass')
Time = base_dimension('time')
Temperature = base_dimension('temperature')
Substance = base_dimension('substance')
Current = base_dimension('current')
Luminosity = base_dimension('luminosity')


# derived dimensionalities
Area = Length**2
Volume = Length**3
NormalVolume = Volume * Normal
Pressure = Mass / Length / Time**2
MassFlow = Mass / Time
VolumeFlow = Volume / Time
NormalVolumeFlow = NormalVolume / Time
Density = Mass / Volume
Energy = Mass * Length**2 / Time**2
Power = Energy / Time
Velocity = Length / Time
DynamicViscosity = Mass / Length / Time
KinematicViscosity = Length**2 / Time
Frequency = 1 / Time
MolarMass = Mass / Substance

# these dimensionalities might have different names depending on the context
HeatingValue = Energy / Mass
LowerHeatingValue = Energy / Mass
HigherHeatingValue = Energy / Mass

HeatCapacity = Energy / Mass / Temperature
ThermalConductivity = Power / Length / Temperature
HeatTransferCoefficient = Power / Area / Temperature

_DIMENSIONALITIES_REV: dict[str, UnitsContainer] = {
    'Dimensionless': Dimensionless,
    'Normal': Normal,
    'Length': Length,
    'Mass': Mass,
    'Time': Time,
    'Temperature': Temperature,
    'Substance': Substance,
    'Current': Current,
    'Luminosity': Luminosity,
    'Area': Area,
    'Volume': Volume,
    'NormalVolume': NormalVolume,
    'Pressure': Pressure,
    'MassFlow': MassFlow,
    'VolumeFlow': VolumeFlow,
    'NormalVolumeFlow': NormalVolumeFlow,
    'Density': Density,
    'Energy': Energy,
    'Power': Power,
    'Velocity': Velocity,
    'DynamicViscosity': DynamicViscosity,
    'KinematicViscosity': KinematicViscosity,
    'Frequency': Frequency,
    'MolarMass': MolarMass,

    'LowerHeatingValue': LowerHeatingValue,
    'HigherHeatingValue': HigherHeatingValue,
    # the most general name last, will overwrite in dict _DIMENSIONALITIES
    'HeatingValue': HeatingValue,

    'HeatCapacity': HeatCapacity,
    'ThermalConductivity': ThermalConductivity,
    'HeatTransferCoefficient': HeatTransferCoefficient
}

# might not contain all elements of _DIMENSIONALITIES_REV
# dimensionalities can have multiple names
_DIMENSIONALITIES = {
    b: a for a, b in _DIMENSIONALITIES_REV.items()
}


_BASE_SI_UNITS: tuple[str, ...] = ('m', 'kg', 's', 'K', 'mol', 'A', 'cd')
