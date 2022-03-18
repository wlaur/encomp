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


# for mypy compatibility
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

Dimensionless = Dimensionality() * _Dimensionless
Length = Dimensionality({'[length]': 1}) * _Dimensionless
Mass = Dimensionality({'[mass]': 1}) * _Dimensionless
Time = Dimensionality({'[time]': 1}) * _Dimensionless
Temperature = Dimensionality({'[temperature]': 1}) * _Dimensionless
Substance = Dimensionality({'[substance]': 1}) * _Dimensionless
Current = Dimensionality({'[current]': 1}) * _Dimensionless
Luminosity = Dimensionality({'[luminosity]': 1}) * _Dimensionless


# derived dimensionalities
Area = Length**2
Volume = Length**3
Pressure = Mass / Length / Time**2
MassFlow = Mass / Time
VolumeFlow = Volume / Time
Density = Mass / Volume
Energy = Mass * Length**2 / Time**2
Power = Energy / Time
Velocity = Length / Time
DynamicViscosity = Mass / Length / Time
KinematicViscosity = Length**2 / Time
Frequency = 1 / Time
MolarMass = Mass / Substance
HeatingValue = Energy / Mass


# these dimensionalities might have different names depending on context
HeatCapacity = Energy / Mass / Temperature
ThermalConductivity = Power / Length / Temperature
HeatTransferCoefficient = Power / Area / Temperature

_DIMENSIONALITIES: dict[UnitsContainer, str] = {
    Dimensionless: 'Dimensionless',
    Length: 'Length',
    Mass: 'Mass',
    Time: 'Time',
    Temperature: 'Temperature',
    Substance: 'Substance',
    Current: 'Current',
    Luminosity: 'Luminosity',
    Area: 'Area',
    Volume: 'Volume',
    Pressure: 'Pressure',
    MassFlow: 'MassFlow',
    VolumeFlow: 'VolumeFlow',
    Density: 'Density',
    Energy: 'Energy',
    Power: 'Power',
    Velocity: 'Velocity',
    DynamicViscosity: 'DynamicViscosity',
    KinematicViscosity: 'KinematicViscosity',
    Frequency: 'Frequency'
}

_DIMENSIONALITIES_REV = {
    b: a for a, b in _DIMENSIONALITIES.items()}


_BASE_SI_UNITS: tuple[str, ...] = ('m', 'kg', 's', 'K', 'mol', 'A', 'cd')


def get_dimensionality_name(dim: UnitsContainer) -> str:
    """
    Returns a readable name for a dimensionality.

    Parameters
    ----------
    dim : UnitsContainer
        input dimensionality

    Returns
    -------
    str
        Readable name, or str representation of the input
    """

    if dim in _DIMENSIONALITIES:
        return _DIMENSIONALITIES[dim]

    else:
        return str(dim)
