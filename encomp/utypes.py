"""
Contains type definitions for :py:class:`encomp.units.Quantity` objects.
If ``encomp.settings.SETTINGS.type_checking`` is ``True``,
these types will be enforced everywhere.

The dimensionalities defined in this module can be combined with ``*`` and ``/``.
Some commonly used derived dimensionalities (like density) are defined for convenience.
"""

from typing import Union, TypeVar, List, Mapping
from decimal import Decimal
import numpy as np

from pint.unit import UnitsContainer

# type alias for the magnitude input to Quantity
Magnitude = Union[float, int, Decimal, np.ndarray, List[float], List[int]]

# base dimensionalities
DimensionlessDim = UnitsContainer()
LengthDim = UnitsContainer({'[length]': 1})
MassDim = UnitsContainer({'[mass]': 1})
TimeDim = UnitsContainer({'[time]': 1})
TemperatureDim = UnitsContainer({'[temperature]': 1})
SubstanceDim = UnitsContainer({'[substance]': 1})
CurrentDim = UnitsContainer({'[current]': 1})
LuminosityDim = UnitsContainer({'[luminosity]': 1})

# derived dimensionalities
AreaDim = LengthDim**2
VolumeDim = LengthDim**3
PressureDim = MassDim / LengthDim / TimeDim**2
MassFlowDim = MassDim / TimeDim
VolumeFlowDim = VolumeDim / TimeDim
DensityDim = MassDim / VolumeDim
EnergyDim = MassDim * LengthDim**2 / TimeDim**2
PowerDim = EnergyDim / TimeDim
VelocityDim = LengthDim / TimeDim
DynamicViscosityDim = MassDim / LengthDim / TimeDim
KinematicViscosityDim = LengthDim**2 / TimeDim
FrequencyDim = 1 / TimeDim

_DIMENSIONALITIES: Mapping[UnitsContainer, str] = {
    DimensionlessDim: 'Dimensionless',
    LengthDim: 'Length',
    MassDim: 'Mass',
    TimeDim: 'Time',
    TemperatureDim: 'Temperature',
    SubstanceDim: 'Substance',
    CurrentDim: 'Current',
    LuminosityDim: 'Luminosity',
    AreaDim: 'Area',
    VolumeDim: 'Volume',
    PressureDim: 'Pressure',
    MassFlowDim: 'MassFlow',
    VolumeFlowDim: 'VolumeFlow',
    DensityDim: 'Density',
    EnergyDim: 'Energy',
    PowerDim: 'Power',
    VelocityDim: 'Velocity',
    DynamicViscosityDim: 'DynamicViscosity',
    KinematicViscosityDim: 'KinematicViscosity',
    FrequencyDim: 'Frequency'
}

_DIMENSIONALITIES_REV = {
    b: a for a, b in _DIMENSIONALITIES.items()}


Dimensionless = TypeVar('Dimensionless')
Dimensionless.dimensionality = DimensionlessDim

Length = TypeVar('Length')
Length.dimensionality = LengthDim

Mass = TypeVar('Mass')
Mass.dimensionality = MassDim

Time = TypeVar('Time')
Time.dimensionality = TimeDim

Temperature = TypeVar('Temperature')
Temperature.dimensionality = TemperatureDim

Substance = TypeVar('Substance')
Substance.dimensionality = SubstanceDim

Current = TypeVar('Current')
Current.dimensionality = CurrentDim

Luminosity = TypeVar('Luminosity')
Luminosity.dimensionality = LuminosityDim

Area = TypeVar('Area')
Area.dimensionality = AreaDim

Volume = TypeVar('Volume')
Volume.dimensionality = VolumeDim

Pressure = TypeVar('Pressure')
Pressure.dimensionality = PressureDim

MassFlow = TypeVar('MassFlow')
MassFlow.dimensionality = MassFlowDim

VolumeFlow = TypeVar('VolumeFlow')
VolumeFlow.dimensionality = VolumeFlowDim

Density = TypeVar('Density')
Density.dimensionality = DensityDim

Energy = TypeVar('Energy')
Energy.dimensionality = EnergyDim

Power = TypeVar('Power')
Power.dimensionality = PowerDim

Velocity = TypeVar('Velocity')
Velocity.dimensionality = VelocityDim

DynamicViscosity = TypeVar('DynamicViscosity')
DynamicViscosity.dimensionality = DynamicViscosityDim

KinematicViscosity = TypeVar('KinematicViscosity')
KinematicViscosity.dimensionality = KinematicViscosityDim

Frequency = TypeVar('Frequency')
Frequency.dimensionality = FrequencyDim


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
