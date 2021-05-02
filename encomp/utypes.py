"""
Contains type definitions for :py:class:`encomp.units.Quantity` objects.
If ``encomp.settings.SETTINGS.type_checking`` is ``True``,
these types will be enforced everywhere.

The dimensionalities defined in this module can be combined with ``*`` and ``/``.
Some commonly used derived dimensionalities (like density) are defined for convenience.
"""

from typing import Union, List, Mapping
from typing_extensions import Annotated

from decimal import Decimal
import numpy.typing as npt

from uncertainties.core import AffineScalarFunc
from pint.unit import UnitsContainer

# type alias for the magnitude input to Quantity
# also accept Decimal and AffineScalarFunc (from uncertainties package)
MagnitudeValue = Union[float, int, Decimal, AffineScalarFunc]
Magnitude = Union[MagnitudeValue, List[MagnitudeValue], npt.ArrayLike]

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
HeatCapacityDim = EnergyDim / MassDim / TemperatureDim
SpecificEnthalpyDim = EnergyDim / MassDim


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
    FrequencyDim: 'Frequency',
    HeatCapacityDim: 'HeatCapacity',
    SpecificEnthalpyDim: 'SpecificEnthalpy'
}

_DIMENSIONALITIES_REV = {
    b: a for a, b in _DIMENSIONALITIES.items()}


# annotated types that contain a reference to the expected dimensionality
Dimensionless = Annotated[UnitsContainer, DimensionlessDim]
Length = Annotated[UnitsContainer, LengthDim]
Mass = Annotated[UnitsContainer, MassDim]
Time = Annotated[UnitsContainer, TimeDim]
Temperature = Annotated[UnitsContainer, TemperatureDim]
Substance = Annotated[UnitsContainer, SubstanceDim]
Current = Annotated[UnitsContainer, CurrentDim]
Luminosity = Annotated[UnitsContainer, LuminosityDim]
Area = Annotated[UnitsContainer, AreaDim]
Volume = Annotated[UnitsContainer, VolumeDim]
Pressure = Annotated[UnitsContainer, PressureDim]
MassFlow = Annotated[UnitsContainer, MassFlowDim]
VolumeFlow = Annotated[UnitsContainer, VolumeFlowDim]
Density = Annotated[UnitsContainer, DensityDim]
Energy = Annotated[UnitsContainer, EnergyDim]
Power = Annotated[UnitsContainer, PowerDim]
Velocity = Annotated[UnitsContainer, VelocityDim]
DynamicViscosity = Annotated[UnitsContainer, DynamicViscosityDim]
KinematicViscosity = Annotated[UnitsContainer, KinematicViscosityDim]
Frequency = Annotated[UnitsContainer, FrequencyDim]
HeatCapacity = Annotated[UnitsContainer, HeatCapacityDim]
SpecificEnthalpy = Annotated[UnitsContainer, SpecificEnthalpyDim]


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
