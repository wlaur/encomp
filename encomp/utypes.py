"""
Contains type definitions for :py:class:`encomp.units.Quantity` objects.
If ``encomp.settings.SETTINGS.type_checking`` is ``True``,
these types will be enforced everywhere.

The dimensionalities defined in this module can be combined with ``*`` and ``/``.
Some commonly used derived dimensionalities (like density) are defined for convenience.
"""

from typing import Union

from decimal import Decimal
import numpy.typing as npt

from uncertainties.core import AffineScalarFunc
from pint.unit import UnitsContainer

# type alias for the magnitude input to Quantity
# also accept Decimal and AffineScalarFunc (from uncertainties package)
MagnitudeValue = Union[float, int, Decimal, AffineScalarFunc]
Magnitude = Union[MagnitudeValue, list[MagnitudeValue], set[MagnitudeValue], npt.ArrayLike]

# base dimensionalities: the 7 base dimensions in the SI system and dimensionless
Dimensionless = UnitsContainer()
Length = UnitsContainer({'[length]': 1})
Mass = UnitsContainer({'[mass]': 1})
Time = UnitsContainer({'[time]': 1})
Temperature = UnitsContainer({'[temperature]': 1})
Substance = UnitsContainer({'[substance]': 1})
Current = UnitsContainer({'[current]': 1})
Luminosity = UnitsContainer({'[luminosity]': 1})

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
