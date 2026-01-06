"""
Contains type definitions for :py:class:`encomp.units.Quantity` objects.

The dimensionalities defined in this module can be combined with ``*`` and ``/``.
Some commonly used derived dimensionalities (like density) are defined for convenience.
"""

from __future__ import annotations

from typing import Literal, TypeVar, cast, get_origin

import numpy as np
import polars as pl
from pint.util import UnitsContainer

BASE_SI_UNITS = (
    "m",
    "kg",
    "s",
    "K",
    "mol",
    "A",
    "cd",
)

# these string literals are used to infer the dimensionality of commonly created quantities
# they are only used by type checkers and ignored at runtime
DimensionlessUnits = Literal[
    "",
    "%",
    "percent",
    "pct",
    "-",
    "dimensionless",
]

CurrencyUnits = Literal[
    "SEK",
    "EUR",
    "USD",
    "kSEK",
    "kEUR",
    "kUSD",
    "MSEK",
    "MEUR",
    "MUSD",
]

CurrencyPerEnergyUnits = Literal[
    "SEK/MWh",
    "EUR/MWh",
    "SEK/kWh",
    "EUR/kWh",
    "SEK/GWh",
    "EUR/GWh",
    "SEK/TWh",
    "EUR/TWh",
]

CurrencyPerMassUnits = Literal[
    "SEK/kg",
    "EUR/kg",
    "SEK/t",
    "EUR/t",
    "SEK/ton",
    "EUR/ton",
    "SEK/g",
    "EUR/g",
    "SEK/mg",
    "EUR/mg",
    "SEK/ug",
    "EUR/ug",
]

CurrencyPerVolumeUnits = Literal[
    "SEK/L",
    "EUR/L",
    "SEK/l",
    "EUR/l",
    "SEK/liter",
    "EUR/liter",
    "SEK/m3",
    "EUR/m3",
    "SEK/m^3",
    "EUR/m^3",
    "SEK/m**3",
    "EUR/m**3",
    "SEK/m³",
    "EUR/m³",
]

CurrencyPerTimeUnits = Literal[
    "SEK/h",
    "EUR/h",
    "SEK/hr",
    "EUR/hr",
    "SEK/hour",
    "EUR/hour",
    "SEK/d",
    "EUR/d",
    "SEK/day",
    "EUR/day",
    "SEK/w",
    "EUR/w",
    "SEK/week",
    "EUR/week",
    "SEK/y",
    "EUR/y",
    "SEK/yr",
    "EUR/yr",
    "SEK/year",
    "EUR/year",
    "SEK/a",
    "EUR/a",
]


LengthUnits = Literal[
    "m",
    "meter",
    "km",
    "cm",
    "mm",
    "um",
]

MassUnits = Literal[
    "kg",
    "g",
    "ton",
    "tonne",
    "t",
    "mg",
    "ug",
]

TimeUnits = Literal[
    "s",
    "second",
    "min",
    "minute",
    "h",
    "hr",
    "hour",
    "d",
    "day",
    "w",
    "week",
    "y",
    "yr",
    "a",
    "year",
    "ms",
    "us",
]

TemperatureUnits = Literal[
    "degC",
    "°C",
    "K",
    "degF",
    "°F",
    "℃",
    "℉",
]

TemperatureDifferenceUnits = Literal[
    "delta_°C",
    "delta_degC",
    "Δ°C",
    "Δ℃",
    "delta_°F",
    "delta_degF",
    "Δ°F",
    "Δ℉",
]

SubstanceUnits = Literal[
    "mol",
    "kmol",
]

MolarMassUnits = Literal[
    "g/mol",
    "kg/kmol",
]

SubstancePerMassUnits = Literal[
    "mol/g",
    "kmol/kg",
]

CurrentUnits = Literal[
    "A",
    "mA",
]

LuminosityUnits = Literal["lm"]

AreaUnits = Literal[
    "m2",
    "m^2",
    "m**2",
    "m²",
    "cm2",
    "cm^2",
    "cm**2",
    "cm²",
]

VolumeUnits = Literal[
    "L",
    "l",
    "liter",
    "m3",
    "m^3",
    "m³",
    "m**3",
    "dm3",
    "dm^3",
    "dm³",
    "dm**3",
    "cm3",
    "cm^3",
    "cm³",
    "cm**3",
]

NormalVolumeUnits = Literal[
    "normal liter",
    "Nm3",
    "nm3",
    "Nm^3",
    "nm^3",
    "Nm³",
    "nm³",
    "Nm**3",
    "nm**3",
]

PressureUnits = Literal[
    "bar",
    "kPa",
    "Pa",
    "MPa",
    "mbar",
    "mmHg",
    "psi",
    "atm",
    "N/m2",
    "N/m^2",
    "N/m**2",
    "N/m²",
]

MassFlowUnits = Literal[
    "kg/s",
    "kg/h",
    "kg/hr",
    "g/s",
    "g/h",
    "g/hr",
    "ton/h",
    "t/h",
    "ton/hr",
    "t/hr",
    "t/d",
    "ton/day",
    "t/w",
    "ton/week",
    "t/y",
    "t/a",
    "t/year",
    "ton/y",
    "ton/a",
    "ton/year",
]

VolumeFlowUnits = Literal[
    "m3/s",
    "m3/h",
    "m3/hr",
    "m**3/s",
    "m**3/h",
    "m**3/hr",
    "m^3/s",
    "m^3/h",
    "m^3/hr",
    "m³/s",
    "m³/h",
    "m³/hr",
    "liter/second",
    "l/s",
    "L/s",
    "liter/s",
    "liter/hour",
    "l/h",
    "L/h",
    "L/hr",
    "l/hr",
]

NormalVolumeFlowUnits = Literal[
    "Nm3/s",
    "Nm3/h",
    "Nm3/hr",
    "nm3/s",
    "nm3/h",
    "nm3/hr",
    "Nm^3/s",
    "Nm^3/h",
    "Nm^3/hr",
    "nm^3/s",
    "nm^3/h",
    "nm^3/hr",
    "Nm³/s",
    "Nm³/h",
    "Nm³/hr",
    "nm³/s",
    "nm³/h",
    "nm³/hr",
    "Nm**3/s",
    "Nm**3/h",
    "Nm**3/hr",
    "nm**3/s",
    "nm**3/h",
    "nm**3/hr",
]

DensityUnits = Literal[
    "kg/m3",
    "kg/m**3",
    "kg/m^3",
    "kg/m³",
    "kg/liter",
    "g/l",
    "g/L",
    "gram/liter",
]

SpecificVolumeUnits = Literal[
    "m3/kg",
    "m^3/kg",
    "m³/kg",
    "l/g",
    "L/g",
]

NormalVolumePerMassUnits = Literal[
    "Nm3/kg",
    "Nm^3/kg",
    "Nm³/kg",
    "nm3/kg",
    "nm^3/kg",
    "nm³/kg",
]

EnergyUnits = Literal[
    "J",
    "kJ",
    "MJ",
    "GJ",
    "TJ",
    "PJ",
    "kWh",
    "MWh",
    "Wh",
    "GWh",
    "TWh",
]

PowerUnits = Literal[
    "W",
    "kW",
    "MW",
    "GW",
    "TW",
    "mW",
    "kWh/d",
    "kWh/w",
    "kWh/y",
    "kWh/yr",
    "kWh/year",
    "MWh/d",
    "MWh/w",
    "MWh/y",
    "MWh/yr",
    "MWh/year",
    "GWh/d",
    "GWh/w",
    "GWh/y",
    "GWh/yr",
    "GWh/year",
    "TWh/d",
    "TWh/w",
    "TWh/y",
    "TWh/yr",
    "TWh/year",
]

VelocityUnits = Literal[
    "m/s",
    "km/s",
    "m/min",
    "cm/s",
    "cm/min",
    "km/h",
    "kmh",
    "kph",
]

ForceUnits = Literal[
    "N",
    "kN",
    "mN",
]

DynamicViscosityUnits = Literal[
    "Pa*s",
    "Pa s",
    "cP",
]

KinematicViscosityUnits = Literal[
    "m2/s",
    "m**2/s",
    "m^2/s",
    "m²/s",
    "cSt",
    "cm2/s",
    "cm**2/s",
    "cm^2/s",
    "cm²/s",
]

EnergyPerMassUnits = Literal[
    "MJ/kg",
    "MWh/kg",
    "kJ/kg",
    "kWh/kg",
    "MJ/t",
    "MWh/t",
    "kJ/t",
    "kWh/t",
    "MJ/ton",
    "MWh/ton",
    "kJ/ton",
    "kWh/ton",
]

SpecificHeatCapacityUnits = Literal[
    "kJ/kg/K",
    "kJ/kg/delta_degC",
    "kJ/kg/Δ°C",
    "kJ/kg/Δ℃",
    "kJ/kg/°C",
    "kJ/kg/℃",
    "kJ/kg/degC",
    "J/kg/K",
    "J/kg/delta_degC",
    "J/kg/Δ°C",
    "J/kg/Δ℃",
    "J/kg/°C",
    "J/kg/℃",
    "J/kg/degC",
    "J/g/K",
    "J/g/delta_degC",
    "J/g/Δ°C",
    "J/g/Δ℃",
    "J/g/°C",
    "J/g/℃",
    "J/g/degC",
]

ThermalConductivityUnits = Literal[
    "W/m/K",
    "W/m/delta_degC",
    "W/m/Δ°C",
    "W/m/Δ℃",
    "kW/m/K",
    "mW/m/K",
]

HeatTransferCoefficientUnits = Literal[
    "W/m2/K",
    "W/m^2/K",
    "W/m**2/K",
    "W/m²/K",
    "W/m2/delta_degC",
    "W/m^2/delta_degC",
    "W/m**2/delta_degC",
    "W/m²/delta_degC",
    "W/m2/Δ°C",
    "W/m^2/Δ°C",
    "W/m**2/Δ°C",
    "W/m²/Δ°C",
    "kW/m2/K",
    "kW/m^2/K",
    "kW/m**2/K",
    "kW/m²/K",
]

AllUnits = (
    DimensionlessUnits
    | CurrencyUnits
    | CurrencyPerEnergyUnits
    | CurrencyPerMassUnits
    | CurrencyPerVolumeUnits
    | CurrencyPerTimeUnits
    | LengthUnits
    | MassUnits
    | TimeUnits
    | TemperatureUnits
    | TemperatureDifferenceUnits
    | SubstanceUnits
    | MolarMassUnits
    | SubstancePerMassUnits
    | CurrentUnits
    | LuminosityUnits
    | AreaUnits
    | VolumeUnits
    | NormalVolumeUnits
    | PressureUnits
    | MassFlowUnits
    | VolumeFlowUnits
    | NormalVolumeFlowUnits
    | DensityUnits
    | SpecificVolumeUnits
    | NormalVolumePerMassUnits
    | EnergyUnits
    | PowerUnits
    | VelocityUnits
    | ForceUnits
    | DynamicViscosityUnits
    | KinematicViscosityUnits
    | EnergyPerMassUnits
    | SpecificHeatCapacityUnits
    | ThermalConductivityUnits
    | HeatTransferCoefficientUnits
)


def get_registered_units() -> dict[str, tuple[str, ...]]:
    ret: dict[str, tuple[str, ...]] = {}

    for k, v in globals().items():
        if get_origin(v) is Literal and k.endswith("Units"):
            ret[k.removesuffix("Units")] = v.__args__

    return ret


class _DimensionalityMeta(type):
    def __eq__(cls, other: object) -> bool:
        if not isinstance(other, type):
            return False

        return cls.__qualname__ == other.__qualname__

    def __hash__(cls) -> int:
        return id(cls)


class Dimensionality(metaclass=_DimensionalityMeta):
    r"""
    Represents the *dimensionality* of a unit, i.e.
    a combination (product) of the base dimensions (with optional rational exponents).

    A dimension ca be expressed as

    .. math::

        \Pi \, d^n_d, d \in \{T, L, M ,I, \Theta, N, J, \ldots\}, n_d \in \mathbb{Q}

    where $\{T, L, M, ...\}$ are the base dimensions (time, length, mass, ...)
    and $n_d$ is a rational number.

    Subclasses of this abstract base class are used
    as type parameters when creating instances of
    :py:class:`encomp.units.Quantity`.

    The ``dimensions`` class attribute defines the dimensions
    of the dimensionality using an instance of
    ``pint.unit.UnitsContainer``.
    """

    # set _distinct to False for dimensionalities that are not distinct
    # purely based on the dimensions
    _distinct: bool | None = None

    # set to True for intermediate subclasses of Dimensionality
    # these cannot be initialized directly, the must be subclassed further
    _intermediate: bool = False

    # sentinel object to indicate unset UnitsContainer
    _UnsetUC = UnitsContainer()

    dimensions: UnitsContainer = _UnsetUC

    # keeps track of all the dimensionalities that have been
    # used in the current process
    # use the class definition as key, since multiple dimensionalities
    # might have the same UnitsContainer
    _registry: dict[type[Dimensionality], UnitsContainer] = {}

    # also store a reversed map, this might not contain all items in _registry
    # dimensionalities where is_distinct() returns True will have precedence
    _registry_reversed: dict[UnitsContainer, type[Dimensionality]] = {}

    def __init_subclass__(cls) -> None:
        if cls._intermediate:
            return

        if cls.dimensions is cls._UnsetUC:
            raise TypeError(
                f"Cannot initialize {cls}, class attribute 'dimensionality' is not defined for this subclass"
            )

        # ensure that the subclass names are unique
        if cls.__name__ in (subcls.__name__ for subcls in cls._registry):
            existing = next(filter(lambda x: x.__name__ == cls.__name__, cls._registry))

            # compare string representations of the UnitsContainer instances
            # might run into issues with float accuracy otherwise
            # the UnitsContainer.__eq__ method checks hash(frozenset(self._d.items()))
            if str(cls.dimensions) != str(existing.dimensions):
                raise TypeError(
                    "Cannot create dimensionality subclass with "
                    f'name "{cls.__name__}", another subclass with '
                    "this name already exists and the dimensions do "
                    f"not match: {cls.dimensions} != {existing.dimensions}"
                )

            # don't create a new subclass with the same name
            return

        # make sure a subclass of an existing Dimensionality has the same dimensions
        # the first element in __mro__ is the class that is being created, the
        # second is the direct parent class
        # parent must be either Dimensionality or a subclass
        parent = cast(type[Dimensionality], cls.__mro__[1])

        # ignore this check if the parent is the base class Dimensionality
        if parent.dimensions is not cls._UnsetUC and parent.dimensions != cls.dimensions:
            raise TypeError(
                f"Cannot create subclass of {parent} where "
                "the dimensions do not match. Tried to "
                f"create subclass with dimensions {cls.dimensions}, but "
                f"the parent has dimensions {parent.dimensions}"
            )

        # this will never happen,
        # since the class name was already checked for duplicates
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
        _Dimensionality = cast(
            type[Dimensionality],
            type(
                f"Dimensionality[{dimensions}]",
                (Dimensionality,),
                {"dimensions": dimensions},
            ),
        )

        return _Dimensionality

    @classmethod
    def is_distinct(cls) -> bool:
        if cls._distinct is None:
            # special case if dimensions was overridden to None
            if getattr(cls, "dimensions", None) is None:
                return True

            ucs = list(cls._registry.values())

            # NOTE: the output of this classmethod
            # might change when the registry is updated
            return ucs.count(cls.dimensions) == 1

        return cls._distinct


_DimensionlessUC = UnitsContainer({})
_CurrencyUC = UnitsContainer({"[currency]": 1})
_NormalUC = UnitsContainer({"[normal]": 1})
_LengthUC = UnitsContainer({"[length]": 1})
_MassUC = UnitsContainer({"[mass]": 1})
_TimeUC = UnitsContainer({"[time]": 1})
_TemperatureUC = UnitsContainer({"[temperature]": 1})
_SubstanceUC = UnitsContainer({"[substance]": 1})
_CurrentUC = UnitsContainer({"[current]": 1})
_LuminosityUC = UnitsContainer({"[luminosity]": 1})


Numpy1DArray = np.ndarray[tuple[int], np.dtype[np.float64]]
Numpy1DBoolArray = np.ndarray[tuple[int], np.dtype[np.bool]]


MT = TypeVar(
    "MT",
    float,
    Numpy1DArray,
    pl.Series,
    pl.Expr,
    default=Numpy1DArray,
)

MT_ = TypeVar(
    "MT_",
    float,
    Numpy1DArray,
    pl.Series,
    pl.Expr,
    default=Numpy1DArray,
)


class UnknownDimensionality(Dimensionality):
    _intermediate = True


# type variables that represent a certain dimensionality
# the DT_ type variable is used to signify a different (possible identical) dimensionality than DT
DT = TypeVar("DT", bound=Dimensionality, default=UnknownDimensionality)
DT_ = TypeVar("DT_", bound=Dimensionality, default=UnknownDimensionality)


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
    _distinct = True
    dimensions = _TemperatureUC


class TemperatureDifference(Dimensionality):
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
_ForceUC = _MassUC * _LengthUC / _TimeUC**2
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
_MolarSpecificEntropyUC = _EnergyUC / _TemperatureUC / _SubstanceUC


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


class Force(Dimensionality):
    dimensions = _ForceUC


class DynamicViscosity(Dimensionality):
    dimensions = _DynamicViscosityUC


class KinematicViscosity(Dimensionality):
    dimensions = _KinematicViscosityUC


class Frequency(Dimensionality):
    dimensions = _FrequencyUC


class MolarMass(Dimensionality):
    dimensions = _MolarMassUC


class SubstancePerMass(Dimensionality):
    dimensions = 1 / _MolarMassUC


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


# not called "SpecificNormalVolume" since that term is not commonly used
# this name is more explicit
class NormalVolumePerMass(Dimensionality):
    dimensions = 1 / _MassPerNormalVolumeUC


class NormalTemperature(Dimensionality):
    dimensions = _TemperatureUC * _NormalUC


# these dimensionalities might have different names depending on the context
# they are not included as inputs for the autogenerated type hint
# (unless the _distinct class attribute is set to True)


class EnergyPerMass(Dimensionality):
    _distinct = True
    dimensions = _EnergyUC / _MassUC


class HeatingValue(Dimensionality):
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
    _intermediate = True
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
