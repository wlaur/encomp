from typing import (
    Any,
    Generic,
    Literal,
    SupportsAbs,
    SupportsRound,
    overload,
)

import numpy as np
import pandas as pd
import polars as pl
import sympy as sp

# this is not consistent with units.py
from pint.errors import DimensionalityError as _DimensionalityError
from pint.facets.formatting.objects import FormattingQuantity, FormattingUnit
from pint.facets.measurement.objects import MeasurementQuantity
from pint.facets.nonmultiplicative.objects import NonMultiplicativeQuantity
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.numpy.unit import NumpyUnit
from pint.facets.plain.quantity import PlainQuantity
from pint.facets.plain.unit import PlainUnit
from pint.registry import UnitRegistry
from pint.util import UnitsContainer

from .sympy import Symbol
from .utypes import (
    DT,
    DT_,
    MT,
    MT_,
    Area,
    AreaUnits,
    Currency,
    CurrencyPerEnergy,
    CurrencyPerEnergyUnits,
    CurrencyPerMass,
    CurrencyPerMassUnits,
    CurrencyPerTime,
    CurrencyPerTimeUnits,
    CurrencyPerVolume,
    CurrencyPerVolumeUnits,
    CurrencyUnits,
    Current,
    CurrentUnits,
    Density,
    DensityUnits,
    Dimensionality,
    Dimensionless,
    DimensionlessUnits,
    DynamicViscosity,
    DynamicViscosityUnits,
    Energy,
    EnergyPerMass,
    EnergyPerMassUnits,
    EnergyUnits,
    Frequency,
    HeatingValue,
    HeatTransferCoefficient,
    HigherHeatingValue,
    KinematicViscosity,
    KinematicViscosityUnits,
    Length,
    LengthUnits,
    LowerHeatingValue,
    Luminosity,
    LuminosityUnits,
    Mass,
    MassFlow,
    MassFlowUnits,
    MassPerEnergy,
    MassPerNormalVolume,
    MassUnits,
    MolarDensity,
    MolarMass,
    MolarMassUnits,
    MolarSpecificEnthalpy,
    MolarSpecificEntropy,
    Normal,
    NormalTemperature,
    NormalVolume,
    NormalVolumeFlow,
    NormalVolumeFlowUnits,
    NormalVolumePerMass,
    NormalVolumeUnits,
    Power,
    PowerPerArea,
    PowerPerLength,
    PowerPerTemperature,
    PowerPerVolume,
    PowerUnits,
    Pressure,
    PressureUnits,
    SpecificHeatCapacity,
    SpecificHeatCapacityUnits,
    SpecificVolume,
    SpecificVolumeUnits,
    Substance,
    SubstancePerMass,
    SubstancePerMassUnits,
    SubstanceUnits,
    Temperature,
    TemperatureDifference,
    TemperatureDifferenceUnits,
    TemperatureUnits,
    ThermalConductivity,
    Time,
    TimeUnits,
    Unknown,
    Velocity,
    VelocityUnits,
    Volume,
    VolumeFlow,
    VolumeFlowUnits,
    VolumeUnits,
)

# this is not consistent with units.py, tries to
# avoid issue where mypy thinks DimensionalityError is not defined
# in units.py (it is directly imported from pint.errors)
class DimensionalityError(_DimensionalityError):
    msg: str
    def __init__(self, msg: str = ...) -> None: ...

class ExpectedDimensionalityError(DimensionalityError): ...
class DimensionalityTypeError(DimensionalityError): ...
class DimensionalityComparisonError(DimensionalityError): ...
class DimensionalityRedefinitionError(ValueError): ...

# this instance is created via the LazyRegistry constructor
# however, it will be correctly set to a UnitRegistry class in the "__init" method
# (note that this is not the same method as "__init__")
ureg: UnitRegistry

CUSTOM_DIMENSIONS: list[str]

def define_dimensionality(name: str, symbol: str = ...) -> None: ...
def set_quantity_format(fmt: str = ...) -> None: ...

class Unit(PlainUnit, NumpyUnit, FormattingUnit, Generic[DT]): ...

class Quantity(
    NonMultiplicativeQuantity,
    PlainQuantity,
    MeasurementQuantity,
    NumpyQuantity,
    FormattingQuantity,
    Generic[DT, MT],
    SupportsAbs,
    SupportsRound,
):
    _magnitude: MT
    _magnitude_type: type[MT]

    def __hash__(self) -> int: ...
    def __class_getitem__(
        cls, types: type[DT] | tuple[type[DT], type[MT]]
    ) -> type[Quantity[DT, MT]]: ...
    @classmethod
    def get_unit(cls, unit_name: str) -> Unit: ...
    def __len__(self) -> int: ...
    @property
    def m(self) -> MT: ...
    @property
    def u(self) -> Unit[DT]: ...
    @property
    def units(self) -> Unit[DT]: ...
    @property
    def is_scalar(self) -> bool: ...
    @property
    def ndim(self) -> int: ...
    def to_reduced_units(self) -> Quantity[DT, MT]: ...
    def to_base_units(self) -> Quantity[DT, MT]: ...
    def asdim(self, other: type[DT_] | Quantity[DT_, MT]) -> Quantity[DT_, MT]: ...
    def astype(self, magnitude_type: type[MT_], **kwargs: Any) -> Quantity[DT, MT_]: ...
    def to(self, unit: Unit[DT] | UnitsContainer | str | dict) -> Quantity[DT, MT]: ...
    def ito(self, unit: Unit[DT] | UnitsContainer | str) -> None: ...
    def check(
        self,
        unit: Quantity[Unknown, Any]
        | UnitsContainer
        | Unit
        | str
        | Dimensionality
        | type[Dimensionality],
    ) -> bool: ...
    def __format__(self, format_type: str) -> str: ...
    @staticmethod
    def correct_unit(unit: str) -> str: ...
    @staticmethod
    def get_unit_symbol(s: str) -> Symbol: ...
    @classmethod
    def from_expr(cls, expr: sp.Basic) -> Quantity[Unknown, float]: ...
    @classmethod
    def validate(cls, qty: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    def is_compatible_with(
        self, other: Quantity | float | int, *contexts, **ctx_kwargs
    ) -> bool: ...
    def check_compatibility(self, other: Quantity | float | int) -> None: ...
    def __round__(self, ndigits: int | None = None) -> Quantity[DT, MT]: ...
    def __abs__(self) -> Quantity[DT, MT]: ...
    def __pos__(self) -> Quantity[DT, MT]: ...
    def __neg__(self) -> Quantity[DT, MT]: ...
    def __eq__(self, other: Any) -> bool: ...
    def __rsub__(
        self: Quantity[Dimensionless, MT], other: float | int
    ) -> Quantity[Dimensionless, MT]: ...
    def __radd__(
        self: Quantity[Dimensionless, MT], other: float | int
    ) -> Quantity[Dimensionless, MT]: ...
    def __rmul__(self, other: float | int) -> Quantity[DT, MT]: ...
    def __rpow__(self: Quantity[Dimensionless, MT], other: float | int) -> float: ...
    def __rfloordiv__(
        self: Quantity[Dimensionless, MT], other: float | int
    ) -> Quantity[Dimensionless, MT]: ...
    def __copy__(self) -> Quantity[DT, MT]: ...
    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Quantity[DT, MT]: ...

    # region: overload __getitem__

    @overload
    def __getitem__(
        self: Quantity[Any, list[float]], index: int
    ) -> Quantity[DT, float]: ...
    @overload
    def __getitem__(
        self: Quantity[Any, pd.Series], index: int
    ) -> Quantity[DT, float]: ...
    @overload
    def __getitem__(
        self: Quantity[Any, pl.Series], index: int
    ) -> Quantity[DT, float]: ...
    @overload
    def __getitem__(
        self: Quantity[Any, pd.DatetimeIndex], index: int
    ) -> Quantity[DT, pd.Timestamp]: ...
    @overload
    def __getitem__(
        self: Quantity[Any, np.ndarray], index: int
    ) -> Quantity[DT, float]: ...

    # endregion

    # region: overload __new__

    @overload
    def __new__(cls, val: MT) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: DimensionlessUnits
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: CurrencyUnits) -> Quantity[Currency, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: CurrencyPerEnergyUnits
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: CurrencyPerVolumeUnits
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: CurrencyPerMassUnits
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: CurrencyPerTimeUnits
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: LengthUnits) -> Quantity[Length, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: MassUnits) -> Quantity[Mass, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: TimeUnits) -> Quantity[Time, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: TemperatureUnits) -> Quantity[Temperature, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: TemperatureDifferenceUnits
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: SubstanceUnits) -> Quantity[Substance, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: MolarMassUnits) -> Quantity[MolarMass, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: SubstancePerMassUnits
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: CurrentUnits) -> Quantity[Current, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: LuminosityUnits) -> Quantity[Luminosity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: AreaUnits) -> Quantity[Area, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: VolumeUnits) -> Quantity[Volume, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: NormalVolumeUnits
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: PressureUnits) -> Quantity[Pressure, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: MassFlowUnits) -> Quantity[MassFlow, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: VolumeFlowUnits) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: NormalVolumeFlowUnits
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: DensityUnits) -> Quantity[Density, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: SpecificVolumeUnits
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: EnergyUnits) -> Quantity[Energy, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: PowerUnits) -> Quantity[Power, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: VelocityUnits) -> Quantity[Velocity, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: DynamicViscosityUnits
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: KinematicViscosityUnits
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: EnergyPerMassUnits
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __new__(
        cls, val: MT, unit: SpecificHeatCapacityUnits
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: Unit[DT]) -> Quantity[DT, MT]: ...
    @overload
    def __new__(cls, val: MT, unit: str | UnitsContainer) -> Quantity[Unknown, MT]: ...
    @overload
    def __new__(cls, val: Quantity[DT, MT]) -> Quantity[DT, MT]: ...

    # endregion

    # region: overload __floordiv__, __pow__, __add__, __sub__, __gt__, __ge__, __lt__, __le__

    @overload
    def __floordiv__(
        self: Quantity[Dimensionless, MT], other: float
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __floordiv__(self, other: Quantity[DT]) -> Quantity[Dimensionless]: ...
    @overload
    def __pow__(self, other: Literal[1]) -> Quantity[DT]: ...
    @overload
    def __pow__(self: Quantity[Length], other: Literal[2]) -> Quantity[Area]: ...
    @overload
    def __pow__(self: Quantity[Length], other: Literal[3]) -> Quantity[Volume]: ...
    @overload
    def __pow__(self: Quantity[Unknown], other: float) -> Quantity[Unknown]: ...
    @overload
    def __pow__(
        self: Quantity[Dimensionless], other: float
    ) -> Quantity[Dimensionless]: ...
    @overload
    def __pow__(self, other: Quantity[Dimensionless]) -> Quantity[Unknown]: ...
    @overload
    def __pow__(self, other: float) -> Quantity[Unknown]: ...
    @overload
    def __add__(self: Quantity[Unknown], other) -> Quantity[Unknown]: ...
    @overload
    def __add__(
        self: Quantity[Dimensionless], other: float
    ) -> Quantity[Dimensionless]: ...
    @overload
    def __add__(
        self: Quantity[Temperature], other: Quantity[TemperatureDifference]
    ) -> Quantity[Temperature]: ...
    @overload
    def __add__(self, other: Quantity[DT]) -> Quantity[DT]: ...
    @overload
    def __sub__(self: Quantity[Unknown], other) -> Quantity[Unknown]: ...
    @overload
    def __sub__(
        self: Quantity[Dimensionless], other: float
    ) -> Quantity[Dimensionless]: ...
    @overload
    def __sub__(
        self: Quantity[Temperature], other: Quantity[TemperatureDifference]
    ) -> Quantity[Temperature]: ...
    @overload
    def __sub__(
        self: Quantity[Temperature], other: Quantity[Temperature]
    ) -> Quantity[TemperatureDifference]: ...
    @overload
    def __sub__(self, other: Quantity[DT]) -> Quantity[DT]: ...
    @overload
    def __gt__(self: Quantity[Dimensionless], other: float) -> bool: ...
    @overload
    def __gt__(self, other: Quantity[DT]) -> bool: ...
    @overload
    def __ge__(self: Quantity[Dimensionless], other: float) -> bool: ...
    @overload
    def __ge__(self, other: Quantity[DT]) -> bool: ...
    @overload
    def __lt__(self: Quantity[Dimensionless], other: float) -> bool: ...
    @overload
    def __lt__(self, other: Quantity[DT]) -> bool: ...
    @overload
    def __le__(self: Quantity[Dimensionless], other: float) -> bool: ...
    @overload
    def __le__(self, other: Quantity[DT]) -> bool: ...

    # endregion

    # region: overload  __rtruediv__

    # region: autogenerated __rtruediv__

    @overload
    def __rtruediv__(
        self: Quantity[Time, MT], other: float
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[Density, MT], other: float
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[SpecificVolume, MT], other: float
    ) -> Quantity[Density, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[Frequency, MT], other: float
    ) -> Quantity[Time, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[MolarMass, MT], other: float
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[SubstancePerMass, MT], other: float
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[MassPerNormalVolume, MT], other: float
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[MassPerEnergy, MT], other: float
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[NormalVolumePerMass, MT], other: float
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[EnergyPerMass, MT], other: float
    ) -> Quantity[MassPerEnergy, MT]: ...

    # endregion

    @overload
    def __rtruediv__(self, other: float) -> Quantity[Unknown, MT]: ...

    # endregion

    # region: overload __mul__

    @overload
    def __mul__(
        self: Quantity[Unknown, MT], other: Quantity | float | int
    ) -> Quantity[Unknown, MT]: ...
    @overload
    def __mul__(self, other: Quantity[Unknown, MT]) -> Quantity[Unknown, MT]: ...

    # region: autogenerated __mul__

    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Mass, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[HeatingValue, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatingValue, MT], other: Quantity[Mass, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[LowerHeatingValue, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[LowerHeatingValue, MT], other: Quantity[Mass, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[HigherHeatingValue, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HigherHeatingValue, MT], other: Quantity[Mass, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[NormalTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[Volume, Any]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[MassPerNormalVolume, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[Length, Any]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[Area, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[PowerPerLength, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[PowerPerArea, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[PowerPerVolume, Any]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[ThermalConductivity, Any]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[HeatTransferCoefficient, Any]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[SubstancePerMass, Any]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[CurrencyPerMass, Any]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[NormalVolumePerMass, Any]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[NormalVolumeFlow, Any]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[Power, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[CurrencyPerTime, Any]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[PowerPerVolume, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT], other: Quantity[Normal, Any]
    ) -> Quantity[NormalTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[PowerPerTemperature, Any],
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[ThermalConductivity, Any],
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[HeatTransferCoefficient, Any],
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[MolarSpecificEntropy, Any],
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[SpecificHeatCapacity, Any],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Substance, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Substance, MT], other: Quantity[MolarSpecificEnthalpy, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[Length, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[PowerPerArea, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[PowerPerVolume, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[HeatTransferCoefficient, Any]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[Normal, Any]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[Density, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[MolarDensity, Any]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[CurrencyPerVolume, Any]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[PowerPerVolume, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolume, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolume, MT], other: Quantity[MassPerNormalVolume, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[Time, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[Volume, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[CurrencyPerEnergy, Any]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[Time, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[CurrencyPerMass, Any]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[NormalVolumePerMass, Any]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[SpecificHeatCapacity, Any]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Normal, Any]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Time, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Density, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[CurrencyPerVolume, Any]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[Time, Any]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[MassPerNormalVolume, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[Volume, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[SubstancePerMass, Any]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[CurrencyPerMass, Any]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[NormalVolumePerMass, Any]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[Normal, Any]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[Mass, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[Density, Any]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[MolarDensity, Any]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[CurrencyPerVolume, Any]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Energy, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Energy, MT], other: Quantity[CurrencyPerEnergy, Any]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Energy, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Power, MT], other: Quantity[Time, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Power, MT], other: Quantity[CurrencyPerEnergy, Any]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Power, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Length, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Time, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Area, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Length, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[SpecificHeatCapacity, Any]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Length, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Time, Any]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Density, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Length, Any]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Mass, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Time, Any]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Area, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Volume, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[NormalVolume, Any]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Energy, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Currency, Any]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[Substance, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[SubstancePerMass, Any]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[MolarDensity, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[SpecificHeatCapacity, Any]
    ) -> Quantity[MolarSpecificEntropy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[Mass, Any]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[Density, Any]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[MolarSpecificEntropy, Any]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT],
        other: Quantity[MolarSpecificEnthalpy, Any],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[Volume, Any]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[MolarSpecificEnthalpy, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Currency, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[Energy, Any]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[Power, Any]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[Mass, Any]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[Density, Any]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[Volume, Any]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[Time, Any]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerLength, MT], other: Quantity[Length, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerLength, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Length, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Area, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Length, Any]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Time, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Area, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Volume, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[TemperatureDifference, Any],
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[Length, Any]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[ThermalConductivity, MT],
        other: Quantity[TemperatureDifference, Any],
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[Length, Any]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT],
        other: Quantity[TemperatureDifference, Any],
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[Area, Any]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT], other: Quantity[Normal, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT], other: Quantity[NormalVolume, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT], other: Quantity[NormalVolumeFlow, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT],
        other: Quantity[NormalVolumePerMass, Any],
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[Energy, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[Power, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[CurrencyPerMass, Any]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[PowerPerLength, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[MolarSpecificEnthalpy, Any]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEntropy, MT],
        other: Quantity[TemperatureDifference, Any],
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEntropy, MT], other: Quantity[SubstancePerMass, Any]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[Mass, Any]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[Density, Any]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT],
        other: Quantity[MassPerNormalVolume, Any],
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Mass, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Time, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Density, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[CurrencyPerEnergy, Any]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[Substance, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT],
        other: Quantity[SubstancePerMass, Any],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[MolarDensity, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT],
        other: Quantity[TemperatureDifference, Any],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[MolarSpecificEntropy, MT]: ...

    # endregion

    @overload
    def __mul__(
        self: Quantity[Dimensionless, MT], other: Quantity[DT_, Any]
    ) -> Quantity[DT_, MT]: ...
    @overload
    def __mul__(self, other: Quantity[Dimensionless, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self, other: float | int) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self, other: Quantity[DT_]) -> Quantity[Unknown, MT]: ...

    # endregion

    # region: overloads __truediv__

    @overload
    def __truediv__(
        self: Quantity[Unknown, MT], other: Quantity[Unknown, Any]
    ) -> Quantity[Unknown, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[DT, Any]) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __truediv__(self: Quantity[Unknown, MT], other) -> Quantity[Unknown, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[Unknown, Any]) -> Quantity[Unknown, MT]: ...

    # region: autogenerated __truediv__

    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Mass, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[Time, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[Density, Any]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[SubstancePerMass, Any]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[MassPerNormalVolume, Any]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[NormalVolumePerMass, Any]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Normal, MT], other: Quantity[Density, Any]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Normal, MT], other: Quantity[NormalVolumePerMass, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Length, MT], other: Quantity[Time, Any]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Length, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Time, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Substance, Any]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Volume, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[NormalVolume, Any]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Density, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Energy, Any]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[MassPerNormalVolume, Any]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Time, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Time, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Substance, MT], other: Quantity[Mass, Any]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Substance, MT], other: Quantity[Volume, Any]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Substance, MT], other: Quantity[SubstancePerMass, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Substance, MT], other: Quantity[MolarDensity, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Area, MT], other: Quantity[Length, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Area, MT], other: Quantity[Time, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Area, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[Length, Any]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[Mass, Any]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[Time, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[Area, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[Normal, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[Mass, Any]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[Time, Any]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[Volume, Any]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[NormalVolumeFlow, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[NormalVolumePerMass, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[Time, Any]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[Density, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[MolarDensity, Any]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[PowerPerVolume, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[MolarSpecificEnthalpy, Any]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Length, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Mass, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[NormalVolumeFlow, Any]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Density, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Power, Any]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[MassPerNormalVolume, Any]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Length, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Area, Any]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Volume, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[Normal, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[NormalVolume, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[NormalVolumePerMass, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[Normal, Any]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[MolarDensity, Any]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[MassPerNormalVolume, Any]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Mass, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Time, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Substance, Any]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Volume, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Power, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[MolarSpecificEnthalpy, Any]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Length, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Area, Any]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Volume, Any]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Energy, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[PowerPerLength, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[PowerPerArea, Any]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[PowerPerVolume, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[PowerPerTemperature, Any]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Velocity, MT], other: Quantity[Length, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Velocity, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Time, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Density, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[PowerPerLength, Any]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Length, Any]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Time, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Area, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarMass, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarMass, MT], other: Quantity[MolarSpecificEnthalpy, Any]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[MolarDensity, Any]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarDensity, MT], other: Quantity[Density, Any]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarDensity, MT], other: Quantity[SubstancePerMass, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[Mass, Any]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[Time, Any]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[Volume, Any]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[Energy, Any]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[CurrencyPerEnergy, Any]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[CurrencyPerMass, Any]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[CurrencyPerVolume, Any]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[CurrencyPerTime, Any]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[CurrencyPerMass, Any]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[MassPerEnergy, Any]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[CurrencyPerEnergy, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[CurrencyPerVolume, Any]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[Density, Any]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[CurrencyPerEnergy, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[CurrencyPerMass, Any]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[VolumeFlow, Any]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[Power, Any]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[Currency, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[CurrencyPerEnergy, Any]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[CurrencyPerMass, Any]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[CurrencyPerVolume, Any]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[Length, Any]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[Area, Any]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[PowerPerArea, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[PowerPerVolume, Any]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[ThermalConductivity, Any]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Length, Any]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[PowerPerVolume, Any]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[HeatTransferCoefficient, Any]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT], other: Quantity[Length, Any]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT], other: Quantity[Area, Any]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT], other: Quantity[MassFlow, Any]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[ThermalConductivity, Any],
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[HeatTransferCoefficient, Any],
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[SpecificHeatCapacity, Any],
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[Length, Any]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[DynamicViscosity, Any]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT],
        other: Quantity[HeatTransferCoefficient, Any],
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT],
        other: Quantity[SpecificHeatCapacity, Any],
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEntropy, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEntropy, MT],
        other: Quantity[SpecificHeatCapacity, Any],
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[Normal, Any]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalTemperature, MT], other: Quantity[Normal, Any]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalTemperature, MT],
        other: Quantity[TemperatureDifference, Any],
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Pressure, Any]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[SpecificVolume, Any]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Velocity, Any]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[KinematicViscosity, Any]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Frequency, Any]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[SubstancePerMass, Any]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MolarSpecificEnthalpy, Any]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[SpecificHeatCapacity, Any]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT],
        other: Quantity[TemperatureDifference, Any],
    ) -> Quantity[MolarSpecificEntropy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[MolarMass, Any]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT],
        other: Quantity[MolarSpecificEntropy, Any],
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[EnergyPerMass, Any]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[SubstancePerMass, Any]
    ) -> Quantity[MolarSpecificEntropy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SpecificHeatCapacity, MT],
        other: Quantity[MolarSpecificEntropy, Any],
    ) -> Quantity[SubstancePerMass, MT]: ...

    # endregion

    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[DT_]
    ) -> Quantity[Unknown, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[Dimensionless]) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self, other: float | int | np.ndarray) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[DT_]) -> Quantity[Unknown, MT]: ...

    # endregion
