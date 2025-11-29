# mypy: disable-error-code="overload-overlap, misc"
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

class Unit(PlainUnit, NumpyUnit, FormattingUnit, Generic[DT]):
    dimensionality: UnitsContainer
    _units: UnitsContainer

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
    dimensionality: UnitsContainer
    dimensionless: bool
    _dimensionality_type: type[Dimensionality]

    def _ok_for_muldiv(self) -> bool: ...
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
        unit: (
            Quantity[Dimensionality, Any]
            | UnitsContainer
            | Unit
            | str
            | Dimensionality
            | type[Dimensionality]
        ),
    ) -> bool: ...
    def __format__(self, format_type: str) -> str: ...
    @staticmethod
    def correct_unit(unit: str) -> str: ...
    @staticmethod
    def get_unit_symbol(s: str) -> Symbol: ...
    @classmethod
    def from_expr(cls, expr: sp.Basic) -> Quantity[Dimensionality, float]: ...
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
        self: Quantity[Dimensionality, pd.Series], index: int
    ) -> Quantity[DT, float]: ...
    @overload
    def __getitem__(
        self: Quantity[Dimensionality, pl.Series], index: int
    ) -> Quantity[DT, float]: ...
    @overload
    def __getitem__(
        self: Quantity[Dimensionality, pl.Expr], index: int
    ) -> Quantity[DT, pl.Expr]: ...
    @overload
    def __getitem__(
        self: Quantity[Dimensionality, pd.DatetimeIndex], index: int
    ) -> Quantity[DT, pd.Timestamp]: ...
    @overload
    def __getitem__(
        self: Quantity[Dimensionality, np.ndarray], index: int
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
    def __new__(
        cls, val: MT, unit: str | UnitsContainer
    ) -> Quantity[Dimensionality, MT]: ...
    @overload
    def __new__(cls, val: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __new__(cls, val: list) -> Quantity[Dimensionless, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: DimensionlessUnits
    ) -> Quantity[Dimensionless, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: CurrencyUnits
    ) -> Quantity[Currency, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: CurrencyPerEnergyUnits
    ) -> Quantity[CurrencyPerEnergy, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: CurrencyPerVolumeUnits
    ) -> Quantity[CurrencyPerVolume, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: CurrencyPerMassUnits
    ) -> Quantity[CurrencyPerMass, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: CurrencyPerTimeUnits
    ) -> Quantity[CurrencyPerTime, np.ndarray]: ...
    @overload
    def __new__(cls, val: list, unit: LengthUnits) -> Quantity[Length, np.ndarray]: ...
    @overload
    def __new__(cls, val: list, unit: MassUnits) -> Quantity[Mass, np.ndarray]: ...
    @overload
    def __new__(cls, val: list, unit: TimeUnits) -> Quantity[Time, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: TemperatureUnits
    ) -> Quantity[Temperature, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: TemperatureDifferenceUnits
    ) -> Quantity[TemperatureDifference, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: SubstanceUnits
    ) -> Quantity[Substance, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: MolarMassUnits
    ) -> Quantity[MolarMass, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: SubstancePerMassUnits
    ) -> Quantity[SubstancePerMass, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: CurrentUnits
    ) -> Quantity[Current, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: LuminosityUnits
    ) -> Quantity[Luminosity, np.ndarray]: ...
    @overload
    def __new__(cls, val: list, unit: AreaUnits) -> Quantity[Area, np.ndarray]: ...
    @overload
    def __new__(cls, val: list, unit: VolumeUnits) -> Quantity[Volume, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: NormalVolumeUnits
    ) -> Quantity[NormalVolume, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: PressureUnits
    ) -> Quantity[Pressure, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: MassFlowUnits
    ) -> Quantity[MassFlow, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: VolumeFlowUnits
    ) -> Quantity[VolumeFlow, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: NormalVolumeFlowUnits
    ) -> Quantity[NormalVolumeFlow, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: DensityUnits
    ) -> Quantity[Density, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: SpecificVolumeUnits
    ) -> Quantity[SpecificVolume, np.ndarray]: ...
    @overload
    def __new__(cls, val: list, unit: EnergyUnits) -> Quantity[Energy, np.ndarray]: ...
    @overload
    def __new__(cls, val: list, unit: PowerUnits) -> Quantity[Power, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: VelocityUnits
    ) -> Quantity[Velocity, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: DynamicViscosityUnits
    ) -> Quantity[DynamicViscosity, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: KinematicViscosityUnits
    ) -> Quantity[KinematicViscosity, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: EnergyPerMassUnits
    ) -> Quantity[EnergyPerMass, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: SpecificHeatCapacityUnits
    ) -> Quantity[SpecificHeatCapacity, np.ndarray]: ...
    @overload
    def __new__(cls, val: list, unit: Unit[DT]) -> Quantity[DT, np.ndarray]: ...
    @overload
    def __new__(
        cls, val: list, unit: str | UnitsContainer
    ) -> Quantity[Dimensionality, np.ndarray]: ...

    # endregion

    # region: overload __floordiv__, __pow__, __add__, __sub__, __gt__, __ge__, __lt__, __le__

    @overload
    def __floordiv__(
        self: Quantity[Dimensionless, MT], other: float | int
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __floordiv__(
        self, other: Quantity[DT, Any]
    ) -> Quantity[Dimensionless, Any]: ...
    @overload
    def __pow__(self, other: Literal[1]) -> Quantity[DT, MT]: ...
    @overload
    def __pow__(
        self: Quantity[Length, MT], other: Literal[2]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __pow__(
        self: Quantity[Length, MT], other: Literal[3]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __pow__(
        self: Quantity[Dimensionality, Any], other: float | int
    ) -> Quantity[Dimensionality, Any]: ...
    @overload
    def __pow__(
        self: Quantity[Dimensionless, Any], other: float | int
    ) -> Quantity[Dimensionless, Any]: ...
    @overload
    def __pow__(
        self, other: Quantity[Dimensionless, Any]
    ) -> Quantity[Dimensionality, Any]: ...
    @overload
    def __pow__(self, other: float | int) -> Quantity[Dimensionality, Any]: ...
    @overload
    @overload
    def __add__(
        self: Quantity[Dimensionless, Any], other: float | int
    ) -> Quantity[Dimensionless, Any]: ...
    @overload
    def __add__(
        self: Quantity[Temperature, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[Temperature, Any]: ...
    @overload
    def __add__(self, other: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __add__(self, other: Quantity[DT, float]) -> Quantity[DT, MT]: ...
    @overload
    def __add__(
        self: Quantity[DT, float], other: Quantity[DT, MT_]
    ) -> Quantity[DT, MT_]: ...
    @overload
    def __add__(self, other: Quantity[DT, Any]) -> Quantity[DT, Any]: ...
    def __add__(
        self: Quantity[Dimensionality, Any], other
    ) -> Quantity[Dimensionality, Any]: ...
    @overload
    def __sub__(
        self: Quantity[Dimensionless, MT], other: float | int
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __sub__(
        self: Quantity[Temperature, MT], other: Quantity[TemperatureDifference, Any]
    ) -> Quantity[Temperature, Any]: ...
    @overload
    def __sub__(
        self: Quantity[Temperature, MT], other: Quantity[Temperature, Any]
    ) -> Quantity[TemperatureDifference, Any]: ...
    @overload
    def __sub__(self, other: Quantity[DT, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __sub__(self, other: Quantity[DT, float]) -> Quantity[DT, MT]: ...
    @overload
    def __sub__(
        self: Quantity[DT, float], other: Quantity[DT, MT_]
    ) -> Quantity[DT, MT_]: ...
    @overload
    def __sub__(self, other: Quantity[DT, Any]) -> Quantity[DT, Any]: ...
    @overload
    def __sub__(
        self: Quantity[Dimensionality, Any], other
    ) -> Quantity[Dimensionality, Any]: ...
    @overload
    def __gt__(self: Quantity[Dimensionless, float], other: float | int) -> bool: ...
    @overload
    def __gt__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    @overload
    def __ge__(self: Quantity[Dimensionless, float], other: float | int) -> bool: ...
    @overload
    def __ge__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    @overload
    def __lt__(self: Quantity[Dimensionless, float], other: float | int) -> bool: ...
    @overload
    def __lt__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...
    @overload
    def __le__(self: Quantity[Dimensionless, float], other: float | int) -> bool: ...
    @overload
    def __le__(self: Quantity[DT, float], other: Quantity[DT, float]) -> bool: ...

    # endregion

    # region: overload  __rtruediv__

    # region: autogenerated __rtruediv__

    @overload
    def __rtruediv__(
        self: Quantity[Time, MT], other: float | int
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[Density, MT], other: float | int
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[SpecificVolume, MT], other: float | int
    ) -> Quantity[Density, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[Frequency, MT], other: float | int
    ) -> Quantity[Time, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[MolarMass, MT], other: float | int
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[SubstancePerMass, MT], other: float | int
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[MassPerNormalVolume, MT], other: float | int
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[MassPerEnergy, MT], other: float | int
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[NormalVolumePerMass, MT], other: float | int
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __rtruediv__(
        self: Quantity[EnergyPerMass, MT], other: float | int
    ) -> Quantity[MassPerEnergy, MT]: ...

    # endregion

    @overload
    def __rtruediv__(self, other: float | int) -> Quantity[Dimensionality, MT]: ...

    # endregion

    # region: overload __mul__

    # region: autogenerated __mul__

    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[HeatingValue, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatingValue, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[LowerHeatingValue, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[LowerHeatingValue, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[HigherHeatingValue, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HigherHeatingValue, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[NormalTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Normal, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[Length, MT_]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[Area, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[PowerPerLength, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[PowerPerArea, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[PowerPerVolume, MT_]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[ThermalConductivity, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Length, MT], other: Quantity[HeatTransferCoefficient, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Mass, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[NormalVolumeFlow, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[Power, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[CurrencyPerTime, MT_]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[PowerPerVolume, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Time, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[NormalTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[PowerPerTemperature, MT_],
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[ThermalConductivity, MT_],
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[HeatTransferCoefficient, MT_],
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[MolarSpecificEntropy, MT_],
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[TemperatureDifference, MT],
        other: Quantity[SpecificHeatCapacity, MT_],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Substance, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Substance, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[Length, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[PowerPerArea, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[PowerPerVolume, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Area, MT], other: Quantity[HeatTransferCoefficient, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[Density, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Volume, MT], other: Quantity[PowerPerVolume, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolume, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolume, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[Time, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Pressure, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassFlow, MT], other: Quantity[SpecificHeatCapacity, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Density, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[VolumeFlow, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[Time, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Density, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[Density, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificVolume, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Energy, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Energy, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Energy, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Power, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Power, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Power, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Length, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Area, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Velocity, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Length, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[SpecificHeatCapacity, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Length, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Density, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Length, MT_]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Area, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[NormalVolume, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Energy, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Frequency, MT], other: Quantity[Currency, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[Substance, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarMass, MT], other: Quantity[SpecificHeatCapacity, MT_]
    ) -> Quantity[MolarSpecificEntropy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[Density, MT_]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[MolarSpecificEntropy, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SubstancePerMass, MT],
        other: Quantity[MolarSpecificEnthalpy, MT_],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarDensity, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[Currency, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[Energy, MT_]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[Power, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[Density, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerLength, MT], other: Quantity[Length, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerLength, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Length, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Area, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Length, MT_]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Area, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[Length, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[ThermalConductivity, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[Length, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __mul__(
        self: Quantity[HeatTransferCoefficient, MT], other: Quantity[Area, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT], other: Quantity[NormalVolume, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT], other: Quantity[NormalVolumeFlow, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerNormalVolume, MT],
        other: Quantity[NormalVolumePerMass, MT_],
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[Energy, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[Power, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[PowerPerLength, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MassPerEnergy, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEntropy, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEntropy, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[Density, MT_]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __mul__(
        self: Quantity[NormalVolumePerMass, MT],
        other: Quantity[MassPerNormalVolume, MT_],
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Time, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Density, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[Substance, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT],
        other: Quantity[SubstancePerMass, MT_],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __mul__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __mul__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[MolarSpecificEntropy, MT]: ...

    # endregion

    @overload
    def __mul__(
        self: Quantity[Dimensionless, MT], other: Quantity[DT_, MT_]
    ) -> Quantity[DT_, MT]: ...
    @overload
    def __mul__(self, other: Quantity[Dimensionless, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self, other: float | int) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self, other: Quantity[DT_, MT_]) -> Quantity[Dimensionality, MT_]: ...
    @overload
    def __mul__(
        self, other: Quantity[Dimensionality, MT_]
    ) -> Quantity[Dimensionality, MT_]: ...
    @overload
    def __mul__(
        self, other: Quantity[Dimensionality, MT]
    ) -> Quantity[Dimensionality, MT]: ...

    # endregion

    # region: overloads __truediv__

    # TODO: this results in Q[Dimensionality] / Q[Dimensionality] becoming Q[Dimensionless]
    # would need to create a separate typevar for any dimensionality except the base "Dimensionality"
    @overload
    def __truediv__(self, other: Quantity[DT, MT_]) -> Quantity[Dimensionless, MT]: ...

    # region: autogenerated __truediv__

    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[Density, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Normal, MT], other: Quantity[Density, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Normal, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Length, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Length, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Time, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Substance, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[NormalVolume, MT_]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Density, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[Energy, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Mass, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Time, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Time, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Substance, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Substance, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Substance, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Substance, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Area, MT], other: Quantity[Length, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Area, MT], other: Quantity[Time, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Area, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[Length, MT_]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[Time, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[Area, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Volume, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[Time, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[NormalVolumeFlow, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolume, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[Time, MT_]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[Density, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[PowerPerVolume, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Pressure, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Length, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[NormalVolumeFlow, MT_]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Density, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Power, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[NormalVolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MassFlow, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Length, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Area, MT_]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[VolumeFlow, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[NormalVolume, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[NormalVolumePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[NormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumeFlow, MT], other: Quantity[NormalVolumePerMass, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[MassPerNormalVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[MassPerNormalVolume, MT_]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Density, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Substance, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[Power, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Energy, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[Substance, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Length, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[PowerPerTemperature, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Area, MT_]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Energy, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[PowerPerLength, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[PowerPerArea, MT_]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[PowerPerVolume, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[PowerPerTemperature, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Power, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Velocity, MT], other: Quantity[Length, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Velocity, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Time, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[Density, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[PowerPerLength, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DynamicViscosity, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[PowerPerLength, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Length, MT_]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Time, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Area, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[KinematicViscosity, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarMass, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarMass, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[MolarDensity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SubstancePerMass, MT], other: Quantity[MolarDensity, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarDensity, MT], other: Quantity[Density, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarDensity, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[Mass, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[Time, MT_]
    ) -> Quantity[CurrencyPerTime, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[Volume, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[Energy, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[Energy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[Mass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[Volume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Currency, MT], other: Quantity[CurrencyPerTime, MT_]
    ) -> Quantity[Time, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[MassPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerEnergy, MT], other: Quantity[MassPerEnergy, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerMass, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[Density, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerVolume, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[Density, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[CurrencyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[VolumeFlow, MT_]
    ) -> Quantity[CurrencyPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[Power, MT_]
    ) -> Quantity[CurrencyPerEnergy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Currency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[Currency, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[CurrencyPerEnergy, MT_]
    ) -> Quantity[Power, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[CurrencyPerMass, MT_]
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[CurrencyPerTime, MT], other: Quantity[CurrencyPerVolume, MT_]
    ) -> Quantity[VolumeFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[Length, MT_]
    ) -> Quantity[PowerPerArea, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[Area, MT_]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[PowerPerArea, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[PowerPerVolume, MT_]
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[ThermalConductivity, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerLength, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Length, MT_]
    ) -> Quantity[PowerPerVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[PowerPerVolume, MT_]
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerArea, MT], other: Quantity[HeatTransferCoefficient, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerVolume, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT], other: Quantity[Length, MT_]
    ) -> Quantity[ThermalConductivity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT], other: Quantity[Area, MT_]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT], other: Quantity[MassFlow, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[ThermalConductivity, MT_],
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[HeatTransferCoefficient, MT_],
    ) -> Quantity[Area, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[PowerPerTemperature, MT],
        other: Quantity[SpecificHeatCapacity, MT_],
    ) -> Quantity[MassFlow, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[Length, MT_]
    ) -> Quantity[HeatTransferCoefficient, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT], other: Quantity[DynamicViscosity, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT],
        other: Quantity[HeatTransferCoefficient, MT_],
    ) -> Quantity[Length, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[ThermalConductivity, MT],
        other: Quantity[SpecificHeatCapacity, MT_],
    ) -> Quantity[DynamicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEntropy, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEntropy, MT],
        other: Quantity[SpecificHeatCapacity, MT_],
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalVolumePerMass, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalTemperature, MT], other: Quantity[Normal, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[NormalTemperature, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[Normal, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[TemperatureDifference, MT_]
    ) -> Quantity[SpecificHeatCapacity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Pressure, MT_]
    ) -> Quantity[SpecificVolume, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[SpecificVolume, MT_]
    ) -> Quantity[Pressure, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Velocity, MT_]
    ) -> Quantity[Velocity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[KinematicViscosity, MT_]
    ) -> Quantity[Frequency, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[Frequency, MT_]
    ) -> Quantity[KinematicViscosity, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[MolarSpecificEnthalpy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[MolarSpecificEnthalpy, MT_]
    ) -> Quantity[SubstancePerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[EnergyPerMass, MT], other: Quantity[SpecificHeatCapacity, MT_]
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT],
        other: Quantity[TemperatureDifference, MT_],
    ) -> Quantity[MolarSpecificEntropy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[MolarMass, MT_]
    ) -> Quantity[EnergyPerMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT],
        other: Quantity[MolarSpecificEntropy, MT_],
    ) -> Quantity[TemperatureDifference, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[MolarSpecificEnthalpy, MT], other: Quantity[EnergyPerMass, MT_]
    ) -> Quantity[MolarMass, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SpecificHeatCapacity, MT], other: Quantity[SubstancePerMass, MT_]
    ) -> Quantity[MolarSpecificEntropy, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[SpecificHeatCapacity, MT],
        other: Quantity[MolarSpecificEntropy, MT_],
    ) -> Quantity[SubstancePerMass, MT]: ...

    # endregion

    @overload
    def __truediv__(self, other: Quantity[Dimensionless, MT_]) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self, other: float | int) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(
        self, other: Quantity[DT_, MT_]
    ) -> Quantity[Dimensionality, MT]: ...
    @overload
    def __truediv__(
        self, other: Quantity[Dimensionality, MT_]
    ) -> Quantity[Dimensionality, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionality, MT], other: Quantity[Dimensionality, MT_]
    ) -> Quantity[Dimensionality, MT]: ...

    # endregion
