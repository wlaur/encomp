from typing import (Generic,
                    Literal,
                    Generator,
                    Any,
                    overload,
                    SupportsAbs,
                    SupportsRound)

import numpy as np
import pandas as pd
import polars as pl
import sympy as sp
from _typeshed import Incomplete


from pint.util import UnitsContainer
from pint.facets.plain.quantity import PlainQuantity
from pint.facets.measurement.objects import MeasurementQuantity
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.nonmultiplicative.objects import NonMultiplicativeQuantity
from pint.facets.plain.unit import PlainUnit
from pint.facets.numpy.unit import NumpyUnit
from pint.facets.formatting.objects import FormattingQuantity, FormattingUnit

from pint.util import UnitsContainer
from pint.registry import UnitRegistry

# this is not consistent with units.py
from pint.errors import DimensionalityError as _DimensionalityError

from .utypes import (DimensionlessUnits,
                     CurrencyUnits,
                     CurrencyPerEnergyUnits,
                     CurrencyPerMassUnits,
                     CurrencyPerVolumeUnits,
                     CurrencyPerTimeUnits,
                     LengthUnits,
                     MassUnits,
                     TimeUnits,
                     TemperatureUnits,
                     TemperatureDifferenceUnits,
                     SubstanceUnits,
                     MolarMassUnits,
                     SubstancePerMassUnits,
                     CurrentUnits,
                     LuminosityUnits,
                     AreaUnits,
                     VolumeUnits,
                     NormalVolumeUnits,
                     PressureUnits,
                     MassFlowUnits,
                     VolumeFlowUnits,
                     NormalVolumeFlowUnits,
                     DensityUnits,
                     SpecificVolumeUnits,
                     EnergyUnits,
                     PowerUnits,
                     VelocityUnits,
                     DynamicViscosityUnits,
                     EnergyPerMassUnits,
                     KinematicViscosityUnits,
                     SpecificHeatCapacityUnits)

from .utypes import (DT,
                     DT_,
                     MT,
                     Dimensionality,
                     Unknown,
                     Dimensionless,
                     Currency,
                     CurrencyPerEnergy,
                     CurrencyPerMass,
                     CurrencyPerVolume,
                     CurrencyPerTime,
                     Substance,
                     MolarMass,
                     SubstancePerMass,
                     Density,
                     Energy,
                     Power,
                     Time,
                     Temperature,
                     TemperatureDifference,
                     Length,
                     Area,
                     Volume,
                     Mass,
                     MassFlow,
                     VolumeFlow,
                     NormalVolume,
                     NormalVolumeFlow,
                     SpecificVolume,
                     Current,
                     Luminosity,
                     Pressure,
                     Velocity,
                     DynamicViscosity,
                     KinematicViscosity,
                     ThermalConductivity,
                     MolarSpecificEnthalpy,
                     EnergyPerMass,
                     HeatingValue,
                     HigherHeatingValue,
                     LowerHeatingValue,
                     Frequency,
                     MassPerEnergy,
                     MassPerNormalVolume,
                     MolarDensity,
                     Normal,
                     SpecificHeatCapacity,
                     HeatTransferCoefficient,
                     PowerPerLength,
                     PowerPerArea,
                     PowerPerTemperature,
                     PowerPerVolume,
                     MolarSpecificEntropy)


# this is not consistent with units.py, tries to
# avoid issue where mypy thinks DimensionalityError is not defined
# in units.py (it is directly imported from pint.errors)
class DimensionalityError(_DimensionalityError):
    msg: str
    def __init__(self, msg: str = ...) -> None: ...


class ExpectedDimensionalityError(DimensionalityError):
    ...


class DimensionalityTypeError(DimensionalityError):
    ...


class DimensionalityComparisonError(DimensionalityError):
    ...


class DimensionalityRedefinitionError(ValueError):
    ...


# this instance is created via the LazyRegistry constructor
# however, it will be correctly set to a UnitRegistry class in the "__init" method
# (note that this is not the same method as "__init__")
ureg: UnitRegistry

CUSTOM_DIMENSIONS: list[str]


def define_dimensionality(name: str, symbol: str = ...) -> None: ...
def set_quantity_format(fmt: str = ...) -> None: ...


class Unit(PlainUnit,
           NumpyUnit,
           FormattingUnit,
           Generic[DT]):
    ...


class Quantity(
    NonMultiplicativeQuantity,
    PlainQuantity,
    MeasurementQuantity,
    NumpyQuantity,
    FormattingQuantity,
    Generic[DT, MT],
    SupportsAbs,
    SupportsRound
):

    _magnitude: MT
    _magnitude_type: type[MT]

    def __hash__(self) -> int: ...
    def __class_getitem__(cls, types: type[DT] | tuple[type[DT], type[MT]]) -> type[Quantity[DT, MT]]: ...
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
    def ndim(self) -> int: ...
    def to_reduced_units(self) -> Quantity[DT, MT]: ...
    def to_base_units(self) -> Quantity[DT, MT]: ...
    def asdim(self, other: type[DT_] | Quantity[DT_, MT]) -> Quantity[DT_, MT]: ...
    def to(self, unit: Unit[DT] | UnitsContainer | str | dict) -> Quantity[DT, MT]: ...
    def ito(self, unit: Unit[DT] | UnitsContainer | str) -> None: ...
    def check(self, unit: Quantity[Unknown, Any] | UnitsContainer | Unit |
              str | Dimensionality | type[Dimensionality]) -> bool: ...

    def __format__(self, format_type: str) -> str: ...
    @staticmethod
    def correct_unit(unit: str) -> str: ...
    @staticmethod
    def get_unit_symbol(s: str) -> sp.Symbol: ...
    @classmethod
    def get_dimension_symbol_map(cls) -> dict[sp.Basic, Unit]: ...
    @classmethod
    def from_expr(cls, expr: sp.Basic) -> Quantity[Unknown, float]: ...
    @classmethod
    def __get_validators__(cls) -> Generator[Incomplete, None, None]: ...
    @classmethod
    def validate(cls, qty: Quantity[DT, MT]) -> Quantity[DT, MT]: ...

    def is_compatible_with(self, other: Quantity | float,
                           *contexts, **ctx_kwargs) -> bool: ...

    def check_compatibility(self, other: Quantity | float) -> None: ...

    @overload
    def __getitem__(self: Quantity[Any, list[int]], index: int) -> Quantity[DT, int]: ...

    @overload
    def __getitem__(self: Quantity[Any, list[float]], index: int) -> Quantity[DT, float]: ...

    @overload
    def __getitem__(self: Quantity[Any, pd.Series], index: int) -> Quantity[DT, float]: ...

    @overload
    def __getitem__(self: Quantity[Any, pl.Series], index: int) -> Quantity[DT, float]: ...

    @overload
    def __getitem__(self: Quantity[Any, pd.DatetimeIndex], index: int) -> Quantity[DT, pd.Timestamp]: ...

    @overload
    def __getitem__(self: Quantity[Any, np.ndarray], index: int) -> Quantity[DT, float]: ...

    def __round__(self, ndigits: int | None = None) -> Quantity[DT, MT]: ...

    def __abs__(self) -> Quantity[DT, MT]: ...
    def __pos__(self) -> Quantity[DT, MT]: ...
    def __neg__(self) -> Quantity[DT, MT]: ...

    def __eq__(self, other: Any) -> bool: ...

    def __rsub__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]:
        ...

    def __radd__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]:
        ...

    def __rmul__(self, other: float) -> Quantity[DT, MT]:
        ...

    def __rpow__(self: Quantity[Dimensionless, MT], other: float) -> float:
        ...

    def __rfloordiv__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]:
        ...

    def __copy__(self) -> Quantity[DT, MT]: ...

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Quantity[DT, MT]: ...

    @overload
    def __new__(cls, val: MT) -> Quantity[Dimensionless, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: DimensionlessUnits
                ) -> Quantity[Dimensionless, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: CurrencyUnits
                ) -> Quantity[Currency, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: CurrencyPerEnergyUnits
                ) -> Quantity[CurrencyPerEnergy, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: CurrencyPerVolumeUnits
                ) -> Quantity[CurrencyPerVolume, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: CurrencyPerMassUnits
                ) -> Quantity[CurrencyPerMass, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: CurrencyPerTimeUnits
                ) -> Quantity[CurrencyPerTime, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: LengthUnits) -> Quantity[Length, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: MassUnits) -> Quantity[Mass, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: TimeUnits) -> Quantity[Time, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: TemperatureUnits) -> Quantity[Temperature, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: TemperatureDifferenceUnits) -> Quantity[TemperatureDifference, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: SubstanceUnits) -> Quantity[Substance, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: MolarMassUnits) -> Quantity[MolarMass, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: SubstancePerMassUnits) -> Quantity[SubstancePerMass, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: CurrentUnits) -> Quantity[Current, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: LuminosityUnits) -> Quantity[Luminosity, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: AreaUnits) -> Quantity[Area, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: VolumeUnits) -> Quantity[Volume, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: NormalVolumeUnits) -> Quantity[NormalVolume, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: PressureUnits) -> Quantity[Pressure, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: MassFlowUnits) -> Quantity[MassFlow, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: VolumeFlowUnits) -> Quantity[VolumeFlow, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: NormalVolumeFlowUnits) -> Quantity[NormalVolumeFlow, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: DensityUnits) -> Quantity[Density, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: SpecificVolumeUnits) -> Quantity[SpecificVolume, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: EnergyUnits) -> Quantity[Energy, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: PowerUnits) -> Quantity[Power, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: VelocityUnits) -> Quantity[Velocity, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: DynamicViscosityUnits) -> Quantity[DynamicViscosity, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: KinematicViscosityUnits) -> Quantity[KinematicViscosity, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: EnergyPerMassUnits) -> Quantity[EnergyPerMass, MT]:
        ...

    @overload
    def __new__(cls, val: MT,
                unit: SpecificHeatCapacityUnits) -> Quantity[SpecificHeatCapacity, MT]:
        ...

    @overload
    def __new__(
        cls,
        val: MT,
        unit: Unit[DT] | UnitsContainer | str | Quantity[DT, Any] | None = None,
        _dt: type[DT] = Unknown,
        _allow_quantity_input: bool = False
    ) -> Quantity[DT, MT]:
        ...

    # region: autogenerated rdiv

    @overload
    def __rtruediv__(self: Quantity[Time], other: float  # type: ignore
                     ) -> Quantity[Frequency]:
        ...

    @overload
    def __rtruediv__(self: Quantity[Density], other: float  # type: ignore
                     ) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __rtruediv__(self: Quantity[SpecificVolume], other: float  # type: ignore
                     ) -> Quantity[Density]:
        ...

    @overload
    def __rtruediv__(self: Quantity[Frequency], other: float  # type: ignore
                     ) -> Quantity[Time]:
        ...

    @overload
    def __rtruediv__(self: Quantity[MassPerEnergy], other: float  # type: ignore
                     ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __rtruediv__(self: Quantity[EnergyPerMass], other: float  # type: ignore
                     ) -> Quantity[MassPerEnergy]:
        ...

    # endregion

    @overload
    def __rtruediv__(self, other: float) -> Quantity[Unknown]:
        ...

    @overload
    def __mul__(self: Quantity[Unknown], other) -> Quantity[Unknown]:  # type: ignore
        ...

    @overload
    def __mul__(self, other: Quantity[Unknown]) -> Quantity[Unknown]:  # type: ignore
        ...

    # region: autogenerated mul

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[EnergyPerMass]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[EnergyPerMass], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[HeatingValue]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[HeatingValue], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[LowerHeatingValue]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[LowerHeatingValue], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[HigherHeatingValue]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[HigherHeatingValue], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Normal], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __mul__(self: Quantity[Normal], other: Quantity[VolumeFlow]  # type: ignore
                ) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Normal], other: Quantity[MassPerNormalVolume]  # type: ignore
                ) -> Quantity[Density]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[Length]  # type: ignore
                ) -> Quantity[Area]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[Area]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[Velocity]  # type: ignore
                ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[DynamicViscosity]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[KinematicViscosity]  # type: ignore
                ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[Velocity]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[PowerPerLength]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[PowerPerArea]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[PowerPerVolume]  # type: ignore
                ) -> Quantity[PowerPerArea]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[ThermalConductivity]  # type: ignore
                ) -> Quantity[PowerPerTemperature]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[HeatTransferCoefficient]  # type: ignore
                ) -> Quantity[ThermalConductivity]:
        ...

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[SpecificVolume]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[CurrencyPerMass]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[EnergyPerMass]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[Pressure]  # type: ignore
                ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[MassFlow]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[VolumeFlow]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[NormalVolumeFlow]  # type: ignore
                ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[Power]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[Velocity]  # type: ignore
                ) -> Quantity[Length]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[KinematicViscosity]  # type: ignore
                ) -> Quantity[Area]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[CurrencyPerTime]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[PowerPerVolume]  # type: ignore
                ) -> Quantity[Pressure]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[EnergyPerMass]  # type: ignore
                ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[TemperatureDifference], other: Quantity[PowerPerTemperature]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[TemperatureDifference], other: Quantity[ThermalConductivity]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[TemperatureDifference], other: Quantity[HeatTransferCoefficient]  # type: ignore
                ) -> Quantity[PowerPerArea]:
        ...

    @overload
    def __mul__(self: Quantity[TemperatureDifference], other: Quantity[MolarSpecificEntropy]  # type: ignore
                ) -> Quantity[MolarSpecificEnthalpy]:
        ...

    @overload
    def __mul__(self: Quantity[TemperatureDifference], other: Quantity[SpecificHeatCapacity]  # type: ignore
                ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[Substance], other: Quantity[MolarMass]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[Substance], other: Quantity[MolarSpecificEnthalpy]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Area], other: Quantity[Length]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Area], other: Quantity[Velocity]  # type: ignore
                ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Area], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[Area], other: Quantity[PowerPerArea]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[Area], other: Quantity[PowerPerVolume]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[Area], other: Quantity[HeatTransferCoefficient]  # type: ignore
                ) -> Quantity[PowerPerTemperature]:
        ...

    @overload
    def __mul__(self: Quantity[Volume], other: Quantity[Normal]  # type: ignore
                ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __mul__(self: Quantity[Volume], other: Quantity[Pressure]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Volume], other: Quantity[Density]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[Volume], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Volume], other: Quantity[MolarDensity]  # type: ignore
                ) -> Quantity[Substance]:
        ...

    @overload
    def __mul__(self: Quantity[Volume], other: Quantity[CurrencyPerVolume]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Volume], other: Quantity[PowerPerVolume]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[NormalVolume], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[NormalVolume], other: Quantity[MassPerNormalVolume]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[Pressure], other: Quantity[Time]  # type: ignore
                ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[Pressure], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Pressure], other: Quantity[VolumeFlow]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[Pressure], other: Quantity[SpecificVolume]  # type: ignore
                ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[Pressure], other: Quantity[Velocity]  # type: ignore
                ) -> Quantity[PowerPerArea]:
        ...

    @overload
    def __mul__(self: Quantity[Pressure], other: Quantity[KinematicViscosity]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[Pressure], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[PowerPerVolume]:
        ...

    @overload
    def __mul__(self: Quantity[Pressure], other: Quantity[CurrencyPerEnergy]  # type: ignore
                ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __mul__(self: Quantity[Pressure], other: Quantity[MassPerEnergy]  # type: ignore
                ) -> Quantity[Density]:
        ...

    @overload
    def __mul__(self: Quantity[MassFlow], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[MassFlow], other: Quantity[SpecificVolume]  # type: ignore
                ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[MassFlow], other: Quantity[KinematicViscosity]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[MassFlow], other: Quantity[CurrencyPerMass]  # type: ignore
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __mul__(self: Quantity[MassFlow], other: Quantity[EnergyPerMass]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[MassFlow], other: Quantity[SpecificHeatCapacity]  # type: ignore
                ) -> Quantity[PowerPerTemperature]:
        ...

    @overload
    def __mul__(self: Quantity[VolumeFlow], other: Quantity[Normal]  # type: ignore
                ) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[VolumeFlow], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[VolumeFlow], other: Quantity[Pressure]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[VolumeFlow], other: Quantity[Density]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[VolumeFlow], other: Quantity[DynamicViscosity]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[VolumeFlow], other: Quantity[CurrencyPerVolume]  # type: ignore
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __mul__(self: Quantity[NormalVolumeFlow], other: Quantity[Time]  # type: ignore
                ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __mul__(self: Quantity[NormalVolumeFlow], other: Quantity[MassPerNormalVolume]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Density], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[Density], other: Quantity[VolumeFlow]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Density], other: Quantity[SpecificVolume]  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __mul__(self: Quantity[Density], other: Quantity[KinematicViscosity]  # type: ignore
                ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[Density], other: Quantity[CurrencyPerMass]  # type: ignore
                ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __mul__(self: Quantity[Density], other: Quantity[EnergyPerMass]  # type: ignore
                ) -> Quantity[Pressure]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificVolume], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificVolume], other: Quantity[Pressure]  # type: ignore
                ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificVolume], other: Quantity[MassFlow]  # type: ignore
                ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificVolume], other: Quantity[Density]  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificVolume], other: Quantity[DynamicViscosity]  # type: ignore
                ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificVolume], other: Quantity[CurrencyPerVolume]  # type: ignore
                ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[Energy], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[Energy], other: Quantity[CurrencyPerEnergy]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Energy], other: Quantity[MassPerEnergy]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[Power], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Power], other: Quantity[CurrencyPerEnergy]  # type: ignore
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __mul__(self: Quantity[Power], other: Quantity[MassPerEnergy]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Velocity], other: Quantity[Length]  # type: ignore
                ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[Velocity], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Length]:
        ...

    @overload
    def __mul__(self: Quantity[Velocity], other: Quantity[Area]  # type: ignore
                ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Velocity], other: Quantity[Pressure]  # type: ignore
                ) -> Quantity[PowerPerArea]:
        ...

    @overload
    def __mul__(self: Quantity[Velocity], other: Quantity[Velocity]  # type: ignore
                ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[DynamicViscosity], other: Quantity[Length]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[DynamicViscosity], other: Quantity[VolumeFlow]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[DynamicViscosity], other: Quantity[SpecificVolume]  # type: ignore
                ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[DynamicViscosity], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[Pressure]:
        ...

    @overload
    def __mul__(self: Quantity[DynamicViscosity], other: Quantity[EnergyPerMass]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[DynamicViscosity], other: Quantity[SpecificHeatCapacity]  # type: ignore
                ) -> Quantity[ThermalConductivity]:
        ...

    @overload
    def __mul__(self: Quantity[KinematicViscosity], other: Quantity[Length]  # type: ignore
                ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[KinematicViscosity], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Area]:
        ...

    @overload
    def __mul__(self: Quantity[KinematicViscosity], other: Quantity[Pressure]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[KinematicViscosity], other: Quantity[MassFlow]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[KinematicViscosity], other: Quantity[Density]  # type: ignore
                ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[KinematicViscosity], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[KinematicViscosity], other: Quantity[MassPerEnergy]  # type: ignore
                ) -> Quantity[Time]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[Length]  # type: ignore
                ) -> Quantity[Velocity]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[Area]  # type: ignore
                ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[NormalVolume]  # type: ignore
                ) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[Pressure]  # type: ignore
                ) -> Quantity[PowerPerVolume]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[Energy]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[DynamicViscosity]  # type: ignore
                ) -> Quantity[Pressure]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[KinematicViscosity]  # type: ignore
                ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[Frequency], other: Quantity[Currency]  # type: ignore
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __mul__(self: Quantity[MolarMass], other: Quantity[Substance]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[MolarMass], other: Quantity[MolarDensity]  # type: ignore
                ) -> Quantity[Density]:
        ...

    @overload
    def __mul__(self: Quantity[MolarMass], other: Quantity[EnergyPerMass]  # type: ignore
                ) -> Quantity[MolarSpecificEnthalpy]:
        ...

    @overload
    def __mul__(self: Quantity[MolarMass], other: Quantity[SpecificHeatCapacity]  # type: ignore
                ) -> Quantity[MolarSpecificEntropy]:
        ...

    @overload
    def __mul__(self: Quantity[MolarDensity], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[Substance]:
        ...

    @overload
    def __mul__(self: Quantity[MolarDensity], other: Quantity[MolarMass]  # type: ignore
                ) -> Quantity[Density]:
        ...

    @overload
    def __mul__(self: Quantity[MolarDensity], other: Quantity[MolarSpecificEnthalpy]  # type: ignore
                ) -> Quantity[Pressure]:
        ...

    @overload
    def __mul__(self: Quantity[Currency], other: Quantity[Frequency]  # type: ignore
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerEnergy], other: Quantity[Pressure]  # type: ignore
                ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerEnergy], other: Quantity[Energy]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerEnergy], other: Quantity[Power]  # type: ignore
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerEnergy], other: Quantity[EnergyPerMass]  # type: ignore
                ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerMass], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerMass], other: Quantity[MassFlow]  # type: ignore
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerMass], other: Quantity[Density]  # type: ignore
                ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerMass], other: Quantity[MassPerEnergy]  # type: ignore
                ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerVolume], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerVolume], other: Quantity[VolumeFlow]  # type: ignore
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerVolume], other: Quantity[SpecificVolume]  # type: ignore
                ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerTime], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[PowerPerLength], other: Quantity[Length]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[PowerPerLength], other: Quantity[MassPerEnergy]  # type: ignore
                ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[PowerPerArea], other: Quantity[Length]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[PowerPerArea], other: Quantity[Area]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[PowerPerVolume], other: Quantity[Length]  # type: ignore
                ) -> Quantity[PowerPerArea]:
        ...

    @overload
    def __mul__(self: Quantity[PowerPerVolume], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Pressure]:
        ...

    @overload
    def __mul__(self: Quantity[PowerPerVolume], other: Quantity[Area]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[PowerPerVolume], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[PowerPerTemperature], other: Quantity[TemperatureDifference]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[ThermalConductivity], other: Quantity[Length]  # type: ignore
                ) -> Quantity[PowerPerTemperature]:
        ...

    @overload
    def __mul__(self: Quantity[ThermalConductivity], other: Quantity[TemperatureDifference]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[HeatTransferCoefficient], other: Quantity[Length]  # type: ignore
                ) -> Quantity[ThermalConductivity]:
        ...

    @overload
    def __mul__(self: Quantity[HeatTransferCoefficient], other: Quantity[TemperatureDifference]  # type: ignore
                ) -> Quantity[PowerPerArea]:
        ...

    @overload
    def __mul__(self: Quantity[HeatTransferCoefficient], other: Quantity[Area]  # type: ignore
                ) -> Quantity[PowerPerTemperature]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerNormalVolume], other: Quantity[Normal]  # type: ignore
                ) -> Quantity[Density]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerNormalVolume], other: Quantity[NormalVolume]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerNormalVolume], other: Quantity[NormalVolumeFlow]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerEnergy], other: Quantity[Pressure]  # type: ignore
                ) -> Quantity[Density]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerEnergy], other: Quantity[Energy]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerEnergy], other: Quantity[Power]  # type: ignore
                ) -> Quantity[MassFlow]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerEnergy], other: Quantity[KinematicViscosity]  # type: ignore
                ) -> Quantity[Time]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerEnergy], other: Quantity[CurrencyPerMass]  # type: ignore
                ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerEnergy], other: Quantity[PowerPerLength]  # type: ignore
                ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerEnergy], other: Quantity[EnergyPerMass]  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __mul__(self: Quantity[MassPerEnergy], other: Quantity[MolarSpecificEnthalpy]  # type: ignore
                ) -> Quantity[MolarMass]:
        ...

    @overload
    def __mul__(self: Quantity[MolarSpecificEntropy], other: Quantity[TemperatureDifference]  # type: ignore
                ) -> Quantity[MolarSpecificEnthalpy]:
        ...

    @overload
    def __mul__(self: Quantity[EnergyPerMass], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[EnergyPerMass], other: Quantity[Time]  # type: ignore
                ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __mul__(self: Quantity[EnergyPerMass], other: Quantity[MassFlow]  # type: ignore
                ) -> Quantity[Power]:
        ...

    @overload
    def __mul__(self: Quantity[EnergyPerMass], other: Quantity[Density]  # type: ignore
                ) -> Quantity[Pressure]:
        ...

    @overload
    def __mul__(self: Quantity[EnergyPerMass], other: Quantity[DynamicViscosity]  # type: ignore
                ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __mul__(self: Quantity[EnergyPerMass], other: Quantity[MolarMass]  # type: ignore
                ) -> Quantity[MolarSpecificEnthalpy]:
        ...

    @overload
    def __mul__(self: Quantity[EnergyPerMass], other: Quantity[CurrencyPerEnergy]  # type: ignore
                ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[EnergyPerMass], other: Quantity[MassPerEnergy]  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __mul__(self: Quantity[MolarSpecificEnthalpy], other: Quantity[Substance]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[MolarSpecificEnthalpy], other: Quantity[MolarDensity]  # type: ignore
                ) -> Quantity[Pressure]:
        ...

    @overload
    def __mul__(self: Quantity[MolarSpecificEnthalpy], other: Quantity[MassPerEnergy]  # type: ignore
                ) -> Quantity[MolarMass]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificHeatCapacity], other: Quantity[TemperatureDifference]  # type: ignore
                ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificHeatCapacity], other: Quantity[MassFlow]  # type: ignore
                ) -> Quantity[PowerPerTemperature]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificHeatCapacity], other: Quantity[DynamicViscosity]  # type: ignore
                ) -> Quantity[ThermalConductivity]:
        ...

    @overload
    def __mul__(self: Quantity[SpecificHeatCapacity], other: Quantity[MolarMass]  # type: ignore
                ) -> Quantity[MolarSpecificEntropy]:
        ...

    # endregion

    @overload
    def __mul__(self: Quantity[Dimensionless], other: Quantity[DT_]  # type: ignore
                ) -> Quantity[DT_]:
        ...

    @overload
    def __mul__(self, other: Quantity[Dimensionless]  # type: ignore
                ) -> Quantity[DT]:
        ...

    @overload
    def __mul__(self, other: float) -> Quantity[DT]:
        ...

    @overload
    def __mul__(self, other: Quantity[DT_]) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self: Quantity[Unknown], other: Quantity[Unknown]  # type: ignore
                    ) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self, other: Quantity[DT]  # type: ignore
                    ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __truediv__(self: Quantity[Unknown], other  # type: ignore
                    ) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self, other: Quantity[Unknown]  # type: ignore
                    ) -> Quantity[Unknown]:
        ...

    # region: autogenerated div

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[Mass]  # type: ignore
                    ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[EnergyPerMass], other: Quantity[TemperatureDifference]  # type: ignore
                    ) -> Quantity[SpecificHeatCapacity]:
        ...

    @overload
    def __truediv__(self: Quantity[Dimensionless], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[Dimensionless], other: Quantity[Density]  # type: ignore
                    ) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Dimensionless], other: Quantity[SpecificVolume]  # type: ignore
                    ) -> Quantity[Density]:
        ...

    @overload
    def __truediv__(self: Quantity[Dimensionless], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[Dimensionless], other: Quantity[MassPerEnergy]  # type: ignore
                    ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[Dimensionless], other: Quantity[EnergyPerMass]  # type: ignore
                    ) -> Quantity[MassPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[Length], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[Velocity]:
        ...

    @overload
    def __truediv__(self: Quantity[Length], other: Quantity[Velocity]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[MassFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[Substance]  # type: ignore
                    ) -> Quantity[MolarMass]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[Density]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[NormalVolume]  # type: ignore
                    ) -> Quantity[MassPerNormalVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[MassFlow]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[Density]  # type: ignore
                    ) -> Quantity[Volume]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[Energy]  # type: ignore
                    ) -> Quantity[MassPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[MolarMass]  # type: ignore
                    ) -> Quantity[Substance]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[MassPerNormalVolume]  # type: ignore
                    ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[MassPerEnergy]  # type: ignore
                    ) -> Quantity[Energy]:
        ...

    @overload
    def __truediv__(self: Quantity[Time], other: Quantity[KinematicViscosity]  # type: ignore
                    ) -> Quantity[MassPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[Time], other: Quantity[MassPerEnergy]  # type: ignore
                    ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[Substance], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[MolarDensity]:
        ...

    @overload
    def __truediv__(self: Quantity[Substance], other: Quantity[MolarDensity]  # type: ignore
                    ) -> Quantity[Volume]:
        ...

    @overload
    def __truediv__(self: Quantity[Area], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[Area], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[Area], other: Quantity[KinematicViscosity]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[Area]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Mass]  # type: ignore
                    ) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Area]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[VolumeFlow]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[SpecificVolume]  # type: ignore
                    ) -> Quantity[Mass]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolume], other: Quantity[Normal]  # type: ignore
                    ) -> Quantity[Volume]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolume], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolume], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[Normal]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolume], other: Quantity[NormalVolumeFlow]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[Pressure], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[PowerPerVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Pressure], other: Quantity[Density]  # type: ignore
                    ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[Pressure], other: Quantity[DynamicViscosity]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[Pressure], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[Pressure], other: Quantity[MolarDensity]  # type: ignore
                    ) -> Quantity[MolarSpecificEnthalpy]:
        ...

    @overload
    def __truediv__(self: Quantity[Pressure], other: Quantity[PowerPerVolume]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[Pressure], other: Quantity[EnergyPerMass]  # type: ignore
                    ) -> Quantity[Density]:
        ...

    @overload
    def __truediv__(self: Quantity[Pressure], other: Quantity[MolarSpecificEnthalpy]  # type: ignore
                    ) -> Quantity[MolarDensity]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[Mass]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[VolumeFlow]  # type: ignore
                    ) -> Quantity[Density]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[NormalVolumeFlow]  # type: ignore
                    ) -> Quantity[MassPerNormalVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[Density]  # type: ignore
                    ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[Power]  # type: ignore
                    ) -> Quantity[MassPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[DynamicViscosity]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[Mass]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[MassPerNormalVolume]  # type: ignore
                    ) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[MassFlow], other: Quantity[MassPerEnergy]  # type: ignore
                    ) -> Quantity[Power]:
        ...

    @overload
    def __truediv__(self: Quantity[VolumeFlow], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[VolumeFlow], other: Quantity[Area]  # type: ignore
                    ) -> Quantity[Velocity]:
        ...

    @overload
    def __truediv__(self: Quantity[VolumeFlow], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[VolumeFlow], other: Quantity[MassFlow]  # type: ignore
                    ) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[VolumeFlow], other: Quantity[SpecificVolume]  # type: ignore
                    ) -> Quantity[MassFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[VolumeFlow], other: Quantity[Velocity]  # type: ignore
                    ) -> Quantity[Area]:
        ...

    @overload
    def __truediv__(self: Quantity[VolumeFlow], other: Quantity[KinematicViscosity]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[VolumeFlow], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[Volume]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolumeFlow], other: Quantity[Normal]  # type: ignore
                    ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolumeFlow], other: Quantity[NormalVolume]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolumeFlow], other: Quantity[VolumeFlow]  # type: ignore
                    ) -> Quantity[Normal]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolumeFlow], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Density], other: Quantity[Normal]  # type: ignore
                    ) -> Quantity[MassPerNormalVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Density], other: Quantity[Pressure]  # type: ignore
                    ) -> Quantity[MassPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[Density], other: Quantity[MolarMass]  # type: ignore
                    ) -> Quantity[MolarDensity]:
        ...

    @overload
    def __truediv__(self: Quantity[Density], other: Quantity[MolarDensity]  # type: ignore
                    ) -> Quantity[MolarMass]:
        ...

    @overload
    def __truediv__(self: Quantity[Density], other: Quantity[MassPerNormalVolume]  # type: ignore
                    ) -> Quantity[Normal]:
        ...

    @overload
    def __truediv__(self: Quantity[Density], other: Quantity[MassPerEnergy]  # type: ignore
                    ) -> Quantity[Pressure]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[Mass]  # type: ignore
                    ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[Power]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[Substance]  # type: ignore
                    ) -> Quantity[MolarSpecificEnthalpy]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[Pressure]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[Pressure]  # type: ignore
                    ) -> Quantity[Volume]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[MassFlow]  # type: ignore
                    ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[VolumeFlow]  # type: ignore
                    ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[Power]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[DynamicViscosity]  # type: ignore
                    ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[KinematicViscosity]  # type: ignore
                    ) -> Quantity[MassFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[EnergyPerMass]  # type: ignore
                    ) -> Quantity[Mass]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[MolarSpecificEnthalpy]  # type: ignore
                    ) -> Quantity[Substance]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[TemperatureDifference]  # type: ignore
                    ) -> Quantity[PowerPerTemperature]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[Area]  # type: ignore
                    ) -> Quantity[PowerPerArea]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[PowerPerVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[Pressure]  # type: ignore
                    ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[MassFlow]  # type: ignore
                    ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[VolumeFlow]  # type: ignore
                    ) -> Quantity[Pressure]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[Energy]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[Energy]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[PowerPerLength]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[PowerPerArea]  # type: ignore
                    ) -> Quantity[Area]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[PowerPerVolume]  # type: ignore
                    ) -> Quantity[Volume]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[PowerPerTemperature]  # type: ignore
                    ) -> Quantity[TemperatureDifference]:
        ...

    @overload
    def __truediv__(self: Quantity[Power], other: Quantity[EnergyPerMass]  # type: ignore
                    ) -> Quantity[MassFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Velocity], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[Velocity], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[DynamicViscosity], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[Pressure]:
        ...

    @overload
    def __truediv__(self: Quantity[DynamicViscosity], other: Quantity[Pressure]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[DynamicViscosity], other: Quantity[Density]  # type: ignore
                    ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[DynamicViscosity], other: Quantity[KinematicViscosity]  # type: ignore
                    ) -> Quantity[Density]:
        ...

    @overload
    def __truediv__(self: Quantity[DynamicViscosity], other: Quantity[PowerPerLength]  # type: ignore
                    ) -> Quantity[MassPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[DynamicViscosity], other: Quantity[MassPerEnergy]  # type: ignore
                    ) -> Quantity[PowerPerLength]:
        ...

    @overload
    def __truediv__(self: Quantity[KinematicViscosity], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[Velocity]:
        ...

    @overload
    def __truediv__(self: Quantity[KinematicViscosity], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[KinematicViscosity], other: Quantity[Area]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[KinematicViscosity], other: Quantity[SpecificVolume]  # type: ignore
                    ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[KinematicViscosity], other: Quantity[Velocity]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[KinematicViscosity], other: Quantity[DynamicViscosity]  # type: ignore
                    ) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[KinematicViscosity], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[Area]:
        ...

    @overload
    def __truediv__(self: Quantity[KinematicViscosity], other: Quantity[EnergyPerMass]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[MolarMass], other: Quantity[MassPerEnergy]  # type: ignore
                    ) -> Quantity[MolarSpecificEnthalpy]:
        ...

    @overload
    def __truediv__(self: Quantity[MolarMass], other: Quantity[MolarSpecificEnthalpy]  # type: ignore
                    ) -> Quantity[MassPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Mass]  # type: ignore
                    ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Energy]  # type: ignore
                    ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[CurrencyPerEnergy]  # type: ignore
                    ) -> Quantity[Energy]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[CurrencyPerMass]  # type: ignore
                    ) -> Quantity[Mass]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[CurrencyPerVolume]  # type: ignore
                    ) -> Quantity[Volume]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[CurrencyPerTime]  # type: ignore
                    ) -> Quantity[Time]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerEnergy], other: Quantity[CurrencyPerMass]  # type: ignore
                    ) -> Quantity[MassPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerEnergy], other: Quantity[MassPerEnergy]  # type: ignore
                    ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerMass], other: Quantity[SpecificVolume]  # type: ignore
                    ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerMass], other: Quantity[CurrencyPerEnergy]  # type: ignore
                    ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerMass], other: Quantity[CurrencyPerVolume]  # type: ignore
                    ) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerMass], other: Quantity[EnergyPerMass]  # type: ignore
                    ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerVolume], other: Quantity[Pressure]  # type: ignore
                    ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerVolume], other: Quantity[Density]  # type: ignore
                    ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerVolume], other: Quantity[CurrencyPerEnergy]  # type: ignore
                    ) -> Quantity[Pressure]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerVolume], other: Quantity[CurrencyPerMass]  # type: ignore
                    ) -> Quantity[Density]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerTime], other: Quantity[MassFlow]  # type: ignore
                    ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerTime], other: Quantity[VolumeFlow]  # type: ignore
                    ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerTime], other: Quantity[Power]  # type: ignore
                    ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerTime], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[Currency]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerTime], other: Quantity[Currency]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerTime], other: Quantity[CurrencyPerEnergy]  # type: ignore
                    ) -> Quantity[Power]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerTime], other: Quantity[CurrencyPerMass]  # type: ignore
                    ) -> Quantity[MassFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[CurrencyPerTime], other: Quantity[CurrencyPerVolume]  # type: ignore
                    ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[PowerPerArea]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[TemperatureDifference]  # type: ignore
                    ) -> Quantity[ThermalConductivity]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[Area]  # type: ignore
                    ) -> Quantity[PowerPerVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[Pressure]  # type: ignore
                    ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[DynamicViscosity]  # type: ignore
                    ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[KinematicViscosity]  # type: ignore
                    ) -> Quantity[Pressure]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[PowerPerArea]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[PowerPerVolume]  # type: ignore
                    ) -> Quantity[Area]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[ThermalConductivity]  # type: ignore
                    ) -> Quantity[TemperatureDifference]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerLength], other: Quantity[EnergyPerMass]  # type: ignore
                    ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerArea], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[PowerPerVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerArea], other: Quantity[TemperatureDifference]  # type: ignore
                    ) -> Quantity[HeatTransferCoefficient]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerArea], other: Quantity[Pressure]  # type: ignore
                    ) -> Quantity[Velocity]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerArea], other: Quantity[Velocity]  # type: ignore
                    ) -> Quantity[Pressure]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerArea], other: Quantity[PowerPerVolume]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerArea], other: Quantity[HeatTransferCoefficient]  # type: ignore
                    ) -> Quantity[TemperatureDifference]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerVolume], other: Quantity[Pressure]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerVolume], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[Pressure]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerTemperature], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[ThermalConductivity]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerTemperature], other: Quantity[Area]  # type: ignore
                    ) -> Quantity[HeatTransferCoefficient]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerTemperature], other: Quantity[MassFlow]  # type: ignore
                    ) -> Quantity[SpecificHeatCapacity]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerTemperature], other: Quantity[ThermalConductivity]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerTemperature], other: Quantity[HeatTransferCoefficient]  # type: ignore
                    ) -> Quantity[Area]:
        ...

    @overload
    def __truediv__(self: Quantity[PowerPerTemperature], other: Quantity[SpecificHeatCapacity]  # type: ignore
                    ) -> Quantity[MassFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[ThermalConductivity], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[HeatTransferCoefficient]:
        ...

    @overload
    def __truediv__(self: Quantity[ThermalConductivity], other: Quantity[DynamicViscosity]  # type: ignore
                    ) -> Quantity[SpecificHeatCapacity]:
        ...

    @overload
    def __truediv__(self: Quantity[ThermalConductivity], other: Quantity[HeatTransferCoefficient]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[ThermalConductivity], other: Quantity[SpecificHeatCapacity]  # type: ignore
                    ) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[MolarSpecificEntropy], other: Quantity[MolarMass]  # type: ignore
                    ) -> Quantity[SpecificHeatCapacity]:
        ...

    @overload
    def __truediv__(self: Quantity[MolarSpecificEntropy], other: Quantity[SpecificHeatCapacity]  # type: ignore
                    ) -> Quantity[MolarMass]:
        ...

    @overload
    def __truediv__(self: Quantity[EnergyPerMass], other: Quantity[TemperatureDifference]  # type: ignore
                    ) -> Quantity[SpecificHeatCapacity]:
        ...

    @overload
    def __truediv__(self: Quantity[EnergyPerMass], other: Quantity[Pressure]  # type: ignore
                    ) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[EnergyPerMass], other: Quantity[SpecificVolume]  # type: ignore
                    ) -> Quantity[Pressure]:
        ...

    @overload
    def __truediv__(self: Quantity[EnergyPerMass], other: Quantity[Velocity]  # type: ignore
                    ) -> Quantity[Velocity]:
        ...

    @overload
    def __truediv__(self: Quantity[EnergyPerMass], other: Quantity[KinematicViscosity]  # type: ignore
                    ) -> Quantity[Frequency]:
        ...

    @overload
    def __truediv__(self: Quantity[EnergyPerMass], other: Quantity[Frequency]  # type: ignore
                    ) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __truediv__(self: Quantity[EnergyPerMass], other: Quantity[SpecificHeatCapacity]  # type: ignore
                    ) -> Quantity[TemperatureDifference]:
        ...

    @overload
    def __truediv__(self: Quantity[MolarSpecificEnthalpy], other: Quantity[TemperatureDifference]  # type: ignore
                    ) -> Quantity[MolarSpecificEntropy]:
        ...

    @overload
    def __truediv__(self: Quantity[MolarSpecificEnthalpy], other: Quantity[MolarMass]  # type: ignore
                    ) -> Quantity[EnergyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[MolarSpecificEnthalpy], other: Quantity[MolarSpecificEntropy]  # type: ignore
                    ) -> Quantity[TemperatureDifference]:
        ...

    @overload
    def __truediv__(self: Quantity[MolarSpecificEnthalpy], other: Quantity[EnergyPerMass]  # type: ignore
                    ) -> Quantity[MolarMass]:
        ...

    # endregion

    @overload
    def __truediv__(self: Quantity[Dimensionless], other: Quantity[DT_]  # type: ignore
                    ) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self, other: Quantity[Dimensionless]  # type: ignore
                    ) -> Quantity[DT]:
        ...

    @overload
    def __truediv__(self, other: float) -> Quantity[DT]:
        ...

    @overload
    def __truediv__(self, other: Quantity[DT_]) -> Quantity[Unknown]:
        ...

    @overload
    def __floordiv__(self: Quantity[Dimensionless], other: float) -> Quantity[Dimensionless]:
        ...

    @overload
    def __floordiv__(self, other: Quantity[DT]) -> Quantity[Dimensionless]:
        ...

    @overload
    def __pow__(self, other: Literal[1]  # type: ignore
                ) -> Quantity[DT]:
        ...

    @overload
    def __pow__(self: Quantity[Length], other: Literal[2]  # type: ignore
                ) -> Quantity[Area]:
        ...

    @overload
    def __pow__(self: Quantity[Length], other: Literal[3]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __pow__(self: Quantity[Unknown], other: float  # type: ignore
                ) -> Quantity[Unknown]:
        ...

    @overload
    def __pow__(self: Quantity[Dimensionless], other: float  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __pow__(self, other: Quantity[Dimensionless]  # type: ignore
                ) -> Quantity[Unknown]:
        ...

    @overload
    def __pow__(self, other: float   # type: ignore
                ) -> Quantity[Unknown]:
        ...

    @overload
    def __add__(self: Quantity[Unknown], other) -> Quantity[Unknown]:
        ...

    @overload
    def __add__(self: Quantity[Dimensionless], other: float) -> Quantity[Dimensionless]:
        ...

    @overload
    def __add__(self: Quantity[Temperature], other: Quantity[TemperatureDifference]  # type: ignore
                ) -> Quantity[Temperature]:
        ...

    @overload
    def __add__(self, other: Quantity[DT]) -> Quantity[DT]:
        ...

    @overload
    def __sub__(self: Quantity[Unknown], other) -> Quantity[Unknown]:
        ...

    @overload
    def __sub__(self: Quantity[Dimensionless], other: float) -> Quantity[Dimensionless]:
        ...

    @overload
    def __sub__(self: Quantity[Temperature], other: Quantity[TemperatureDifference]  # type: ignore
                ) -> Quantity[Temperature]:
        ...

    @overload
    def __sub__(self: Quantity[Temperature], other: Quantity[Temperature]  # type: ignore
                ) -> Quantity[TemperatureDifference]:
        ...

    @overload
    def __sub__(self, other: Quantity[DT]) -> Quantity[DT]:
        ...

    @overload  # type: ignore[override]
    def __gt__(self: Quantity[Dimensionless], other: float) -> bool:
        ...

    @overload
    def __gt__(self, other: Quantity[DT]) -> bool:
        ...

    @overload  # type: ignore[override]
    def __ge__(self: Quantity[Dimensionless], other: float) -> bool:
        ...

    @overload
    def __ge__(self, other: Quantity[DT]) -> bool:
        ...

    @overload  # type: ignore[override]
    def __lt__(self: Quantity[Dimensionless], other: float) -> bool:
        ...

    @overload
    def __lt__(self, other: Quantity[DT]) -> bool:
        ...

    @overload  # type: ignore[override]
    def __le__(self: Quantity[Dimensionless], other: float) -> bool:
        ...

    @overload
    def __le__(self, other: Quantity[DT]) -> bool:
        ...
