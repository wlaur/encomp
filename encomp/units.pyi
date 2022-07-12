# from __future__ import annotations
from typing import Generic, Union, Optional, Literal, Generator, Any, overload

import sympy as sp
from _typeshed import Incomplete

import pint
from pint.unit import UnitsContainer, Unit

# this is not consistent with units.py
from pint.errors import DimensionalityError as _DimensionalityError

from encomp.utypes import (DimensionlessUnits,
                           CurrencyUnits,
                           CurrencyPerEnergyUnits,
                           CurrencyPerMassUnits,
                           CurrencyPerVolumeUnits,
                           CurrencyPerTimeUnits,
                           LengthUnits,
                           MassUnits,
                           TimeUnits,
                           TemperatureUnits,
                           SubstanceUnits,
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
                           KinematicViscosityUnits)

from encomp.utypes import (Magnitude,
                           MagnitudeInput,
                           MagnitudeScalar,
                           DT,
                           DT_,
                           Dimensionality,
                           Unknown,
                           Impossible,
                           Dimensionless,
                           Currency,
                           CurrencyPerEnergy,
                           CurrencyPerMass,
                           CurrencyPerVolume,
                           CurrencyPerTime,
                           Substance,
                           Density,
                           Energy,
                           Power,
                           Time,
                           Temperature,
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
                           KinematicViscosity)


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


class CustomRegistry:
    def __init__(self, args: Incomplete | None = ...,
                 kwargs: Incomplete | None = ...) -> None: ...

    def __getattr__(self, item): ...
    def __setattr__(self, key, value) -> None: ...
    def __getitem__(self, item): ...
    def __call__(self, *args, **kwargs): ...


ureg: Incomplete
wraps: Incomplete
check: Incomplete


def define_dimensionality(name: str, symbol: str = ...) -> None: ...
def set_quantity_format(fmt: str = ...) -> None: ...


class Quantity(pint.quantity.Quantity, Generic[DT]):

    def __hash__(self) -> int: ...
    def __class_getitem__(cls, dim: type[DT]) -> type[Quantity[DT]]: ...
    @classmethod
    def get_unit(cls, unit_name: str) -> Unit: ...
    def __len__(self) -> int: ...
    @property
    def m(self) -> Magnitude: ...
    def to_reduced_units(self) -> Quantity[DT]: ...
    def to_base_units(self) -> Quantity[DT]: ...

    def to(self, unit: Union[Unit, UnitsContainer,  # type: ignore
           str, Quantity[DT], dict]) -> Quantity[DT]: ...

    def ito(self, unit: Union[Unit, UnitsContainer,  # type: ignore
            str, Quantity[DT]]) -> None: ...
    def check(self, unit: Union[Quantity, UnitsContainer, Unit,
              str, Dimensionality, type[Dimensionality]]) -> bool: ...

    def __format__(self, format_type: str) -> str: ...
    @staticmethod
    def correct_unit(unit: str) -> str: ...
    @staticmethod
    def get_unit_symbol(s: str) -> sp.Symbol: ...
    @classmethod
    def get_dimension_symbol_map(cls) -> dict[sp.Basic, Unit]: ...
    @classmethod
    def from_expr(cls, expr: sp.Basic) -> Quantity: ...
    @property
    def dim(self) -> UnitsContainer: ...
    @property
    def dimensionality_name(self) -> str: ...
    @property
    def dim_name(self) -> str: ...
    @classmethod
    def __get_validators__(cls) -> Generator[Incomplete, None, None]: ...
    @classmethod
    def validate(cls, qty: Quantity[DT]) -> Quantity[DT]: ...
    def check_compatibility(
        self, other: Union[Quantity, MagnitudeScalar]) -> None: ...

    def __eq__(self, other: Any) -> bool: ...

    def __rsub__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
        ...

    def __radd__(self: Quantity[Dimensionless],  # type: ignore[override]
                 other: MagnitudeScalar) -> Quantity[Dimensionless]:
        ...

    def __rmul__(self, other: MagnitudeScalar  # type: ignore[override]
                 ) -> Quantity[DT]:
        ...

    def __rtruediv__(self, other: MagnitudeScalar) -> Quantity[Unknown]:
        ...

    def __rpow__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> MagnitudeScalar:
        ...

    def __rfloordiv__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
        ...

    # overloaded methods

    @overload
    def __new__(cls, val: str) -> Quantity[Unknown]:  # type: ignore
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[Dimensionless]]  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: DimensionlessUnits
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyUnits
                ) -> Quantity[Currency]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyPerEnergyUnits
                ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyPerVolumeUnits
                ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyPerMassUnits
                ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrencyPerTimeUnits
                ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: LengthUnits) -> Quantity[Length]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: MassUnits) -> Quantity[Mass]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: TimeUnits) -> Quantity[Time]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: TemperatureUnits) -> Quantity[Temperature]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: SubstanceUnits) -> Quantity[Substance]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: CurrentUnits) -> Quantity[Current]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: LuminosityUnits) -> Quantity[Luminosity]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: AreaUnits) -> Quantity[Area]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: VolumeUnits) -> Quantity[Volume]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: NormalVolumeUnits) -> Quantity[NormalVolume]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: PressureUnits) -> Quantity[Pressure]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: MassFlowUnits) -> Quantity[MassFlow]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: VolumeFlowUnits) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: NormalVolumeFlowUnits) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: DensityUnits) -> Quantity[Density]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: SpecificVolumeUnits) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: EnergyUnits) -> Quantity[Energy]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: PowerUnits) -> Quantity[Power]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: VelocityUnits) -> Quantity[Velocity]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: DynamicViscosityUnits) -> Quantity[DynamicViscosity]:
        ...

    @overload
    def __new__(cls, val: Union[MagnitudeInput, Quantity[DT], str],  # type: ignore
                unit: KinematicViscosityUnits) -> Quantity[KinematicViscosity]:
        ...

    @overload
    def __new__(
        cls,
        val: Union[MagnitudeInput, Quantity[DT], str],
        unit: Union[Unit, UnitsContainer,
                    str, Quantity[DT], None] = None,

        # this is a hack to force the type checker to default to Unknown
        # in case the generic type is not specified at all
        _dt: type[DT] = Unknown  # type: ignore
    ) -> Quantity[DT]:
        ...

    @overload
    def __mul__(self: Quantity[Unknown], other) -> Quantity[Unknown]:
        ...

    @overload
    def __mul__(self, other: Quantity[Unknown]) -> Quantity[Unknown]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[Length]  # type: ignore
                ) -> Quantity[Area]:
        ...

    @overload
    def __mul__(self: Quantity[Area], other: Quantity[Length]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Length], other: Quantity[Area]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[MassFlow]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[MassFlow], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[VolumeFlow]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[VolumeFlow], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Volume]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[NormalVolumeFlow]  # type: ignore
                ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __mul__(self: Quantity[NormalVolumeFlow], other: Quantity[Time]  # type: ignore
                ) -> Quantity[NormalVolume]:
        ...

    @overload
    def __mul__(self: Quantity[Power], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[Power]  # type: ignore
                ) -> Quantity[Energy]:
        ...

    @overload
    def __mul__(self: Quantity[Density], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[Mass]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerEnergy], other: Quantity[Energy]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerMass], other: Quantity[Mass]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerVolume], other: Quantity[Volume]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[CurrencyPerTime], other: Quantity[Time]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Energy], other: Quantity[CurrencyPerEnergy]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Mass], other: Quantity[CurrencyPerMass]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Volume], other: Quantity[CurrencyPerVolume]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Time], other: Quantity[CurrencyPerTime]  # type: ignore
                ) -> Quantity[Currency]:
        ...

    @overload
    def __mul__(self: Quantity[Dimensionless], other: Quantity[DT_]  # type: ignore
                ) -> Quantity[DT_]:
        ...

    @overload
    def __mul__(self, other: Quantity[Dimensionless]  # type: ignore
                ) -> Quantity[DT]:
        ...

    @overload
    def __mul__(self, other: MagnitudeScalar) -> Quantity[DT]:
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
    def __truediv__(self: Quantity[Unknown], other) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self, other: Quantity[Unknown]) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[Area]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Area]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[Area], other: Quantity[Length]  # type: ignore
                    ) -> Quantity[Length]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[MassFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[VolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[NormalVolume], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[NormalVolumeFlow]:
        ...

    @overload
    def __truediv__(self: Quantity[Mass], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[Density]:
        ...

    @overload
    def __truediv__(self: Quantity[Volume], other: Quantity[Mass]  # type: ignore
                    ) -> Quantity[SpecificVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Energy], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[Power]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Energy]  # type: ignore
                    ) -> Quantity[CurrencyPerEnergy]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Mass]  # type: ignore
                    ) -> Quantity[CurrencyPerMass]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Volume]  # type: ignore
                    ) -> Quantity[CurrencyPerVolume]:
        ...

    @overload
    def __truediv__(self: Quantity[Currency], other: Quantity[Time]  # type: ignore
                    ) -> Quantity[CurrencyPerTime]:
        ...

    @overload
    def __truediv__(self: Quantity[Dimensionless], other: Quantity[DT_]  # type: ignore
                    ) -> Quantity[Unknown]:
        ...

    @overload
    def __truediv__(self, other: Quantity[Dimensionless]  # type: ignore
                    ) -> Quantity[DT]:
        ...

    @overload
    def __truediv__(self, other: MagnitudeScalar) -> Quantity[DT]:
        ...

    @overload
    def __truediv__(self, other: Quantity[DT_]) -> Quantity[Unknown]:
        ...

    @overload
    def __floordiv__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
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
    def __pow__(self: Quantity[Unknown], other: MagnitudeScalar  # type: ignore
                ) -> Quantity[Unknown]:
        ...

    @overload
    def __pow__(self: Quantity[Dimensionless], other: MagnitudeScalar  # type: ignore
                ) -> Quantity[Dimensionless]:
        ...

    @overload
    def __pow__(self, other: Quantity[Dimensionless]  # type: ignore
                ) -> Quantity[Unknown]:
        ...

    @overload
    def __pow__(self, other: MagnitudeScalar   # type: ignore
                ) -> Quantity[Unknown]:
        ...

    @overload
    def __add__(self: Quantity[Unknown], other) -> Quantity[Impossible]:
        ...

    @overload
    def __add__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
        ...

    @overload
    def __add__(self, other: Quantity[DT]) -> Quantity[DT]:
        ...

    @overload
    def __sub__(self: Quantity[Unknown], other) -> Quantity[Impossible]:
        ...

    @overload
    def __sub__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> Quantity[Dimensionless]:
        ...

    @overload
    def __sub__(self, other: Quantity[DT]) -> Quantity[DT]:
        ...

    @overload  # type: ignore[override]
    def __gt__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> bool:
        ...

    @overload
    def __gt__(self, other: Quantity[DT]) -> bool:
        ...

    @overload  # type: ignore[override]
    def __ge__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> bool:
        ...

    @overload
    def __ge__(self, other: Quantity[DT]) -> bool:
        ...

    @overload  # type: ignore[override]
    def __lt__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> bool:
        ...

    @overload
    def __lt__(self, other: Quantity[DT]) -> bool:
        ...

    @overload  # type: ignore[override]
    def __le__(self: Quantity[Dimensionless], other: MagnitudeScalar) -> bool:
        ...

    @overload
    def __le__(self, other: Quantity[DT]) -> bool:
        ...


@overload
def convert_volume_mass(inp: Quantity[Mass],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[Volume]:
    ...


@overload
def convert_volume_mass(inp: Quantity[MassFlow],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[VolumeFlow]:
    ...


@overload
def convert_volume_mass(inp: Quantity[Volume],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[Mass]:
    ...


@overload
def convert_volume_mass(inp: Quantity[VolumeFlow],
                        rho: Optional[Quantity[Density]] = None) -> Quantity[MassFlow]:
    ...


@overload
def convert_volume_mass(inp: Quantity,
                        rho: Optional[Quantity[Density]] = None
                        ) -> Union[Quantity[Mass],
                                   Quantity[MassFlow],
                                   Quantity[Volume],
                                   Quantity[VolumeFlow]
                                   ]:
    ...
