from typing import assert_type

import numpy as np
import polars as pl

from .. import utypes as ut
from ..units import Quantity as Q


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_inference_basic() -> None:
    assert_type(Q([1]), Q[ut.Dimensionless])
    assert_type(Q(1), Q[ut.Dimensionless, float])
    assert_type(Q(np.array([1])), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q(np.array([1])), Q[ut.Dimensionless, ut.Numpy1DArray])
    assert_type(Q(pl.Series([1])), Q[ut.Dimensionless, pl.Series])
    assert_type(Q(pl.lit(1)), Q[ut.Dimensionless, pl.Expr])

    assert_type(Q(1) * 1, Q[ut.Dimensionless, float])
    assert_type(Q(1) * Q(1), Q[ut.Dimensionless, float])
    assert_type(Q(1), Q[ut.Dimensionless, float])
    assert_type(Q([1]) * Q(1), Q[ut.Dimensionless, np.ndarray])

    assert_type(Q(1) * Q(1, "kg"), Q[ut.Mass, float])
    assert_type(Q(1, "kg") * Q(1), Q[ut.Mass, float])
    assert_type(Q(1, "kg") / Q(1), Q[ut.Mass, float])
    assert_type(Q(1, "kg") * 1, Q[ut.Mass, float])
    assert_type(Q(1, "kg") / 1, Q[ut.Mass, float])

    assert_type(Q([1]) * Q(1, "kg"), Q[ut.Mass, np.ndarray])


def test_inference_multiplication() -> None:
    # scalar * scalar
    assert_type(Q(1) * Q(2), Q[ut.Dimensionless, float])
    assert_type(Q(1, "m") * Q(2, "m"), Q[ut.Area, float])

    # array * scalar
    assert_type(Q([1, 2]) * Q(3), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q([1, 2], "m") * Q(3, "m"), Q[ut.Area, np.ndarray])

    # scalar * array (dimensionless)
    assert_type(Q(3) * Q([1, 2]), Q[ut.Dimensionless, np.ndarray])

    # array * array
    assert_type(Q([1, 2]) * Q([3, 4]), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q([1, 2], "m") * Q([3, 4], "m"), Q[ut.Area, np.ndarray])

    # dimensionless * dimensional
    assert_type(Q(2) * Q(1, "kg"), Q[ut.Mass, float])
    assert_type(Q([1, 2]) * Q(3, "kg"), Q[ut.Mass, np.ndarray])

    # dimensional * dimensionless (scalar only)
    assert_type(Q(1, "kg") * Q(2), Q[ut.Mass, float])


def test_inference_division() -> None:
    # scalar / scalar
    assert_type(Q(4) / Q(2), Q[ut.Dimensionless, float])
    assert_type(Q(4, "m") / Q(2, "m"), Q[ut.Dimensionless, float])
    assert_type(Q(4, "m") / Q(2, "s"), Q[ut.Velocity, float])

    # array / scalar
    assert_type(Q([4, 6]) / Q(2), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q([4, 6], "m") / Q(2, "s"), Q[ut.Velocity, np.ndarray])

    # scalar / array (dimensionless)
    assert_type(Q(4) / Q([2, 4]), Q[ut.Dimensionless, np.ndarray])

    # array / array
    assert_type(Q([4, 6]) / Q([2, 3]), Q[ut.Dimensionless, np.ndarray])

    # dimensionless / dimensional
    assert_type(Q(2) / Q(1, "s"), Q[ut.UnknownDimensionality, float])
    assert_type(Q([2, 4]) / Q(1, "s"), Q[ut.UnknownDimensionality, np.ndarray])

    # dimensional / dimensionless (scalar only)
    assert_type(Q(4, "m") / Q(2), Q[ut.Length, float])
    assert_type(Q(4, "m") / 2, Q[ut.Length, float])
    assert_type(Q(4, "m") / 2.0, Q[ut.Length, float])
    assert_type(Q(4, "m") // Q(2), Q[ut.Length, float])
    assert_type(Q(4, "m") // 2, Q[ut.Length, float])
    assert_type(Q(4, "m") // 2.0, Q[ut.Length, float])


def test_inference_addition_subtraction() -> None:
    # scalar + scalar
    assert_type(Q(1) + Q(2), Q[ut.Dimensionless, float])
    assert_type(Q(1, "m") + Q(2, "m"), Q[ut.Length, float])

    # array + scalar
    assert_type(Q([1, 2]) + Q(3), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q([1, 2], "m") + Q(3, "m"), Q[ut.Length, np.ndarray])

    # array + array
    assert_type(Q([1, 2]) + Q([3, 4]), Q[ut.Dimensionless, np.ndarray])

    # subtraction
    assert_type(Q(5) - Q(2), Q[ut.Dimensionless, float])
    assert_type(Q([5, 6]) - Q(2), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q(5, "m") - Q(2, "m"), Q[ut.Length, float])


def test_inference_comparisons() -> None:
    # scalar comparisons return bool (not np.bool_ - __eq__ is overridden)
    assert_type(Q(1) > Q(2), bool)
    assert_type(Q(1) < Q(2), bool)
    assert_type(Q(1) >= Q(2), bool)
    assert_type(Q(1) <= Q(2), bool)
    assert_type(Q(1) == Q(2), bool)

    # array comparisons return arrays
    # Note: these currently don't type check due to missing operator overloads
    # but work at runtime
    result1 = Q([1, 2]) > Q(2)
    assert isinstance(result1, np.ndarray)

    result2 = Q([1, 2]) < Q([2, 3])
    assert isinstance(result2, np.ndarray)

    result3 = Q([1, 2], "m") >= Q(2, "m")
    assert isinstance(result3, np.ndarray)


def test_inference_polars() -> None:
    # polars Series construction
    assert_type(Q(pl.Series([1, 2])), Q[ut.Dimensionless, pl.Series])

    # polars Expr
    assert_type(Q(pl.lit(1)), Q[ut.Dimensionless, pl.Expr])
    assert_type(Q(pl.lit(1)) * Q(2, "kg"), Q[ut.Mass, pl.Expr])
    assert_type(Q(pl.lit(1), "m") * Q(2, "m"), Q[ut.Area, pl.Expr])


def test_inference_complex_units() -> None:
    # velocity = length / time
    assert_type(Q(10, "m") / Q(2, "s"), Q[ut.Velocity, float])
    assert_type(Q([10, 20], "m") / Q(2, "s"), Q[ut.Velocity, np.ndarray])

    # energy = mass * velocity^2
    velocity = Q(5, "m/s")
    assert_type(Q(2, "kg") * velocity * velocity, Q[ut.UnknownDimensionality, float])

    # power = energy / time
    assert_type(Q(100, "J") / Q(2, "s"), Q[ut.Power, float])

    # density = mass / volume
    assert_type(Q(1000, "kg") / Q(1, "m^3"), Q[ut.Density, float])

    # pressure = force / area
    assert_type(Q(100, "N") / Q(2, "m^2"), Q[ut.UnknownDimensionality, float])


def test_inference_dimensional_multiplication() -> None:
    # Length * Length = Area
    assert_type(Q(5.0, "m") * Q(3.0, "m"), Q[ut.Area, float])
    assert_type(Q([5.0], "m") * Q(3.0, "m"), Q[ut.Area, np.ndarray])
    assert_type(Q(pl.Series([5.0]), "m") * Q(3.0, "m"), Q[ut.Area, pl.Series])

    # Length * Area = Volume
    assert_type(Q(2.0, "m") * Q(10.0, "m^2"), Q[ut.Volume, float])
    assert_type(Q([2.0, 3.0], "m") * Q(10.0, "m^2"), Q[ut.Volume, np.ndarray])

    # Area * Length = Volume (commutative)
    assert_type(Q(10.0, "m^2") * Q(2.0, "m"), Q[ut.Volume, float])

    # Mass * SpecificVolume = Volume
    assert_type(Q(100.0, "kg") * Q(0.001, "m^3/kg"), Q[ut.UnknownDimensionality, float])
    assert_type(Q(pl.lit(100.0), "kg") * Q(0.001, "m^3/kg"), Q[ut.UnknownDimensionality, pl.Expr])

    # Time * Power = Energy
    assert_type(Q(3600.0, "s") * Q(1000.0, "W"), Q[ut.Energy, float])

    # Time * MassFlow = Mass
    assert_type(Q(10.0, "s") * Q(5.0, "kg/s"), Q[ut.Mass, float])
    assert_type(Q([10.0, 20.0], "s") * Q(5.0, "kg/s"), Q[ut.Mass, np.ndarray])

    # Time * VolumeFlow = Volume
    assert_type(Q(60.0, "s") * Q(2.0, "m^3/s"), Q[ut.Volume, float])

    # Volume * Density = Mass
    assert_type(Q(1.0, "m^3") * Q(1000.0, "kg/m^3"), Q[ut.Mass, float])
    assert_type(Q(pl.Series([1.0, 2.0]), "m^3") * Q(1000.0, "kg/m^3"), Q[ut.Mass, pl.Series])

    # Pressure * Volume = Energy
    assert_type(Q(101325.0, "Pa") * Q(1.0, "m^3"), Q[ut.UnknownDimensionality, float])

    # Velocity * Area = VolumeFlow
    assert_type(Q(5.0, "m/s") * Q(2.0, "m^2"), Q[ut.UnknownDimensionality, float])
    assert_type(Q([5.0, 10.0], "m/s") * Q(2.0, "m^2"), Q[ut.UnknownDimensionality, np.ndarray])


def test_inference_dimensional_division() -> None:
    # Mass / Time = MassFlow
    assert_type(Q(100.0, "kg") / Q(10.0, "s"), Q[ut.MassFlow, float])
    assert_type(Q([100.0, 200.0], "kg") / Q(10.0, "s"), Q[ut.MassFlow, np.ndarray])
    assert_type(Q(pl.Series([100.0]), "kg") / Q(10.0, "s"), Q[ut.MassFlow, pl.Series])

    # Volume / Time = VolumeFlow
    assert_type(Q(10.0, "m^3") / Q(5.0, "s"), Q[ut.VolumeFlow, float])
    assert_type(Q(pl.lit(10.0), "m^3") / Q(5.0, "s"), Q[ut.VolumeFlow, pl.Expr])

    # Mass / Volume = Density
    assert_type(Q(1000.0, "kg") / Q(1.0, "m^3"), Q[ut.Density, float])
    assert_type(Q([1000.0, 2000.0], "kg") / Q(1.0, "m^3"), Q[ut.Density, np.ndarray])

    # Length / Time = Velocity
    assert_type(Q(100.0, "m") / Q(10.0, "s"), Q[ut.Velocity, float])

    # Energy / Time = Power
    assert_type(Q(3600000.0, "J") / Q(3600.0, "s"), Q[ut.Power, float])

    # Energy / Mass = EnergyPerMass (specific energy)
    assert_type(Q(1000.0, "J") / Q(1.0, "kg"), Q[ut.EnergyPerMass, float])
    assert_type(Q([1000.0, 2000.0], "J") / Q(1.0, "kg"), Q[ut.EnergyPerMass, np.ndarray])

    # Volume / Area = Length
    assert_type(Q(100.0, "m^3") / Q(10.0, "m^2"), Q[ut.UnknownDimensionality, float])

    # Area / Length = Length
    assert_type(Q(50.0, "m^2") / Q(5.0, "m"), Q[ut.UnknownDimensionality, float])
    assert_type(Q(pl.Series([50.0]), "m^2") / Q(5.0, "m"), Q[ut.UnknownDimensionality, pl.Series])

    # VolumeFlow / Area = Velocity
    assert_type(Q(10.0, "m^3/s") / Q(2.0, "m^2"), Q[ut.UnknownDimensionality, float])

    # Power / Area = PowerPerArea (heat flux)
    assert_type(Q(1000.0, "W") / Q(1.0, "m^2"), Q[ut.UnknownDimensionality, float])
    assert_type(Q([1000.0, 2000.0], "W") / Q(1.0, "m^2"), Q[ut.UnknownDimensionality, np.ndarray])

    # MassFlow / Density = VolumeFlow
    assert_type(Q(10.0, "kg/s") / Q(1000.0, "kg/m^3"), Q[ut.UnknownDimensionality, float])


def test_inference_dimensional_derived_units() -> None:
    # Pressure = Force / Area with different notations for m²
    assert_type(Q(1000.0, "N/m2"), Q[ut.Pressure, float])
    assert_type(Q(1000.0, "N/m^2"), Q[ut.Pressure, float])
    assert_type(Q(1000.0, "N/m**2"), Q[ut.Pressure, float])
    assert_type(Q(1000.0, "N/m²"), Q[ut.Pressure, float])

    # Test with various magnitude types
    assert_type(Q([1000.0], "N/m2"), Q[ut.Pressure, np.ndarray])
    assert_type(Q(pl.Series([1000.0]), "N/m**2"), Q[ut.Pressure, pl.Series])
    assert_type(Q(pl.lit(1000.0), "N/m²"), Q[ut.Pressure, pl.Expr])

    # Kinematic Viscosity = Length * Velocity (mul returns ut.UnknownDimensionality dimensionality)
    assert_type(Q(0.1, "m") * Q(0.5, "m/s"), Q[ut.UnknownDimensionality, float])
    assert_type(Q(pl.Series([0.1]), "m") * Q(0.5, "m/s"), Q[ut.UnknownDimensionality, pl.Series])
    assert_type(Q([0.1], "m") * Q(0.5, "m/s"), Q[ut.UnknownDimensionality, np.ndarray])

    # Dynamic Viscosity = Density * Kinematic Viscosity
    assert_type(Q(1000.0, "kg/m^3") * Q(0.001, "m^2/s"), Q[ut.UnknownDimensionality, float])
    assert_type(Q([1000.0], "kg/m^3") * Q(0.001, "m^2/s"), Q[ut.UnknownDimensionality, np.ndarray])
    assert_type(Q(pl.lit(1000.0), "kg/m3") * Q(0.001, "m2/s"), Q[ut.UnknownDimensionality, pl.Expr])

    # Specific Volume = Dimensionless / Density
    assert_type(Q(1.0) / Q(1000.0, "kg/m^3"), Q[ut.UnknownDimensionality, float])
    assert_type(Q([1.0]) / Q(1000.0, "kg/m³"), Q[ut.UnknownDimensionality, np.ndarray])
    assert_type(Q(pl.Series([1.0])) / Q(1000.0, "kg/m3"), Q[ut.UnknownDimensionality, pl.Series])

    # Thermal Conductivity = Power / (Length * TemperatureDifference) = W/(m·K)
    assert_type(Q(0.6, "W/m/K"), Q[ut.ThermalConductivity, float])
    assert_type(Q([0.6, 0.8], "W/m/K"), Q[ut.ThermalConductivity, np.ndarray])
    assert_type(Q(pl.lit(0.6), "W/m/K"), Q[ut.ThermalConductivity, pl.Expr])

    # Heat Transfer Coefficient = PowerPerArea / TemperatureDifference = W/(m²·K)
    assert_type(Q(10.0, "W/m^2/K"), Q[ut.HeatTransferCoefficient, float])
    assert_type(Q([10.0, 20.0], "W/m²/K"), Q[ut.HeatTransferCoefficient, np.ndarray])
    assert_type(Q(pl.Series([10.0]), "W/m**2/K"), Q[ut.HeatTransferCoefficient, pl.Series])

    # Frequency = Dimensionless / Time (div returns ut.UnknownDimensionality dimensionality)
    assert_type(Q(1.0) / Q(0.5, "s"), Q[ut.UnknownDimensionality, float])
    assert_type(Q([1.0, 2.0]) / Q(0.5, "s"), Q[ut.UnknownDimensionality, np.ndarray])
    assert_type(Q(pl.Series([1.0])) / Q(0.5, "s"), Q[ut.UnknownDimensionality, pl.Series])

    # MolarMass = Mass / Substance
    assert_type(Q(18.0, "g") / Q(1.0, "mol"), Q[ut.UnknownDimensionality, float])
    assert_type(Q([18.0, 44.0], "g") / Q(1.0, "mol"), Q[ut.UnknownDimensionality, np.ndarray])
    assert_type(Q(pl.lit(18.0), "g") / Q(1.0, "mol"), Q[ut.UnknownDimensionality, pl.Expr])

    # SubstancePerMass = Substance / Mass (reciprocal of MolarMass)
    assert_type(Q(1.0, "mol") / Q(18.0, "g"), Q[ut.UnknownDimensionality, float])
    assert_type(Q([1.0, 2.0], "kmol") / Q(18.0, "kg"), Q[ut.UnknownDimensionality, np.ndarray])


def test_various() -> None:
    assert_type(Q(1.0, "m"), Q[ut.Length, float])
    assert_type(Q(1, "m"), Q[ut.Length, float])

    assert_type(Q(1, str("m")), Q)  # pyright: ignore[reportAssertTypeFailure] # noqa: UP018

    assert_type(Q([1, 2, 3], "kg"), Q[ut.Mass, np.ndarray])

    assert_type(Q(pl.col.asd, "kg"), Q[ut.Mass, pl.Expr])
    assert_type(Q(pl.DataFrame({"test": []})["test"], "kg"), Q[ut.Mass, pl.Series])

    assert_type(Q(pl.col.asd, "kg") / Q(25, "min"), Q[ut.MassFlow, pl.Expr])


def test_inference_magnitude_type_promotion() -> None:
    # ISSUE: scalar with units * array with units
    # Runtime correctly produces ndarray, but static type is inferred as float
    result1 = Q(3, "m") * Q([1, 2], "m")
    assert isinstance(result1.m, np.ndarray)  # Runtime: ndarray ✓
    # Static: Q[Area, float] ✗

    # ISSUE: dimensional scalar * dimensionless array
    result2 = Q(1, "kg") * Q([2, 3])
    assert isinstance(result2.m, np.ndarray)  # Runtime: ndarray ✓
    # Static: Q[Mass, ndarray] would be correct but currently Any

    # ISSUE: scalar / array with units
    result3 = Q(4, "m") / Q([2, 4], "s")
    assert isinstance(result3.m, np.ndarray)  # Runtime: ndarray ✓
    # Static: Q[Velocity, float] ✗

    # ISSUE: dimensional scalar / dimensionless array
    result4 = Q(4, "m") / Q([2, 4])
    assert isinstance(result4.m, np.ndarray)  # Runtime: ndarray ✓
    # Static: Q[Length, float] ✗

    # ISSUE: scalar + array
    result5 = Q(3) + Q([1, 2])
    assert isinstance(result5.m, np.ndarray)  # Runtime: ndarray ✓
    # Static: Q[Dimensionless, float] ✗

    result6 = Q(3, "m") + Q([1, 2], "m")
    assert isinstance(result6.m, np.ndarray)  # Runtime: ndarray ✓
    # Static: Q[Length, float] ✗

    _ = Q(pl.Series([1, 2])) * Q(2)
    # Runtime behavior varies for polars Series
    # Static type: Q[Dimensionless, pl.Series]

    _ = Q(pl.Series([1, 2]), "m") / Q(2, "s")
    # Runtime behavior varies
    # Static type mismatch


def test_inference_mul_truediv() -> None:
    result1 = Q(1) * Q(1)
    assert_type(result1, Q[ut.Dimensionless, float])

    result2 = Q(1) * Q([1])
    assert_type(result2, Q[ut.Dimensionless, ut.Numpy1DArray])

    result3 = Q(1) / Q([2])
    assert_type(result3, Q[ut.Dimensionless, ut.Numpy1DArray])

    result4 = Q([1]) * Q(1)
    assert_type(result4, Q[ut.Dimensionless, np.ndarray])

    result5 = Q([1]) * Q([1])
    assert_type(result5, Q[ut.Dimensionless, np.ndarray])

    result6 = Q([1]) / Q([1])
    assert_type(result6, Q[ut.Dimensionless, np.ndarray])

    result7 = Q([1]) // Q([1])
    assert_type(result7, Q[ut.Dimensionless, np.ndarray])


def test_mul_dimensionless_float_float() -> None:
    assert_type(Q(1.0) * Q(2.0), Q[ut.Dimensionless, float])
    assert_type(Q(1.0) * 2, Q[ut.Dimensionless, float])
    assert_type(Q(1.0) * 2.0, Q[ut.Dimensionless, float])


def test_mul_dimensionless_float_array() -> None:
    assert_type(Q(1.0) * Q([2.0]), Q[ut.Dimensionless, ut.Numpy1DArray])
    assert_type(Q(1.0) * Q(np.array([2.0])), Q[ut.Dimensionless, ut.Numpy1DArray])


def test_mul_dimensionless_float_series() -> None:
    assert_type(Q(1.0) * Q(pl.Series([2.0])), Q[ut.Dimensionless, pl.Series])


def test_mul_dimensionless_float_expr() -> None:
    assert_type(Q(1.0) * Q(pl.lit(2.0)), Q[ut.Dimensionless, pl.Expr])


def test_mul_dimensionless_array_float() -> None:
    assert_type(Q([1.0]) * Q(2.0), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q(np.array([1.0])) * Q(2.0), Q[ut.Dimensionless, np.ndarray])


def test_mul_dimensionless_array_array() -> None:
    assert_type(Q([1.0]) * Q([2.0]), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q(np.array([1.0])) * Q(np.array([2.0])), Q[ut.Dimensionless, np.ndarray])


def test_mul_dimensionless_series_float() -> None:
    assert_type(Q(pl.Series([1.0])) * Q(2.0), Q[ut.Dimensionless, pl.Series])


def test_mul_dimensionless_series_series() -> None:
    assert_type(Q(pl.Series([1.0])) * Q(pl.Series([2.0])), Q[ut.Dimensionless, pl.Series])


def test_mul_dimensionless_expr_float() -> None:
    assert_type(Q(pl.lit(1.0)) * Q(2.0), Q[ut.Dimensionless, pl.Expr])


def test_mul_dimensionless_expr_expr() -> None:
    assert_type(Q(pl.lit(1.0)) * Q(pl.lit(2.0)), Q[ut.Dimensionless, pl.Expr])


def test_mul_dimensionless_propagates_dimensionality_float() -> None:
    assert_type(Q(2.0) * Q(1.0, "kg"), Q[ut.Mass, float])
    assert_type(Q(2.0) * Q(1.0, "m"), Q[ut.Length, float])
    assert_type(Q(2.0) * Q(1.0, "s"), Q[ut.Time, float])


def test_mul_dimensionless_propagates_dimensionality_array() -> None:
    assert_type(Q([2.0]) * Q(1.0, "kg"), Q[ut.Mass, np.ndarray])
    assert_type(Q([2.0]) * Q(1.0, "m"), Q[ut.Length, np.ndarray])
    assert_type(Q([2.0]) * Q(1.0, "s"), Q[ut.Time, np.ndarray])


def test_mul_dimensionless_propagates_dimensionality_series() -> None:
    assert_type(Q(pl.Series([2.0])) * Q(1.0, "kg"), Q[ut.Mass, pl.Series])
    assert_type(Q(pl.Series([2.0])) * Q(1.0, "m"), Q[ut.Length, pl.Series])


def test_mul_dimensionless_propagates_dimensionality_expr() -> None:
    assert_type(Q(pl.lit(2.0)) * Q(1.0, "kg"), Q[ut.Mass, pl.Expr])
    assert_type(Q(pl.lit(2.0)) * Q(1.0, "m"), Q[ut.Length, pl.Expr])


def test_mul_dimensional_by_dimensionless_float() -> None:
    assert_type(Q(1.0, "kg") * Q(2.0), Q[ut.Mass, float])
    assert_type(Q(1.0, "kg") * 2, Q[ut.Mass, float])
    assert_type(Q(1.0, "kg") * 2.0, Q[ut.Mass, float])
    assert_type(Q(1.0, "m") * Q(2.0), Q[ut.Length, float])
    assert_type(Q(1.0, "s") * Q(2.0), Q[ut.Time, float])


def test_mul_dimensional_by_dimensionless_array() -> None:
    assert_type(Q([1.0], "kg") * Q(2.0), Q[ut.Mass, np.ndarray])
    assert_type(Q([1.0], "m") * Q(2.0), Q[ut.Length, np.ndarray])


def test_mul_dimensional_by_dimensionless_series() -> None:
    assert_type(Q(pl.Series([1.0]), "kg") * Q(2.0), Q[ut.Mass, pl.Series])


def test_mul_dimensional_by_dimensionless_expr() -> None:
    assert_type(Q(pl.lit(1.0), "kg") * Q(2.0), Q[ut.Mass, pl.Expr])


def test_mul_dimensional_by_dimensional_float() -> None:
    assert_type(Q(1.0, "kg") * Q(2.0, "m"), Q[ut.UnknownDimensionality, float])
    assert_type(Q(1.0, "m") * Q(2.0, "m"), Q[ut.Area, float])
    assert_type(Q(1.0, "m") * Q(2.0, "s"), Q[ut.UnknownDimensionality, float])


def test_mul_dimensional_by_dimensional_array() -> None:
    assert_type(Q([1.0], "kg") * Q(2.0, "m"), Q[ut.UnknownDimensionality, np.ndarray])
    assert_type(Q([1.0], "m") * Q(2.0, "m"), Q[ut.Area, np.ndarray])


def test_mul_dimensional_by_dimensional_series() -> None:
    assert_type(Q(pl.Series([1.0]), "kg") * Q(2.0, "m"), Q[ut.UnknownDimensionality, pl.Series])


def test_mul_dimensional_by_dimensional_expr() -> None:
    assert_type(Q(pl.lit(1.0), "kg") * Q(2.0, "m"), Q[ut.UnknownDimensionality, pl.Expr])


def test_rmul_float_by_quantity() -> None:
    assert_type(2 * Q(1.0, "kg"), Q[ut.Mass, float])
    assert_type(2.0 * Q(1.0, "kg"), Q[ut.Mass, float])
    assert_type(2 * Q(1.0), Q[ut.Dimensionless, float])
    assert_type(2.0 * Q(1.0), Q[ut.Dimensionless, float])


def test_rmul_float_by_array_quantity() -> None:
    assert_type(2 * Q([1.0], "kg"), Q[ut.Mass, np.ndarray])
    assert_type(2.0 * Q([1.0], "kg"), Q[ut.Mass, np.ndarray])
    assert_type(2 * Q([1.0]), Q[ut.Dimensionless, np.ndarray])


def test_rmul_float_by_series_quantity() -> None:
    assert_type(2 * Q(pl.Series([1.0]), "kg"), Q[ut.Mass, pl.Series])
    assert_type(2.0 * Q(pl.Series([1.0]), "kg"), Q[ut.Mass, pl.Series])


def test_rmul_float_by_expr_quantity() -> None:
    assert_type(2 * Q(pl.lit(1.0), "kg"), Q[ut.Mass, pl.Expr])
    assert_type(2.0 * Q(pl.lit(1.0), "kg"), Q[ut.Mass, pl.Expr])


def test_truediv_dimensionless_float_float() -> None:
    assert_type(Q(4.0) / Q(2.0), Q[ut.Dimensionless, float])
    assert_type(Q(4.0) / 2, Q[ut.Dimensionless, float])
    assert_type(Q(4.0) / 2.0, Q[ut.Dimensionless, float])


def test_truediv_dimensionless_float_array() -> None:
    assert_type(Q(4.0) / Q([2.0]), Q[ut.Dimensionless, ut.Numpy1DArray])


def test_truediv_dimensionless_float_series() -> None:
    assert_type(Q(4.0) / Q(pl.Series([2.0])), Q[ut.Dimensionless, pl.Series])


def test_truediv_dimensionless_float_expr() -> None:
    assert_type(Q(4.0) / Q(pl.lit(2.0)), Q[ut.Dimensionless, pl.Expr])


def test_truediv_dimensionless_array_float() -> None:
    assert_type(Q([4.0]) / Q(2.0), Q[ut.Dimensionless, np.ndarray])


def test_truediv_dimensionless_array_array() -> None:
    assert_type(Q([4.0]) / Q([2.0]), Q[ut.Dimensionless, np.ndarray])


def test_truediv_dimensionless_series_float() -> None:
    assert_type(Q(pl.Series([4.0])) / Q(2.0), Q[ut.Dimensionless, pl.Series])


def test_truediv_dimensionless_expr_float() -> None:
    assert_type(Q(pl.lit(4.0)) / Q(2.0), Q[ut.Dimensionless, pl.Expr])


def test_truediv_dimensional_by_dimensionless_float() -> None:
    assert_type(Q(4.0, "kg") / Q(2.0), Q[ut.Mass, float])
    assert_type(Q(4.0, "kg") / 2, Q[ut.Mass, float])
    assert_type(Q(4.0, "kg") / 2.0, Q[ut.Mass, float])
    assert_type(Q(4.0, "m") / Q(2.0), Q[ut.Length, float])


def test_truediv_dimensional_by_dimensionless_array() -> None:
    assert_type(Q([4.0], "kg") / Q(2.0), Q[ut.Mass, np.ndarray])
    assert_type(Q([4.0], "m") / Q(2.0), Q[ut.Length, np.ndarray])


def test_truediv_dimensional_by_dimensionless_series() -> None:
    assert_type(Q(pl.Series([4.0]), "kg") / Q(2.0), Q[ut.Mass, pl.Series])


def test_truediv_dimensional_by_dimensionless_expr() -> None:
    assert_type(Q(pl.lit(4.0), "kg") / Q(2.0), Q[ut.Mass, pl.Expr])


def test_truediv_same_dimensional_float() -> None:
    assert_type(Q(4.0, "kg") / Q(2.0, "kg"), Q[ut.Dimensionless, float])
    assert_type(Q(4.0, "m") / Q(2.0, "m"), Q[ut.Dimensionless, float])


def test_truediv_same_dimensional_array() -> None:
    assert_type(Q([4.0], "kg") / Q(2.0, "kg"), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q([4.0], "m") / Q(2.0, "m"), Q[ut.Dimensionless, np.ndarray])


def test_truediv_same_dimensional_series() -> None:
    assert_type(Q(pl.Series([4.0]), "kg") / Q(2.0, "kg"), Q[ut.Dimensionless, pl.Series])


def test_truediv_same_dimensional_expr() -> None:
    assert_type(Q(pl.lit(4.0), "kg") / Q(2.0, "kg"), Q[ut.Dimensionless, pl.Expr])


def test_truediv_different_dimensional_float() -> None:
    assert_type(Q(4.0, "kg") / Q(2.0, "s"), Q[ut.MassFlow, float])
    assert_type(Q(4.0, "m") / Q(2.0, "s"), Q[ut.Velocity, float])


def test_truediv_different_dimensional_array() -> None:
    assert_type(Q([4.0], "kg") / Q(2.0, "s"), Q[ut.MassFlow, np.ndarray])
    assert_type(Q([4.0], "m") / Q(2.0, "s"), Q[ut.Velocity, np.ndarray])


def test_truediv_different_dimensional_series() -> None:
    assert_type(Q(pl.Series([4.0]), "kg") / Q(2.0, "s"), Q[ut.MassFlow, pl.Series])


def test_truediv_different_dimensional_expr() -> None:
    assert_type(Q(pl.lit(4.0), "kg") / Q(2.0, "s"), Q[ut.MassFlow, pl.Expr])


def test_truediv_dimensionless_by_dimensional_float() -> None:
    assert_type(Q(4.0) / Q(2.0, "s"), Q[ut.UnknownDimensionality, float])
    assert_type(Q(4.0) / Q(2.0, "kg"), Q[ut.UnknownDimensionality, float])


def test_truediv_dimensionless_by_dimensional_array() -> None:
    assert_type(Q([4.0]) / Q(2.0, "s"), Q[ut.UnknownDimensionality, np.ndarray])


def test_truediv_dimensionless_by_dimensional_series() -> None:
    assert_type(Q(pl.Series([4.0])) / Q(2.0, "s"), Q[ut.UnknownDimensionality, pl.Series])


def test_truediv_dimensionless_by_dimensional_expr() -> None:
    assert_type(Q(pl.lit(4.0)) / Q(2.0, "s"), Q[ut.UnknownDimensionality, pl.Expr])


def test_rtruediv_float_by_quantity() -> None:
    assert_type(2 / Q(1.0, "s"), Q[ut.UnknownDimensionality, float])
    assert_type(2.0 / Q(1.0, "s"), Q[ut.UnknownDimensionality, float])
    assert_type(2 / Q(1.0), Q[ut.Dimensionless, float])


def test_rtruediv_float_by_array_quantity() -> None:
    assert_type(2 / Q([1.0], "s"), Q[ut.UnknownDimensionality, np.ndarray])
    assert_type(2.0 / Q([1.0], "s"), Q[ut.UnknownDimensionality, np.ndarray])


def test_rtruediv_float_by_series_quantity() -> None:
    assert_type(2 / Q(pl.Series([1.0]), "s"), Q[ut.UnknownDimensionality, pl.Series])


def test_rtruediv_float_by_expr_quantity() -> None:
    assert_type(2 / Q(pl.lit(1.0), "s"), Q[ut.UnknownDimensionality, pl.Expr])


def test_floordiv_dimensional_by_dimensionless() -> None:
    assert_type(Q(10.0, "m") // Q(3.0), Q[ut.Length, float])
    assert_type(Q(10.0, "m") // 3, Q[ut.Length, float])
    assert_type(Q(10.0, "m") // 3.0, Q[ut.Length, float])


def test_floordiv_same_dimensional() -> None:
    assert_type(Q(10.0, "m") // Q(3.0, "m"), Q[ut.Dimensionless, float])
    assert_type(Q(10.0, "kg") // Q(3.0, "kg"), Q[ut.Dimensionless, float])


def test_floordiv_array() -> None:
    assert_type(Q([10.0], "m") // Q(3.0), Q[ut.Length, np.ndarray])
    assert_type(Q([10.0], "m") // Q(3.0, "m"), Q[ut.Dimensionless, np.ndarray])


def test_add_same_dimensional_float() -> None:
    assert_type(Q(1.0, "kg") + Q(2.0, "kg"), Q[ut.Mass, float])
    assert_type(Q(1.0, "m") + Q(2.0, "m"), Q[ut.Length, float])


def test_add_same_dimensional_array() -> None:
    assert_type(Q([1.0], "kg") + Q(2.0, "kg"), Q[ut.Mass, np.ndarray])
    assert_type(Q([1.0], "m") + Q(2.0, "m"), Q[ut.Length, np.ndarray])


def test_add_same_dimensional_series() -> None:
    assert_type(Q(pl.Series([1.0]), "kg") + Q(2.0, "kg"), Q[ut.Mass, pl.Series])


def test_add_same_dimensional_expr() -> None:
    assert_type(Q(pl.lit(1.0), "kg") + Q(2.0, "kg"), Q[ut.Mass, pl.Expr])


def test_add_dimensionless_float() -> None:
    assert_type(Q(1.0) + Q(2.0), Q[ut.Dimensionless, float])
    assert_type(Q(1.0) + 2, Q[ut.Dimensionless, float])
    assert_type(Q(1.0) + 2.0, Q[ut.Dimensionless, float])


def test_add_dimensionless_array() -> None:
    assert_type(Q([1.0]) + Q(2.0), Q[ut.Dimensionless, np.ndarray])
    assert_type(Q([1.0]) + Q([2.0]), Q[ut.Dimensionless, np.ndarray])


def test_sub_same_dimensional_float() -> None:
    assert_type(Q(3.0, "kg") - Q(1.0, "kg"), Q[ut.Mass, float])
    assert_type(Q(3.0, "m") - Q(1.0, "m"), Q[ut.Length, float])


def test_sub_same_dimensional_array() -> None:
    assert_type(Q([3.0], "kg") - Q(1.0, "kg"), Q[ut.Mass, np.ndarray])


def test_sub_same_dimensional_series() -> None:
    assert_type(Q(pl.Series([3.0]), "kg") - Q(1.0, "kg"), Q[ut.Mass, pl.Series])


def test_sub_same_dimensional_expr() -> None:
    assert_type(Q(pl.lit(3.0), "kg") - Q(1.0, "kg"), Q[ut.Mass, pl.Expr])


def test_sub_dimensionless_float() -> None:
    assert_type(Q(3.0) - Q(1.0), Q[ut.Dimensionless, float])
    assert_type(Q(3.0) - 1, Q[ut.Dimensionless, float])
    assert_type(Q(3.0) - 1.0, Q[ut.Dimensionless, float])


def test_pow_length_squared() -> None:
    assert_type(Q(2.0, "m") ** 2, Q[ut.Area, float])


def test_pow_length_cubed() -> None:
    assert_type(Q(2.0, "m") ** 3, Q[ut.Volume, float])


def test_pow_dimensionless() -> None:
    assert_type(Q(2.0) ** 2, Q[ut.Dimensionless, float])
    assert_type(Q(2.0) ** 3, Q[ut.Dimensionless, float])
    assert_type(Q(2.0) ** 0.5, Q[ut.Dimensionless, float])


def test_pow_general() -> None:
    assert_type(Q(2.0, "kg") ** 2, Q[ut.UnknownDimensionality, float])
    assert_type(Q(2.0, "s") ** -1, Q[ut.UnknownDimensionality, float])


def test_mul_float_scalar_by_array_dimensional() -> None:
    assert_type(Q(2.0, "kg") * Q([1.0, 2.0], "m"), Q[ut.UnknownDimensionality, np.ndarray])


def test_truediv_float_scalar_by_array_dimensional() -> None:
    assert_type(Q(4.0, "m") / Q([2.0, 4.0], "s"), Q[ut.Velocity, np.ndarray])


def test_mul_array_dimensional_by_array_dimensional() -> None:
    assert_type(Q([1.0, 2.0], "kg") * Q([3.0, 4.0], "m"), Q[ut.UnknownDimensionality, np.ndarray])


def test_truediv_array_dimensional_by_array_dimensional() -> None:
    assert_type(Q([4.0, 8.0], "m") / Q([2.0, 4.0], "s"), Q[ut.Velocity, np.ndarray])


def test_truediv_mass_time_to_massflow_float() -> None:
    assert_type(Q(10.0, "kg") / Q(2.0, "s"), Q[ut.MassFlow, float])
    assert_type(Q(10.0, "kg") / Q(2.0, "min"), Q[ut.MassFlow, float])
    assert_type(Q(10.0, "kg") / Q(2.0, "h"), Q[ut.MassFlow, float])


def test_truediv_mass_time_to_massflow_array() -> None:
    assert_type(Q([10.0], "kg") / Q(2.0, "s"), Q[ut.MassFlow, np.ndarray])
    assert_type(Q(np.array([10.0, 20.0]), "kg") / Q(2.0, "s"), Q[ut.MassFlow, np.ndarray])


def test_truediv_mass_time_to_massflow_series() -> None:
    assert_type(Q(pl.Series([10.0]), "kg") / Q(2.0, "s"), Q[ut.MassFlow, pl.Series])


def test_truediv_mass_time_to_massflow_expr() -> None:
    assert_type(Q(pl.lit(10.0), "kg") / Q(2.0, "s"), Q[ut.MassFlow, pl.Expr])
    assert_type(Q(pl.col("mass"), "kg") / Q(2.0, "s"), Q[ut.MassFlow, pl.Expr])


def test_truediv_volume_time_to_volumeflow_float() -> None:
    assert_type(Q(10.0, "m^3") / Q(2.0, "s"), Q[ut.VolumeFlow, float])
    assert_type(Q(10.0, "L") / Q(2.0, "min"), Q[ut.VolumeFlow, float])


def test_truediv_volume_time_to_volumeflow_array() -> None:
    assert_type(Q([10.0], "m^3") / Q(2.0, "s"), Q[ut.VolumeFlow, np.ndarray])


def test_truediv_volume_time_to_volumeflow_series() -> None:
    assert_type(Q(pl.Series([10.0]), "m^3") / Q(2.0, "s"), Q[ut.VolumeFlow, pl.Series])


def test_truediv_volume_time_to_volumeflow_expr() -> None:
    assert_type(Q(pl.lit(10.0), "m^3") / Q(2.0, "s"), Q[ut.VolumeFlow, pl.Expr])


def test_truediv_energy_time_to_power_float() -> None:
    assert_type(Q(1000.0, "J") / Q(2.0, "s"), Q[ut.Power, float])
    assert_type(Q(1000.0, "kJ") / Q(2.0, "h"), Q[ut.Power, float])


def test_truediv_energy_time_to_power_array() -> None:
    assert_type(Q([1000.0], "J") / Q(2.0, "s"), Q[ut.Power, np.ndarray])


def test_truediv_energy_time_to_power_series() -> None:
    assert_type(Q(pl.Series([1000.0]), "J") / Q(2.0, "s"), Q[ut.Power, pl.Series])


def test_truediv_energy_time_to_power_expr() -> None:
    assert_type(Q(pl.lit(1000.0), "J") / Q(2.0, "s"), Q[ut.Power, pl.Expr])


def test_truediv_length_time_to_velocity_float() -> None:
    assert_type(Q(100.0, "m") / Q(10.0, "s"), Q[ut.Velocity, float])
    assert_type(Q(100.0, "km") / Q(1.0, "h"), Q[ut.Velocity, float])


def test_truediv_length_time_to_velocity_array() -> None:
    assert_type(Q([100.0], "m") / Q(10.0, "s"), Q[ut.Velocity, np.ndarray])


def test_truediv_length_time_to_velocity_series() -> None:
    assert_type(Q(pl.Series([100.0]), "m") / Q(10.0, "s"), Q[ut.Velocity, pl.Series])


def test_truediv_length_time_to_velocity_expr() -> None:
    assert_type(Q(pl.lit(100.0), "m") / Q(10.0, "s"), Q[ut.Velocity, pl.Expr])


def test_truediv_energy_mass_to_energypermass_float() -> None:
    assert_type(Q(1000.0, "J") / Q(1.0, "kg"), Q[ut.EnergyPerMass, float])
    assert_type(Q(1000.0, "kJ") / Q(1.0, "g"), Q[ut.EnergyPerMass, float])


def test_truediv_energy_mass_to_energypermass_array() -> None:
    assert_type(Q([1000.0], "J") / Q(1.0, "kg"), Q[ut.EnergyPerMass, np.ndarray])


def test_truediv_energy_mass_to_energypermass_series() -> None:
    assert_type(Q(pl.Series([1000.0]), "J") / Q(1.0, "kg"), Q[ut.EnergyPerMass, pl.Series])


def test_truediv_energy_mass_to_energypermass_expr() -> None:
    assert_type(Q(pl.lit(1000.0), "J") / Q(1.0, "kg"), Q[ut.EnergyPerMass, pl.Expr])


def test_truediv_mass_volume_to_density_float() -> None:
    assert_type(Q(1000.0, "kg") / Q(1.0, "m^3"), Q[ut.Density, float])
    assert_type(Q(1.0, "g") / Q(1.0, "cm^3"), Q[ut.Density, float])


def test_truediv_mass_volume_to_density_array() -> None:
    assert_type(Q([1000.0], "kg") / Q(1.0, "m^3"), Q[ut.Density, np.ndarray])


def test_truediv_mass_volume_to_density_series() -> None:
    assert_type(Q(pl.Series([1000.0]), "kg") / Q(1.0, "m^3"), Q[ut.Density, pl.Series])


def test_truediv_mass_volume_to_density_expr() -> None:
    assert_type(Q(pl.lit(1000.0), "kg") / Q(1.0, "m^3"), Q[ut.Density, pl.Expr])


def test_mul_massflow_time_to_mass_float() -> None:
    assert_type(Q(5.0, "kg/s") * Q(10.0, "s"), Q[ut.Mass, float])
    assert_type(Q(5.0, "kg/h") * Q(2.0, "h"), Q[ut.Mass, float])


def test_mul_massflow_time_to_mass_array() -> None:
    assert_type(Q([5.0], "kg/s") * Q(10.0, "s"), Q[ut.Mass, np.ndarray])


def test_mul_massflow_time_to_mass_series() -> None:
    assert_type(Q(pl.Series([5.0]), "kg/s") * Q(10.0, "s"), Q[ut.Mass, pl.Series])


def test_mul_massflow_time_to_mass_expr() -> None:
    assert_type(Q(pl.lit(5.0), "kg/s") * Q(10.0, "s"), Q[ut.Mass, pl.Expr])


def test_mul_time_massflow_to_mass_float() -> None:
    assert_type(Q(10.0, "s") * Q(5.0, "kg/s"), Q[ut.Mass, float])


def test_mul_time_massflow_to_mass_array() -> None:
    assert_type(Q([10.0], "s") * Q(5.0, "kg/s"), Q[ut.Mass, np.ndarray])


def test_mul_time_massflow_to_mass_series() -> None:
    assert_type(Q(pl.Series([10.0]), "s") * Q(5.0, "kg/s"), Q[ut.Mass, pl.Series])


def test_mul_time_massflow_to_mass_expr() -> None:
    assert_type(Q(pl.lit(10.0), "s") * Q(5.0, "kg/s"), Q[ut.Mass, pl.Expr])


def test_mul_volumeflow_time_to_volume_float() -> None:
    assert_type(Q(2.0, "m^3/s") * Q(30.0, "s"), Q[ut.Volume, float])


def test_mul_volumeflow_time_to_volume_array() -> None:
    assert_type(Q([2.0], "m^3/s") * Q(30.0, "s"), Q[ut.Volume, np.ndarray])


def test_mul_volumeflow_time_to_volume_series() -> None:
    assert_type(Q(pl.Series([2.0]), "m^3/s") * Q(30.0, "s"), Q[ut.Volume, pl.Series])


def test_mul_volumeflow_time_to_volume_expr() -> None:
    assert_type(Q(pl.lit(2.0), "m^3/s") * Q(30.0, "s"), Q[ut.Volume, pl.Expr])


def test_mul_time_volumeflow_to_volume_float() -> None:
    assert_type(Q(30.0, "s") * Q(2.0, "m^3/s"), Q[ut.Volume, float])


def test_mul_time_volumeflow_to_volume_array() -> None:
    assert_type(Q([30.0], "s") * Q(2.0, "m^3/s"), Q[ut.Volume, np.ndarray])


def test_mul_power_time_to_energy_float() -> None:
    assert_type(Q(1000.0, "W") * Q(3600.0, "s"), Q[ut.Energy, float])
    assert_type(Q(1.0, "kW") * Q(1.0, "h"), Q[ut.Energy, float])


def test_mul_power_time_to_energy_array() -> None:
    assert_type(Q([1000.0], "W") * Q(3600.0, "s"), Q[ut.Energy, np.ndarray])


def test_mul_power_time_to_energy_series() -> None:
    assert_type(Q(pl.Series([1000.0]), "W") * Q(3600.0, "s"), Q[ut.Energy, pl.Series])


def test_mul_power_time_to_energy_expr() -> None:
    assert_type(Q(pl.lit(1000.0), "W") * Q(3600.0, "s"), Q[ut.Energy, pl.Expr])


def test_mul_time_power_to_energy_float() -> None:
    assert_type(Q(3600.0, "s") * Q(1000.0, "W"), Q[ut.Energy, float])


def test_mul_time_power_to_energy_array() -> None:
    assert_type(Q([3600.0], "s") * Q(1000.0, "W"), Q[ut.Energy, np.ndarray])


def test_mul_velocity_time_to_length_float() -> None:
    assert_type(Q(10.0, "m/s") * Q(5.0, "s"), Q[ut.Length, float])


def test_mul_velocity_time_to_length_array() -> None:
    assert_type(Q([10.0], "m/s") * Q(5.0, "s"), Q[ut.Length, np.ndarray])


def test_mul_velocity_time_to_length_series() -> None:
    assert_type(Q(pl.Series([10.0]), "m/s") * Q(5.0, "s"), Q[ut.Length, pl.Series])


def test_mul_velocity_time_to_length_expr() -> None:
    assert_type(Q(pl.lit(10.0), "m/s") * Q(5.0, "s"), Q[ut.Length, pl.Expr])


def test_mul_time_velocity_to_length_float() -> None:
    assert_type(Q(5.0, "s") * Q(10.0, "m/s"), Q[ut.Length, float])


def test_mul_time_velocity_to_length_array() -> None:
    assert_type(Q([5.0], "s") * Q(10.0, "m/s"), Q[ut.Length, np.ndarray])


def test_mul_density_volume_to_mass_float() -> None:
    assert_type(Q(1000.0, "kg/m^3") * Q(1.0, "m^3"), Q[ut.Mass, float])


def test_mul_density_volume_to_mass_array() -> None:
    assert_type(Q([1000.0], "kg/m^3") * Q(1.0, "m^3"), Q[ut.Mass, np.ndarray])


def test_mul_density_volume_to_mass_series() -> None:
    assert_type(Q(pl.Series([1000.0]), "kg/m^3") * Q(1.0, "m^3"), Q[ut.Mass, pl.Series])


def test_mul_density_volume_to_mass_expr() -> None:
    assert_type(Q(pl.lit(1000.0), "kg/m^3") * Q(1.0, "m^3"), Q[ut.Mass, pl.Expr])


def test_mul_volume_density_to_mass_float() -> None:
    assert_type(Q(1.0, "m^3") * Q(1000.0, "kg/m^3"), Q[ut.Mass, float])


def test_mul_volume_density_to_mass_array() -> None:
    assert_type(Q([1.0], "m^3") * Q(1000.0, "kg/m^3"), Q[ut.Mass, np.ndarray])


def test_mul_length_length_to_area_float() -> None:
    assert_type(Q(5.0, "m") * Q(3.0, "m"), Q[ut.Area, float])
    assert_type(Q(5.0, "cm") * Q(3.0, "cm"), Q[ut.Area, float])


def test_mul_length_length_to_area_array() -> None:
    assert_type(Q([5.0], "m") * Q(3.0, "m"), Q[ut.Area, np.ndarray])


def test_mul_length_length_to_area_series() -> None:
    assert_type(Q(pl.Series([5.0]), "m") * Q(3.0, "m"), Q[ut.Area, pl.Series])


def test_mul_length_length_to_area_expr() -> None:
    assert_type(Q(pl.lit(5.0), "m") * Q(3.0, "m"), Q[ut.Area, pl.Expr])


def test_mul_length_area_to_volume_float() -> None:
    assert_type(Q(2.0, "m") * Q(10.0, "m^2"), Q[ut.Volume, float])


def test_mul_length_area_to_volume_array() -> None:
    assert_type(Q([2.0], "m") * Q(10.0, "m^2"), Q[ut.Volume, np.ndarray])


def test_mul_length_area_to_volume_series() -> None:
    assert_type(Q(pl.Series([2.0]), "m") * Q(10.0, "m^2"), Q[ut.Volume, pl.Series])


def test_mul_length_area_to_volume_expr() -> None:
    assert_type(Q(pl.lit(2.0), "m") * Q(10.0, "m^2"), Q[ut.Volume, pl.Expr])


def test_mul_area_length_to_volume_float() -> None:
    assert_type(Q(10.0, "m^2") * Q(2.0, "m"), Q[ut.Volume, float])


def test_mul_area_length_to_volume_array() -> None:
    assert_type(Q([10.0], "m^2") * Q(2.0, "m"), Q[ut.Volume, np.ndarray])


def test_mul_area_length_to_volume_series() -> None:
    assert_type(Q(pl.Series([10.0]), "m^2") * Q(2.0, "m"), Q[ut.Volume, pl.Series])


def test_mul_area_length_to_volume_expr() -> None:
    assert_type(Q(pl.lit(10.0), "m^2") * Q(2.0, "m"), Q[ut.Volume, pl.Expr])


def test_unknown_second_unit_float() -> None:
    unknown_q = Q(30.0, "s * m")
    v = Q(2.0, "m^3/s") * unknown_q
    assert_type(v, Q[ut.UnknownDimensionality, float])


def test_pow_inference() -> None:
    assert_type(Q(25, "kg"), Q[ut.Mass, float])
    assert_type(Q(25, "kg") ** 1, Q[ut.Mass, float])
    assert_type(Q(25, "kg") ** 2, Q[ut.UnknownDimensionality, float])
    assert_type(Q(25, "m") ** 2, Q[ut.Area, float])
    assert_type(Q(25, "m") ** 3, Q[ut.Volume, float])
    assert_type(Q(25, "m") ** 2 * Q(2, "m"), Q[ut.Volume, float])
    assert_type(Q(25, "m") * Q(2, "m") ** 2, Q[ut.Volume, float])
    assert_type(Q(25, "m") * Q(2, "m") * Q(25, "cm"), Q[ut.Volume, float])

    assert_type(Q(25, "kg") ** 0.5, Q[ut.UnknownDimensionality, float])
    assert_type(Q(25, "m") ** 0.5, Q[ut.UnknownDimensionality, float])


def test_unknown() -> None:
    assert_type(Q(25, "kg").unknown(), Q[ut.UnknownDimensionality, float])
    assert_type(Q([1, 23], "kg").unknown(), Q[ut.UnknownDimensionality, ut.Numpy1DArray])
    assert_type(Q([1, 23], "kg").unknown(), Q[ut.UnknownDimensionality, np.ndarray])
