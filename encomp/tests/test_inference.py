from typing import assert_type

import numpy as np
import pandas as pd
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
    assert_type(Q(pd.Series([1])), Q[ut.Dimensionless, pd.Series])

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
    """Test multiplication type inference with various magnitude types."""
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
    """Test division type inference with various magnitude types."""
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
    assert_type(Q(2) / Q(1, "s"), Q[ut.Frequency, float])
    assert_type(Q([2, 4]) / Q(1, "s"), Q[ut.Frequency, np.ndarray])

    # dimensional / dimensionless (scalar only)
    assert_type(Q(4, "m") / Q(2), Q[ut.Length, float])


def test_inference_addition_subtraction() -> None:
    """Test addition and subtraction type inference."""
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
    """Test comparison type inference."""
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


def test_inference_pandas_polars() -> None:
    """Test type inference with pandas and polars types."""
    # pandas Series construction
    assert_type(Q(pd.Series([1, 2])), Q[ut.Dimensionless, pd.Series])

    # polars Series construction
    assert_type(Q(pl.Series([1, 2])), Q[ut.Dimensionless, pl.Series])

    # polars Expr
    assert_type(Q(pl.lit(1)), Q[ut.Dimensionless, pl.Expr])
    assert_type(Q(pl.lit(1)) * Q(2, "kg"), Q[ut.Mass, pl.Expr])
    assert_type(Q(pl.lit(1), "m") * Q(2, "m"), Q[ut.Area, pl.Expr])


def test_inference_complex_units() -> None:
    """Test type inference with complex unit combinations."""
    # velocity = length / time
    assert_type(Q(10, "m") / Q(2, "s"), Q[ut.Velocity, float])
    assert_type(Q([10, 20], "m") / Q(2, "s"), Q[ut.Velocity, np.ndarray])

    # energy = mass * velocity^2
    velocity = Q(5, "m/s")
    assert_type(Q(2, "kg") * velocity * velocity, Q[ut.Energy, float])

    # power = energy / time
    assert_type(Q(100, "J") / Q(2, "s"), Q[ut.Power, float])

    # density = mass / volume
    assert_type(Q(1000, "kg") / Q(1, "m^3"), Q[ut.Density, float])

    # pressure = force / area
    assert_type(Q(100, "N") / Q(2, "m^2"), Q[ut.Pressure, float])


def test_inference_dimensional_multiplication() -> None:
    """Test dimensional type inference for multiplication operations."""
    # Length * Length = Area
    assert_type(Q(5.0, "m") * Q(3.0, "m"), Q[ut.Area, float])
    assert_type(Q([5.0], "m") * Q(3.0, "m"), Q[ut.Area, np.ndarray])
    assert_type(Q(pl.Series([5.0]), "m") * Q(3.0, "m"), Q[ut.Area, pl.Series])
    assert_type(Q(pd.Series([5.0]), "m") * Q(3.0, "m"), Q[ut.Area, pd.Series])

    # Length * Area = Volume
    assert_type(Q(2.0, "m") * Q(10.0, "m^2"), Q[ut.Volume, float])
    assert_type(Q([2.0, 3.0], "m") * Q(10.0, "m^2"), Q[ut.Volume, np.ndarray])

    # Area * Length = Volume (commutative)
    assert_type(Q(10.0, "m^2") * Q(2.0, "m"), Q[ut.Volume, float])

    # Mass * SpecificVolume = Volume
    assert_type(Q(100.0, "kg") * Q(0.001, "m^3/kg"), Q[ut.Volume, float])
    assert_type(Q(pl.lit(100.0), "kg") * Q(0.001, "m^3/kg"), Q[ut.Volume, pl.Expr])

    # Time * Power = Energy
    assert_type(Q(3600.0, "s") * Q(1000.0, "W"), Q[ut.Energy, float])
    assert_type(Q(pd.Series([3600.0]), "s") * Q(1000.0, "W"), Q[ut.Energy, pd.Series])

    # Time * MassFlow = Mass
    assert_type(Q(10.0, "s") * Q(5.0, "kg/s"), Q[ut.Mass, float])
    assert_type(Q([10.0, 20.0], "s") * Q(5.0, "kg/s"), Q[ut.Mass, np.ndarray])

    # Time * VolumeFlow = Volume
    assert_type(Q(60.0, "s") * Q(2.0, "m^3/s"), Q[ut.Volume, float])

    # Volume * Density = Mass
    assert_type(Q(1.0, "m^3") * Q(1000.0, "kg/m^3"), Q[ut.Mass, float])
    assert_type(Q(pl.Series([1.0, 2.0]), "m^3") * Q(1000.0, "kg/m^3"), Q[ut.Mass, pl.Series])

    # Pressure * Volume = Energy
    assert_type(Q(101325.0, "Pa") * Q(1.0, "m^3"), Q[ut.Energy, float])

    # Velocity * Area = VolumeFlow
    assert_type(Q(5.0, "m/s") * Q(2.0, "m^2"), Q[ut.VolumeFlow, float])
    assert_type(Q([5.0, 10.0], "m/s") * Q(2.0, "m^2"), Q[ut.VolumeFlow, np.ndarray])


def test_inference_dimensional_division() -> None:
    """Test dimensional type inference for division operations."""
    # Mass / Time = MassFlow
    assert_type(Q(100.0, "kg") / Q(10.0, "s"), Q[ut.MassFlow, float])
    assert_type(Q([100.0, 200.0], "kg") / Q(10.0, "s"), Q[ut.MassFlow, np.ndarray])
    assert_type(Q(pl.Series([100.0]), "kg") / Q(10.0, "s"), Q[ut.MassFlow, pl.Series])
    assert_type(Q(pd.Series([100.0]), "kg") / Q(10.0, "s"), Q[ut.MassFlow, pd.Series])

    # Volume / Time = VolumeFlow
    assert_type(Q(10.0, "m^3") / Q(5.0, "s"), Q[ut.VolumeFlow, float])
    assert_type(Q(pl.lit(10.0), "m^3") / Q(5.0, "s"), Q[ut.VolumeFlow, pl.Expr])

    # Mass / Volume = Density
    assert_type(Q(1000.0, "kg") / Q(1.0, "m^3"), Q[ut.Density, float])
    assert_type(Q([1000.0, 2000.0], "kg") / Q(1.0, "m^3"), Q[ut.Density, np.ndarray])

    # Length / Time = Velocity
    assert_type(Q(100.0, "m") / Q(10.0, "s"), Q[ut.Velocity, float])
    assert_type(Q(pd.Series([100.0]), "m") / Q(10.0, "s"), Q[ut.Velocity, pd.Series])

    # Energy / Time = Power
    assert_type(Q(3600000.0, "J") / Q(3600.0, "s"), Q[ut.Power, float])

    # Energy / Mass = EnergyPerMass (specific energy)
    assert_type(Q(1000.0, "J") / Q(1.0, "kg"), Q[ut.EnergyPerMass, float])
    assert_type(Q([1000.0, 2000.0], "J") / Q(1.0, "kg"), Q[ut.EnergyPerMass, np.ndarray])

    # Volume / Area = Length
    assert_type(Q(100.0, "m^3") / Q(10.0, "m^2"), Q[ut.Length, float])

    # Area / Length = Length
    assert_type(Q(50.0, "m^2") / Q(5.0, "m"), Q[ut.Length, float])
    assert_type(Q(pl.Series([50.0]), "m^2") / Q(5.0, "m"), Q[ut.Length, pl.Series])

    # VolumeFlow / Area = Velocity
    assert_type(Q(10.0, "m^3/s") / Q(2.0, "m^2"), Q[ut.Velocity, float])

    # Power / Area = PowerPerArea (heat flux)
    assert_type(Q(1000.0, "W") / Q(1.0, "m^2"), Q[ut.PowerPerArea, float])
    assert_type(Q([1000.0, 2000.0], "W") / Q(1.0, "m^2"), Q[ut.PowerPerArea, np.ndarray])

    # MassFlow / Density = VolumeFlow
    assert_type(Q(10.0, "kg/s") / Q(1000.0, "kg/m^3"), Q[ut.VolumeFlow, float])


def test_inference_dimensional_derived_units() -> None:
    """Test dimensional inference for derived units and complex operations."""
    # Pressure = Force / Area with different notations for m²
    assert_type(Q(1000.0, "N/m2"), Q[ut.Pressure, float])
    assert_type(Q(1000.0, "N/m^2"), Q[ut.Pressure, float])
    assert_type(Q(1000.0, "N/m**2"), Q[ut.Pressure, float])
    assert_type(Q(1000.0, "N/m²"), Q[ut.Pressure, float])

    # Test with various magnitude types
    assert_type(Q([1000.0], "N/m2"), Q[ut.Pressure, np.ndarray])
    assert_type(Q(pd.Series([1000.0]), "N/m^2"), Q[ut.Pressure, pd.Series])
    assert_type(Q(pl.Series([1000.0]), "N/m**2"), Q[ut.Pressure, pl.Series])
    assert_type(Q(pl.lit(1000.0), "N/m²"), Q[ut.Pressure, pl.Expr])

    # Kinematic Viscosity = Length * Velocity
    assert_type(Q(0.1, "m") * Q(0.5, "m/s"), Q[ut.KinematicViscosity, float])
    assert_type(Q(pl.Series([0.1]), "m") * Q(0.5, "m/s"), Q[ut.KinematicViscosity, pl.Series])
    assert_type(Q(pd.Series([0.1]), "m") * Q(0.5, "m/s"), Q[ut.KinematicViscosity, pd.Series])
    assert_type(Q([0.1], "m") * Q(0.5, "m/s"), Q[ut.KinematicViscosity, np.ndarray])

    # Dynamic Viscosity = Density * Kinematic Viscosity
    assert_type(Q(1000.0, "kg/m^3") * Q(0.001, "m^2/s"), Q[ut.DynamicViscosity, float])
    assert_type(Q([1000.0], "kg/m^3") * Q(0.001, "m^2/s"), Q[ut.DynamicViscosity, np.ndarray])
    assert_type(Q(pd.Series([1000.0]), "kg/m³") * Q(0.001, "m²/s"), Q[ut.DynamicViscosity, pd.Series])
    assert_type(Q(pl.lit(1000.0), "kg/m3") * Q(0.001, "m2/s"), Q[ut.DynamicViscosity, pl.Expr])

    # Specific Volume = Dimensionless / Density
    assert_type(Q(1.0) / Q(1000.0, "kg/m^3"), Q[ut.SpecificVolume, float])
    assert_type(Q([1.0]) / Q(1000.0, "kg/m³"), Q[ut.SpecificVolume, np.ndarray])
    assert_type(Q(pl.Series([1.0])) / Q(1000.0, "kg/m3"), Q[ut.SpecificVolume, pl.Series])

    # Thermal Conductivity = Power / (Length * TemperatureDifference) = W/(m·K)
    # Using the relation: Length * ThermalConductivity = PowerPerTemperature
    assert_type(Q(0.6, "W/m/K"), Q[ut.ThermalConductivity, float])
    assert_type(Q([0.6, 0.8], "W/m/K"), Q[ut.ThermalConductivity, np.ndarray])
    assert_type(Q(pd.Series([0.6]), "W/m/K"), Q[ut.ThermalConductivity, pd.Series])
    assert_type(Q(pl.lit(0.6), "W/m/K"), Q[ut.ThermalConductivity, pl.Expr])

    # Heat Transfer Coefficient = PowerPerArea / TemperatureDifference = W/(m²·K)
    assert_type(Q(10.0, "W/m^2/K"), Q[ut.HeatTransferCoefficient, float])
    assert_type(Q([10.0, 20.0], "W/m²/K"), Q[ut.HeatTransferCoefficient, np.ndarray])
    assert_type(Q(pd.Series([10.0]), "W/m2/K"), Q[ut.HeatTransferCoefficient, pd.Series])
    assert_type(Q(pl.Series([10.0]), "W/m**2/K"), Q[ut.HeatTransferCoefficient, pl.Series])

    # Frequency = Dimensionless / Time
    assert_type(Q(1.0) / Q(0.5, "s"), Q[ut.Frequency, float])
    assert_type(Q([1.0, 2.0]) / Q(0.5, "s"), Q[ut.Frequency, np.ndarray])
    assert_type(Q(pl.Series([1.0])) / Q(0.5, "s"), Q[ut.Frequency, pl.Series])

    # MolarMass = Mass / Substance
    assert_type(Q(18.0, "g") / Q(1.0, "mol"), Q[ut.MolarMass, float])
    assert_type(Q([18.0, 44.0], "g") / Q(1.0, "mol"), Q[ut.MolarMass, np.ndarray])
    assert_type(Q(pl.lit(18.0), "g") / Q(1.0, "mol"), Q[ut.MolarMass, pl.Expr])

    # SubstancePerMass = Substance / Mass (reciprocal of MolarMass)
    assert_type(Q(1.0, "mol") / Q(18.0, "g"), Q[ut.SubstancePerMass, float])
    assert_type(Q([1.0, 2.0], "kmol") / Q(18.0, "kg"), Q[ut.SubstancePerMass, np.ndarray])


def test_various() -> None:
    assert_type(Q(1.0, "m"), Q[ut.Length, float])
    assert_type(Q(1, "m"), Q[ut.Length, float])

    assert_type(Q(1, str("m")), Q)  # pyright: ignore[reportAssertTypeFailure] # noqa: UP018

    assert_type(Q([1, 2, 3], "kg"), Q[ut.Mass, np.ndarray])

    assert_type(Q(pl.col.asd, "kg"), Q[ut.Mass, pl.Expr])
    assert_type(Q(pl.DataFrame({"test": []})["test"], "kg"), Q[ut.Mass, pl.Series])

    assert_type(Q(pl.col.asd, "kg") / Q(25, "min"), Q[ut.MassFlow, pl.Expr])
    assert_type(Q(pl.col.asd, "kg") / Q([1, 3, 4], "day"), Q[ut.MassFlow, pl.Expr])


def test_inference_magnitude_type_promotion() -> None:
    """Test cases where magnitude type promotion is not correctly inferred.

    These tests document issues where operations between scalars and arrays
    don't properly promote to array types. Both runtime checks and static
    type assertions are included to show the mismatch.
    """
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

    # ISSUE: pandas/polars Series operations - type preservation
    # Some operations preserve Series type at runtime, but static inference doesn't reflect this
    _ = Q(pd.Series([1, 2])) * Q(2)
    # Runtime preserves pd.Series in some cases
    # Static type: Q[Dimensionless, pd.Series] (actually correct at runtime sometimes)

    _ = Q(pd.Series([1, 2]), "m") * Q(2, "s")
    # May preserve pd.Series at runtime
    # Static type mismatch: expected Q[KinematicViscosity, pd.Series] but got Q[Any, Any]

    _ = Q(2) * Q(pd.Series([1, 2]), "m")
    # Static type: Q[Length, float] but runtime may have pd.Series
    # Type promotion issue: scalar * Series

    _ = Q(pl.Series([1, 2])) * Q(2)
    # Runtime behavior varies for polars Series
    # Static type: Q[Dimensionless, pl.Series]

    _ = Q(pl.Series([1, 2]), "m") / Q(2, "s")
    # Runtime behavior varies
    # Static type mismatch


def test_inference_static_vs_runtime_mismatch() -> None:
    """Test cases where static type inference doesn't match runtime behavior.

    These document cases where the static type checker infers one type,
    but the runtime produces a different (more specific) type. This is
    primarily due to numpy's broadcasting rules not being captured in
    the type system.
    """
    # ISSUE: Q(1) * Q([1]) is statically inferred as float but runtime is ndarray
    # Static inference sees: Quantity[Dimensionless, float] * Quantity[Dimensionless, ndarray]
    # and infers the left operand's magnitude type (float)
    # Runtime: 1 * array([1]) = array([1]) which is ndarray
    result = Q(1) * Q([1])
    assert isinstance(result.m, np.ndarray)  # Runtime: ndarray
    # But static type is Q[Dimensionless, float] - this is the inconsistency

    # Similar issue with division
    result2 = Q(1) / Q([2])
    assert isinstance(result2.m, np.ndarray)  # Runtime: ndarray
    # But static type would be Q[Dimensionless, float]

    # This works correctly (array on left)
    result3 = Q([1]) * Q(1)
    assert isinstance(result3.m, np.ndarray)
    assert_type(result3, Q[ut.Dimensionless, np.ndarray])  # This matches!
