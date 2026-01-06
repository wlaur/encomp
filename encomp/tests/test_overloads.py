from typing import assert_type

import numpy as np
import polars as pl

from ..units import Quantity as Q
from ..utypes import (
    Area,
    Density,
    Dimensionless,
    Energy,
    EnergyPerMass,
    Length,
    Mass,
    MassFlow,
    Numpy1DArray,
    Numpy1DBoolArray,
    Power,
    UnknownDimensionality,
    Velocity,
    Volume,
    VolumeFlow,
)


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_eq() -> None:
    r0 = Q(5, "kg") == Q(5_000, "g")
    assert_type(r0, bool)
    assert isinstance(r0, bool)
    assert r0

    r1 = Q([1, 2, 3], "kg") == Q([1, 2, 3], "kg")
    assert_type(r1, Numpy1DBoolArray)
    assert isinstance(r1, np.ndarray)
    assert r1.all()

    r2 = Q([1, 2, 3], "kg") == Q([1_000, 2_000, 3_000], "g")
    assert_type(r2, Numpy1DBoolArray)
    assert isinstance(r2, np.ndarray)
    assert r2.all()

    r3 = Q([1, 2, 3], "kg") == Q([1_001, 2_000, 3_000], "g")
    assert_type(r3, Numpy1DBoolArray)
    assert isinstance(r3, np.ndarray)
    assert not r3.all()

    r4 = Q([5, 5, 5], "kg") == Q(5_000, "g")
    assert_type(r4, Numpy1DBoolArray)
    assert isinstance(r4, np.ndarray)
    assert r4.all()

    expr = Q(pl.col.mass_kg, "kg") == Q(3_000, "g")
    assert_type(expr, pl.Expr)

    assert pl.DataFrame({"mass_kg": [1, 2, 3]}).select(expr).sum().item() == 1


def test_unknown_mul_div() -> None:
    q1 = Q(2, "kilogram")
    assert_type(q1, Q[UnknownDimensionality, float])

    q2 = Q(2, "meter/kg * kg")
    assert_type(q2, Q[UnknownDimensionality, float])

    assert_type(q1 * 2, Q[UnknownDimensionality, float])
    assert_type(2 * q2, Q[UnknownDimensionality, float])

    assert_type(q1 * q2, Q[UnknownDimensionality, float])
    assert_type(q2 * q1, Q[UnknownDimensionality, float])

    assert_type(q1 / 2, Q[UnknownDimensionality, float])
    assert_type(2 / q2, Q[UnknownDimensionality, float])

    assert_type(q1 / q2, Q[UnknownDimensionality, float])
    assert_type(q2 / q1, Q[UnknownDimensionality, float])

    q3 = Q(pl.Series([1, 2, 3]), "meter/kg * kg")
    assert_type(q3 * 2, Q[UnknownDimensionality, pl.Series])
    assert_type(q3 * q1, Q[UnknownDimensionality, pl.Series])
    assert_type(q1 * q3, Q[UnknownDimensionality, pl.Series])

    assert_type(q3 / 2, Q[UnknownDimensionality, pl.Series])
    assert_type(q3 / q1, Q[UnknownDimensionality, pl.Series])
    assert_type(q1 / q3, Q[UnknownDimensionality, pl.Series])

    assert_type(Q(2, "m") / Q(25, "km"), Q[Dimensionless, float])
    assert_type(Q(2, "kg") / Q(25, "g"), Q[Dimensionless, float])

    assert_type(Q(25, "kg") / Q([1, 2, 3], "s"), Q[MassFlow, Numpy1DArray])
    assert_type(Q(25, "kg") / Q(pl.col.test, "s"), Q[MassFlow, pl.Expr])


def test_magnitude_type_broadcasting() -> None:
    assert_type(Q(25, "kg") + Q([1, 2, 3], "g"), Q[Mass, Numpy1DArray])
    assert_type(Q(25, "kg") - Q([1, 2, 3], "g"), Q[Mass, Numpy1DArray])
    assert_type(Q([1, 2, 3], "g") - Q(25, "kg"), Q[Mass, Numpy1DArray])
    assert_type(Q([1, 2, 3], "g") + Q(25, "kg"), Q[Mass, Numpy1DArray])

    assert_type(Q(25, "kg") * Q([1, 2, 3], "m"), Q[UnknownDimensionality, Numpy1DArray])
    assert_type(Q(25, "kg") / Q([1, 2, 3], "m"), Q[UnknownDimensionality, Numpy1DArray])
    assert_type(Q([1, 2, 3], "g") * Q(25, "m"), Q[UnknownDimensionality, Numpy1DArray])
    assert_type(Q([1, 2, 3], "g") / Q(25, "m"), Q[UnknownDimensionality, Numpy1DArray])

    assert_type(Q(25, "kg") + Q(pl.col.test, "g"), Q[Mass, pl.Expr])
    assert_type(Q(25, "kg") - Q(pl.col.test, "g"), Q[Mass, pl.Expr])
    assert_type(Q(pl.col.test, "g") - Q(25, "kg"), Q[Mass, pl.Expr])
    assert_type(Q(pl.col.test, "g") + Q(25, "kg"), Q[Mass, pl.Expr])

    assert_type(Q(25, "kg") * Q(pl.col.test, "m"), Q[UnknownDimensionality, pl.Expr])
    assert_type(Q(25, "kg") / Q(pl.col.test, "m"), Q[UnknownDimensionality, pl.Expr])
    assert_type(Q(pl.col.test, "g") * Q(25, "m"), Q[UnknownDimensionality, pl.Expr])
    assert_type(Q(pl.col.test, "g") / Q(25, "m"), Q[UnknownDimensionality, pl.Expr])


def test_mul_massflow_time_all_magnitude_types() -> None:
    assert_type(Q(2.5, "kg/s") * Q(10.0, "s"), Q[Mass, float])
    assert Q(2.5, "kg/s") * Q(10.0, "s") == Q(25.0, "kg")

    assert_type(Q([1.0, 2.0, 3.0], "kg/s") * Q(10.0, "s"), Q[Mass, Numpy1DArray])
    assert (Q([1.0, 2.0, 3.0], "kg/s") * Q(10.0, "s") == Q([10.0, 20.0, 30.0], "kg")).all()

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "kg/s") * Q(10.0, "s"), Q[Mass, pl.Series])

    assert_type(Q(pl.col("flow"), "kg/s") * Q(10.0, "s"), Q[Mass, pl.Expr])

    assert_type(Q(2.5, "kg/s") * Q([1.0, 2.0, 3.0], "s"), Q[Mass, Numpy1DArray])
    assert (Q(2.5, "kg/s") * Q([1.0, 2.0, 3.0], "s") == Q([2.5, 5.0, 7.5], "kg")).all()


def test_mul_time_massflow_all_magnitude_types() -> None:
    assert_type(Q(10.0, "s") * Q(2.5, "kg/s"), Q[Mass, float])
    assert Q(10.0, "s") * Q(2.5, "kg/s") == Q(25.0, "kg")

    assert_type(Q([1.0, 2.0, 3.0], "s") * Q(10.0, "kg/s"), Q[Mass, Numpy1DArray])
    assert (Q([1.0, 2.0, 3.0], "s") * Q(10.0, "kg/s") == Q([10.0, 20.0, 30.0], "kg")).all()

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "s") * Q(10.0, "kg/s"), Q[Mass, pl.Series])

    assert_type(Q(pl.col("time"), "s") * Q(10.0, "kg/s"), Q[Mass, pl.Expr])

    assert_type(Q([1.0, 2.0, 3.0], "s") * Q([10.0, 20.0, 30.0], "kg/s"), Q[Mass, Numpy1DArray])


def test_mul_volumeflow_time_all_magnitude_types() -> None:
    assert_type(Q(2.5, "m3/s") * Q(10.0, "s"), Q[Volume, float])
    assert Q(2.5, "m3/s") * Q(10.0, "s") == Q(25.0, "m3")

    assert_type(Q([1.0, 2.0, 3.0], "m3/s") * Q(10.0, "s"), Q[Volume, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "m3/s") * Q(10.0, "s"), Q[Volume, pl.Series])

    assert_type(Q(pl.col("flow"), "m3/s") * Q(10.0, "s"), Q[Volume, pl.Expr])

    assert_type(Q(2.5, "m3/s") * Q([1.0, 2.0, 3.0], "s"), Q[Volume, Numpy1DArray])


def test_mul_time_volumeflow_all_magnitude_types() -> None:
    assert_type(Q(10.0, "s") * Q(2.5, "m3/s"), Q[Volume, float])

    assert_type(Q([1.0, 2.0, 3.0], "s") * Q(10.0, "m3/s"), Q[Volume, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "s") * Q(10.0, "m3/s"), Q[Volume, pl.Series])

    assert_type(Q(pl.col("time"), "s") * Q(10.0, "m3/s"), Q[Volume, pl.Expr])


def test_mul_power_time_all_magnitude_types() -> None:
    assert_type(Q(2.5, "kW") * Q(10.0, "s"), Q[Energy, float])
    assert Q(2.5, "kW") * Q(10.0, "s") == Q(25.0, "kJ")

    assert_type(Q([1.0, 2.0, 3.0], "kW") * Q(10.0, "s"), Q[Energy, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "kW") * Q(10.0, "s"), Q[Energy, pl.Series])

    assert_type(Q(pl.col("power"), "kW") * Q(10.0, "s"), Q[Energy, pl.Expr])

    assert_type(Q(2.5, "kW") * Q([1.0, 2.0, 3.0], "s"), Q[Energy, Numpy1DArray])


def test_mul_time_power_all_magnitude_types() -> None:
    assert_type(Q(10.0, "s") * Q(2.5, "kW"), Q[Energy, float])

    assert_type(Q([1.0, 2.0, 3.0], "s") * Q(10.0, "kW"), Q[Energy, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "s") * Q(10.0, "kW"), Q[Energy, pl.Series])

    assert_type(Q(pl.col("time"), "s") * Q(10.0, "kW"), Q[Energy, pl.Expr])


def test_mul_velocity_time_all_magnitude_types() -> None:
    assert_type(Q(2.5, "m/s") * Q(10.0, "s"), Q[Length, float])
    assert Q(2.5, "m/s") * Q(10.0, "s") == Q(25.0, "m")

    assert_type(Q([1.0, 2.0, 3.0], "m/s") * Q(10.0, "s"), Q[Length, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "m/s") * Q(10.0, "s"), Q[Length, pl.Series])

    assert_type(Q(pl.col("velocity"), "m/s") * Q(10.0, "s"), Q[Length, pl.Expr])

    assert_type(Q(2.5, "m/s") * Q([1.0, 2.0, 3.0], "s"), Q[Length, Numpy1DArray])


def test_mul_time_velocity_all_magnitude_types() -> None:
    assert_type(Q(10.0, "s") * Q(2.5, "m/s"), Q[Length, float])

    assert_type(Q([1.0, 2.0, 3.0], "s") * Q(10.0, "m/s"), Q[Length, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "s") * Q(10.0, "m/s"), Q[Length, pl.Series])

    assert_type(Q(pl.col("time"), "s") * Q(10.0, "m/s"), Q[Length, pl.Expr])


def test_mul_density_volume_all_magnitude_types() -> None:
    assert_type(Q(1000.0, "kg/m3") * Q(10.0, "m3"), Q[Mass, float])
    assert Q(1000.0, "kg/m3") * Q(10.0, "m3") == Q(10000.0, "kg")

    assert_type(Q([1000.0, 2000.0, 3000.0], "kg/m3") * Q(10.0, "m3"), Q[Mass, Numpy1DArray])

    assert_type(Q(pl.Series([1000.0, 2000.0, 3000.0]), "kg/m3") * Q(10.0, "m3"), Q[Mass, pl.Series])

    assert_type(Q(pl.col("density"), "kg/m3") * Q(10.0, "m3"), Q[Mass, pl.Expr])

    assert_type(Q(1000.0, "kg/m3") * Q([1.0, 2.0, 3.0], "m3"), Q[Mass, Numpy1DArray])


def test_mul_volume_density_all_magnitude_types() -> None:
    assert_type(Q(10.0, "m3") * Q(1000.0, "kg/m3"), Q[Mass, float])

    assert_type(Q([1.0, 2.0, 3.0], "m3") * Q(1000.0, "kg/m3"), Q[Mass, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "m3") * Q(1000.0, "kg/m3"), Q[Mass, pl.Series])

    assert_type(Q(pl.col("volume"), "m3") * Q(1000.0, "kg/m3"), Q[Mass, pl.Expr])


def test_mul_length_length_all_magnitude_types() -> None:
    assert_type(Q(5.0, "m") * Q(10.0, "m"), Q[Area, float])
    assert Q(5.0, "m") * Q(10.0, "m") == Q(50.0, "m2")

    assert_type(Q([1.0, 2.0, 3.0], "m") * Q(10.0, "m"), Q[Area, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "m") * Q(10.0, "m"), Q[Area, pl.Series])

    assert_type(Q(pl.col("length"), "m") * Q(10.0, "m"), Q[Area, pl.Expr])

    assert_type(Q(5.0, "m") * Q([1.0, 2.0, 3.0], "m"), Q[Area, Numpy1DArray])


def test_mul_length_area_all_magnitude_types() -> None:
    assert_type(Q(5.0, "m") * Q(10.0, "m2"), Q[Volume, float])
    assert Q(5.0, "m") * Q(10.0, "m2") == Q(50.0, "m3")

    assert_type(Q([1.0, 2.0, 3.0], "m") * Q(10.0, "m2"), Q[Volume, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "m") * Q(10.0, "m2"), Q[Volume, pl.Series])

    assert_type(Q(pl.col("length"), "m") * Q(10.0, "m2"), Q[Volume, pl.Expr])

    assert_type(Q(5.0, "m") * Q([1.0, 2.0, 3.0], "m2"), Q[Volume, Numpy1DArray])


def test_mul_area_length_all_magnitude_types() -> None:
    assert_type(Q(10.0, "m2") * Q(5.0, "m"), Q[Volume, float])

    assert_type(Q([1.0, 2.0, 3.0], "m2") * Q(10.0, "m"), Q[Volume, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "m2") * Q(10.0, "m"), Q[Volume, pl.Series])

    assert_type(Q(pl.col("area"), "m2") * Q(10.0, "m"), Q[Volume, pl.Expr])


def test_mul_dimensionless_all_magnitude_types() -> None:
    assert_type(Q(2.0, "") * Q(10.0, "kg"), Q[Mass, float])
    assert Q(2.0, "") * Q(10.0, "kg") == Q(20.0, "kg")

    assert_type(Q([1.0, 2.0, 3.0], "") * Q(10.0, "kg"), Q[Mass, Numpy1DArray])

    assert_type(Q(10.0, "kg") * Q(2.0, ""), Q[Mass, float])

    assert_type(Q([1.0, 2.0, 3.0], "kg") * Q(2.0, ""), Q[Mass, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "kg") * Q(2.0, ""), Q[Mass, pl.Series])


def test_mul_scalar_all_magnitude_types() -> None:
    assert_type(Q(10.0, "kg") * 2, Q[Mass, float])
    assert Q(10.0, "kg") * 2 == Q(20.0, "kg")

    assert_type(Q([1.0, 2.0, 3.0], "kg") * 2, Q[Mass, Numpy1DArray])

    assert_type(Q(pl.Series([1.0, 2.0, 3.0]), "kg") * 2, Q[Mass, pl.Series])

    assert_type(Q(pl.col("mass"), "kg") * 2, Q[Mass, pl.Expr])


def test_div_mass_time_all_magnitude_types() -> None:
    assert_type(Q(25.0, "kg") / Q(10.0, "s"), Q[MassFlow, float])
    assert Q(25.0, "kg") / Q(10.0, "s") == Q(2.5, "kg/s")

    assert_type(Q([10.0, 20.0, 30.0], "kg") / Q(10.0, "s"), Q[MassFlow, Numpy1DArray])
    assert (Q([10.0, 20.0, 30.0], "kg") / Q(10.0, "s") == Q([1.0, 2.0, 3.0], "kg/s")).all()

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "kg") / Q(10.0, "s"), Q[MassFlow, pl.Series])

    assert_type(Q(pl.col("mass"), "kg") / Q(10.0, "s"), Q[MassFlow, pl.Expr])

    assert_type(Q(25.0, "kg") / Q([1.0, 2.0, 5.0], "s"), Q[MassFlow, Numpy1DArray])


def test_div_volume_time_all_magnitude_types() -> None:
    assert_type(Q(25.0, "m3") / Q(10.0, "s"), Q[VolumeFlow, float])
    assert Q(25.0, "m3") / Q(10.0, "s") == Q(2.5, "m3/s")

    assert_type(Q([10.0, 20.0, 30.0], "m3") / Q(10.0, "s"), Q[VolumeFlow, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "m3") / Q(10.0, "s"), Q[VolumeFlow, pl.Series])

    assert_type(Q(pl.col("volume"), "m3") / Q(10.0, "s"), Q[VolumeFlow, pl.Expr])

    assert_type(Q(25.0, "m3") / Q([1.0, 2.0, 5.0], "s"), Q[VolumeFlow, Numpy1DArray])


def test_div_energy_time_all_magnitude_types() -> None:
    assert_type(Q(25.0, "kJ") / Q(10.0, "s"), Q[Power, float])
    assert Q(25.0, "kJ") / Q(10.0, "s") == Q(2.5, "kW")

    assert_type(Q([10.0, 20.0, 30.0], "kJ") / Q(10.0, "s"), Q[Power, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "kJ") / Q(10.0, "s"), Q[Power, pl.Series])

    assert_type(Q(pl.col("energy"), "kJ") / Q(10.0, "s"), Q[Power, pl.Expr])

    assert_type(Q(25.0, "kJ") / Q([1.0, 2.0, 5.0], "s"), Q[Power, Numpy1DArray])


def test_div_length_time_all_magnitude_types() -> None:
    assert_type(Q(25.0, "m") / Q(10.0, "s"), Q[Velocity, float])
    assert Q(25.0, "m") / Q(10.0, "s") == Q(2.5, "m/s")

    assert_type(Q([10.0, 20.0, 30.0], "m") / Q(10.0, "s"), Q[Velocity, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "m") / Q(10.0, "s"), Q[Velocity, pl.Series])

    assert_type(Q(pl.col("length"), "m") / Q(10.0, "s"), Q[Velocity, pl.Expr])

    assert_type(Q(25.0, "m") / Q([1.0, 2.0, 5.0], "s"), Q[Velocity, Numpy1DArray])


def test_div_energy_mass_all_magnitude_types() -> None:
    assert_type(Q(100.0, "MJ") / Q(10.0, "kg"), Q[EnergyPerMass, float])
    assert Q(100.0, "MJ") / Q(10.0, "kg") == Q(10.0, "MJ/kg")

    assert_type(Q([10.0, 20.0, 30.0], "MJ") / Q(10.0, "kg"), Q[EnergyPerMass, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "MJ") / Q(10.0, "kg"), Q[EnergyPerMass, pl.Series])

    assert_type(Q(pl.col("energy"), "MJ") / Q(10.0, "kg"), Q[EnergyPerMass, pl.Expr])

    assert_type(Q(100.0, "MJ") / Q([1.0, 2.0, 5.0], "kg"), Q[EnergyPerMass, Numpy1DArray])


def test_div_mass_volume_all_magnitude_types() -> None:
    assert_type(Q(10000.0, "kg") / Q(10.0, "m3"), Q[Density, float])
    assert Q(10000.0, "kg") / Q(10.0, "m3") == Q(1000.0, "kg/m3")

    assert_type(Q([1000.0, 2000.0, 3000.0], "kg") / Q(10.0, "m3"), Q[Density, Numpy1DArray])

    assert_type(Q(pl.Series([1000.0, 2000.0, 3000.0]), "kg") / Q(10.0, "m3"), Q[Density, pl.Series])

    assert_type(Q(pl.col("mass"), "kg") / Q(10.0, "m3"), Q[Density, pl.Expr])

    assert_type(Q(10000.0, "kg") / Q([1.0, 2.0, 5.0], "m3"), Q[Density, Numpy1DArray])


def test_div_same_dimension_all_magnitude_types() -> None:
    assert_type(Q(100.0, "kg") / Q(10.0, "kg"), Q[Dimensionless, float])
    assert Q(100.0, "kg") / Q(10.0, "kg") == Q(10.0, "")

    assert_type(Q([10.0, 20.0, 30.0], "kg") / Q(10.0, "kg"), Q[Dimensionless, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "kg") / Q(10.0, "kg"), Q[Dimensionless, pl.Series])

    assert_type(Q(pl.col("mass"), "kg") / Q(10.0, "kg"), Q[Dimensionless, pl.Expr])

    assert Q(2000.0, "g") / Q(1.0, "kg") == Q(2.0, "")


def test_div_by_dimensionless_all_magnitude_types() -> None:
    assert_type(Q(10.0, "kg") / Q(2.0, ""), Q[Mass, float])
    assert Q(10.0, "kg") / Q(2.0, "") == Q(5.0, "kg")

    assert_type(Q([10.0, 20.0, 30.0], "kg") / Q(2.0, ""), Q[Mass, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "kg") / Q(2.0, ""), Q[Mass, pl.Series])

    assert_type(Q(pl.col("mass"), "kg") / Q(2.0, ""), Q[Mass, pl.Expr])


def test_div_scalar_all_magnitude_types() -> None:
    assert_type(Q(10.0, "kg") / 2, Q[Mass, float])
    assert Q(10.0, "kg") / 2 == Q(5.0, "kg")

    assert_type(Q([10.0, 20.0, 30.0], "kg") / 2, Q[Mass, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "kg") / 2, Q[Mass, pl.Series])

    assert_type(Q(pl.col("mass"), "kg") / 2, Q[Mass, pl.Expr])


def test_add_same_dimension_all_magnitude_types() -> None:
    assert_type(Q(10.0, "kg") + Q(5.0, "kg"), Q[Mass, float])
    assert Q(10.0, "kg") + Q(5.0, "kg") == Q(15.0, "kg")

    assert_type(Q([10.0, 20.0, 30.0], "kg") + Q(5.0, "kg"), Q[Mass, Numpy1DArray])
    assert (Q([10.0, 20.0, 30.0], "kg") + Q(5.0, "kg") == Q([15.0, 25.0, 35.0], "kg")).all()

    assert_type(Q(5.0, "kg") + Q([10.0, 20.0, 30.0], "kg"), Q[Mass, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "kg") + Q(5.0, "kg"), Q[Mass, pl.Series])

    assert_type(Q(pl.col("mass"), "kg") + Q(5.0, "kg"), Q[Mass, pl.Expr])


def test_sub_same_dimension_all_magnitude_types() -> None:
    assert_type(Q(10.0, "kg") - Q(5.0, "kg"), Q[Mass, float])
    assert Q(10.0, "kg") - Q(5.0, "kg") == Q(5.0, "kg")

    assert_type(Q([10.0, 20.0, 30.0], "kg") - Q(5.0, "kg"), Q[Mass, Numpy1DArray])
    assert (Q([10.0, 20.0, 30.0], "kg") - Q(5.0, "kg") == Q([5.0, 15.0, 25.0], "kg")).all()

    assert_type(Q(35.0, "kg") - Q([10.0, 20.0, 30.0], "kg"), Q[Mass, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "kg") - Q(5.0, "kg"), Q[Mass, pl.Series])

    assert_type(Q(pl.col("mass"), "kg") - Q(5.0, "kg"), Q[Mass, pl.Expr])


def test_add_dimensionless_scalar_all_magnitude_types() -> None:
    assert_type(Q(10.0, "") + 5, Q[Dimensionless, float])
    assert Q(10.0, "") + 5 == Q(15.0, "")

    assert_type(Q([10.0, 20.0, 30.0], "") + 5, Q[Dimensionless, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "") + 5, Q[Dimensionless, pl.Series])

    assert_type(Q(pl.col("value"), "") + 5, Q[Dimensionless, pl.Expr])


def test_sub_dimensionless_scalar_all_magnitude_types() -> None:
    assert_type(Q(10.0, "") - 5, Q[Dimensionless, float])
    assert Q(10.0, "") - 5 == Q(5.0, "")

    assert_type(Q([10.0, 20.0, 30.0], "") - 5, Q[Dimensionless, Numpy1DArray])

    assert_type(Q(pl.Series([10.0, 20.0, 30.0]), "") - 5, Q[Dimensionless, pl.Series])

    assert_type(Q(pl.col("value"), "") - 5, Q[Dimensionless, pl.Expr])


def test_add_unknown_dimension_all_magnitude_types() -> None:
    q1 = Q(10.0, "kg*m")
    q2 = Q(5.0, "kg*m")
    assert_type(q1, Q[UnknownDimensionality, float])
    assert_type(q2, Q[UnknownDimensionality, float])
    assert_type(q1 + q2, Q[UnknownDimensionality, float])

    q3 = Q([10.0, 20.0, 30.0], "kg*m")
    assert_type(q3, Q[UnknownDimensionality, Numpy1DArray])
    assert_type(q3 + q2, Q[UnknownDimensionality, Numpy1DArray])

    q4 = Q(pl.Series([10.0, 20.0, 30.0]), "kg*m")
    assert_type(q4, Q[UnknownDimensionality, pl.Series])
    assert_type(q4 + q2, Q[UnknownDimensionality, pl.Series])

    q5 = Q(pl.col("value"), "kg*m")
    assert_type(q5, Q[UnknownDimensionality, pl.Expr])
    assert_type(q5 + q2, Q[UnknownDimensionality, pl.Expr])


def test_sub_unknown_dimension_all_magnitude_types() -> None:
    q1 = Q(10.0, "kg*m")
    q2 = Q(5.0, "kg*m")
    assert_type(q1 - q2, Q[UnknownDimensionality, float])

    q3 = Q([10.0, 20.0, 30.0], "kg*m")
    assert_type(q3 - q2, Q[UnknownDimensionality, Numpy1DArray])

    q4 = Q(pl.Series([10.0, 20.0, 30.0]), "kg*m")
    assert_type(q4 - q2, Q[UnknownDimensionality, pl.Series])

    q5 = Q(pl.col("value"), "kg*m")
    assert_type(q5 - q2, Q[UnknownDimensionality, pl.Expr])


def test_mul_unknown_unknown_all_magnitude_types() -> None:
    q1 = Q(10.0, "kg*m")
    q2 = Q(5.0, "kg*m")
    assert_type(q1 * q2, Q[UnknownDimensionality, float])

    q3 = Q([10.0, 20.0, 30.0], "kg*m")
    assert_type(q3 * q2, Q[UnknownDimensionality, Numpy1DArray])

    assert_type(q1 * q3, Q[UnknownDimensionality, Numpy1DArray])

    q4 = Q(pl.Series([10.0, 20.0, 30.0]), "kg*m")
    assert_type(q4 * q2, Q[UnknownDimensionality, pl.Series])

    q5 = Q(pl.col("value"), "kg*m")
    assert_type(q5 * q2, Q[UnknownDimensionality, pl.Expr])


def test_div_unknown_unknown_all_magnitude_types() -> None:
    q1 = Q(10.0, "kg*m")
    q2 = Q(5.0, "kg*s")
    assert_type(q1 / q2, Q[UnknownDimensionality, float])

    q3 = Q([10.0, 20.0, 30.0], "kg*m")
    assert_type(q3 / q2, Q[UnknownDimensionality, Numpy1DArray])

    assert_type(q1 / q3, Q[UnknownDimensionality, Numpy1DArray])

    q4 = Q(pl.Series([10.0, 20.0, 30.0]), "kg*m")
    assert_type(q4 / q2, Q[UnknownDimensionality, pl.Series])

    q5 = Q(pl.col("value"), "kg*m")
    assert_type(q5 / q2, Q[UnknownDimensionality, pl.Expr])


def test_mul_known_unknown_all_magnitude_types() -> None:
    q_unknown = Q(5.0, "kg*m")

    assert_type(Q(10.0, "kg") * q_unknown, Q[UnknownDimensionality, float])
    assert_type(Q([10.0, 20.0], "kg") * q_unknown, Q[UnknownDimensionality, Numpy1DArray])
    assert_type(Q(pl.Series([10.0, 20.0]), "kg") * q_unknown, Q[UnknownDimensionality, pl.Series])
    assert_type(Q(pl.col("mass"), "kg") * q_unknown, Q[UnknownDimensionality, pl.Expr])


def test_div_known_unknown_all_magnitude_types() -> None:
    q_unknown = Q(5.0, "kg*m")

    assert_type(Q(10.0, "kg") / q_unknown, Q[UnknownDimensionality, float])
    assert_type(Q([10.0, 20.0], "kg") / q_unknown, Q[UnknownDimensionality, Numpy1DArray])
    assert_type(Q(pl.Series([10.0, 20.0]), "kg") / q_unknown, Q[UnknownDimensionality, pl.Series])
    assert_type(Q(pl.col("mass"), "kg") / q_unknown, Q[UnknownDimensionality, pl.Expr])
