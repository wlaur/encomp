from typing import assert_type

import numpy as np
import polars as pl
from typeguard import typechecked

from ..units import Quantity as Q
from ..utypes import (
    DT,
    Area,
    Density,
    Dimensionless,
    Energy,
    Length,
    Mass,
    MassFlow,
    Numpy1DArray,
    Power,
    Time,
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


@typechecked
def _energy_dag_energy() -> Q[Energy, float]:
    return Q(1500.0, "MJ")


@typechecked
def _energy_dag_time() -> Q[Time, float]:
    return Q(3600.0, "s")


@typechecked
def _energy_dag_power(energy: Q[Energy, float], time: Q[Time, float]) -> Q[Power, float]:
    return energy / time


@typechecked
def _energy_dag_duration() -> Q[Time, float]:
    return Q(7200.0, "s")


@typechecked
def _energy_dag_energy_back(power: Q[Power, float], duration: Q[Time, float]) -> Q[Energy, float]:
    return power * duration


def test_energy_dag() -> None:
    energy = _energy_dag_energy()
    assert_type(energy, Q[Energy, float])

    time = _energy_dag_time()
    assert_type(time, Q[Time, float])

    power = _energy_dag_power(energy, time)
    assert_type(power, Q[Power, float])

    duration = _energy_dag_duration()
    assert_type(duration, Q[Time, float])

    energy_back = _energy_dag_energy_back(power, duration)
    assert_type(energy_back, Q[Energy, float])

    assert power.to("kW").m == 416.6666666666667
    assert energy_back.to("MJ").m == 3000.0


@typechecked
def _flow_dag_mass() -> Q[Mass, Numpy1DArray]:
    return Q([100.0, 200.0, 300.0], "kg")


@typechecked
def _flow_dag_time() -> Q[Time, float]:
    return Q(60.0, "s")


@typechecked
def _flow_dag_massflow(mass: Q[Mass, Numpy1DArray], time: Q[Time, float]) -> Q[MassFlow, Numpy1DArray]:
    return mass / time


@typechecked
def _flow_dag_duration() -> Q[Time, Numpy1DArray]:
    return Q([10.0, 20.0, 30.0], "s")


@typechecked
def _flow_dag_mass_back(flow: Q[MassFlow, Numpy1DArray], duration: Q[Time, Numpy1DArray]) -> Q[Mass, Numpy1DArray]:
    return flow * duration


def test_flow_dag() -> None:
    mass = _flow_dag_mass()
    assert_type(mass, Q[Mass, Numpy1DArray])

    time = _flow_dag_time()
    assert_type(time, Q[Time, float])

    flow = _flow_dag_massflow(mass, time)
    assert_type(flow, Q[MassFlow, Numpy1DArray])

    duration = _flow_dag_duration()
    assert_type(duration, Q[Time, Numpy1DArray])

    mass_back = _flow_dag_mass_back(flow, duration)
    assert_type(mass_back, Q[Mass, Numpy1DArray])

    assert np.allclose(mass_back.to("kg").m, [16.666667, 66.666667, 150.0])


@typechecked
def _density_dag_mass() -> Q[Mass, pl.Series]:
    return Q(pl.Series([1000.0, 2000.0, 3000.0]), "kg")


@typechecked
def _density_dag_volume() -> Q[Volume, float]:
    return Q(10.0, "m3")


@typechecked
def _density_dag_density(mass: Q[Mass, pl.Series], volume: Q[Volume, float]) -> Q[Density, pl.Series]:
    return mass / volume


@typechecked
def _density_dag_volumes() -> Q[Volume, pl.Series]:
    return Q(pl.Series([1.0, 2.0, 3.0]), "m3")


@typechecked
def _density_dag_masses(density: Q[Density, pl.Series], volumes: Q[Volume, pl.Series]) -> Q[Mass, pl.Series]:
    return density * volumes


def test_density_dag() -> None:
    mass = _density_dag_mass()
    assert_type(mass, Q[Mass, pl.Series])

    volume = _density_dag_volume()
    assert_type(volume, Q[Volume, float])

    density = _density_dag_density(mass, volume)
    assert_type(density, Q[Density, pl.Series])

    volumes = _density_dag_volumes()
    assert_type(volumes, Q[Volume, pl.Series])

    masses = _density_dag_masses(density, volumes)
    assert_type(masses, Q[Mass, pl.Series])

    assert masses.to("kg").m.to_list() == [100.0, 400.0, 900.0]


@typechecked
def _geometry_dag_length() -> Q[Length, float]:
    return Q(5.0, "m")


@typechecked
def _geometry_dag_width() -> Q[Length, float]:
    return Q(3.0, "m")


@typechecked
def _geometry_dag_area(length: Q[Length, float], width: Q[Length, float]) -> Q[Area, float]:
    return length * width


@typechecked
def _geometry_dag_height() -> Q[Length, float]:
    return Q(2.0, "m")


@typechecked
def _geometry_dag_volume(area: Q[Area, float], height: Q[Length, float]) -> Q[Volume, float]:
    return area * height


@typechecked
def _geometry_dag_density() -> Q[Density, float]:
    return Q(1000.0, "kg/m3")


@typechecked
def _geometry_dag_mass(volume: Q[Volume, float], density: Q[Density, float]) -> Q[Mass, float]:
    return volume * density


def test_geometry_dag() -> None:
    length = _geometry_dag_length()
    assert_type(length, Q[Length, float])

    width = _geometry_dag_width()
    assert_type(width, Q[Length, float])

    area = _geometry_dag_area(length, width)
    assert_type(area, Q[Area, float])

    height = _geometry_dag_height()
    assert_type(height, Q[Length, float])

    volume = _geometry_dag_volume(area, height)
    assert_type(volume, Q[Volume, float])

    density = _geometry_dag_density()
    assert_type(density, Q[Density, float])

    mass = _geometry_dag_mass(volume, density)
    assert_type(mass, Q[Mass, float])

    assert mass.to("kg").m == 30000.0


@typechecked
def _velocity_dag_distance() -> Q[Length, Numpy1DArray]:
    return Q([100.0, 200.0, 300.0], "m")


@typechecked
def _velocity_dag_time() -> Q[Time, Numpy1DArray]:
    return Q([10.0, 20.0, 30.0], "s")


@typechecked
def _velocity_dag_velocity(distance: Q[Length, Numpy1DArray], time: Q[Time, Numpy1DArray]) -> Q[Velocity, Numpy1DArray]:
    return distance / time


@typechecked
def _velocity_dag_duration() -> Q[Time, float]:
    return Q(5.0, "s")


@typechecked
def _velocity_dag_distance_traveled(
    velocity: Q[Velocity, Numpy1DArray], duration: Q[Time, float]
) -> Q[Length, Numpy1DArray]:
    return velocity * duration


def test_velocity_dag() -> None:
    distance = _velocity_dag_distance()
    assert_type(distance, Q[Length, Numpy1DArray])

    time = _velocity_dag_time()
    assert_type(time, Q[Time, Numpy1DArray])

    velocity = _velocity_dag_velocity(distance, time)
    assert_type(velocity, Q[Velocity, Numpy1DArray])

    duration = _velocity_dag_duration()
    assert_type(duration, Q[Time, float])

    distance_traveled = _velocity_dag_distance_traveled(velocity, duration)
    assert_type(distance_traveled, Q[Length, Numpy1DArray])

    assert np.allclose(distance_traveled.to("m").m, [50.0, 50.0, 50.0])


@typechecked
def _volumeflow_dag_volume() -> Q[Volume, float]:
    return Q(1000.0, "L")


@typechecked
def _volumeflow_dag_time() -> Q[Time, float]:
    return Q(3600.0, "s")


@typechecked
def _volumeflow_dag_flow(volume: Q[Volume, float], time: Q[Time, float]) -> Q[VolumeFlow, float]:
    return volume / time


@typechecked
def _volumeflow_dag_duration() -> Q[Time, float]:
    return Q(7200.0, "s")


@typechecked
def _volumeflow_dag_volume_back(flow: Q[VolumeFlow, float], duration: Q[Time, float]) -> Q[Volume, float]:
    return flow * duration


def test_volumeflow_dag() -> None:
    volume = _volumeflow_dag_volume()
    assert_type(volume, Q[Volume, float])

    time = _volumeflow_dag_time()
    assert_type(time, Q[Time, float])

    flow = _volumeflow_dag_flow(volume, time)
    assert_type(flow, Q[VolumeFlow, float])

    duration = _volumeflow_dag_duration()
    assert_type(duration, Q[Time, float])

    volume_back = _volumeflow_dag_volume_back(flow, duration)
    assert_type(volume_back, Q[Volume, float])

    assert flow.to("L/s").m == 0.2777777777777778
    assert volume_back.to("L").m == 2000.0


@typechecked
def _dimensionless_dag_mass1() -> Q[Mass, float]:
    return Q(500.0, "kg")


@typechecked
def _dimensionless_dag_mass2() -> Q[Mass, float]:
    return Q(250.0, "kg")


@typechecked
def _dimensionless_dag_ratio(m1: Q[Mass, float], m2: Q[Mass, float]) -> Q[Dimensionless, float]:
    return m1 / m2


@typechecked
def _dimensionless_dag_base_value() -> Q[Energy, float]:
    return Q(1000.0, "MJ")


@typechecked
def _dimensionless_dag_scaled_value(base: Q[Energy, float], ratio: Q[Dimensionless, float]) -> Q[Energy, float]:
    return base * ratio


def test_dimensionless_dag() -> None:
    mass1 = _dimensionless_dag_mass1()
    assert_type(mass1, Q[Mass, float])

    mass2 = _dimensionless_dag_mass2()
    assert_type(mass2, Q[Mass, float])

    ratio = _dimensionless_dag_ratio(mass1, mass2)
    assert_type(ratio, Q[Dimensionless, float])

    base = _dimensionless_dag_base_value()
    assert_type(base, Q[Energy, float])

    scaled = _dimensionless_dag_scaled_value(base, ratio)
    assert_type(scaled, Q[Energy, float])

    assert scaled.to("MJ").m == 2000.0


@typechecked
def _mixed_types_dag_base_mass() -> Q[Mass, float]:
    return Q(100.0, "kg")


@typechecked
def _mixed_types_dag_mass_array() -> Q[Mass, Numpy1DArray]:
    return Q([1.0, 2.0, 3.0], "kg")


@typechecked
def _mixed_types_dag_masses_broadcasted(base: Q[Mass, float], arr: Q[Mass, Numpy1DArray]) -> Q[Mass, Numpy1DArray]:
    return base + arr


@typechecked
def _mixed_types_dag_time() -> Q[Time, float]:
    return Q(10.0, "s")


@typechecked
def _mixed_types_dag_massflows(masses: Q[Mass, Numpy1DArray], time: Q[Time, float]) -> Q[MassFlow, Numpy1DArray]:
    return masses / time


def test_mixed_types_dag() -> None:
    base = _mixed_types_dag_base_mass()
    assert_type(base, Q[Mass, float])

    arr = _mixed_types_dag_mass_array()
    assert_type(arr, Q[Mass, Numpy1DArray])

    masses = _mixed_types_dag_masses_broadcasted(base, arr)
    assert_type(masses, Q[Mass, Numpy1DArray])

    time = _mixed_types_dag_time()
    assert_type(time, Q[Time, float])

    flows = _mixed_types_dag_massflows(masses, time)
    assert_type(flows, Q[MassFlow, Numpy1DArray])

    assert np.allclose(flows.to("kg/s").m, [10.1, 10.2, 10.3])


def _power_undecorated_dag_energy() -> Q[Energy, Numpy1DArray]:
    return Q([1000.0, 2000.0, 3000.0], "MJ")


def _power_undecorated_dag_time() -> Q[Time, Numpy1DArray]:
    return Q([3600.0, 7200.0, 10800.0], "s")


def _power_undecorated_dag_power(
    energy: Q[Energy, Numpy1DArray], time: Q[Time, Numpy1DArray]
) -> Q[Power, Numpy1DArray]:
    return energy / time


def _power_undecorated_dag_duration() -> Q[Time, float]:
    return Q(1800.0, "s")


def _power_undecorated_dag_energy_consumed(
    power: Q[Power, Numpy1DArray], duration: Q[Time, float]
) -> Q[Energy, Numpy1DArray]:
    return power * duration


def test_power_undecorated_dag() -> None:
    energy = typechecked(_power_undecorated_dag_energy)()
    assert_type(energy, Q[Energy, Numpy1DArray])

    time = typechecked(_power_undecorated_dag_time)()
    assert_type(time, Q[Time, Numpy1DArray])

    power = typechecked(_power_undecorated_dag_power)(energy, time)
    assert_type(power, Q[Power, Numpy1DArray])

    duration = typechecked(_power_undecorated_dag_duration)()
    assert_type(duration, Q[Time, float])

    energy_consumed = typechecked(_power_undecorated_dag_energy_consumed)(power, duration)
    assert_type(energy_consumed, Q[Energy, Numpy1DArray])

    assert np.allclose(power.to("kW").m, [277.77777778, 277.77777778, 277.77777778])
    assert np.allclose(energy_consumed.to("MJ").m, [500.0, 500.0, 500.0])


def _nested_undecorated_dag_length1() -> Q[Length, float]:
    return Q(10.0, "m")


def _nested_undecorated_dag_length2() -> Q[Length, float]:
    return Q(5.0, "m")


def _nested_undecorated_dag_area(l1: Q[Length, float], l2: Q[Length, float]) -> Q[Area, float]:
    return l1 * l2


def _nested_undecorated_dag_height() -> Q[Length, float]:
    return Q(3.0, "m")


def _nested_undecorated_dag_volume(area: Q[Area, float], height: Q[Length, float]) -> Q[Volume, float]:
    return area * height


def test_nested_undecorated_dag() -> None:
    area = typechecked(_nested_undecorated_dag_area)(
        typechecked(_nested_undecorated_dag_length1)(), typechecked(_nested_undecorated_dag_length2)()
    )
    assert_type(area, Q[Area, float])

    volume = typechecked(_nested_undecorated_dag_volume)(area, typechecked(_nested_undecorated_dag_height)())
    assert_type(volume, Q[Volume, float])

    assert area.to("m2").m == 50.0
    assert volume.to("m3").m == 150.0


def _chained_undecorated_dag_initial_mass() -> Q[Mass, pl.Series]:
    return Q(pl.Series([100.0, 200.0, 300.0]), "kg")


def _chained_undecorated_dag_time1() -> Q[Time, float]:
    return Q(10.0, "s")


def _chained_undecorated_dag_massflow(mass: Q[Mass, pl.Series], time: Q[Time, float]) -> Q[MassFlow, pl.Series]:
    return mass / time


def _chained_undecorated_dag_time2() -> Q[Time, pl.Series]:
    return Q(pl.Series([5.0, 10.0, 15.0]), "s")


def _chained_undecorated_dag_mass_back(flow: Q[MassFlow, pl.Series], time: Q[Time, pl.Series]) -> Q[Mass, pl.Series]:
    return flow * time


def test_chained_undecorated_dag() -> None:
    flow = typechecked(_chained_undecorated_dag_massflow)(
        typechecked(_chained_undecorated_dag_initial_mass)(), typechecked(_chained_undecorated_dag_time1)()
    )
    assert_type(flow, Q[MassFlow, pl.Series])

    mass = typechecked(_chained_undecorated_dag_mass_back)(flow, typechecked(_chained_undecorated_dag_time2)())
    assert_type(mass, Q[Mass, pl.Series])

    assert flow.to("kg/s").m.to_list() == [10.0, 20.0, 30.0]
    assert mass.to("kg").m.to_list() == [50.0, 200.0, 450.0]


@typechecked
def _default_mt_dag_masses() -> Q[Mass]:
    return Q([100.0, 200.0, 300.0], "kg")


@typechecked
def _default_mt_dag_time_scalar() -> Q[Time, float]:
    return Q(10.0, "s")


@typechecked
def _default_mt_dag_flows(mass: Q[Mass], time: Q[Time, float]) -> Q[MassFlow]:
    return mass / time


@typechecked
def _default_mt_dag_additional_mass() -> Q[Mass, float]:
    return Q(50.0, "kg")


@typechecked
def _default_mt_dag_total_masses(mass_array: Q[Mass], mass_scalar: Q[Mass, float]) -> Q[Mass]:
    return mass_array + mass_scalar


@typechecked
def _default_mt_dag_volumes() -> Q[Volume]:
    return Q([10.0, 20.0, 30.0], "m3")


@typechecked
def _default_mt_dag_densities(masses: Q[Mass], volumes: Q[Volume]) -> Q[Density]:
    return masses / volumes


@typechecked
def _default_mt_dag_single_volume() -> Q[Volume, float]:
    return Q(5.0, "m3")


@typechecked
def _default_mt_dag_masses_from_density(densities: Q[Density], volume: Q[Volume, float]) -> Q[Mass]:
    return densities * volume


def test_default_mt_dag() -> None:
    masses = _default_mt_dag_masses()
    assert_type(masses, Q[Mass, Numpy1DArray])

    time_scalar = _default_mt_dag_time_scalar()
    assert_type(time_scalar, Q[Time, float])

    flows = _default_mt_dag_flows(masses, time_scalar)
    assert_type(flows, Q[MassFlow, Numpy1DArray])

    additional_mass = _default_mt_dag_additional_mass()
    assert_type(additional_mass, Q[Mass, float])

    total_masses = _default_mt_dag_total_masses(masses, additional_mass)
    assert_type(total_masses, Q[Mass, Numpy1DArray])

    volumes = _default_mt_dag_volumes()
    assert_type(volumes, Q[Volume, Numpy1DArray])

    densities = _default_mt_dag_densities(total_masses, volumes)
    assert_type(densities, Q[Density, Numpy1DArray])

    single_volume = _default_mt_dag_single_volume()
    assert_type(single_volume, Q[Volume, float])

    masses_from_density = _default_mt_dag_masses_from_density(densities, single_volume)
    assert_type(masses_from_density, Q[Mass, Numpy1DArray])

    assert np.allclose(flows.to("kg/s").m, [10.0, 20.0, 30.0])
    assert np.allclose(total_masses.to("kg").m, [150.0, 250.0, 350.0])
    assert np.allclose(densities.to("kg/m3").m, [15.0, 12.5, 11.666667])
    assert np.allclose(masses_from_density.to("kg").m, [75.0, 62.5, 58.333333])


@typechecked
def _times_2_unknown(q: Q[UnknownDimensionality, float]) -> Q[UnknownDimensionality, float]:
    return q * 2


def test_unknown() -> None:
    assert_type(_times_2_unknown(Q(25, "kg").unknown()), Q[UnknownDimensionality, float])


@typechecked
def _times_2_unset(q: Q) -> Q:
    return q * 2


def test_unset() -> None:
    v = Q(25, "kg").astype(np.ndarray).unknown()
    assert_type(_times_2_unset(v), Q[UnknownDimensionality, np.ndarray])


@typechecked
def _asdim(q: Q[Mass]) -> Q[Power]:
    r = q * Q(2, "kW/kg")
    return r.asdim(Power)


def test_asdim() -> None:
    assert_type(_asdim(Q([25], "kg")), Q[Power, np.ndarray])


@typechecked
def _dynamic(*q: Q[DT]) -> Q[DT]:
    assert len(q)
    return sum(q[1:], start=q[0])


def test_dynamic() -> None:
    v = _dynamic(Q([25], "kg"), Q([25], "kg"))
    assert_type(v, Q[Mass, np.ndarray])


@typechecked
def _dynamic_scalar(*q: Q[DT, float]) -> Q[DT, float]:
    assert len(q)
    return sum(q[1:], start=q[0])


def test_dynamic_scalar() -> None:
    v = _dynamic_scalar(Q(25, "kg"), Q(25, "g"))
    assert_type(v, Q[Mass, float])
