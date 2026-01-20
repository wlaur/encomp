from typing import assert_type

import numpy as np
import pytest
from typeguard import TypeCheckError, typechecked

from ..units import Quantity as Q
from ..utypes import (
    Density,
    Dimensionless,
    Energy,
    Length,
    Mass,
    MassFlow,
    Numpy1DArray,
    Power,
    Time,
    Velocity,
    Volume,
)


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


@typechecked
def _wrong_dim_dag_mass() -> Q[Mass, float]:
    return Q(100.0, "kg")


@typechecked
def _wrong_dim_dag_time() -> Q[Time, float]:
    return Q(10.0, "s")


@typechecked
def _wrong_dim_dag_power_expects_energy(energy: Q[Energy, float], time: Q[Time, float]) -> Q[Power, float]:
    return energy / time


def test_wrong_dimensionality_runtime_error() -> None:
    mass = _wrong_dim_dag_mass()
    assert_type(mass, Q[Mass, float])

    time = _wrong_dim_dag_time()
    assert_type(time, Q[Time, float])

    with pytest.raises(TypeCheckError, match=r"is not an instance of.*Energy"):
        _wrong_dim_dag_power_expects_energy(mass, time)  # pyright: ignore[reportArgumentType]


@typechecked
def _wrong_mt_dag_masses() -> Q[Mass, Numpy1DArray]:
    return Q([100.0, 200.0, 300.0], "kg")


@typechecked
def _wrong_mt_dag_time() -> Q[Time, float]:
    return Q(10.0, "s")


@typechecked
def _wrong_mt_dag_flow_expects_float(mass: Q[Mass, float], time: Q[Time, float]) -> Q[MassFlow, float]:
    return mass / time


def test_wrong_magnitude_type_runtime_error() -> None:
    masses = _wrong_mt_dag_masses()
    assert_type(masses, Q[Mass, Numpy1DArray])

    time = _wrong_mt_dag_time()
    assert_type(time, Q[Time, float])

    with pytest.raises(TypeCheckError, match=r"is not an instance of.*float"):
        _wrong_mt_dag_flow_expects_float(masses, time)  # pyright: ignore[reportArgumentType]


@typechecked
def _incompatible_add_dag_mass() -> Q[Mass, float]:
    return Q(100.0, "kg")


@typechecked
def _incompatible_add_dag_time() -> Q[Time, float]:
    return Q(10.0, "s")


@typechecked
def _incompatible_add_dag_add(mass: Q[Mass, float], time: Q[Time, float]) -> Q[Mass, float]:
    result = mass + time  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    return result  # pyright: ignore[reportUnknownVariableType]


def test_incompatible_addition_runtime_error() -> None:
    mass = _incompatible_add_dag_mass()
    assert_type(mass, Q[Mass, float])

    time = _incompatible_add_dag_time()
    assert_type(time, Q[Time, float])

    with pytest.raises((TypeCheckError, ValueError, TypeError)):
        _incompatible_add_dag_add(mass, time)


@typechecked
def _return_wrong_type_dag_mass() -> Q[Mass, float]:
    return Q(100.0, "kg")


@typechecked
def _return_wrong_type_dag_time() -> Q[Time, float]:
    return Q(10.0, "s")


@typechecked
def _return_wrong_type_dag_should_return_power(mass: Q[Mass, float], time: Q[Time, float]) -> Q[Power, float]:
    flow = mass / time
    return flow  # pyright: ignore[reportReturnType]


def test_wrong_return_type_runtime_error() -> None:
    mass = _return_wrong_type_dag_mass()
    time = _return_wrong_type_dag_time()

    with pytest.raises(TypeCheckError, match=r"is not an instance of.*Power"):
        _return_wrong_type_dag_should_return_power(mass, time)


def _undecorated_wrong_arg_dag_energy() -> Q[Energy, float]:
    return Q(1500.0, "MJ")


def _undecorated_wrong_arg_dag_time() -> Q[Time, float]:
    return Q(3600.0, "s")


def _undecorated_wrong_arg_dag_expects_length(length: Q[Length, float], time: Q[Time, float]) -> Q[Velocity, float]:
    return length / time


def test_undecorated_wrong_arg_with_typechecked() -> None:
    energy = _undecorated_wrong_arg_dag_energy()
    time = _undecorated_wrong_arg_dag_time()

    with pytest.raises(TypeCheckError, match=r"is not an instance of.*Length"):
        typechecked(_undecorated_wrong_arg_dag_expects_length)(energy, time)  # pyright: ignore[reportArgumentType]


@typechecked
def _mixed_errors_dag_mass() -> Q[Mass, float]:
    return Q(100.0, "kg")


@typechecked
def _mixed_errors_dag_volume_array() -> Q[Volume, Numpy1DArray]:
    return Q([10.0, 20.0, 30.0], "m3")


@typechecked
def _mixed_errors_dag_density_expects_both_float(mass: Q[Mass, float], volume: Q[Volume, float]) -> Q[Density, float]:
    return mass / volume


def test_mixed_errors_dag() -> None:
    mass = _mixed_errors_dag_mass()
    assert_type(mass, Q[Mass, float])

    volume_array = _mixed_errors_dag_volume_array()
    assert_type(volume_array, Q[Volume, Numpy1DArray])

    with pytest.raises(TypeCheckError, match=r"is not an instance of.*float"):
        _mixed_errors_dag_density_expects_both_float(mass, volume_array)  # pyright: ignore[reportArgumentType]


@typechecked
def _mostly_correct_dag_masses() -> Q[Mass, Numpy1DArray]:
    return Q([100.0, 200.0, 300.0], "kg")


@typechecked
def _mostly_correct_dag_time() -> Q[Time, float]:
    return Q(10.0, "s")


@typechecked
def _mostly_correct_dag_flows(mass: Q[Mass, Numpy1DArray], time: Q[Time, float]) -> Q[MassFlow, Numpy1DArray]:
    return mass / time


@typechecked
def _mostly_correct_dag_duration_wrong() -> Q[Length, float]:
    return Q(5.0, "m")


@typechecked
def _mostly_correct_dag_mass_back(flow: Q[MassFlow, Numpy1DArray], duration: Q[Time, float]) -> Q[Mass, Numpy1DArray]:
    return flow * duration


def test_mostly_correct_dag_with_one_error() -> None:
    masses = _mostly_correct_dag_masses()
    assert_type(masses, Q[Mass, Numpy1DArray])

    time = _mostly_correct_dag_time()
    assert_type(time, Q[Time, float])

    flows = _mostly_correct_dag_flows(masses, time)
    assert_type(flows, Q[MassFlow, Numpy1DArray])

    duration_wrong = _mostly_correct_dag_duration_wrong()

    with pytest.raises(TypeCheckError, match=r"is not an instance of.*Time"):
        _mostly_correct_dag_mass_back(flows, duration_wrong)  # pyright: ignore[reportArgumentType]

    duration_correct = Q(5.0, "s")
    mass_back = _mostly_correct_dag_mass_back(flows, duration_correct)
    assert_type(mass_back, Q[Mass, Numpy1DArray])

    assert np.allclose(flows.to("kg/s").m, [10.0, 20.0, 30.0])
    assert np.allclose(mass_back.to("kg").m, [50.0, 100.0, 150.0])


@typechecked
def _scalar_array_mismatch_dag_base() -> Q[Mass, float]:
    return Q(100.0, "kg")


@typechecked
def _scalar_array_mismatch_dag_array() -> Q[Mass, Numpy1DArray]:
    return Q([1.0, 2.0, 3.0], "kg")


@typechecked
def _scalar_array_mismatch_dag_expects_both_arrays(
    m1: Q[Mass, Numpy1DArray], m2: Q[Mass, Numpy1DArray]
) -> Q[Mass, Numpy1DArray]:
    return m1 + m2


def test_scalar_array_mismatch() -> None:
    base = _scalar_array_mismatch_dag_base()
    assert_type(base, Q[Mass, float])

    array = _scalar_array_mismatch_dag_array()
    assert_type(array, Q[Mass, Numpy1DArray])

    with pytest.raises(TypeCheckError, match=r"is not an instance of.*ndarray"):
        _scalar_array_mismatch_dag_expects_both_arrays(base, array)  # pyright: ignore[reportArgumentType]


def _dimensionless_error_dag_mass1() -> Q[Mass, float]:
    return Q(500.0, "kg")


def _dimensionless_error_dag_mass2() -> Q[Mass, float]:
    return Q(250.0, "kg")


def _dimensionless_error_dag_ratio_expects_energy(
    e1: Q[Energy, float], e2: Q[Energy, float]
) -> Q[Dimensionless, float]:
    return e1 / e2


def test_dimensionless_wrong_inputs() -> None:
    m1 = _dimensionless_error_dag_mass1()
    m2 = _dimensionless_error_dag_mass2()

    with pytest.raises(TypeCheckError, match=r"is not an instance of.*Energy"):
        typechecked(_dimensionless_error_dag_ratio_expects_energy)(m1, m2)  # pyright: ignore[reportArgumentType]
