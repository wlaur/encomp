# pyright: reportConstantRedefinition=false

import copy
import inspect
import pickle
import subprocess
import sys
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, TypedDict, assert_never, assert_type, cast

import numpy as np
import polars as pl
import pytest
from pint.errors import OffsetUnitCalculusError, UnitStrippedWarning
from pydantic import BaseModel, ConfigDict, ValidationError
from typeguard import typechecked

from ..conversion import convert_volume_mass
from ..misc import isinstance_types
from ..units import (
    CUSTOM_DIMENSIONS,
    UNIT_REGISTRY,
    DimensionalityComparisonError,
    DimensionalityError,
    DimensionalityRedefinitionError,
    DimensionalityTypeError,
    ExpectedDimensionalityError,
    Quantity,
    Unit,
    define_dimensionality,
    set_quantity_format,
)
from ..units import Quantity as Q
from ..utypes import (
    DT,
    MT,
    Area,
    Dimensionality,
    Dimensionless,
    EnergyPerMass,
    HeatingValue,
    Length,
    LowerHeatingValue,
    Mass,
    MassFlow,
    Normal,
    NormalVolume,
    NormalVolumeFlow,
    Numpy1DArray,
    Power,
    Pressure,
    SpecificEnthalpy,
    Temperature,
    TemperatureDifference,
    UnitsContainer,
    UnknownDimensionality,
    Velocity,
    Volume,
    VolumeFlow,
    get_registered_units,
)

# pytest.approx is loosely typed; expose it as an Any-typed alias
approx = cast(Any, pytest).approx


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_core_quantity_api_has_docstrings() -> None:
    documented_api = (
        Quantity,
        Quantity.to,
        Quantity.ito,
        Quantity.check,
        Quantity.asdim,
        Quantity.astype,
        set_quantity_format,
    )

    for obj in documented_api:
        assert inspect.getdoc(obj)


def test_registry() -> None:
    import pint
    from pint import application_registry

    _DEFAULT_REGISTRY = cast(Any, pint)._DEFAULT_REGISTRY

    us: list[Any] = [UNIT_REGISTRY, _DEFAULT_REGISTRY, application_registry.get()]

    # check that all these objects are the same
    assert len(set(map(id, us))) == 1

    # check that units from all objects can be combined
    # NOTE: there is not typing for quantities created by this method
    q = 1 * cast(Any, UNIT_REGISTRY).kg / _DEFAULT_REGISTRY.s**2 / cast(Any, application_registry.get()).m
    assert isinstance_types(q, Q[Pressure, Any])

    # options cannot be overridden once set
    UNIT_REGISTRY.force_ndarray_like = True
    assert not UNIT_REGISTRY.force_ndarray_like


def test_define_dimensionality() -> None:
    assert "normal" in CUSTOM_DIMENSIONS

    with pytest.raises(DimensionalityRedefinitionError):
        define_dimensionality(CUSTOM_DIMENSIONS[0])

    # if_exists="warn" logs and keeps the existing definition instead of raising
    define_dimensionality(CUSTOM_DIMENSIONS[0], if_exists="warn")

    with pytest.raises(ValueError, match="if_exists"):
        define_dimensionality(CUSTOM_DIMENSIONS[0], if_exists=cast(Any, "invalid"))


def test_format() -> None:
    assert "{:.2f~P}".format(Q(25, "delta_degC") / Q(255, "m3")) == "0.10 Δ°C/m³"
    assert f"{Q(25, 'meter'):D}" == "25.0 meter"
    assert f"{Q(25, 'meter'):.2fD}" == "25.00 meter"
    assert f"{Q(25, 'meter'):.2f}" == "25.00 m"


@contextmanager
def _reset_dimensionality_registry() -> Generator[None]:
    # NOTE: this is a hack, only use this for tests

    import importlib

    from .. import utypes

    # this does not completely reload the module,
    # since there are multiple references to encomp.utypes
    utypes_reloaded = importlib.reload(utypes)

    # this is a new class definition...
    assert Dimensionality is not utypes_reloaded.Dimensionality

    # explicitly replace the registry dicts on the version of
    # Dimensionality that was loaded on module-level in this test module

    _registry_orig = cast(Any, Dimensionality)._registry
    _registry_reversed_orig = cast(Any, Dimensionality)._registry_reversed

    cast(Any, Dimensionality)._registry = utypes_reloaded.Dimensionality._registry
    cast(Any, Dimensionality)._registry_reversed = utypes_reloaded.Dimensionality._registry_reversed

    try:
        yield

    finally:
        # reset to original registries
        # otherwise any code executed after this context manager
        # will have issues with isinstance() and issubclass()
        cast(Any, Dimensionality)._registry = _registry_orig
        cast(Any, Dimensionality)._registry_reversed = _registry_reversed_orig

        # clear existing mapping from dimensionality subclass name to Quantity subclass
        # this will be dynamically rebuilt
        cast(Any, Q)._subclasses.clear()


def test_dimensionality_subtype_protocol() -> None:
    with _reset_dimensionality_registry():
        # the subclass is not checked, only the "dimensions" attribute
        # however, doing this causes a lot of type errors, so make sure to inherit from Dimensionality instead
        class Test:
            dimensions = Dimensionless.dimensions

        cast(Any, Q)[Test]
        cast(Any, Q)[Test](1)


def test_asdim() -> None:
    q = Q(2)
    _q = q.asdim(Dimensionless)
    # if the dimension already matches, return the same object
    assert q is _q

    _q_unknown = q.asdim(UnknownDimensionality)
    _q_unknown_alt = q.unknown()

    assert q is _q_unknown
    assert q is _q_unknown_alt

    with pytest.raises(TypeError):
        Q(2).asdim(Dimensionality)

    Q(1).asdim(Dimensionless)

    # remains the actual dimensionality at runtime, this is to "trick" the type checker
    assert_type(Q(2).asdim(UnknownDimensionality), Q[UnknownDimensionality, float])

    with _reset_dimensionality_registry():
        # default dimensionality for kJ/kg is EnergyPerMass
        q1 = Q(15, "kJ/kg")
        q2 = Q(15, "kJ/kg").asdim(LowerHeatingValue)

        assert type(q1) is not type(q2)
        assert not q1.is_compatible_with(q2)

        assert type(q1) is type(q2.asdim(EnergyPerMass))
        assert type(q2) is type(q1.asdim(LowerHeatingValue))

        assert type(q1) is type(q2.asdim(q1))
        assert type(q2) is type(q1.asdim(q2))

        assert q1 == q2.asdim(EnergyPerMass)
        assert q2 == q1.asdim(LowerHeatingValue)

        assert q1 == q2.asdim(q1)
        assert q2 == q1.asdim(q2)

        with pytest.raises(ExpectedDimensionalityError):
            q1.asdim(Temperature)

        with pytest.raises(ExpectedDimensionalityError):
            q1.asdim(cast(Any, Q(25, "kg")))


def test_custom_dimensionality() -> None:
    with _reset_dimensionality_registry():

        class Custom1(Dimensionality):
            dimensions = Temperature.dimensions**2 / Length.dimensions

        _custom = Custom1

        q1 = cast(Any, Q[Custom1])(1, "degC**2/m")

        class Custom2(Dimensionality):
            dimensions = Temperature.dimensions**2 / Length.dimensions

        # the classes are not identical
        assert Custom2 is not _custom
        assert Q[Custom2] is not Q[_custom]

        q2 = cast(Any, Q[Custom2])(1, "degC**2/m")

        # the values and units are equivalent, but the dimensionality
        # types don't match: not equal (== answers False, it does not raise)
        assert q1 != q2
        assert (q1 == q2) is False

        assert isinstance(q1.to("degC**2/km"), type(q1))
        assert isinstance(q1.to_base_units(), type(q1))
        assert isinstance(q1.to_reduced_units(), type(q1))

        with pytest.raises(TypeError):
            type(
                "Custom2",
                (Dimensionality,),
                # cannot create a duplicate (based on classname) dimensionality
                # with different dimensions
                {"dimensions": Temperature.dimensions**3 / Length.dimensions},
            )


def test_function_annotations() -> None:
    # this results in Quantity[Any]
    a = cast(Any, Q)[DT]

    def return_input(q: Q[DT, MT]) -> Q[DT, MT]:
        return q

    a = return_input(cast(Any, Q(25, "m")))

    assert isinstance_types(a, Q[Length])

    # not possible to determine output dimensionality for this
    def divide_by_time(q: Q[DT, MT]) -> Q[Any, MT]:
        return q / Q(1, "h")

    # this will be resolved to MassFlow at runtime
    assert isinstance_types(divide_by_time(cast(Any, Q(25, "kg"))), Q[MassFlow])


def test_dimensionality_type_hierarchy() -> None:
    with _reset_dimensionality_registry():
        # NOTE: this method of differentiating between different
        # types of the same dimensionality is not very robust, and
        # will break down once quantities are combined
        # it is better to create a completely new dimensionality

        class Estimation(Dimensionality):
            _intermediate = True

        class EstimatedLength(Estimation):
            dimensions = Length.dimensions

        class EstimatedMass(Estimation):
            dimensions = Mass.dimensions

        # a direct subclass will be compatible with parent classes
        class EstimatedDistance(EstimatedLength):
            dimensions = Length.dimensions

        # the Estimation subtype cannot be used directly
        # it's possible to create the subclass, but not create an instance
        EstimatedQuantity = Q[Estimation, float]

        # TODO: this does not currently work
        # with pytest.raises(TypeError):
        EstimatedQuantity(25, "m")

        # these quantities are not compatible with normal Length/Mass
        s = Q[EstimatedLength, float](25, "m")
        m = Q[EstimatedMass, float](25, "kg")

        assert issubclass(cast(Any, s)._dimensionality_type, Estimation)
        assert issubclass(cast(Any, m)._dimensionality_type, Estimation)

        # the dimensionality type is preserved for add, sub and
        # mul, div with scalars and Q[Dimensionless]

        assert isinstance_types(s, Q[EstimatedLength])
        assert isinstance_types(s.to_root_units(), Q[EstimatedLength])
        assert isinstance_types(s.to_base_units(), Q[EstimatedLength])
        assert isinstance_types(s.to(s.u), Q[EstimatedLength])
        assert isinstance_types(s.to("cm"), Q[EstimatedLength])

        assert isinstance_types(s * 2, Q[EstimatedLength])
        assert isinstance_types(2 * s, Q[EstimatedLength])

        assert isinstance_types(s / 2, Q[EstimatedLength])
        assert isinstance_types(s // 2, Q[EstimatedLength])

        # inverted dimensionality 1/Length
        assert not isinstance_types(2 / s, Q[EstimatedLength])

        assert isinstance_types(s + s, Q[EstimatedLength])
        assert isinstance_types(s - s, Q[EstimatedLength])
        assert isinstance_types(s - s * 2, Q[EstimatedLength])
        assert isinstance_types(s + s / 2, Q[EstimatedLength])

        assert isinstance_types(2 * s + s, Q[EstimatedLength])
        assert isinstance_types(2 * s - s / 2, Q[EstimatedLength])

        assert isinstance_types(1 * s, Q[EstimatedLength])

        assert isinstance_types(Q(1) * s, Q[EstimatedLength])
        assert isinstance_types(s * Q(1), Q[EstimatedLength])

        assert not isinstance_types(Q(1) / s, Q[EstimatedLength])
        assert isinstance_types(s / Q(1), Q[EstimatedLength])

        assert isinstance_types(s // Q(1), Q[EstimatedLength])

        s_arr = Q([25], "m")
        s_arr_first = s_arr[0]
        s_arr_slice = s_arr[0:1]

        assert_type(s_arr, Q[Length, Numpy1DArray])
        assert_type(s_arr_first, Q[Length, float])
        assert_type(s_arr_slice, Q[Length, Numpy1DArray])

        s_series = Q(pl.Series([25]), "m")
        s_series_first = s_series[0]
        s_series_slice = s_series[0:1]

        assert_type(s_series, Q[Length, pl.Series])
        assert_type(s_series_first, Q[Length, float])
        assert_type(s_series_slice, Q[Length, pl.Series])

        s_arr = Q([25], "m").asdim(EstimatedLength)
        s_arr_first = s_arr[0]

        assert_type(s_arr, Q[EstimatedLength, Numpy1DArray])
        assert_type(s_arr_first, Q[EstimatedLength, float])

        s_series = Q(pl.Series([25]), "m").asdim(EstimatedLength)
        s_series_first = s_series[0]

        assert_type(s_series, Q[EstimatedLength, pl.Series])
        assert_type(s_series_first, Q[EstimatedLength, float])

        # these quantities are not compatible with normal Length/Mass
        # TODO: use a more specific exception here
        with pytest.raises(Exception):  # noqa: B017
            _ = cast(Any, s) + Q(25, "m")

        with pytest.raises(Exception):  # noqa: B017
            _ = m + Q(25, "kg")

        # EstimatedDistance is a direct subclass of EstimatedLength, so this works
        _ = Q[EstimatedDistance, float](2, "m") + s
        _ = cast(Any, s) - Q[EstimatedDistance, float](2, "m")

        assert Q[EstimatedDistance](s.m, s.u) == s

        assert isinstance_types(Q[EstimatedDistance, float](2, "m") + s, Q[EstimatedDistance])
        assert isinstance_types(cast(Any, s) - Q[EstimatedDistance, float](2, "m"), Q[EstimatedLength])


def test_type_eq() -> None:
    q = Q(25, "m")
    q_arr = Q([25], "m")

    assert isinstance(q, Q)
    assert isinstance_types(q, Q[Length, float])
    assert isinstance_types(q, Q[Length, Any])

    # the isinstance check is statically redundant (q is always a Quantity),
    # but the narrowing it produces lets the else branch exercise assert_never
    if isinstance(q, Q):  # pyright: ignore[reportUnnecessaryIsInstance]
        assert_type(q, Q[Length, float])  # pyrefly: ignore[assert-type]
    else:
        assert_never(q)

    if isinstance_types(q, Q[Length, Any]):
        assert_type(q, Q[Length, float])  # pyrefly: ignore[assert-type]
    else:
        assert_never(q)

    if isinstance_types(q, Q[Length, Numpy1DArray]):
        assert_never(q)  # pyrefly: ignore[bad-argument-type]

    if isinstance_types(q_arr, Q[Length]):
        assert_type(q_arr, Q[Length])
    else:
        assert_never(q_arr)

    if isinstance_types(q, Q[Length, float]):
        assert_type(q, Q[Length, float])  # pyrefly: ignore[assert-type]
    else:
        assert_never(q)

    # this is overloaded to work for the Quantity base class
    # for compatibility with other libraries

    assert type(q) == Q  # noqa: E721
    assert type(q) == Q  # noqa: E721

    assert type(Q(2)) == Q  # noqa: E721
    assert type(Q(25, "bar")) == Q  # noqa: E721

    # __eq__ is overloaded, but these are still different types
    assert type(q) is not Q

    # subclasses behave as expected

    assert type(q) == Q[Length, float]  # noqa: E721
    assert Q[Length, float] == type(q)  # noqa: E721

    assert type(q) != Q[Dimensionless, float]  # noqa: E721
    assert Q[Dimensionless, float] != type(q)  # noqa: E721

    assert type(q) != Q[Length]  # noqa: E721
    assert Q[Length] != type(q)  # noqa: E721


def test_class_eq_hash_contract() -> None:
    # Dimensionality classes compare by qualname, so equal classes (e.g. the same
    # class statement re-executed, as under Jupyter autoreload) must hash equal
    with _reset_dimensionality_registry():
        X1 = type("XDimContract", (Dimensionality,), {"dimensions": Length.dimensions})
        X2 = type("XDimContract", (Dimensionality,), {"dimensions": Length.dimensions})

        assert X1 is not X2
        assert cast(object, X1) == X2
        assert hash(X1) == hash(X2)

    assert (Temperature == TemperatureDifference) is False

    # Quantity classes: `subclass == Quantity` is a convenience predicate, but the
    # hash stays identity-based so class-keyed caches (typeguard, pydantic, pint)
    # keep every subclass a distinct key and never alias it to the base class
    assert type(Q(1.0, "m")) == Q  # noqa: E721
    assert (Q[Mass, float] == Q[Length, float]) is False
    keyed = {Q[Mass, float]: "mass", Q[Length, float]: "length"}
    assert keyed[Q[Mass, float]] == "mass"
    assert keyed[Q[Length, float]] == "length"
    assert Q not in keyed


def test_hash_eq_consistency() -> None:
    # __eq__ compares across units and with a tolerance, so the eq/hash contract
    # (a == b => hash(a) == hash(b)) requires hashing a canonical (root-unit) form,
    # otherwise a Quantity is unusable as a dict key / set member.
    def _eq_and_hash(a: Q[Any, float], b: Q[Any, float]) -> None:
        assert a == b
        assert hash(a) == hash(b), f"{a} == {b} but hashes differ"

    _eq_and_hash(Q(1.0, "m"), Q(100.0, "cm"))
    _eq_and_hash(Q(1.0, "bar"), Q(100.0, "kPa"))
    _eq_and_hash(Q(0.0, "degC"), Q(273.15, "K"))
    _eq_and_hash(Q(3600.0, "s"), Q(1.0, "h"))

    # usable as dict key / set member regardless of the unit it was written in
    assert Q(100.0, "cm") in {Q(1.0, "m"): "one metre"}
    assert len({Q(1.0, "m"), Q(100.0, "cm"), Q(1000.0, "mm")}) == 1

    # a TemperatureDifference must stay hashable -- __hash__ uses the root-unit
    # representation, which expresses the ΔT in K without changing its meaning
    dt = Q(20.0, "degC") - Q(15.0, "degC")
    assert dt.check(TemperatureDifference)
    assert hash(dt) == hash(Q(5.0, "delta_degC"))

    # non-float magnitudes remain unhashable (unchanged behaviour)
    with pytest.raises(TypeError, match="unhashable"):
        hash(Q(np.array([1.0, 2.0]), "m"))

    # dimensionless quantities compare equal to plain numbers, so they must also
    # hash like them: {0.5, Q(0.5, "")} is ONE element, and dict lookups with a
    # plain-number key find the Quantity entry (and vice versa)
    assert Q(0.5, "") == 0.5
    assert hash(Q(0.5, "")) == hash(0.5)
    assert len({0.5, Q(0.5, "")}) == 1
    assert Q(50.0, "%") in {0.5: "half"}


def test_magnitude_setter() -> None:
    q = Q(1.0, "m")

    # same-type replacement is validated like the constructor (int -> float)
    q.m = cast(Any, 5)
    assert isinstance(q.m, float)
    assert q.m == 5.0

    # switching the magnitude type in place would desync the instance from its
    # Quantity[DT, MT] subclass -- refused, use .astype() / a new Quantity
    with pytest.raises(TypeError, match="astype"):
        q.m = cast(Any, [1.0, 2.0, 3.0])

    qa = Q(np.array([1.0, 2.0]), "m")
    qa.m = cast(Any, [3.0, 4.0])  # list -> ndarray via constructor validation
    assert isinstance(qa.m, np.ndarray)

    with pytest.raises(TypeError, match="astype"):
        qa.m = cast(Any, 1.0)


def test_Q() -> None:
    # test that Quantity objects can be constructed
    Q(1, "dimensionless")
    Q(1)
    Q(1, None)
    Q(1, "kg")
    Q(1, "bar")
    Q(1, "h")
    Q(1, "newton")
    Q(1, "cSt")
    Q([])

    units = [
        "Δ%",
        "ΔK",
        "ΔdegC",
        "Δ℃",
        "℃",
        "Δ°C",
    ]

    for unit in units:
        Q(1, unit)

    assert Q(1, "meter/kilometer").to_reduced_units().m == 0.001
    assert Q(1, "km").to_base_units().m == 1000

    # make sure that the alias Q behaves identically to Quantity
    assert Q(1) == Quantity(1)
    assert type(Q(1)) is type(Quantity(1))
    assert type(Q) is type(Quantity)

    # == across incomparable operands answers False (Python convention);
    # only the ordering comparisons raise
    assert Q(2, "kg") != 2
    assert (Q(2, "kg") == 2) is False
    assert 2 != Q(2, "kg")  # noqa: SIM300
    assert (2 == Q(2, "kg")) is False  # noqa: SIM300
    assert cast(Any, Q(2, "kg")) != Q(25, "m")
    assert (cast(Any, Q(2, "kg")) == Q(25, "m")) is False

    # inputs can be nested
    Q(Q(1, "kg"))

    mass = Q(12, "kg")

    Q(Q(Q(Q(mass))))

    assert Q(Q(Q(Q(mass), "lbs"))).u == Unit("lbs")

    Q(Q(Q(Q(mass), "lbs")), "stone")

    # no unit input defaults to dimensionless
    assert Q(12).check("")
    assert Q(1) == Q(100, "%")
    Q[Dimensionless, float](21)
    assert isinstance_types(Q(21), Q[Dimensionless])
    assert isinstance_types(Q(21), Q[Dimensionless, float])
    assert isinstance_types(Q(21), Q[Dimensionless, Any])
    assert Q(1) == Q(1.0)

    assert isinstance(Q(1, "meter").m, float)
    assert isinstance(Q(2.3, "meter").m, float)
    assert isinstance(Q([2], "meter").m, np.ndarray)
    assert isinstance(Q([3.4], "meter").m, np.ndarray)
    assert isinstance(Q([3.4, 2], "meter").m, np.ndarray)
    assert isinstance(Q([2, 3.4], "meter").m, np.ndarray)
    assert isinstance(Q(np.array([2, 3.4]), "meter").m, np.ndarray)

    with pytest.raises(TypeError, match="unit must be"):
        Q(1.0, cast(Any, 5))

    Q(1, Q(2, "bar").u)
    Q(Q(2, "bar").to("kPa").m, "kPa")

    # check that the dimensionality constraints work
    Q[Length, float](1, "m")
    Q[Pressure, float](1, "kPa")
    Q[Temperature, float](1, "°C")
    Q[Temperature, Any](1, "°C")
    P = Q(1, "bar")
    # this Quantity must have the same dimensionality as P
    Q(2, "kPa").check(P)

    # TODO: not possible to raise on constructing inconsistent Quantity
    # if the complete subclass is pre-defined

    Q[Temperature, float](1, "kg")
    Q[Pressure, float](1, "meter")
    Q[Mass, float](1, str(P.u))
    Q[Mass, float](P)
    # in-place conversion
    # NOTE: don't use this for objects that are passed in by the user
    P3 = Q(1, "bar")
    P3.ito("kPa")
    P3.ito(Q(123123, "kPa").u)

    assert P3.m == approx(100, rel=1e-12)

    # test conversions to np.ndarray with int/float dtypes
    a = Q([1, 2, 3], "bar")
    a.ito("kPa")

    a = Q(np.array([1, 2, 3.0]), "bar")
    a.ito("kPa")

    a = Q(np.array([1.0, 2.0, 3.0]), "bar")
    a.ito("kPa")

    # conversion to new object
    P4 = Q(1, "bar")
    P4_b = P4.to("kPa")
    P4_b = P4.to(Q(123123, "kPa").u)

    assert P4_b.m == approx(100, rel=1e-12)

    assert Q(1, "bar") == Q(100, "kPa") == Q(0.1, "MPa") == Q(1e5, "Pa")

    P2 = Q(2, "feet_water")

    # floating point math might make this off at the N:th decimal
    assert P2.m == approx(2, rel=1e-12)
    assert isinstance_types(P2, Q[Pressure])
    assert_type(Q(Q(2, "feet_water"), Q(2, "kPa").u), Q[Pressure, float])
    assert_type(
        Q(Q(2, "feet_water"), Q(321321, "psi").u).to(Q(123123, "feet_water").asdim(Pressure)),
        Q[Pressure, float],
    )

    with pytest.raises(DimensionalityError):
        Q(Q(2, "feet_water"), Q(321321, "kg").u).to(cast(Any, Q(123123, "feet_water").asdim(Pressure)))

    # the UnitsContainer objects can be used to construct new dimensionalities
    # NOTE: custom dimensionalities must have unique names

    class Custom(Dimensionality):
        dimensions = Length.dimensions * Length.dimensions * Length.dimensions / Temperature.dimensions

    Q[Custom, float](1, "m³/K")
    with pytest.raises(Exception):  # noqa: B017
        cast(Any, Q)[cast(Any, Pressure) / Area, float](1, "bar/m")

    # percent or %
    Q(1.124124e-3, "").to("%").to("percent")
    Q(1.124124e-3).to("%").to("percent")
    assert Q(1.0, "mol%").u == Q(1.0, "mol percent").u

    # np.ndarray magnitudes equality check
    assert (Q([1, 2, 3], "kg") == Q([1000, 2000, 3000], "g")).all()
    assert not (Q([1, 2, 3], "kg") == Q([1000, 2000, 3001], "g")).all()

    # incompatible dimensionalities: == answers a plain False, even for arrays
    incompatible_eq = cast(Any, Q([1, 2, 3], "kg")) == Q([1000, 2000, 300], "g * meter")
    assert incompatible_eq is False

    vals = [1, 2, 3]

    # compare scalar and vector will return a vector
    assert (Q(2, "bar") == Q(vals, "bar").to("kPa")).any()

    assert not (Q(5, "bar") == Q(vals, "bar").to("kPa")).any()


def test_custom_units() -> None:
    assert_type(Q(1, "kilogram"), Q[UnknownDimensionality, float])
    assert Q(1, "kg") == Q(1, "kilogram")

    assert cast(Any, Q(1, "kg")) != Q(1, "m")
    assert (cast(Any, Q(1, "kg")) == Q(1, "m")) is False
    assert Q(1, "kilogram") != Q(1, "m")

    # "ton" should always default to metric ton
    assert Q(1, "ton") == Q(1, "Ton") == Q(1, "TON") == Q(1, "tonne") == Q(1, "metric_ton") == Q(1000, "kg")

    assert Q(1, "US_ton") == Q(907.1847400000001, "kg")

    assert (
        Q(1, "ton/hour")
        == Q(1, "Ton/hour")
        == Q(1, "TON/hour")
        == Q(1, "tonne/hour")
        == Q(1, "metric_ton/hour")
        == Q(1000, "kg/hour")
    )

    v1 = (Q(1000, "liter") * Q(1, "normal")).to_base_units().m
    v2 = Q(1000, "normal liter").to_base_units().m
    v3 = Q(1, "nm3").m
    v4 = Q(1, "Nm3").m

    # floating point accuracy
    assert round(v1, 10) == round(v2, 10) == round(v3, 10) == round(v4, 10)

    define_dimensionality("air")

    factor = Q(12, "Nm3 water/ (normal liter air)")
    (Q(1, "kg water") / factor).to("pound air")

    Q[NormalVolume, Any](2, "nm**3")

    # with pytest.raises(ExpectedDimensionalityError):
    #     Q[NormalVolumeFlow](2, 'm**3/hour')

    Q[NormalVolumeFlow, Any](2, "Nm**3/hour").to("normal liter/sec")

    class _NormalVolumeFlow(NormalVolumeFlow):
        dimensions = Normal.dimensions * VolumeFlow.dimensions

    Q[_NormalVolumeFlow, Any](2, "Nm**3/hour").to("normal liter/sec")

    Q(2, "normal liter air / day")
    Q(2, "1/Nm3").to("1 / (liter normal)")


def test_wraps() -> None:
    # @UNIT_REGISTRY.wraps(ret, args, strict=True|False) is a convenience
    # decorator for making the input/output of a function into Quantity
    # however, it does not enforce the return value
    # NOTE: do not use this, does not support typing at all

    @cast(Any, UNIT_REGISTRY).wraps("kg", ("m", "kg"), strict=True)
    def func(a: Any, b: Any) -> Any:  # noqa: ANN401
        # this is incorrect, cannot add 1 to a dimensional Quantity
        return a * b**2 + 1

    assert isinstance_types(func(Q(1, "yd"), Q(20, "lbs")), Q[Mass])
    assert Q(1, "bar").check(Pressure)


def test_numpy_integration() -> None:
    assert isinstance_types(Q(np.linspace(0, 1), "kg"), Q[Mass])
    assert isinstance_types(np.linspace(Q(0, "kg"), Q(1, "kg")), Q[Mass])

    assert isinstance_types(np.linspace(Q(2), Q(25, "%")), Q[Dimensionless])
    assert isinstance_types(np.linspace(Q(2, "cm"), Q(25, "km")), Q[Length])

    with pytest.raises(DimensionalityError):
        np.linspace(Q(2, "kg"), Q(25, "m"))
    q1 = Q(np.linspace(0, 1), "kg")
    q2 = np.linspace(Q(0, "kg"), Q(1, "kg"))
    comp = q1 == q2
    assert comp.all()

    assert isinstance_types(list(Q(np.linspace(0, 1), "degC")), list[Q[Temperature]])


def test_units_import_does_not_install_warning_filter() -> None:
    code = """
import warnings
from pint.errors import UnitStrippedWarning

warnings.simplefilter("error", UnitStrippedWarning)
import encomp.units  # noqa: F401

bad_filters = [
    filt for filt in warnings.filters
    if filt[0] == "ignore" and filt[2] is UnitStrippedWarning
]
raise SystemExit(1 if bad_filters else 0)
"""

    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_unit_stripped_warning_suppression_is_scoped_to_quantity_array() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UnitStrippedWarning)
        np.asarray(Q(1.0, "m"))

    assert not caught


def test_check() -> None:
    assert Q(25, "kg").check(Mass)
    assert Q(25, "m").check("[length]")
    assert not Q(25, "kg").check("[length]")

    assert Q(25, "kg").check(Q(25, "g"))
    assert Q(25, "kg").check(Q([25, 25], "g"))

    epm = Q(1.0, "kJ/kg").asdim(EnergyPerMass)
    hv = Q(1.0, "kJ/kg").asdim(LowerHeatingValue)

    assert hv.check("kJ/kg")
    assert hv.check(Q(1.0, "J/kg"))
    assert hv.check(EnergyPerMass)
    assert hv.check(epm)

    with pytest.raises(DimensionalityTypeError):
        _ = cast(Any, hv) + epm


def test_typechecked() -> None:
    @typechecked
    def func_a(a: Quantity[Temperature, Any]) -> Quantity[Pressure, Any]:  # noqa: ARG001
        return Q(2, "bar")

    assert func_a(Q(2, "degC")) == Q(2, "bar")

    with pytest.raises(Exception):  # noqa: B017
        func_a(cast(Any, Q(2, "meter")))

    @typechecked
    def func_b(a: Quantity[Pressure, Any]) -> Quantity[Pressure, Any]:
        return a

    assert func_b(Q(2, "bar")) == Q(2, "bar")
    assert func_b(Q(2, "psi")) == Q(2, "psi")
    assert func_b(Q(2, "mmHg")) == Q(2, "mmHg")

    with pytest.raises(Exception):  # noqa: B017
        func_a(cast(Any, Q(2, "meter")))

    @typechecked
    def func_c(a: Quantity[Temperature]) -> Quantity[Pressure]:  # noqa: ARG001
        return Q([2], "bar")

    assert func_c(Q([2], "degC")) == Q([2], "bar")

    func_c(cast(Any, Q(2, "degC")))

    with pytest.raises(Exception):  # noqa: B017
        func_c(cast(Any, Q([2], "meter")))


def test_generic_dimensionality() -> None:
    assert issubclass(Q[Pressure], Q)
    assert not issubclass(Q[Pressure], cast(type, Q[Temperature]))

    assert Q[Pressure] is Q[Pressure]
    assert Q[Pressure] == Q[Pressure]

    assert Q[Pressure] is not Q[Temperature]
    assert Q[Pressure] != Q[Temperature]

    assert isinstance_types([Q, Q[Pressure], Q[Temperature]], list[type[Q]])

    assert not isinstance_types([Q, Q[Pressure], Q[Temperature]], list[Q])

    assert isinstance_types([Q[Pressure], Q[Pressure]], list[type[Q[Pressure]]])

    _ = Q[Any]
    _ = Q[Any, Any]

    with pytest.raises(TypeError):
        cast(Any, Q)[1]

    with pytest.raises(TypeError):
        Q["Temperature"]  # pyrefly: ignore[not-a-type]

    with pytest.raises(TypeError):
        cast(Any, Q)["string"]

    with pytest.raises(TypeError):
        cast(Any, Q)[None]

    with pytest.raises(TypeError):
        cast(Any, Q)[None, None]

    with pytest.raises(TypeError):
        cast(Any, Q)[Any, Any, Any]

    cast(Any, Q)[DT]
    cast(Any, Q)[DT, MT]
    cast(Any, Q)[DT, Any]
    cast(Any, Q)[Any, MT]


def test_dynamic_dimensionalities() -> None:
    # this will create a new dimensionality type
    q1 = Q(1, "kg^2/K^4")

    # this will reuse the previously created one
    q2 = Q(25, "g*g/(K^2*K^2)")

    assert type(q1) is type(q2)

    # NOTE: don't use __class__ when checking this, always use type()
    assert isinstance(q1, type(q2))
    assert isinstance(q2, type(q1))
    assert isinstance(q2, Q)
    assert not isinstance_types(q2, Q[Pressure])

    q3 = Q(25, "m*g*g/(K^2*K^2)")

    assert type(q3) is not type(q2)
    assert not isinstance(q3, type(q1))


def test_instance_checks() -> None:
    assert isinstance_types(Q(2), Q[Dimensionless])
    assert isinstance(Q(2), Q)

    class Fraction(Dimensionless):
        pass

    # Q(2) will default to Dimensionless, since that was
    # defined before Fraction
    assert not isinstance_types(Q(2), Q[Fraction])
    assert isinstance_types(Q(2), Q[Dimensionless])

    # Q[Fraction] will override the subclass
    assert isinstance_types(Q[Fraction, float](2), Q[Fraction])

    assert isinstance_types(Q(25, "bar"), Q[Pressure])

    assert not isinstance_types(Q(25, "°C"), Q[Pressure])

    assert isinstance_types(Q(25, "°C"), Q[Temperature])

    assert isinstance_types(Q(25, "kg"), Q[Mass])

    assert isinstance(Q(25, "kg"), Q)
    assert isinstance_types(Q(25, "kg"), Q)

    assert isinstance_types(Q(25, "°C"), Q)
    assert isinstance_types(Q(25, "°C"), Q[Temperature])

    assert isinstance_types([Q(25, "°C")], list[Q[Temperature]])
    assert isinstance_types([Q[Temperature, float](25, "°C"), Q(25, "°F")], list[Q[Temperature]])
    assert isinstance_types([Q(25, "°C"), Q(25, "bar")], list[Q])

    # NOTE: the name CustomDimensionality must be globally unique

    class CustomDimensionality(Dimensionality):
        dimensions = UnitsContainer({"[mass]": 1, "[temperature]": -2})

    # Q(25, 'g/K^2') will find the correct subclass since there
    # are no other dimensionalities with these dimensions
    assert isinstance_types(Q(25, "g/K^2"), Q[CustomDimensionality])

    assert isinstance_types(Q[CustomDimensionality, float](25, "g/K^2"), Q[CustomDimensionality])

    assert isinstance_types(Q[CustomDimensionality, float](25, "g/K^2"), Q)

    assert isinstance_types({Q(25, "g/K^2")}, set[Q[CustomDimensionality]])

    assert not isinstance_types((Q(25, "m*g/K^2"),), tuple[Q[CustomDimensionality], ...])

    assert not isinstance_types([Q(25, "°C")], list[Q[Pressure]])

    assert isinstance_types([Q[Temperature, float](25, "°C"), Q(25, "°F")], list[Q[Temperature]])

    assert not isinstance_types([Q(25, "F/day")], list[Q[Temperature]])

    assert not isinstance_types([Q(25, "°C"), Q(25, "bar")], list[Q[CustomDimensionality]])


def test_typed_dict() -> None:
    d = {
        "P": Q[Pressure, float](25, "bar"),
        "T": Q[Temperature, float](25, "degC"),
        "x": Q[Dimensionless, float](0.5),
    }

    class PTxDict1(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    # cannot use isinstance with complex types
    with pytest.raises(TypeError):
        isinstance(d, cast(Any, PTxDict1))

    assert isinstance_types(d, PTxDict1)

    d = {
        "P": Q[Pressure, float](25, "bar"),
        "T": Q[Temperature, float](25, "degC"),
        "x": Q[Dimensionless, float](0.5),
        "extra": Q,
    }

    class PTxDict2(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    assert not isinstance_types(d, PTxDict2)

    d = {
        "P": Q[Pressure, float](25, "bar"),
        "T": Q[Temperature, float](25, "degC"),
        "x": Q[Dimensionless, float](0.5),
    }

    class PTxDict3(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]
        missing: Q

    assert not isinstance_types(d, PTxDict3)

    d = {
        "P": Q(25, "bar"),
        "T": Q(25, "degC"),
        "x": Q(0.5),
    }

    class PTxDict4(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    assert isinstance_types(d, PTxDict4)

    d = {
        "P": Q(25, "bar"),
        "T": Q(25, "meter"),
        "x": Q(0.5),
    }

    class PTxDict5(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    assert not isinstance_types(d, PTxDict5)


def test_convert_volume_mass() -> None:
    V = convert_volume_mass(Q(125, "kg/s"), Q(25, "kg/m**3"))

    assert V.check(VolumeFlow)

    m = convert_volume_mass(Q(125, "liter/day").asdim(VolumeFlow))

    assert m.check(MassFlow)


def test_compatibility() -> None:
    _ = Q(1) + Q(2)

    _ = Q(1, "%") - Q(2)

    _ = Q(25, "m") + Q(25, "cm")

    # this subtype inherits from Dimensionless, which
    # means that it is compatible (can be added and subtracted)

    class Fraction(Dimensionless):
        pass

    q1 = Q(2)
    q2 = Q[Fraction, float](0.6)

    _ = q1 + q2
    _ = q2 + q1

    _ = q1 - q2
    _ = q2 - q1

    # this is a new dimensionality that happens to have
    # the same dimensions as Dimensionless
    # however, it is not compatible with Dimensionless or any of its subclasses

    class IncompatibleFraction(Dimensionality):
        dimensions = Dimensionless.dimensions

    q3 = Q[IncompatibleFraction, float](0.2)

    # TODO: use more specific exception here

    with pytest.raises(Exception):  # noqa: B017
        _ = q3 + q1

    with pytest.raises(Exception):  # noqa: B017
        _ = q3 + q2

    with pytest.raises(Exception):  # noqa: B017
        _ = q1 + q3

    with pytest.raises(Exception):  # noqa: B017
        _ = q2 + q3

    s = Q(25, "m")

    # this dimensionality means something specific,
    # it cannot be added to a normal length
    class DistanceAlongPath(Dimensionality):
        dimensions = Length.dimensions

    # need to override the overload based on unit "km"
    # this is not very elegant
    d = Q[DistanceAlongPath, float](25, "km")
    d2 = Q[DistanceAlongPath, float](5, "km")

    _ = d + d
    _ = d - d
    _ = d + d2
    _ = d2 - d

    with pytest.raises(Exception):  # noqa: B017
        _ = s + d

    with pytest.raises(Exception):  # noqa: B017
        _ = d - s

    assert str(round((Q(25, "MSEK/GWh") * Q(25, "kWh")).to_reduced_units(), 8)) == "0.000625 MSEK"

    assert str((Q(25, "MSEK/GWh") * Q(25, "kWh")).to_base_units()) == "625.0 currency"

    with pytest.raises(DimensionalityTypeError):
        _ = cast(Any, Q(25, "kg")) + Q(2, "m")

    with pytest.raises(DimensionalityTypeError):
        _ = cast(Any, Q(25, "kg")) - Q(2, "m")

    # if the _distinct class attribute is True, an unspecified
    # dimensionality will default to this
    # for example, EnergyPerMass has _distinct=True even though
    # it shares dimensions with other dimensionalities
    # like SpecificEnthalpy and HeatingValue

    # q4 will be become Quantity[EnergyPerMass] by default

    q4 = Q(25, "kJ/kg")

    # override the Literal['kJ/kg'] overload
    q5 = Q[SpecificEnthalpy, float](25, "kJ/kg")

    # prefer to use the asdim method
    q5_ = Q(25, "kJ/kg").asdim(SpecificEnthalpy)

    assert q5 == q5_

    q6 = Q[EnergyPerMass, float](25, "kJ/kg")

    # NOTE: changed to not consider SpecificEnthalpy and EnergyPerMass distinct,
    # this caused a lot of extra asdim() calls with no apparent advantage
    _ = q4 - q5
    _ = q5 - q4

    _ = q4 + q6
    _ = q6 - q4


def test_distinct_dimensionality() -> None:
    # TODO: this test does not seem correct

    unit = "m**6/kg**2"
    uc = UnitsContainer({"[length]": 6, "[mass]": -2})

    class Indistinct(Dimensionality):
        dimensions = uc
        _distinct = False

    class Distinct(Dimensionality):
        dimensions = uc
        _distinct = True

    assert type(Q(1, unit)) is Q[Distinct, float]

    # TODO: this should actually be Q[Distinct, float]
    assert type(Q[Distinct, float](1, unit)) is Q[Distinct, float]
    assert type(Q[Indistinct, float](1, unit)) is Q[Indistinct, float]


def test_literal_units() -> None:
    for d, units in get_registered_units().items():
        for u in units:
            assert cast(Any, Q(1, u))._dimensionality_type.__name__ == d


def test_indexing() -> None:
    qs = Q([1, 2, 3], "kg")

    assert isinstance_types(qs, Q[Mass])

    qi = qs[1]

    assert isinstance_types(qi, Q[Mass])
    assert qi == Q(2, "kg")


def test_round() -> None:
    # TODO: should this even work?
    # type numpy.ndarray doesn't define __round__ method

    q = Q(25.12312312312, "kg/s")

    q_r = round(q, 1)

    assert q_r.m == 25.1

    q2 = Q([25.12312312312, 25.12312312312], "kg/s")

    q_r2 = round(q2, 1)

    assert q_r2.m[0] == 25.1
    assert q_r2.m[1] == 25.1


def test_abs() -> None:
    q = Q(-25, "kg/s")

    q_a = abs(q)

    assert q_a.m == 25

    q2 = Q([-25, -25], "kg/s")

    q_a2 = abs(q2)

    assert q_a2.m[0] == 25
    assert q_a2.m[1] == 25


def test_unit_compatibility() -> None:
    # the UNIT_REGISTRY registry object contains unit attributes
    # that can be multiplied and divided by a magnitude
    # to create Quantity instances

    assert isinstance_types(cast(Any, UNIT_REGISTRY).m * 1, Q[Length])
    assert isinstance_types(1 * cast(Any, UNIT_REGISTRY).m / cast(Any, UNIT_REGISTRY).s, Q[Velocity])
    assert isinstance_types([1, 2, 3] * cast(Any, UNIT_REGISTRY).m / cast(Any, UNIT_REGISTRY).s, Q[Velocity])
    assert isinstance_types((1, 2, 3) * cast(Any, UNIT_REGISTRY).m / cast(Any, UNIT_REGISTRY).s, Q[Velocity])
    # assert isinstance_types(np.array([1, 2, 3]) * UNIT_REGISTRY.m / UNIT_REGISTRY.s, Q[Velocity])
    assert isinstance_types([1, 2, 3] * cast(Any, UNIT_REGISTRY).m / cast(Any, UNIT_REGISTRY).s, Q[Velocity])


def test_mul_rmul_initialization() -> None:
    assert isinstance_types(cast(Any, UNIT_REGISTRY).m * np.array([1, 2]), Q[Length])
    # this returns array([<Quantity(1.0, 'meter')>, <Quantity(2.0, 'meter')>], dtype=object) instead
    # assert isinstance_types(np.array([1, 2]) * cast(Any, UNIT_REGISTRY).m, Q[Length])
    assert isinstance_types([1, 2] * cast(Any, UNIT_REGISTRY).m, Q[Length])
    assert isinstance_types(cast(Any, [1, 2]) * Q(1, "m"), Q[Length])
    assert isinstance_types(np.array([1, 2]) * Q(1, "m"), Q[Length])


def test_copy() -> None:
    q = Q(25, "m")

    assert isinstance_types(q, Q[Length])
    assert isinstance_types(q.__copy__(), Q[Length])

    assert isinstance_types(q.__deepcopy__(), Q[Length])
    assert isinstance_types(q.__deepcopy__({}), Q[Length])

    assert isinstance_types(copy.copy(q), Q[Length])
    assert isinstance_types(copy.deepcopy(q), Q[Length])


def test_pickle_preserves_dimensionality_type() -> None:
    dt = Q(5.0, "delta_degC").to("K")
    dt_roundtrip = pickle.loads(pickle.dumps(dt))

    assert isinstance_types(dt_roundtrip, Q[TemperatureDifference])
    assert dt_roundtrip.u == Unit("K")
    assert dt_roundtrip == dt

    hv = Q(25.0, "kJ/kg").asdim(HeatingValue)
    hv_roundtrip = pickle.loads(pickle.dumps(hv))

    assert isinstance_types(hv_roundtrip, Q[HeatingValue])
    assert hv_roundtrip.u == Unit("kilojoule / kilogram")
    assert hv_roundtrip == hv


def test_pydantic_integration() -> None:
    class Model(BaseModel):
        # a can be any dimensionality
        a: Q[UnknownDimensionality, float]

        m: Q[Mass, float]
        s: Q[Length, float]

        # float can be converted to Quantity[Dimensionless]
        r: Q[Dimensionless, float] = cast(Any, 0.5)

        # float cannot be converted to Quantity[Length]
        # d: Q[Length] = 0.5

        model_config = ConfigDict(validate_default=True)

    Model(a=Q(25, "cSt").asdim(UnknownDimensionality), m=Q(25, "kg"), s=Q(25, "cm"))

    with pytest.raises(ValidationError):
        Model(
            a=Q(25, "cSt").asdim(UnknownDimensionality),
            m=cast(Any, Q(25, "kg/day").asdim(MassFlow)),
            s=Q(25, "cm"),
        )


def test_float_cast() -> None:
    assert isinstance(Q([False, False]).m[0], float)

    assert (Q([False, True]) == Q(np.array([False, True]))).all()


def test_temperature_difference() -> None:
    # .to() preserves the dimensionality type: an absolute temperature cannot be
    # converted to a delta unit (use .asdim(TemperatureDifference) to reinterpret)
    with pytest.raises(DimensionalityTypeError):
        _ = Q(25, "K").to("delta_degC")

    T0 = Q(25, "K").asdim(TemperatureDifference).to("delta_degC")
    assert isinstance_types(T0, Q[TemperatureDifference])
    assert T0.m == approx(25.0)

    td_offset = Q(20, "degC").asdim(TemperatureDifference)
    assert isinstance_types(td_offset, Q[TemperatureDifference])
    assert td_offset.u == Unit("delta_degC")
    assert td_offset.to("K").m == approx(20.0)
    assert (Q(30, "degC") + td_offset).m == approx(50.0)

    with pytest.raises(DimensionalityTypeError, match="delta unit"):
        Q[TemperatureDifference, float](5, "degC")

    T1 = Q(25, "degC")
    T2 = Q(35, "degC")

    dT1 = T1 - T2
    dT2 = T2 - T1

    assert isinstance_types(dT1, Q[TemperatureDifference])
    assert isinstance_types(dT2, Q[TemperatureDifference])

    assert dT1.u == Unit("delta_degC")
    assert dT2.u == Unit("delta_degC")

    assert (Q(25, "degF") - Q(30, "degF")).u == Unit("delta_degF")

    with pytest.raises(OffsetUnitCalculusError):
        _ = T1 + T2

    with pytest.raises(OffsetUnitCalculusError):
        _ = T2 + T1

    with pytest.raises(OffsetUnitCalculusError):
        _ = T1 * 2

    with pytest.raises(OffsetUnitCalculusError):
        _ = T1 / 2

    assert isinstance_types(dT1, Q[TemperatureDifference])
    assert isinstance_types(dT1.to("delta_degF"), Q[TemperatureDifference])

    assert isinstance_types(dT2, Q[TemperatureDifference])
    assert isinstance_types(dT2 + dT1, Q[TemperatureDifference])

    assert isinstance_types(T1 + dT1, Q[Temperature])
    assert isinstance_types(dT1 + T1, Q[Temperature])

    with pytest.raises(DimensionalityTypeError):
        (T1 - T2).to("degC")

    # a ΔT converts to any multiplicative [temperature] unit (K, degR, ...)
    # and keeps the TemperatureDifference dimensionality; only offset scales
    # (degC, degF) are refused
    dk = (T1.to("K") - T2.to("K")).to("K")
    assert isinstance_types(dk, Q[TemperatureDifference])
    assert dk.m == approx(-10.0)

    dk2 = (T1 - T2).to("K")
    assert isinstance_types(dk2, Q[TemperatureDifference])
    assert dk2.m == approx(-10.0)

    with pytest.raises(DimensionalityTypeError):
        (T1 - T2).to("degF")

    T1 = Q(800, "degC")
    T2 = T1.to("K") - Q(100, "degC").to("K")

    dk3 = T2.to("K")
    assert isinstance_types(dk3, Q[TemperatureDifference])
    assert dk3.m == approx(700.0)

    T2_ = T1.to("K") - Q(100, "delta_degC")

    assert isinstance_types(T2_, Q[Temperature])


def test_ordering_comparisons_across_dimensionalities() -> None:
    import operator

    t = Q(300.0, "K")
    dt = Q(5.0, "delta_degC")

    # == / != answer False / True across incompatible dimensionality types
    # (the static layer rejects these comparisons outright, hence the casts)...
    assert cast(Any, t) != dt
    assert (cast(Any, t) == dt) is False
    assert cast(Any, Q(1.0, "m")) != Q(1.0, "kg")

    # ...but ordering raises: no answer is correct (pint would otherwise silently
    # scale-convert, e.g. 5 delta_degC -> 5 K, and compare 5 < 300)
    for op in (operator.lt, operator.le, operator.gt, operator.ge):
        with pytest.raises(DimensionalityComparisonError):
            op(dt, t)
        with pytest.raises(DimensionalityComparisonError):
            op(t, dt)
        with pytest.raises(DimensionalityComparisonError):
            op(Q(1.0, "m"), Q(1.0, "kg"))
        with pytest.raises(DimensionalityComparisonError):
            op(Q(1.0, "m"), 5.0)

    # same dimensionality still orders normally, across units
    assert Q(1.0, "kg") > Q(25.0, "g")
    assert Q(5.0, "delta_degC") > Q(5.0, "delta_degF")


def test_to_preserves_temperature_dimensionality() -> None:
    # Temperature -> delta unit is refused in BOTH .to() and .ito()
    with pytest.raises(DimensionalityTypeError):
        _ = Q(300.0, "K").to("delta_degC")

    with pytest.raises(DimensionalityTypeError):
        _ = Q(25.0, "degC").to("delta_degF")

    with pytest.raises(DimensionalityTypeError):
        Q(np.array([300.0]), "K").ito("delta_degC")

    # .asdim() is the explicit, value-preserving escape hatch
    td = Q(300.0, "K").asdim(TemperatureDifference)
    assert isinstance_types(td, Q[TemperatureDifference])
    assert td.to("delta_degC").m == approx(300.0)
    assert td.to("delta_degF").m == approx(540.0)

    # non-temperature -> delta unit stays a plain dimensionality error
    with pytest.raises(DimensionalityError):
        _ = Q(1.0, "m").to("delta_degC")


def test_inplace_add_sub_checks_dimensionality() -> None:
    epm = Q(1.0, "kJ/kg").asdim(EnergyPerMass)
    hv = Q(2.0, "kJ/kg").asdim(LowerHeatingValue)

    with pytest.raises(DimensionalityTypeError):
        epm += hv

    with pytest.raises(DimensionalityTypeError):
        epm -= hv

    td = Q(np.array([10.0]), "delta_degC")
    td += Q(np.array([20.0]), "degC")
    assert isinstance_types(td, Q[Temperature])
    assert td.u == Unit("degC")
    assert td.m == approx(np.array([30.0]))

    td = Q(np.array([10.0]), "delta_degC")
    with pytest.raises(DimensionalityTypeError):
        td -= Q(np.array([20.0]), "degC")


def test_temperature_add_sub_preserves_unit() -> None:
    # T ± ΔT keeps the temperature operand's original unit (the value is the
    # same absolute temperature either way; only the display unit differs)
    res = Q(300.0, "K") + Q(10.0, "delta_degC")
    assert res.u == Unit("K")
    assert res.m == approx(310.0)
    assert isinstance_types(res, Q[Temperature])

    res = Q(10.0, "delta_degC") + Q(300.0, "K")
    assert res.u == Unit("K")
    assert res.m == approx(310.0)

    res = Q(300.0, "K") - Q(10.0, "delta_degC")
    assert res.u == Unit("K")
    assert res.m == approx(290.0)

    res = Q(25.0, "degC") + Q(10.0, "delta_degC")
    assert res.u == Unit("degC")
    assert res.m == approx(35.0)

    res = Q(50.0, "degF") + Q(9.0, "delta_degF")
    assert res.u == Unit("degF")
    assert res.m == approx(59.0)

    # vector magnitudes behave the same
    res = Q(np.array([300.0, 400.0]), "K") + Q(10.0, "delta_degC")
    assert res.u == Unit("K")
    assert res.m == approx(np.array([310.0, 410.0]))


def test_numpy_scalar_magnitudes() -> None:
    # numpy scalar magnitudes (e.g. arr.sum() on an integer array) normalize to
    # a plain float instead of failing with a confusing 1-D-array shape error
    for val in (np.int64(5), np.int32(5), np.float32(5.0), np.float64(5.0)):
        q = Q(cast(Any, val), "m")
        assert type(q.m) is float
        assert q.m == approx(5.0)

    assert Q(np.array([1, 2, 3]).sum(), "kg").m == approx(6.0)


def test_ambiguous_textile_units_removed() -> None:
    # pint's default registry defines the textile unit "Nm" (number meter,
    # km/kg) -- in a library where Nm³ prominently means *normal* cubic meter,
    # a torque-intending "Nm" silently parsed as km/kg. The textile group is
    # removed from defs/units.txt, so "Nm" is an explicit error instead
    from pint.errors import UndefinedUnitError

    for unit in ("Nm", "tex", "denier", "number_meter"):
        with pytest.raises(UndefinedUnitError):
            _ = Q(1.0, unit)

    # the normal-volume shorthand is unaffected
    assert Q(1.0, "Nm3/h").check(NormalVolumeFlow)
    assert Q(1.0, "Nm³").check(NormalVolume)


def test_temperature_difference_to_multiplicative_units() -> None:
    # a ΔT is a scale-only quantity: it can be expressed in any multiplicative
    # [temperature] unit (K, degR, mK, ...) and keeps its dimensionality type
    dt = Q(10.0, "delta_degC")

    k = dt.to("K")
    assert isinstance_types(k, Q[TemperatureDifference])
    assert k.m == approx(10.0)
    assert k.u == Unit("K")

    assert dt.to("degR").m == approx(18.0)
    assert dt.to("mK").m == approx(10000.0)

    # round trip back to a delta unit
    assert k.to("delta_degF").m == approx(18.0)

    # to_base_units agrees with to_root_units (both express the ΔT in K)
    base = dt.to_base_units()
    assert isinstance_types(base, Q[TemperatureDifference])
    assert base.m == approx(10.0)
    assert base.m == approx(dt.to_root_units().m)

    # in-place conversion keeps the dimensionality type too
    arr = Q(np.array([10.0, 20.0]), "delta_degC")
    arr.ito("K")
    assert isinstance_types(arr, Q[TemperatureDifference])
    assert arr.m == approx(np.array([10.0, 20.0]))

    # offset scales stay refused: their zero point would silently reinterpret
    # the difference as an absolute temperature
    with pytest.raises(DimensionalityTypeError):
        _ = dt.to("degC")
    with pytest.raises(DimensionalityTypeError):
        _ = dt.to("degF")

    # the absolute -> delta direction is unchanged (asdim is the escape hatch)
    with pytest.raises(DimensionalityTypeError):
        _ = Q(300.0, "K").to("delta_degC")


def test_temperature_unit_inputs() -> None:
    for unit in [
        "degC",
        "delta_degC",
        "ΔdegC",
        "degF",
        "delta_degF",
        "ΔdegF",
        "delta_°C",
        "delta_℃",
        "Δ℃",
        "Δ°C",
        "delta_℉",
        "Δ℉",
        "Δ°F",
    ]:
        qty = Q(1, unit)

        assert isinstance_types(qty, Q[Temperature]) or isinstance_types(qty, Q[TemperatureDifference])
        assert qty.check(Temperature)
        assert qty.check(TemperatureDifference)

        # this will automatically be converted to delta_temperature per length,
        # even if the input is temperature (not delta_temperature)
        dT_per_length = Q(1, f"{unit} / m")
        assert dT_per_length.dimensionality == Temperature.dimensions / Length.dimensions

        vol_per_dT = Q(1, f"m3/{unit}")
        assert vol_per_dT.dimensionality == Volume.dimensions / Temperature.dimensions


def test_nested_quantity_input() -> None:
    q = Q(25, "bar")
    q2 = Q(q)

    assert type(q) is type(q2)
    assert q == q2

    assert Q(Q(Q(Q(15, "m")) * 2)) * 2 == Q(60, "m")


def test_getitem() -> None:
    ms = Q([1.2, 1.3], "kg")
    assert isinstance_types(ms, Q[Mass, np.ndarray])

    m0 = ms[0]
    assert isinstance_types(m0, Q[Mass, float])


def test_class_getitem() -> None:
    Q[Length]
    Q[Length, float]

    with pytest.raises(TypeError):
        Q[Dimensionality]

    with pytest.raises(TypeError):
        cast(Any, Q)[Length, str]

    with pytest.raises(TypeError):
        cast(Any, Q)[None]

    with pytest.raises(TypeError):
        cast(Any, Q)[None, float]


def test_astype() -> None:
    assert isinstance(Q(25).astype(np.ndarray).m[0], float)

    assert isinstance(Q([1, 2, 3]).astype(np.ndarray).m, np.ndarray)
    assert isinstance(Q([1, 2, 3]).astype(pl.Series).m, pl.Series)

    qe = Q(2).astype(pl.Expr)
    assert isinstance(qe.m, pl.Expr)

    with pytest.raises(TypeError):
        Q([1, 2, 3]).astype(pl.Expr)


def test_single_element_array_magnitude() -> None:
    s1_list = [1.0]
    s2_list = [1.0, 2.0]

    _ = Q(s1_list, "kg") * Q(s2_list, "m") / Q(s2_list, "kg")

    s1_arr = np.array([1])
    s2_arr = np.array([1, 2])

    _ = Q(s1_arr, "kg") * Q(s2_arr, "m") / Q(s2_arr, "kg")


def test_check_temperature_difference() -> None:
    assert not Q(1, "degC").check(Q(12, "kg"))

    assert Q(1, "degC").check(Q(12, "degC"))
    assert Q(1, "degC").check(Q(12, "degC").u)

    assert Q(1, "delta_degC").check(Q(12, "degC"))
    assert Q(1, "delta_degC").check(Q(12, "degC").u)

    assert Q(1, "delta_degC").check(Q(12, "delta_degC"))
    assert Q(1, "delta_degC").check(Q(12, "delta_degC").u)

    with pytest.raises(DimensionalityTypeError):
        _ = cast(Any, Q(1, "delta_degC")) - Q(12, "degC")


def test_complex_units() -> None:
    Q(8.3144598, "kg*m²/K/mol/s²")

    val = Q(0.0315, "MW**0.3") * Q(2, "MW") ** 0.7

    val.to("MW")

    # Test to() with fractional units
    Q(1, "megawatt ** 0.3 * kilojoule ** 0.7 / second ** 0.7").to("MW")

    # Test ito() with fractional units
    q = Q(1, "megawatt ** 0.3 * kilojoule ** 0.7 / second ** 0.7")
    q.ito("MW")
    assert q.units == Q(1, "MW").units

    # Test with real-world scenario: fractional power law with mixed units
    def Q_flow_RC(Q_flow_B: Q[Any, Any]) -> Q[Any, Any]:
        C = Q(0.0315, "MW**0.3")
        return C * Q_flow_B**0.7

    result_mw = Q_flow_RC(Q(2, "MW")).to("MW")
    result_kjs = Q_flow_RC(Q(2000, "kJ/s")).to("MW")  # 2000 kJ/s = 2 MW
    assert abs(result_mw.magnitude - result_kjs.magnitude) < 1e-10


def test_magnitude_type_scalar_operations() -> None:
    q = Q(pl.Series([1, 2, 3]), "kg/s")

    assert_type(q, Q[MassFlow, pl.Series])
    assert_type(q * 2, Q[MassFlow, pl.Series])
    assert_type(2 * q, Q[MassFlow, pl.Series])
    assert_type(q * Q(2), Q[MassFlow, pl.Series])
    assert_type(Q(2) * q, Q[MassFlow, pl.Series])

    assert_type(q / 2, Q[MassFlow, pl.Series])
    assert_type(q / Q(2), Q[MassFlow, pl.Series])

    q2 = Q([1, 2, 3], "kg/s")

    assert_type(q2, Q[MassFlow, Numpy1DArray])
    assert_type(q2 * 2, Q[MassFlow, Numpy1DArray])
    assert_type(2 * q2, Q[MassFlow, Numpy1DArray])
    assert_type(q2 * Q(2), Q[MassFlow, Numpy1DArray])
    assert_type(Q(2) * q2, Q[MassFlow, Numpy1DArray])

    assert_type(q2 / 2, Q[MassFlow, Numpy1DArray])
    assert_type(q2 / Q(2), Q[MassFlow, Numpy1DArray])

    q3 = Q(pl.col.test, "kg/s")

    assert_type(q3, Q[MassFlow, pl.Expr])
    assert_type(q3 * 2, Q[MassFlow, pl.Expr])
    assert_type(2 * q3, Q[MassFlow, pl.Expr])
    assert_type(q3 * Q(2), Q[MassFlow, pl.Expr])
    assert_type(Q(2) * q3, Q[MassFlow, pl.Expr])

    assert_type(q3 / 2, Q[MassFlow, pl.Expr])
    assert_type(q3 / Q(2), Q[MassFlow, pl.Expr])

    q4 = Q(123, "kg/s")

    assert_type(q4, Q[MassFlow, float])
    assert_type(q4 * 2, Q[MassFlow, float])
    assert_type(2 * q4, Q[MassFlow, float])
    assert_type(q4 * Q(2), Q[MassFlow, float])
    assert_type(Q(2) * q4, Q[MassFlow, float])

    assert_type(q4 / 2, Q[MassFlow, float])
    assert_type(q4 / Q(2), Q[MassFlow, float])


def test_magnitude_type_scalar_operations_explicit_type() -> None:
    q = Q(pl.Series([1, 2, 3]), "kg/s")

    assert type(q) is Q[MassFlow, pl.Series]
    assert type(q * 2) is Q[MassFlow, pl.Series]
    assert type(2 * q) is Q[MassFlow, pl.Series]
    assert type(q * Q(2)) is Q[MassFlow, pl.Series]
    assert type(Q(2) * q) is Q[MassFlow, pl.Series]

    assert type(q / 2) is Q[MassFlow, pl.Series]
    assert type(q / Q(2)) is Q[MassFlow, pl.Series]

    q2 = Q([1, 2, 3], "kg/s")

    assert type(q2) is Q[MassFlow, Numpy1DArray]
    assert type(q2 * 2) is Q[MassFlow, Numpy1DArray]
    assert type(2 * q2) is Q[MassFlow, Numpy1DArray]
    assert type(q2 * Q(2)) is Q[MassFlow, Numpy1DArray]
    assert type(Q(2) * q2) is Q[MassFlow, Numpy1DArray]

    assert type(q2 / 2) is Q[MassFlow, Numpy1DArray]
    assert type(q2 / Q(2)) is Q[MassFlow, Numpy1DArray]

    q3 = Q(pl.col.test, "kg/s")

    assert type(q3) is Q[MassFlow, pl.Expr]
    assert type(q3 * 2) is Q[MassFlow, pl.Expr]
    assert type(2 * q3) is Q[MassFlow, pl.Expr]
    assert type(q3 * Q(2)) is Q[MassFlow, pl.Expr]
    assert type(Q(2) * q3) is Q[MassFlow, pl.Expr]

    assert type(q3 / 2) is Q[MassFlow, pl.Expr]
    assert type(q3 / Q(2)) is Q[MassFlow, pl.Expr]

    q4 = Q(123, "kg/s")

    assert type(q4) is Q[MassFlow, float]
    assert type(q4 * 2) is Q[MassFlow, float]
    assert type(2 * q4) is Q[MassFlow, float]
    assert type(q4 * Q(2)) is Q[MassFlow, float]
    assert type(Q(2) * q4) is Q[MassFlow, float]

    assert type(q4 / 2) is Q[MassFlow, float]
    assert type(q4 / Q(2)) is Q[MassFlow, float]


def test_temperature_difference_add_sub() -> None:
    assert_type(Q(2, "degC") + Q(2, "delta_degC"), Q[Temperature, float])
    assert_type(Q([2, 3], "degC") + Q(2, "delta_degC"), Q[Temperature, Numpy1DArray])

    assert_type(Q([2, 3], "degC").astype(pl.Series) + Q(2, "delta_degC"), Q[Temperature, pl.Series])
    assert_type(Q(2, "delta_degC") + Q([2, 3], "degC").astype(pl.Series), Q[Temperature, pl.Series])

    assert_type(Q(2, "degC") - Q(2, "delta_degC"), Q[Temperature, float])
    assert_type(Q([2, 3], "degC") - Q(2, "delta_degC"), Q[Temperature, Numpy1DArray])

    assert_type(Q([2, 3], "degC").astype(pl.Series) - Q(2, "delta_degC"), Q[Temperature, pl.Series])

    # ΔT - T is not a temperature and is not defined (T ± ΔT and ΔT + T are)
    with pytest.raises(DimensionalityTypeError):
        _ = Q(2, "delta_degC") - cast(Any, Q([2, 3], "degC").astype(pl.Series))
    with pytest.raises(DimensionalityTypeError):
        _ = Q(2, "delta_degC") - cast(Any, Q(20, "degC"))


def test_dimensionless_add_sub() -> None:
    assert_type(45 + Q(2), Q[Dimensionless, float])
    assert_type(Q(2) + 45, Q[Dimensionless, float])
    assert_type(Q(45) + Q(2), Q[Dimensionless, float])

    assert_type(45 - Q(2), Q[Dimensionless, float])
    assert_type(Q(2) - 45, Q[Dimensionless, float])
    assert_type(Q(45) - Q(2), Q[Dimensionless, float])

    with pytest.raises(DimensionalityError):
        _ = 2 + Q(2, "kg")

    with pytest.raises(DimensionalityError):
        _ = 2 - Q(2, "kg")

    with pytest.raises(TypeError):
        _ = cast(Any, "asd") - Q(2, "kg")


def test_magnitude_type_name() -> None:
    assert Q(2).mt_name == "float"
    assert Q([2]).mt_name == "ndarray"
    assert Q(pl.lit(25)).mt_name == "pl.Expr"
    assert Q(pl.Series([1, 2, 3])).mt_name == "pl.Series"

    q = Q(25)
    assert q.mt is float
    assert q.mt_name == "float"
    assert cast(Any, q)._get_magnitude_type_name(q.mt) == q.mt_name


def test_astype_inference() -> None:
    assert_type(Q(25).astype("pl.Expr"), Q[Dimensionless, pl.Expr])
    assert_type(Q(25).astype("pl.Series"), Q[Dimensionless, pl.Series])
    assert_type(Q(25).astype("ndarray"), Q[Dimensionless, Numpy1DArray])
    assert_type(Q(25).astype("float"), Q[Dimensionless, float])

    assert_type(Q(25).astype(pl.Expr), Q[Dimensionless, pl.Expr])
    assert_type(Q(25).astype(pl.Series), Q[Dimensionless, pl.Series])
    assert_type(Q(25).astype(Numpy1DArray), Q[Dimensionless, Numpy1DArray])
    assert_type(Q(25).astype(float), Q[Dimensionless, float])

    assert_type(Q(25).astype(np.ndarray), Q[Dimensionless, Numpy1DArray])

    with pytest.raises(AssertionError):
        Q(25).astype(cast(Any, "invalid"))


def test_unary_pos_neg() -> None:
    assert -Q(-1) == 1

    assert (-Q(25, "kg")).mt_name == "float"
    assert isinstance_types(-Q(25, "kg"), Q[Mass, float])
    assert isinstance_types(-1 * -Q(25, "kg"), Q[Mass, float])
    assert isinstance_types(-Q(25, "kg") * 1, Q[Mass, float])
    assert isinstance_types(-Q(25, "kg") * -1, Q[Mass, float])
    assert isinstance_types(-(1 * -Q(25, "kg")), Q[Mass, float])

    assert +Q(+1) == 1

    assert (+Q(25, "kg")).mt_name == "float"
    assert isinstance_types(+Q(25, "kg"), Q[Mass, float])
    assert isinstance_types(+1 * +Q(25, "kg"), Q[Mass, float])
    assert isinstance_types(+Q(25, "kg") * 1, Q[Mass, float])
    assert isinstance_types(+Q(25, "kg") * +1, Q[Mass, float])
    assert isinstance_types(+(1 * +Q(25, "kg")), Q[Mass, float])


def test_dimensionless_division() -> None:
    assert_type(1 / Q(1), Q[Dimensionless, float])
    assert_type(1.0 / Q(1), Q[Dimensionless, float])
    assert_type(1 / Q([1, 2, 3]), Q[Dimensionless, Numpy1DArray])
    assert_type(1 / Q(pl.Series([1, 2, 3])), Q[Dimensionless, pl.Series])
    assert_type(1 / Q(pl.col.test), Q[Dimensionless, pl.Expr])


def test_massflow_energypermass_power_mul() -> None:
    mf = Q(10, "kg/s")
    epm = Q(1000, "kJ/kg")

    result = mf * epm
    assert isinstance_types(result, Q[Power])
    assert_type(mf * epm, Q[Power, float])

    # commutative
    result2 = epm * mf
    assert isinstance_types(result2, Q[Power])
    assert_type(epm * mf, Q[Power, float])

    # array magnitude types
    mf_arr = Q([10, 20], "kg/s")
    assert isinstance_types(mf_arr * epm, Q[Power])
    assert_type(mf_arr * epm, Q[Power, Numpy1DArray])

    mf_series = Q(pl.Series([10, 20]), "kg/s")
    assert isinstance_types(mf_series * epm, Q[Power])
    assert_type(mf_series * epm, Q[Power, pl.Series])

    mf_expr = Q(pl.col.test, "kg/s")
    assert_type(mf_expr * epm, Q[Power, pl.Expr])

    # commutative with array types
    epm_arr = Q([1000, 2000], "kJ/kg")
    assert isinstance_types(epm_arr * mf, Q[Power])
    assert_type(epm_arr * mf, Q[Power, Numpy1DArray])


def test_power_div_massflow_energypermass() -> None:
    p = Q(10000, "kW")
    mf = Q(10, "kg/s")
    epm = Q(1000, "kJ/kg")

    result1 = p / mf
    assert isinstance_types(result1, Q[EnergyPerMass])
    assert_type(p / mf, Q[EnergyPerMass, float])

    result2 = p / epm
    assert isinstance_types(result2, Q[MassFlow])
    assert_type(p / epm, Q[MassFlow, float])

    # array magnitude types
    p_arr = Q([10000, 20000], "kW")
    assert isinstance_types(p_arr / mf, Q[EnergyPerMass])
    assert_type(p_arr / mf, Q[EnergyPerMass, Numpy1DArray])

    assert isinstance_types(p_arr / epm, Q[MassFlow])
    assert_type(p_arr / epm, Q[MassFlow, Numpy1DArray])

    p_series = Q(pl.Series([10000, 20000]), "kW")
    assert isinstance_types(p_series / mf, Q[EnergyPerMass])
    assert_type(p_series / mf, Q[EnergyPerMass, pl.Series])

    assert isinstance_types(p_series / epm, Q[MassFlow])
    assert_type(p_series / epm, Q[MassFlow, pl.Series])

    p_expr = Q(pl.col.test, "kW")
    assert_type(p_expr / mf, Q[EnergyPerMass, pl.Expr])
    assert_type(p_expr / epm, Q[MassFlow, pl.Expr])
