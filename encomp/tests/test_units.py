# pyright: reportConstantRedefinition=false

import copy
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, TypedDict, assert_never, assert_type

import numpy as np
import polars as pl
import pytest
from pint.errors import OffsetUnitCalculusError
from pydantic import BaseModel, ConfigDict
from pytest import approx  # pyright: ignore[reportUnknownVariableType]
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
)
from ..units import Quantity as Q
from ..utypes import (
    DT,
    MT,
    Area,
    Dimensionality,
    Dimensionless,
    EnergyPerMass,
    Length,
    LowerHeatingValue,
    Mass,
    MassFlow,
    Normal,
    NormalVolume,
    NormalVolumeFlow,
    Numpy1DArray,
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


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_registry() -> None:
    from pint import (
        _DEFAULT_REGISTRY,  # pyright: ignore[reportUnknownVariableType, reportPrivateUsage]
        application_registry,
    )

    us: list[Any] = [UNIT_REGISTRY, _DEFAULT_REGISTRY, application_registry.get()]

    # check that all these objects are the same
    assert len(set(map(id, us))) == 1

    # check that units from all objects can be combined
    # NOTE: there is not typing for quantities created by this method
    q = 1 * UNIT_REGISTRY.kg / _DEFAULT_REGISTRY.s**2 / application_registry.get().m  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]
    assert isinstance_types(q, Q[Pressure, Any])

    # options cannot be overridden once set
    UNIT_REGISTRY.force_ndarray_like = True
    assert not UNIT_REGISTRY.force_ndarray_like


def test_define_dimensionality() -> None:
    assert "normal" in CUSTOM_DIMENSIONS

    with pytest.raises(DimensionalityRedefinitionError):
        define_dimensionality(CUSTOM_DIMENSIONS[0])


def test_format() -> None:
    assert "{:.2f~P}".format(Q(25, "delta_degC") / Q(255, "m3")) == "0.10 Δ°C/m³"


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

    _registry_orig = Dimensionality._registry  # pyright: ignore[reportPrivateUsage]
    _registry_reversed_orig = Dimensionality._registry_reversed  # pyright: ignore[reportPrivateUsage]

    Dimensionality._registry = utypes_reloaded.Dimensionality._registry  # pyright: ignore[reportPrivateUsage]
    Dimensionality._registry_reversed = utypes_reloaded.Dimensionality._registry_reversed  # pyright: ignore[reportPrivateUsage]

    try:
        yield

    finally:
        # reset to original registries
        # otherwise any code executed after this context manager
        # will have issues with isinstance() and issubclass()
        Dimensionality._registry = _registry_orig  # pyright: ignore[reportPrivateUsage]
        Dimensionality._registry_reversed = _registry_reversed_orig  # pyright: ignore[reportPrivateUsage]

        # clear existing mapping from dimensionality subclass name to Quantity subclass
        # this will be dynamically rebuilt
        Q._subclasses.clear()  # pyright: ignore[reportPrivateUsage]


def test_dimensionality_subtype_protocol() -> None:
    with _reset_dimensionality_registry():
        # the subclass is not checked, only the "dimensions" attribute
        # however, doing this causes a lot of type errors, so make sure to inherit from Dimensionality instead
        class Test:
            dimensions = Dimensionless.dimensions

        Q[Test]  # pyright: ignore[reportInvalidTypeArguments]
        Q[Test](1)  # pyright: ignore[reportArgumentType, reportCallIssue, reportInvalidTypeArguments]


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
            q1.asdim(Q(25, "kg"))


def test_custom_dimensionality() -> None:
    with _reset_dimensionality_registry():

        class Custom1(Dimensionality):
            dimensions = Temperature.dimensions**2 / Length.dimensions

        _custom = Custom1

        q1 = Q[Custom1](1, "degC**2/m")  # pyright: ignore[reportArgumentType, reportCallIssue]

        class Custom2(Dimensionality):  # pyright: ignore[reportRedeclaration]
            dimensions = Temperature.dimensions**2 / Length.dimensions

        # the classes are not identical
        assert Custom2 is not _custom
        assert Q[Custom2] is not Q[_custom]

        q2 = Q[Custom2](1, "degC**2/m")  # pyright: ignore[reportArgumentType, reportCallIssue]

        # the values and units are equivalent
        # but the dimensionality types don't match
        with pytest.raises(DimensionalityComparisonError):
            assert q1 == q2  # pyright: ignore[reportOperatorIssue]

        assert isinstance(q1.to("degC**2/km"), type(q1))
        assert isinstance(q1.to_base_units(), type(q1))
        assert isinstance(q1.to_reduced_units(), type(q1))

        with pytest.raises(TypeError):

            class Custom2(Dimensionality):
                # cannot create a duplicate (based on classname) dimensionality
                # with different dimensions
                dimensions = Temperature.dimensions**3 / Length.dimensions


def test_function_annotations() -> None:
    # this results in Quantity[Any]
    a = Q[DT]  # pyright: ignore[reportGeneralTypeIssues]

    def return_input(q: Q[DT, MT]) -> Q[DT, MT]:
        return q

    a = return_input(Q(25, "m"))

    assert isinstance_types(a, Q[Length])

    # not possible to determine output dimensionality for this
    def divide_by_time(q: Q[DT, MT]) -> Q[Any, MT]:
        return q / Q(1, "h")

    # this will be resolved to MassFlow at runtime
    assert isinstance_types(divide_by_time(Q(25, "kg")), Q[MassFlow])


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
        # (override the str literal unit overloads for mypy)
        s = Q[EstimatedLength, float](25, "m")
        m = Q[EstimatedMass, float](25, "kg")

        assert issubclass(s._dimensionality_type, Estimation)  # pyright: ignore[reportPrivateUsage]
        assert issubclass(m._dimensionality_type, Estimation)  # pyright: ignore[reportPrivateUsage]

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

        assert_type(s_arr, Q[Length, np.ndarray])
        assert_type(s_arr_first, Q[Length, float])

        s_series = Q(pl.Series([25]), "m")
        s_series_first = s_series[0]

        assert_type(s_series, Q[Length, pl.Series])
        assert_type(s_series_first, Q[Length, float])

        s_arr = Q([25], "m").asdim(EstimatedLength)
        s_arr_first = s_arr[0]

        assert_type(s_arr, Q[EstimatedLength, np.ndarray])
        assert_type(s_arr_first, Q[EstimatedLength, float])

        s_series = Q(pl.Series([25]), "m").asdim(EstimatedLength)
        s_series_first = s_series[0]

        assert_type(s_series, Q[EstimatedLength, pl.Series])
        assert_type(s_series_first, Q[EstimatedLength, float])

        # these quantities are not compatible with normal Length/Mass
        # TODO: use a more specific exception here
        with pytest.raises(Exception):  # noqa: B017
            _ = s + Q(25, "m")

        with pytest.raises(Exception):  # noqa: B017
            _ = m + Q(25, "kg")

        # EstimatedDistance is a direct subclass of EstimatedLength, so this works
        _ = Q[EstimatedDistance, float](2, "m") + s
        _ = s - Q[EstimatedDistance, float](2, "m")

        assert Q[EstimatedDistance](s.m, s.u) == s

        assert isinstance_types(Q[EstimatedDistance, float](2, "m") + s, Q[EstimatedDistance])
        assert isinstance_types(s - Q[EstimatedDistance, float](2, "m"), Q[EstimatedLength])


def test_type_eq() -> None:
    q = Q(25, "m")
    q_arr = Q([25], "m")

    assert isinstance(q, Q)
    assert isinstance_types(q, Q[Length, float])
    assert isinstance_types(q, Q[Length, Any])

    if isinstance(q, Q):  # pyright: ignore[reportUnnecessaryIsInstance]
        assert_type(q, Q[Length, float])
    else:
        assert_never(q)

    if isinstance_types(q, Q[Length, Any]):
        assert_type(q, Q[Length, float])
    else:
        assert_never(q)

    if isinstance_types(q, Q[Length, Numpy1DArray]):
        assert_never(q)

    if isinstance_types(q_arr, Q[Length]):
        assert_type(q_arr, Q[Length])
    else:
        assert_never(q_arr)

    if isinstance_types(q, Q[Length, float]):
        assert_type(q, Q[Length, float])
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

    with pytest.raises(DimensionalityComparisonError):
        _ = Q(2, "kg") == 2

    with pytest.raises(DimensionalityComparisonError):
        _ = 2 == Q(2, "kg")  # noqa: SIM300

    with pytest.raises(DimensionalityComparisonError):
        _ = Q(2, "kg") == Q(25, "m")  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]

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
        Q(Q(2, "feet_water"), Q(321321, "psi").u).to(Q(123123, "feet_water").asdim(Pressure)), Q[Pressure, float]
    )

    with pytest.raises(DimensionalityError):
        Q(Q(2, "feet_water"), Q(321321, "kg").u).to(Q(123123, "feet_water").asdim(Pressure))  # pyright: ignore[reportArgumentType]

    # the UnitsContainer objects can be used to construct new dimensionalities
    # NOTE: custom dimensionalities must have unique names

    class Custom(Dimensionality):
        dimensions = Length.dimensions * Length.dimensions * Length.dimensions / Temperature.dimensions

    Q[Custom, float](1, "m³/K")

    with pytest.raises(Exception):  # noqa: B017
        Q[Pressure / Area, float](1, "bar/m")  # pyright: ignore[reportOperatorIssue]

    # percent or %
    Q(1.124124e-3, "").to("%").to("percent")
    Q(1.124124e-3).to("%").to("percent")

    # np.ndarray magnitudes equality check
    assert (Q([1, 2, 3], "kg") == Q([1000, 2000, 3000], "g")).all()
    assert not (Q([1, 2, 3], "kg") == Q([1000, 2000, 3001], "g")).all()

    with pytest.raises(DimensionalityComparisonError):
        (Q([1, 2, 3], "kg") == Q([1000, 2000, 300], "g * meter")).any()

    vals = [1, 2, 3]

    # compare scalar and vector will return a vector
    assert (Q(2, "bar") == Q(vals, "bar").to("kPa")).any()

    assert not (Q(5, "bar") == Q(vals, "bar").to("kPa")).any()


def test_custom_units() -> None:
    assert_type(Q(1, "kilogram"), Q[UnknownDimensionality, float])
    assert Q(1, "kg") == Q(1, "kilogram")

    with pytest.raises(DimensionalityComparisonError):
        assert Q(1, "kg") == Q(1, "m")  # pyright: ignore[reportOperatorIssue]

    with pytest.raises(DimensionalityComparisonError):
        assert Q(1, "kilogram") == Q(1, "m")

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

    @UNIT_REGISTRY.wraps("kg", ("m", "kg"), strict=True)  # pyright: ignore[reportUnknownMemberType]
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


def test_check() -> None:
    assert Q(25, "kg").check(Mass)

    assert Q(25, "kg").check(Q(25, "g"))
    assert Q(25, "kg").check(Q([25, 25], "g"))


def test_typechecked() -> None:
    @typechecked
    def func_a(a: Quantity[Temperature, Any]) -> Quantity[Pressure, Any]:  # noqa: ARG001
        return Q(2, "bar")

    assert func_a(Q(2, "degC")) == Q(2, "bar")

    with pytest.raises(Exception):  # noqa: B017
        func_a(Q(2, "meter"))  # pyright: ignore[reportArgumentType]

    @typechecked
    def func_b(a: Quantity[Pressure, Any]) -> Quantity[Pressure, Any]:
        return a

    assert func_b(Q(2, "bar")) == Q(2, "bar")
    assert func_b(Q(2, "psi")) == Q(2, "psi")
    assert func_b(Q(2, "mmHg")) == Q(2, "mmHg")

    with pytest.raises(Exception):  # noqa: B017
        func_a(Q(2, "meter"))  # pyright: ignore[reportArgumentType]

    @typechecked
    def func_c(a: Quantity[Temperature]) -> Quantity[Pressure]:  # noqa: ARG001
        return Q([2], "bar")

    assert func_c(Q([2], "degC")) == Q([2], "bar")

    func_c(Q(2, "degC"))  # pyright: ignore[reportArgumentType]

    with pytest.raises(Exception):  # noqa: B017
        func_c(Q([2], "meter"))  # pyright: ignore[reportArgumentType]


def test_generic_dimensionality() -> None:
    assert issubclass(Q[Pressure], Q)
    assert not issubclass(Q[Pressure], Q[Temperature])  # pyright: ignore[reportArgumentType]

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
        Q[1]  # pyright: ignore[reportInvalidTypeArguments]

    with pytest.raises(TypeError):
        Q["Temperature"]

    with pytest.raises(TypeError):
        Q["string"]  # pyright: ignore[reportUndefinedVariable]

    with pytest.raises(TypeError):
        Q[None]  # pyright: ignore[reportInvalidTypeArguments]

    with pytest.raises(TypeError):
        Q[None, None]  # pyright: ignore[reportInvalidTypeArguments]

    with pytest.raises(TypeError):
        Q[Any, Any, Any]  # pyright: ignore[reportInvalidTypeArguments]

    Q[DT]  # pyright: ignore[reportGeneralTypeIssues]
    Q[DT, MT]  # pyright: ignore[reportGeneralTypeIssues]
    Q[DT, Any]  # pyright: ignore[reportGeneralTypeIssues]
    Q[Any, MT]  # pyright: ignore[reportGeneralTypeIssues]


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
        isinstance(d, PTxDict1)  # pyright: ignore[reportArgumentType]

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
        _ = Q(25, "kg") + Q(2, "m")  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]

    with pytest.raises(DimensionalityTypeError):
        _ = Q(25, "kg") - Q(2, "m")  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]

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

    # this is not allowed on the type level, but works at runtime
    assert q5 == q5_  # pyright: ignore[reportOperatorIssue]

    q6 = Q[EnergyPerMass, float](25, "kJ/kg")

    with pytest.raises(Exception):  # noqa: B017
        _ = q4 - q5

    with pytest.raises(Exception):  # noqa: B017
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
            assert Q(1, u)._dimensionality_type.__name__ == d  # pyright: ignore[reportPrivateUsage]


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

    assert isinstance_types(UNIT_REGISTRY.m * 1, Q[Length])
    assert isinstance_types(1 * UNIT_REGISTRY.m / UNIT_REGISTRY.s, Q[Velocity])
    assert isinstance_types([1, 2, 3] * UNIT_REGISTRY.m / UNIT_REGISTRY.s, Q[Velocity])
    assert isinstance_types((1, 2, 3) * UNIT_REGISTRY.m / UNIT_REGISTRY.s, Q[Velocity])
    # assert isinstance_types(np.array([1, 2, 3]) * UNIT_REGISTRY.m / UNIT_REGISTRY.s, Q[Velocity])
    assert isinstance_types([1, 2, 3] * UNIT_REGISTRY.m / UNIT_REGISTRY.s, Q[Velocity])


def test_mul_rmul_initialization() -> None:
    assert isinstance_types(UNIT_REGISTRY.m * np.array([1, 2]), Q[Length])
    # this returns array([<Quantity(1.0, 'meter')>, <Quantity(2.0, 'meter')>], dtype=object) instead
    # assert isinstance_types(np.array([1, 2]) * UNIT_REGISTRY.m, Q[Length])
    assert isinstance_types([1, 2] * UNIT_REGISTRY.m, Q[Length])
    assert isinstance_types([1, 2] * Q(1, "m"), Q[Length])  # pyright: ignore[reportOperatorIssue]
    assert isinstance_types(np.array([1, 2]) * Q(1, "m"), Q[Length])


def test_copy() -> None:
    q = Q(25, "m")

    assert isinstance_types(q, Q[Length])
    assert isinstance_types(q.__copy__(), Q[Length])

    assert isinstance_types(q.__deepcopy__(), Q[Length])
    assert isinstance_types(q.__deepcopy__({}), Q[Length])

    assert isinstance_types(copy.copy(q), Q[Length])
    assert isinstance_types(copy.deepcopy(q), Q[Length])


def test_pydantic_integration() -> None:
    class Model(BaseModel):
        # a can be any dimensionality
        a: Q[UnknownDimensionality, float]

        m: Q[Mass, float]
        s: Q[Length, float]

        # float can be converted to Quantity[Dimensionless]
        r: Q[Dimensionless, float] = 0.5  # pyright: ignore[reportAssignmentType]

        # float cannot be converted to Quantity[Length]
        # d: Q[Length] = 0.5

        model_config = ConfigDict(validate_default=True)

    Model(a=Q(25, "cSt").asdim(UnknownDimensionality), m=Q(25, "kg"), s=Q(25, "cm"))

    with pytest.raises(ExpectedDimensionalityError):
        Model(
            a=Q(25, "cSt").asdim(UnknownDimensionality),
            m=Q(25, "kg/day").asdim(MassFlow),  # pyright: ignore[reportArgumentType]
            s=Q(25, "cm"),
        )


def test_float_cast() -> None:
    assert isinstance(Q([False, False]).m[0], float)

    assert (Q([False, True]) == Q(np.array([False, True]))).all()


def test_temperature_difference() -> None:
    T0 = Q(25, "K").to("delta_degC")
    assert isinstance_types(T0, Q[TemperatureDifference])

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

    with pytest.raises(DimensionalityTypeError):
        (T1.to("K") - T2.to("K")).to("K")

    with pytest.raises(DimensionalityTypeError):
        (T1 - T2).to("K")

    with pytest.raises(DimensionalityTypeError):
        (T1 - T2).to("degC")

    T1 = Q(800, "degC")
    T2 = T1.to("K") - Q(100, "degC").to("K")

    with pytest.raises(DimensionalityTypeError):
        T2.to("K")

    T2_ = T1.to("K") - Q(100, "delta_degC")

    assert isinstance_types(T2_, Q[Temperature])


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
        assert qty.check(Temperature) or qty.check(TemperatureDifference)
        assert not (qty.check(Temperature) and qty.check(TemperatureDifference))

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
        Q[Length, str]  # pyright: ignore[reportInvalidTypeArguments]

    with pytest.raises(TypeError):
        Q[None]  # pyright: ignore[reportInvalidTypeArguments]

    with pytest.raises(TypeError):
        Q[None, float]  # pyright: ignore[reportInvalidTypeArguments]


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

    assert not Q(1, "delta_degC").check(Q(12, "degC"))
    assert not Q(1, "delta_degC").check(Q(12, "degC").u)

    assert Q(1, "delta_degC").check(Q(12, "delta_degC"))
    assert Q(1, "delta_degC").check(Q(12, "delta_degC").u)

    assert not Q(1, "delta_degC").check(Q(12, "degC"))
    assert not Q(1, "delta_degC").check(Q(12, "degC").u)


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
    assert_type(Q(2, "delta_degC") - Q([2, 3], "degC").astype(pl.Series), Q[Temperature, pl.Series])


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
        _ = "asd" - Q(2, "kg")  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]


def test_magnitude_type_name() -> None:
    assert Q(2).mt_name == "float"
    assert Q([2]).mt_name == "ndarray"
    assert Q(pl.lit(25)).mt_name == "pl.Expr"
    assert Q(pl.Series([1, 2, 3])).mt_name == "pl.Series"

    q = Q(25)
    assert q.mt is float
    assert q.mt_name == "float"
    assert q._get_magnitude_type_name(q.mt) == q.mt_name  # pyright: ignore[reportPrivateUsage]


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
        Q(25).astype("invalid")  # pyright: ignore[reportArgumentType, reportCallIssue]


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
