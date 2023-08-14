import copy
from contextlib import contextmanager
from decimal import Decimal
from typing import TypedDict

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.api.types import is_list_like as pandas_is_list_like  # type: ignore
from pint.errors import OffsetUnitCalculusError
from pydantic import BaseModel, ConfigDict
from pytest import approx
from typeguard import typechecked

from ..conversion import convert_volume_mass
from ..fluids import Water
from ..misc import isinstance_types
from ..serialize import decode
from ..units import (
    CUSTOM_DIMENSIONS,
    DimensionalityError,
    DimensionalityRedefinitionError,
    DimensionalityTypeError,
    ExpectedDimensionalityError,
    Quantity,
    Unit,
    define_dimensionality,
    ureg,
)
from ..units import Quantity as Q
from ..utypes import (
    DT,
    DT_,
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
    Pressure,
    SpecificEnthalpy,
    Temperature,
    TemperatureDifference,
    UnitsContainer,
    Unknown,
    Unset,
    Variable,
    Velocity,
    Volume,
    VolumeFlow,
    get_registered_units,
)


def test_registry():
    from pint import _DEFAULT_REGISTRY, application_registry

    us = [ureg, _DEFAULT_REGISTRY, application_registry.get()]

    # check that all these objects are the same
    assert len(set(map(id, us))) == 1

    # check that units from all objects can be combined
    q = 1 * ureg.kg / _DEFAULT_REGISTRY.s**2 / application_registry.get().m
    assert isinstance(q, Q[Pressure])

    # options cannot be overridden once set
    ureg.force_ndarray_like = True
    assert not ureg.force_ndarray_like


def test_define_dimensionality():
    assert "normal" in CUSTOM_DIMENSIONS

    with pytest.raises(DimensionalityRedefinitionError):
        define_dimensionality(CUSTOM_DIMENSIONS[0])


def test_format():
    assert "{:.2f~P}".format(Q(25, "delta_degC") / Q(255, "m3")) == "0.10 Δ°C/m³"


@contextmanager
def _reset_dimensionality_registry():
    # NOTE: this is a hack, only use this for tests

    import importlib

    from encomp import utypes

    # this does not completely reload the module,
    # since there are multiple references to encomp.utypes
    utypes_reloaded = importlib.reload(utypes)

    # this is a new class definition...
    assert Dimensionality is not utypes_reloaded.Dimensionality

    # explicitly replace the registry dicts on the version of
    # Dimensionality that was loaded on module-level in this test module

    _registry_orig = Dimensionality._registry
    _registry_reversed_orig = Dimensionality._registry_reversed

    Dimensionality._registry = utypes_reloaded.Dimensionality._registry
    Dimensionality._registry_reversed = (
        utypes_reloaded.Dimensionality._registry_reversed
    )

    try:
        yield

    finally:
        # reset to original registries
        # otherwise any code executed after this context manager
        # will have issues with isinstance() and issubclass()
        Dimensionality._registry = _registry_orig
        Dimensionality._registry_reversed = _registry_reversed_orig

        # clear existing mapping from dimensionality subclass name to Quantity subclass
        # this will be dynamically rebuilt
        Q._subclasses.clear()


def test_dimensionality_subtype_protocol():
    with _reset_dimensionality_registry():
        # the subclass is not checked, only the "dimensions" attribute
        class Test:
            dimensions = Dimensionless.dimensions

        Q[Test]
        Q[Test](1)


def test_asdim():
    with _reset_dimensionality_registry():
        # default dimensionality for kJ/kg is EnergyPerMass
        q1 = Q(15, "kJ/kg")
        q2 = Q(15, "kJ/kg").asdim(LowerHeatingValue)

        assert type(q1) is not type(q2)
        assert q1 != q2

        assert type(q1) is type(q2.asdim(EnergyPerMass))  # noqa: E721
        assert type(q2) is type(q1.asdim(LowerHeatingValue))  # noqa: E721

        assert type(q1) is type(q2.asdim(q1))  # noqa: E721
        assert type(q2) is type(q1.asdim(q2))  # noqa: E721

        assert q1 == q2.asdim(EnergyPerMass)
        assert q2 == q1.asdim(LowerHeatingValue)

        assert q1 == q2.asdim(q1)
        assert q2 == q1.asdim(q2)

        # TODO: this does not currently work

        # with pytest.raises(ExpectedDimensionalityError):
        #     q1.asdim(Temperature)

        # with pytest.raises(ExpectedDimensionalityError):
        #     q1.asdim(Q(25, 'kg'))


def test_custom_dimensionality():
    with _reset_dimensionality_registry():

        class Custom(Dimensionality):
            dimensions = Temperature.dimensions**2 / Length.dimensions

        _custom = Custom

        q1 = Q[Custom](1, "degC**2/m")

        class Custom(Dimensionality):
            dimensions = Temperature.dimensions**2 / Length.dimensions

        # the classes are not identical
        assert Custom is not _custom

        # the Q[DT] type only considers the class name,
        # which is identical
        assert Q[Custom] is Q[_custom]

        q2 = Q[Custom](1, "degC**2/m")

        # the values and units are equivalent
        # but the dimensionality types don't match
        assert q1 == q2

        assert type(q1) == type(q2) == Q[Custom]
        assert isinstance(q1, type(q2))
        assert isinstance(q2, type(q1))
        assert isinstance(q1 + q2, type(q1))
        assert isinstance(q2 - q1, type(q1))

        assert isinstance(q1.to("degC**2/km"), type(q1))
        assert isinstance(q1.to_base_units(), type(q1))
        assert isinstance(q1.to_reduced_units(), type(q1))

        with pytest.raises(TypeError):

            class Custom(Dimensionality):
                # cannot create a duplicate (based on classname) dimensionality
                # with different dimensions
                dimensions = Temperature.dimensions**3 / Length.dimensions


def test_function_annotations():
    # this results in Quantity[Variable]
    a = Q[DT]

    # Quantity[Variable] works as type annotations at runtime

    def return_input(q: Q[DT]) -> Q[DT]:
        return q

    a = return_input(Q(25, "m"))

    assert isinstance(a, Q[Length])

    # not possible to determine output dimensionality for this
    def divide_by_time(q: Q[DT]) -> Q:
        return q / Q(1, "h")

    # this will be resolved to MassFlow at runtime
    assert isinstance(divide_by_time(Q(25, "kg")), Q[MassFlow])


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
        EstimatedQuantity = Q[Estimation]
        EstimatedQuantity

        # TODO: this does not currently work
        # with pytest.raises(TypeError):
        #     EstimatedQuantity(25, 'm')

        # these quantities are not compatible with normal Length/Mass
        # (override the str literal unit overloads for mypy)
        s = Q[EstimatedLength](25, str("m"))
        m = Q[EstimatedMass](25, str("kg"))

        assert issubclass(s._dimensionality_type, Estimation)
        assert issubclass(m._dimensionality_type, Estimation)

        # the dimensionality type is preserved for add, sub and
        # mul, div with scalars (not with Q[Dimensionless])

        assert isinstance(s, Q[EstimatedLength])
        assert isinstance(s * 2, Q[EstimatedLength])
        assert isinstance(2 * s, Q[EstimatedLength])

        assert isinstance(s / 2, Q[EstimatedLength])

        # inverted dimensionality 1/Length
        assert not isinstance(2 / s, Q[EstimatedLength])

        assert isinstance(s + s, Q[EstimatedLength])
        assert isinstance(s - s, Q[EstimatedLength])
        assert isinstance(s - s * 2, Q[EstimatedLength])
        assert isinstance(s + s / 2, Q[EstimatedLength])

        assert isinstance(2 * s + s, Q[EstimatedLength])
        assert isinstance(2 * s - s / 2, Q[EstimatedLength])

        assert isinstance(Q(1) * s, Q[EstimatedLength])
        assert isinstance(s * Q(1), Q[EstimatedLength])

        # these quantities are not compatible with normal Length/Mass
        # TODO: use a more specific exception here
        with pytest.raises(Exception):
            s + Q(25, "m")

        with pytest.raises(Exception):
            m + Q(25, "kg")

        # EstimatedDistance is a direct subclass of EstimatedLength, so this works
        Q[EstimatedDistance](2, "m") + s
        s - Q[EstimatedDistance](2, "m")

        assert Q[EstimatedDistance](s.m, s.u) == s

        # however, the type will be determined by the first object
        assert isinstance(Q[EstimatedDistance](2, "m") + s, Q[EstimatedDistance])

        assert isinstance(s - Q[EstimatedDistance](2, "m"), Q[EstimatedLength])


def test_type_eq():
    q = Q(25, "m")

    # this is the recommended way of checking type
    assert isinstance(q, Q)

    # this is overloaded to work for the Quantity base class
    # for compatibility with other libraries

    assert type(q) == Q
    assert Q == type(q)

    assert type(Q(2)) == Q
    assert Q == type(Q(25, "bar"))  # noqa: E721

    # __eq__ is overloaded, but these are still different types
    assert type(q) is not Q

    # subclasses behave as expected

    assert type(q) == Q[Length, float]
    assert Q[Length, float] == type(q)

    assert not type(q) == Q[Dimensionless, float]
    assert not Q[Dimensionless, float] == type(q)

    assert not type(q) == Q[Length]
    assert not Q[Length] == type(q)


def test_Q():
    # test that Quantity objects can be constructed
    Q(1, "dimensionless")
    Q(1, "kg")
    Q(1, "bar")
    Q(1, "h")
    Q(1, "newton")
    Q(1, "cSt")

    assert Q(1, "meter/kilometer").to_reduced_units().m == 0.001
    assert Q(1, "km").to_base_units().m == 1000

    # make sure that the alias Q behaves identically to Quantity
    assert Q(1) == Quantity(1)
    assert type(Q(1)) is type(Quantity(1))  # noqa: E721
    assert type(Q) is type(Quantity)

    # inputs can be nested
    Q(Q(1, "kg"))

    mass = Q(12, "kg")

    Q(Q(Q(Q(mass))))

    # mixing Quantity and unit input is not allowed

    with pytest.raises(ValueError):
        Q(Q(Q(Q(mass), "lbs")))

    with pytest.raises(ValueError):
        Q(Q(Q(Q(mass), "lbs")), "stone")

    # no unit input defaults to dimensionless
    assert Q(12).check("")
    assert Q(1) == Q(100, "%")
    Q[Dimensionless](21)
    assert isinstance(Q(21), Q[Dimensionless])

    assert Q(1) == Q(1.0)

    # check type of "m"
    # inputs are converted to float
    assert isinstance(Q(1, "meter").m, float)
    assert isinstance(Q(2.3, "meter").m, float)
    assert isinstance(Q([2, 3.4], "meter").m, np.ndarray)
    assert isinstance(Q((2, 3.4), "meter").m, np.ndarray)
    assert isinstance(Q(np.array([2, 3.4]), "meter").m, np.ndarray)

    Q(1, Q(2, "bar").u)
    Q(Q(2, "bar").to("kPa").m, "kPa")

    # TODO: these do not currently work
    # # check that the dimensionality constraints work
    # Q[Length](1, 'm')
    # Q[Pressure](1, 'kPa')
    # Q[Temperature](1, '°C')

    # # the dimensionalities can also be specified as strings
    # Q[Temperature](1, '°C')

    # P = Q(1, 'bar')
    # # this Quantity must have the same dimensionality as P
    # Q(2, 'kPa').check(P)

    # with pytest.raises(ExpectedDimensionalityError):
    #     Q[Temperature](1, 'kg')

    # with pytest.raises(ExpectedDimensionalityError):
    #     Q[Pressure](1, 'meter')

    # with pytest.raises(ExpectedDimensionalityError):
    #     Q[Mass](1, P)

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
    assert isinstance(P2, Q[Pressure])

    with pytest.raises(Exception):
        # incorrect dimensionalities should raise Exception
        Q(Q(2, "feet_water"), Q(321321, "kg")).to(Q(123123, "feet_water"))

    # the UnitsContainer objects can be used to construct new dimensionalities
    # NOTE: custom dimensionalities must have unique names

    class Custom(Dimensionality):
        dimensions = (
            Length.dimensions
            * Length.dimensions
            * Length.dimensions
            / Temperature.dimensions
        )

    Q[Custom](1, "m³/K")

    with pytest.raises(Exception):
        Q[Pressure / Area](1, "bar/m")

    # percent or %
    Q(1.124124e-3, "").to("%").to("percent")
    Q(1.124124e-3).to("%").to("percent")

    vals = [2, 3, 4]
    s = pd.Series(vals, name="Pressure")
    arr = Q(s, "bar").to("kPa").m
    assert isinstance(arr, pd.Series)
    assert arr[0] == 200

    # np.ndarray magnitudes equality check
    assert (Q(s, "bar") == Q(vals, "bar").to("kPa")).all()
    assert (Q([1, 2, 3], "kg") == Q([1000, 2000, 3000], "g")).all()
    assert not (Q([1, 2, 3], "kg") == Q([1000, 2000, 3001], "g")).all()
    assert not (Q([1, 2, 3], "kg") == Q([1000, 2000, 300], "g * meter")).any()

    # compare scalar and vector will return a vector
    assert (Q(2, "bar") == Q(vals, "bar").to("kPa")).any()

    assert not (Q(5, "bar") == Q(vals, "bar").to("kPa")).any()


def test_custom_units():
    # "ton" should always default to metric ton
    assert (
        Q(1, "ton")
        == Q(1, "Ton")
        == Q(1, "TON")
        == Q(1, "tonne")
        == Q(1, "metric_ton")
        == Q(1000, "kg")
    )

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

    Q[NormalVolume](2, "nm**3")

    # with pytest.raises(ExpectedDimensionalityError):
    #     Q[NormalVolumeFlow](2, 'm**3/hour')

    Q[NormalVolumeFlow](2, "Nm**3/hour").to("normal liter/sec")

    class _NormalVolumeFlow(NormalVolumeFlow):
        dimensions = Normal.dimensions * VolumeFlow.dimensions

    Q[_NormalVolumeFlow](2, "Nm**3/hour").to("normal liter/sec")

    Q(2, "normal liter air / day")
    Q(2, "1/Nm3").to("1 / (liter normal)")


def test_wraps():
    # @ureg.wraps(ret, args, strict=True|False) is a convenience
    # decorator for making the input/output of a function into Quantity
    # however, it does not enforce the return value

    @ureg.wraps("kg", ("m", "kg"), strict=True)
    def func(a, b):
        # this is incorrect, cannot add 1 to a dimensional Quantity
        return a * b**2 + 1

    assert isinstance(func(Q(1, "yd"), Q(20, "lbs")), Q[Mass])
    assert Q(1, "bar").check(Pressure)


def test_numpy_integration():
    assert isinstance(Q(np.linspace(0, 1), "kg"), Q[Mass])
    assert isinstance(np.linspace(Q(0, "kg"), Q(1, "kg")), Q[Mass])

    assert isinstance(np.linspace(Q(2), Q(25, "%")), Q[Dimensionless])
    assert isinstance(np.linspace(Q(2, "cm"), Q(25, "km")), Q[Length])

    with pytest.raises(DimensionalityError):
        np.linspace(Q(2, "kg"), Q(25, "m"))

    assert (Q(np.linspace(0, 1), "kg") == np.linspace(Q(0, "kg"), Q(1, "kg"))).all()

    assert isinstance_types(list(Q(np.linspace(0, 1), "degC")), list[Q[Temperature]])


def test_series_integration():
    # indirectly support Polars via "to_numpy method"

    s_pd = pd.Series([1, 2, 3], name="name")

    # the "name" attribute it lost when creating a Quantity
    qty = Q(s_pd, "kg")
    assert qty.to("g")[0] == Q(1000, "g")

    # ignore these tests if Polars is not installed
    try:
        import polars as pl
    except ImportError:
        return

    s_pl = pl.Series("name", [1, 2, 3])

    qty = Q(s_pl, "kg")

    assert qty.to("g")[0] == Q(1000, "g")


def test_check():
    pass

    # TODO: the ureg.check decorator does not work since
    # it uses nested Quantity inputs

    # assert not Q(1, 'kg').check('[energy]')
    # assert Q(1, 'kg').check(Mass)
    # assert not Q(1, 'kg').check(Energy)

    # @ureg.check('[length]', '[mass]')
    # def func(a, b):
    #     return a * b

    # func(Q(1, 'yd'), Q(20, 'lbs'))


def test_typechecked():
    @typechecked
    def func_a(a: Quantity[Temperature]) -> Quantity[Pressure]:
        return Q(2, "bar")

    assert func_a(Q(2, "degC")) == Q(2, "bar")

    with pytest.raises(Exception):
        func_a(Q(2, "meter"))

    @typechecked
    def func_b(a: Quantity) -> Quantity[Pressure]:
        return a

    assert func_b(Q(2, "bar")) == Q(2, "bar")
    assert func_b(Q(2, "psi")) == Q(2, "psi")
    assert func_b(Q(2, "mmHg")) == Q(2, "mmHg")

    with pytest.raises(Exception):
        func_a(Q(2, "meter"))


def test_dataframe_assign():
    df_multiple_rows = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1, 2, 3],
        }
    )

    df_single_row = pd.DataFrame(
        {
            "A": [1],
            "B": [1],
        }
    )

    df_empty = pd.DataFrame(
        {
            "A": [],
            "B": [],
        }
    )

    for df in [df_multiple_rows, df_single_row, df_empty]:
        df["C"] = Q(df.A, "bar") * Q(25, "meter")

        df["Temp"] = Q(df.A, "degC")

        with pytest.raises(AttributeError):
            density = Water(
                # this is pd.Series[float]
                T=df.Temp.to_numpy(),
                Q=Q(0.5),
            ).D

        density = Water(T=Q(df.Temp.to_numpy(), "degC"), Q=Q(0.5)).D

        # implicitly strips magnitude in whatever unit the Quantity happens to have
        df["density"] = density

        # assigns a column with specific unit
        df["density_with_unit"] = density.to("kg/m3")

        # the .m accessor is not necessary for vectors
        df["density_with_unit_magnitude"] = density.to("kg/m3").m

        # this does not work -- pandas function is_list_like(Q(4, 'bar')) -> True
        # which means that this fails internally in pandas
        # ValueError: Length of values (1) does not match length of index (3)
        # df['D'] = Q(4, 'bar')

        # i.e. the .m accessor must be used for scalar Quantity assignment

        df["E"] = Q(df.A, "bar").m
        df["F"] = Q(4, "bar").m


def test_generic_dimensionality():
    assert issubclass(Q[Pressure], Q)
    assert not issubclass(Q[Pressure], Q[Temperature])

    assert Q[Pressure] is Q[Pressure]
    assert Q[Pressure] == Q[Pressure]

    assert Q[Pressure] is not Q[Temperature]
    assert Q[Pressure] != Q[Temperature]

    assert isinstance_types([Q, Q[Pressure], Q[Temperature]], list[type[Q]])

    assert not isinstance_types([Q, Q[Pressure], Q[Temperature]], list[Q])

    assert isinstance_types([Q[Pressure], Q[Pressure]], list[type[Q[Pressure]]])

    with pytest.raises(TypeError):
        Q[1]

    with pytest.raises(TypeError):
        Q["Temperature"]

    with pytest.raises(TypeError):
        Q["string"]

    with pytest.raises(TypeError):
        Q[None]

    Q[Unknown]
    Q[Unset]
    Q[Variable]
    Q[DT]
    Q[DT_]


def test_dynamic_dimensionalities():
    # this will create a new dimensionality type
    q1 = Q(1, "kg^2/K^4")

    # this will reuse the previously created one
    q2 = Q(25, "g*g/(K^2*K^2)")

    assert type(q1) is type(q2)

    # NOTE: don't use __class__ when checking this, always use type()
    assert isinstance(q1, type(q2))
    assert isinstance(q2, type(q1))
    assert isinstance(q2, Q)
    assert not isinstance(q2, Q[Pressure])

    q3 = Q(25, "m*g*g/(K^2*K^2)")

    assert type(q3) is not type(q2)
    assert not isinstance(q3, type(q1))


def test_instance_checks():
    assert isinstance(Q(2), Q[Dimensionless])
    assert isinstance(Q(2), Q)

    class Fraction(Dimensionless):
        pass

    # Q(2) will default to Dimensionless, since that was
    # defined before Fraction
    assert not isinstance(Q(2), Q[Fraction])
    assert isinstance(Q(2), Q[Dimensionless])

    # Q[Fraction] will override the subclass
    assert isinstance(Q[Fraction](2), Q[Fraction])

    assert isinstance(Q(25, "bar"), Q[Pressure])

    assert not isinstance(Q(25, "°C"), Q[Pressure])

    assert isinstance(Q(25, "°C"), Q[Temperature])

    assert isinstance(Q(25, "kg"), Q[Mass])

    assert isinstance(Q(25, "kg"), Q)

    assert isinstance_types(Q(25, "°C"), Q)
    assert isinstance_types(Q(25, "°C"), Q[Temperature])

    assert isinstance_types([Q(25, "°C")], list[Q[Temperature]])
    assert isinstance_types(
        [Q[Temperature](25, "°C"), Q(25, "°F")], list[Q[Temperature]]
    )
    assert isinstance_types([Q(25, "°C"), Q(25, "bar")], list[Q])

    # NOTE: the name CustomDimensionality must be globally unique

    class CustomDimensionality(Dimensionality):
        dimensions = UnitsContainer({"[mass]": 1, "[temperature]": -2})

    # Q(25, 'g/K^2') will find the correct subclass since there
    # are no other dimensionalities with these dimensions
    assert isinstance(Q(25, "g/K^2"), Q[CustomDimensionality])

    assert isinstance(Q[CustomDimensionality](25, "g/K^2"), Q[CustomDimensionality])

    assert isinstance(Q[CustomDimensionality](25, "g/K^2"), Q)

    assert isinstance_types({Q(25, "g/K^2")}, set[Q[CustomDimensionality]])

    assert not isinstance_types(
        (Q(25, "m*g/K^2"),), tuple[Q[CustomDimensionality], ...]
    )

    assert not isinstance_types([Q(25, "°C")], list[Q[Pressure]])

    assert isinstance_types(
        [Q[Temperature](25, "°C"), Q(25, "°F")], list[Q[Temperature]]
    )

    assert not isinstance_types([Q(25, "F/day")], list[Q[Temperature]])

    assert not isinstance_types(
        [Q(25, "°C"), Q(25, "bar")], list[Q[CustomDimensionality]]
    )


def test_typed_dict():
    d = {
        "P": Q[Pressure](25, "bar"),
        "T": Q[Temperature](25, "degC"),
        "x": Q[Dimensionless](0.5),
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    # cannot use isinstance with complex types
    with pytest.raises(TypeError):
        isinstance(d, PTxDict)

    assert isinstance_types(d, PTxDict)

    d = {
        "P": Q[Pressure](25, "bar"),
        "T": Q[Temperature](25, "degC"),
        "x": Q[Dimensionless](0.5),
        "extra": Q,
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    assert not isinstance_types(d, PTxDict)

    d = {
        "P": Q[Pressure](25, "bar"),
        "T": Q[Temperature](25, "degC"),
        "x": Q[Dimensionless](0.5),
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]
        missing: Q

    assert not isinstance_types(d, PTxDict)

    d = {
        "P": Q(25, "bar"),
        "T": Q(25, "degC"),
        "x": Q(0.5),
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    assert isinstance_types(d, PTxDict)

    d = {
        "P": Q(25, "bar"),
        "T": Q(25, "meter"),
        "x": Q(0.5),
    }

    class PTxDict(TypedDict):
        P: Q[Pressure]
        T: Q[Temperature]
        x: Q[Dimensionless]

    assert not isinstance_types(d, PTxDict)


def test_convert_volume_mass():
    V = convert_volume_mass(Q(125, "kg/s"), Q(25, "kg/m**3"))

    assert V.check(VolumeFlow)

    m = convert_volume_mass(Q(125, "liter/day"))

    assert m.check(MassFlow)


def test_compatibility():
    Q(1) + Q(2)

    Q(1, "%") - Q(2)

    Q(25, "m") + Q(25, "cm")

    # this subtype inherits from Dimensionless, which
    # means that it is compatible (can be added and subtracted)

    class Fraction(Dimensionless):
        pass

    q1 = Q(2)
    q2 = Q[Fraction](0.6)

    q1 + q2
    q2 + q1

    q1 - q2
    q2 - q1

    # this is a new dimensionality that happens to have
    # the same dimensions as Dimensionless
    # however, it is not compatible with Dimensionless or any of its subclasses

    class IncompatibleFraction(Dimensionality):
        dimensions = Dimensionless.dimensions

    q3 = Q[IncompatibleFraction](0.2)

    # TODO: use more specific exception here

    with pytest.raises(Exception):
        q3 + q1

    with pytest.raises(Exception):
        q3 + q2

    with pytest.raises(Exception):
        q1 + q3

    with pytest.raises(Exception):
        q2 + q3

    s = Q(25, "m")

    # this dimensionality means something specific,
    # it cannot be added to a normal length
    class DistanceAlongPath(Dimensionality):
        dimensions = Length.dimensions

    # need to override the overload based on unit "km"
    # this is not very elegant
    d = Q[DistanceAlongPath](25, str("km"))
    d2 = Q[DistanceAlongPath](5, str("km"))

    d + d
    d - d
    d + d2
    d2 - d

    with pytest.raises(Exception):
        s + d

    with pytest.raises(Exception):
        d - s

    assert str((Q(25, "MSEK/GWh") * Q(25, "kWh")).to_reduced_units()) == "0.000625 MSEK"

    assert str((Q(25, "MSEK/GWh") * Q(25, "kWh")).to_base_units()) == "625.0 currency"

    # if the _distinct class attribute is True, an unspecified
    # dimensionality will default to this
    # for example, EnergyPerMass has _distinct=True even though
    # it shares dimensions with other dimensionalities like SpecificEnthalpy and HeatingValue

    # q4 will be become Quantity[EnergyPerMass] by default

    q4 = Q(25, "kJ/kg")

    # override the Literal['kJ/kg'] overload
    q5 = Q[SpecificEnthalpy](25, str("kJ/kg"))

    # prefer to use the asdim method
    q5_ = Q(25, "kJ/kg").asdim(SpecificEnthalpy)

    assert q5 == q5_

    q6 = Q[EnergyPerMass](25, str("kJ/kg"))

    with pytest.raises(Exception):
        q4 - q5

    with pytest.raises(Exception):
        q5 - q4

    q4 + q6
    q6 - q4


def test_distinct_dimensionality():
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
    assert type(Q[Distinct](1, unit)) is Q[Distinct]
    assert type(Q[Indistinct](1, unit)) is Q[Indistinct]


def test_literal_units():
    for d, units in get_registered_units().items():
        for u in units:
            decoded = decode({"type": "Quantity", "dimensionality": d, "data": [1, u]})

            assert decoded._dimensionality_type.__name__ == d


def test_indexing():
    qs = Q([1, 2, 3], "kg")

    assert isinstance(qs, Q[Mass])

    qi = qs[1]

    assert isinstance(qi, Q[Mass])
    assert qi == Q(2, "kg")


def test_plus_minus():
    # might be better to use the uncertainties package for this
    length = Q(2, "m")

    # TODO: add type hints for this
    l_e = length.plus_minus(Q(1, "cm"))

    l2_e = (l_e**2).to("km**2")

    assert l2_e.error == Q(4e-8, "km**2")
    assert isinstance(l2_e.error, Q[Area])


def test_round():
    pass

    # TODO: should this even work?
    # type numpy.ndarray doesn't define __round__ method

    # q = Q(25.12312312312, 'kg/s')

    # q_r = round(q, 1)

    # assert q_r.m == 25.1

    # q = Q([25.12312312312, 25.12312312312], 'kg/s')

    # q_r = round(q, 1)

    # assert q_r.m[0] == 25.1
    # assert q_r.m[1] == 25.1


def test_abs():
    q = Q(-25, "kg/s")

    q_a = abs(q)

    assert q_a.m == 25

    q = Q([-25, -25], "kg/s")

    q_a = abs(q)

    assert q_a.m[0] == 25
    assert q_a.m[1] == 25


def test_pandas_is_list_like():
    # scalar magnitude is not list like

    assert pandas_is_list_like(Q([25]))
    assert pandas_is_list_like(Q([25, 25]))
    assert pandas_is_list_like(Q(np.linspace(0, 1), "kg"))

    assert not pandas_is_list_like(Q(25))
    assert not pandas_is_list_like(Q(0.2))
    assert not pandas_is_list_like(Q(0.2, "kg/s"))


def test_pandas_integration():
    index = pd.date_range("2020-01-01", "2020-01-02", freq="h")
    df = pd.DataFrame(index=index)

    index_qty = Q(index)

    assert isinstance(index_qty.m, pd.DatetimeIndex)

    with pytest.raises(ValueError):
        Q(index, "kg")

    df["input"] = np.linspace(0, 1, len(df))

    q_vector = Q(df["input"], "m/s")

    # assigns a float array, as expected
    df["A"] = q_vector.to("kmh")

    q_scalar = Q(25, "ton/h")

    # assigns a repeated array of Quantity objects
    df["B"] = q_scalar

    # identical to the previous assignment
    df["C"] = [q_scalar] * len(df)

    # this will be correctly broadcasted to a repeated array
    df["D"] = q_scalar.m

    assert df.dtypes.A == np.float64
    assert df.dtypes.B == object
    assert df.dtypes.C == object
    # all inputs are cast to float when constructing a Quantity
    assert df.dtypes.D == np.float64

    assert isinstance(df.A.values[0], np.float64)
    assert isinstance(df.B.values[-1], Q[MassFlow])
    assert isinstance(df.C.values[0], Q[MassFlow])
    assert isinstance(df.D.values[0], np.float64)


def test_unit_compatibility():
    # the ureg registry object contains unit attributes
    # that can be multiplied and divided by a magnitude
    # to create Quantity instances

    assert isinstance(ureg.m * 1, Q[Length])
    assert isinstance(1 * ureg.m / ureg.s, Q[Velocity])
    assert isinstance([1, 2, 3] * ureg.m / ureg.s, Q[Velocity])
    assert isinstance((1, 2, 3) * ureg.m / ureg.s, Q[Velocity])
    assert isinstance(np.array([1, 2, 3]) * ureg.m / ureg.s, Q[Velocity])


def test_mul_rmul_initialization():
    assert isinstance(ureg.m * np.array([1, 2]), Q[Length])
    assert isinstance(np.array([1, 2]) * ureg.m, Q[Length])
    assert isinstance([1, 2] * Q(1, "m"), Q[Length])
    assert isinstance(np.array([1, 2]) * Q(1, "m"), Q[Length])


def test_decimal():
    # decimal.Decimal works, but it's not included in the type hints
    # TODO: inputs are converted to float, don't use this

    assert Q(Decimal("1.5"), "MSEK").to("SEK").m == Decimal("1500000.00")

    q = Q([Decimal("1.5"), Decimal("1.5")], "kg")

    q_gram = q.to("g")

    assert (q_gram.m == 1000 * q.m).all()


def test_copy():
    q = Q(25, "m")

    assert isinstance(q, Q[Length])
    assert isinstance(q.__copy__(), Q[Length])

    assert isinstance(q.__deepcopy__(), Q[Length])
    assert isinstance(q.__deepcopy__({}), Q[Length])

    assert isinstance(copy.copy(q), Q[Length])
    assert isinstance(copy.deepcopy(q), Q[Length])


def test_pydantic_integration():
    class Model(BaseModel):
        # a can be any dimensionality
        a: Q

        m: Q[Mass]
        s: Q[Length]

        # float can be converted to Quantity[Dimensionless]
        r: Q[Dimensionless] = 0.5

        # float cannot be converted to Quantity[Length]
        # d: Q[Length] = 0.5

        model_config = ConfigDict(validate_default=True)

    Model(a=Q(25, "cSt"), m=Q(25, "kg"), s=Q(25, "cm"))

    with pytest.raises(ExpectedDimensionalityError):
        Model(a=Q(25, "cSt"), m=Q(25, "kg/day"), s=Q(25, "cm"))


def test_float_cast():
    assert isinstance(Q([False, False]).m[0], float)

    assert (Q([False, True]) == Q(np.array([False, True]))).all()


def test_temperature_difference():
    T1 = Q(25, "degC")
    T2 = Q(35, "degC")

    dT1 = T1 - T2
    dT2 = T2 - T1

    assert isinstance(dT1, Q[TemperatureDifference])
    assert isinstance(dT2, Q[TemperatureDifference])

    assert dT1.u == Unit("delta_degC")
    assert dT2.u == Unit("delta_degC")

    assert (Q(25, "degF") - Q(30, "degF")).u == Unit("delta_degF")

    with pytest.raises(OffsetUnitCalculusError):
        T1 + T2

    with pytest.raises(OffsetUnitCalculusError):
        T2 + T1

    with pytest.raises(OffsetUnitCalculusError):
        T1 * 2

    with pytest.raises(OffsetUnitCalculusError):
        T1 / 2

    assert isinstance(dT1, Q[TemperatureDifference])
    assert isinstance(dT1.to("delta_degF"), Q[TemperatureDifference])

    assert isinstance(dT2, Q[TemperatureDifference])
    assert isinstance(dT2 + dT1, Q[TemperatureDifference])

    assert isinstance(T1 + dT1, Q[Temperature])

    with pytest.raises(DimensionalityTypeError):
        assert isinstance(dT1 + T1, Q[Temperature])

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

    assert isinstance(T2_, Q[Temperature])


def test_temperature_unit_inputs():
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

        # NOTE: qty.check(Temperature) and qty.check(TemperatureDifference)
        # are equivalent since both dimensionalitites have the same unitscontainer
        assert isinstance(qty, (Q[Temperature], Q[TemperatureDifference]))

        # this will automatically be converted to delta_temperature per length,
        # even if the input is temperature (not delta_temperature)
        dT_per_length = Q(1, f"{unit} / m")
        assert (
            dT_per_length.dimensionality == Temperature.dimensions / Length.dimensions
        )

        vol_per_dT = Q(1, f"m3/{unit}")
        assert vol_per_dT.dimensionality == Volume.dimensions / Temperature.dimensions


def test_nested_quantity_input():
    q = Q(25, "bar")
    q2 = Q(q)

    assert type(q) is type(q2)
    assert q == q2

    assert Q(Q(Q(Q(15, "m")) * 2)) * 2 == Q(60, "m")


def test_getitem():
    ms = Q([1.2, 1.3], "kg")
    assert isinstance(ms, Q[Mass, list[float]])

    m0 = ms[0]
    assert isinstance(m0, Q[Mass, float])

    ts = Q(pd.DatetimeIndex([pd.Timestamp.now(), pd.Timestamp.now()]))
    assert isinstance(ts, Q[Dimensionless, pd.DatetimeIndex])

    t0 = ts[0]
    assert isinstance(t0, Q[Dimensionless, pd.Timestamp])


def test_astype():
    assert isinstance(Q(25).astype(list[float]).m[0], float)

    assert isinstance(Q([1, 2, 3]).astype(np.ndarray).m, np.ndarray)
    assert isinstance(Q([1, 2, 3]).astype(pd.Series).m, pd.Series)
    assert isinstance(Q([1, 2, 3]).astype(pl.Series).m, pl.Series)

    assert Q([1, 2, 3]).astype(pd.Series, name="s1").m.name == "s1"
    assert Q([1, 2, 3]).astype(pl.Series, name="s1").m.name == "s1"


def test_single_element_array_magnitude():
    s1_list = [1.0]
    s2_list = [1.0, 2.0]

    Q(s1_list, "kg") * Q(s2_list, "m") / Q(s2_list, "kg")

    s1_arr = np.array([1])
    s2_arr = np.array([1, 2])

    Q(s1_arr, "kg") * Q(s2_arr, "m") / Q(s2_arr, "kg")

    s1_series = pd.Series([1], name="one")
    s2_series = pd.Series([1, 2], name="two")

    Q(s1_series, "kg") * Q(s2_series, "m") / Q(s2_series, "kg")


def test_check_temperature_difference():
    assert not Q(1, "degC").check(Q(12, "kg"))

    assert Q(1, "degC").check(Q(12, "degC"))
    assert Q(1, "degC").check(Q(12, "degC").u)

    assert not Q(1, "delta_degC").check(Q(12, "degC"))
    assert not Q(1, "delta_degC").check(Q(12, "degC").u)

    assert Q(1, "delta_degC").check(Q(12, "delta_degC"))
    assert Q(1, "delta_degC").check(Q(12, "delta_degC").u)

    assert not Q(1, "delta_degC").check(Q(12, "degC"))
    assert not Q(1, "delta_degC").check(Q(12, "degC").u)
