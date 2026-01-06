# ruff: noqa: B018
# pyright: reportConstantRedefinition=false

from typing import assert_type

import numpy as np
import polars as pl
import pytest
from pytest import approx  # pyright: ignore[reportUnknownVariableType]

from .. import utypes as ut
from ..fluids import Fluid, HumidAir, Water
from ..units import Quantity as Q
from ..utypes import DT, Density, SpecificEntropy


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def _approx_equal(q1: Q[DT, float], q2: Q[DT, float]) -> bool:
    if not q1.is_compatible_with(q2):
        return False

    return q1.to(q2.u).m == approx(q2.m)


def test_Fluid() -> None:
    fld = Fluid("R123", P=Q(2, "bar"), T=Q(25, "°C"))

    repr(fld)

    fld.describe("P")
    fld.search("pressure")

    # using __getattr__ will not call asdim(), these are Q[SpecificHeatCapacity]
    # (default for "J/(K kg)")
    assert _approx_equal(fld.__getattr__("S"), Q(1087.7758824621442, "J/(K kg)"))

    assert _approx_equal(Q(1087.7758824621442, "J/(K kg)").asdim(SpecificEntropy), fld.S)

    assert _approx_equal(fld.__getattr__("D"), fld.D)

    water = Fluid("water", P=Q(2, "bar"), T=Q(25, "°C"))
    assert water.T.u == Q.get_unit("degC")
    assert water.T.m == 25

    HumidAir(T=Q(25, "degC"), P=Q(125, "kPa"), R=Q(0.2, "dimensionless"))

    Water(P=Q(1, "bar"), Q=Q(0.9, ""))
    Water(P=Q(1, "bar"), T=Q(0.9, "degC"))
    Water(T=Q(1, "bar"), Q=Q(0.9, ""))

    repr(Water(T=Q(np.nan, "degC"), Q=Q(0.9)))
    repr(Water(T=Q(np.inf, "degC"), Q=Q(0.9)))
    repr(Water(T=Q(-np.inf, "degC"), Q=Q(0.9)))

    repr(Water(T=Q([np.nan, np.nan], "degC"), Q=Q(0.9)))
    repr(Water(T=Q([np.inf, np.inf], "degC"), Q=Q(0.9)))
    repr(Water(T=Q([-np.inf, -np.inf], "degC"), Q=Q(0.9)))
    repr(Water(T=Q([-np.inf, np.inf], "degC"), Q=Q(0.9)))

    with pytest.raises(ValueError):
        # cannot fix all of P, T, Q
        Water(P=Q(1, "bar"), T=Q(150, "degC"), Q=Q(0.4, ""))

    with pytest.raises(ValueError):
        # incorrect argument name
        Water(T=Q(1, "bar"), p=Q(9, "degC"))

    Fluid("water", T=Q([25, 95], "°C"), P=Q([1, 2], "bar")).H
    Fluid("water", T=Q([25, np.nan], "°C"), P=Q([1, 2], "bar")).H
    Fluid("water", T=Q([np.nan, np.nan], "°C"), P=Q([1, 2], "bar")).H
    Fluid("water", T=Q([np.nan, np.nan], "°C"), P=Q([np.nan, np.nan], "bar")).H
    Fluid("water", P=Q([1, 2], "bar"), T=Q(23, "°C")).H
    Fluid("water", P=Q([1], "bar"), T=Q(23, "°C")).H
    Fluid("water", T=Q([23, 25], "°C"), P=Q([1], "bar")).H
    Fluid("water", T=Q([23, 25], "°C"), P=Q(np.nan, "bar")).H
    Fluid("water", T=Q([23, 25], "°C"), P=Q([1, np.nan], "bar")).H

    Water(T=Q([25, 25, 63], "°C"), Q=Q([np.nan, np.nan, 0.4], "")).H
    Water(T=Q([25, np.nan, 63], "°C"), Q=Q([np.nan, 0.2, 0.5], "")).H
    Water(T=Q([25, np.nan, np.nan], "°C"), Q=Q([np.nan, 0.2, np.nan], "")).H

    # returns empty array (not nan)
    ret = Fluid("water", T=Q([], "°C"), P=Q([], "bar")).H.m
    assert isinstance(ret, np.ndarray) and ret.size == 0
    ret = Fluid("water", T=Q([], "°C"), P=Q([], "bar")).H.m
    assert isinstance(ret, np.ndarray) and ret.size == 0
    ret = Fluid("water", T=Q([], "°C"), P=Q(np.array([]), "bar")).H.m
    assert isinstance(ret, np.ndarray) and ret.size == 0

    # 1-element list or array works in the same way as scalar,
    # except that the output is also a 1-element list or array
    ret = Water(P=Q([2, 3], "bar"), Q=Q([0.5])).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 2

    ret = Water(P=Q([2, 3], "bar"), Q=Q(0.5)).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 2

    ret = Water(P=Q([2], "bar"), Q=Q([0.5])).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 1

    ret = Water(P=Q([2], "bar"), Q=Q(0.5)).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 1

    ret = Water(Q=Q([0.5]), P=Q(2, "bar")).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 1

    ret = Water(P=Q(2, "bar"), Q=Q(0.5)).D.m

    assert isinstance(ret, float)

    ret = Water(P=Q([], "bar"), Q=Q([0.5])).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 0

    ret = Water(P=Q([], "bar"), Q=Q([])).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 0

    ret = Water(P=Q(np.array([]), "bar"), Q=Q(np.array([]))).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 0

    # returns 1-element list
    assert isinstance(Fluid("water", T=Q([23], "°C"), P=Q([1], "bar")).H.m, np.ndarray)

    assert isinstance(Fluid("water", P=Q([1], "bar"), T=Q(23, "°C")).H.m, np.ndarray)

    assert isinstance(Fluid("water", T=Q([23], "°C"), P=Q(1, "bar")).H.m, np.ndarray)

    # returns float
    assert isinstance(Fluid("water", T=Q(23, "°C"), P=Q(1, "bar")).H.m, float)

    with pytest.raises(ValueError):
        Fluid("water", T=Q([np.nan, np.nan], "°C"), P=Q([np.nan, np.nan, np.nan], "bar")).H

    with pytest.raises(ValueError):
        Fluid("water", T=Q([np.nan, np.nan], "°C"), P=Q([], "bar")).H


def test_incorrect_inputs() -> None:
    # NOTE: the name cannot be checked until CoolProp is actually
    # called, so the name is not validated in __init__
    invalid = Fluid("this fluid name does not exist", P=Q(2, "bar"), T=Q(25, "°C"))

    with pytest.raises(ValueError):
        invalid.P

    p = np.zeros((5, 5))
    t = np.zeros(5)

    with pytest.raises(ValueError):
        Fluid("water", P=Q(p, "bar"), T=Q(t, "degC")).D  # pyright: ignore[reportArgumentType, reportCallIssue]

    p = np.zeros((5, 5))
    t = np.zeros(5 * 5)

    with pytest.raises(ValueError):
        Fluid("water", P=Q(p, "bar"), T=Q(t, "degC")).D  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(ValueError):
        Fluid("water", P=Q(p, "bar"), T=Q(t, "degC"), H=Q(25, "kJ/kg"))  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(ValueError):
        Water(P=Q(p, "bar"), T=Q(t, "degC"), H=Q(25, "kJ/kg"))  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(ValueError):
        Water(P=Q(p, "bar"))  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(AttributeError):
        Fluid("water", P=Q(2, "bar"), T=Q(25, "°C")).THIS_ATTRIBUTE_DOES_NOT_EXIST


def test_Water() -> None:
    water_single = Water(T=Q(25, "°C"), P=Q(5, "bar"))

    repr(water_single)

    water_multi = Water(T=Q(np.linspace(25, 50), "°C"), P=Q(5, "bar"))

    repr(water_multi)

    water_mixed_phase = Water(T=Q(np.linspace(25, 500, 10), "°C"), P=Q(np.linspace(0.5, 10, 10), "bar"))

    repr(water_mixed_phase)

    with pytest.raises(Exception):  # noqa: B017
        # mismatching sizes
        # must access an attribute before it's actually evaluated
        Water(T=Q(np.linspace(25, 500, 10), "°C"), P=Q(np.linspace(0.5, 10, 50), "bar")).P


def test_HumidAir() -> None:
    T = Q(20, "°C")
    P = Q(20, "bar")
    R = Q(20, "%")

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([25, 34], "°C")
    P = Q(20, "bar")
    R = Q(20, "%")

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([25, 34], "°C")
    P = Q([20, 30], "bar")
    R = Q([20, 40], "%")

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([25, 34], "°C")
    P = Q([20, 30], "bar")
    R = Q([20, np.nan], "%")

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([np.nan, 34], "°C")
    P = Q([np.nan, 30], "bar")
    R = Q([20, np.nan], "%")

    ha = HumidAir(T=T, P=P, R=R)
    ha.V

    T = Q([20, 40], "°C")
    P = Q([20, 1], "bar")
    R = Q([20, 101], "%")

    ha = HumidAir(T=T, P=P, R=R)
    val = ha.V.m
    assert not np.isnan(val[0])
    assert np.isnan(val[1])


def test_shapes() -> None:
    # NOTE: Quantity magnitudes must be 1D, these tests are not relevant
    N = 16

    T = Q(np.linspace(50, 60, N), "°C")
    P = Q(np.linspace(2, 4, N), "bar")

    water = Fluid("water", T=T, P=P)

    assert water.D.m.shape == P.m.shape
    assert water.D.m.shape == T.m.shape

    N = 27

    T = Q(np.linspace(50, 60, N), "°C")
    P = Q(np.linspace(2, 4, N), "bar")

    water = Fluid("water", T=T, P=P)

    assert water.D.m.shape == P.m.shape
    assert water.D.m.shape == T.m.shape


def test_invalid_areas() -> None:
    N = 10
    T = Q(np.linspace(-100, -50, N), "K")
    P = Q(np.linspace(-1, -2, N), "bar")

    water = Fluid("water", T=T, P=P)

    assert water.D.check(Density)
    assert isinstance(water.D.m, np.ndarray)

    T = Q(np.linspace(-100, 300, N), "K")
    P = Q(np.linspace(-1, 2, N), "bar")

    water = Fluid("water", T=T, P=P)

    assert water.D.check(Density)
    assert isinstance(water.D.m, np.ndarray)
    assert np.isnan(water.D.m[0])
    assert not np.isnan(water.D.m[-1])

    arr1 = np.linspace(-100, 400, N)
    arr2 = np.linspace(-1, 2, N)

    arr1[-2] = np.nan
    arr2[-1] = np.nan
    arr2[-3] = np.nan

    T = Q(arr1, "K")
    P = Q(arr2, "bar")

    water = Fluid("water", T=T, P=P)

    assert water.D.m.size == N


def test_properties_Fluid() -> None:
    props = Fluid.ALL_PROPERTIES

    fluid_names = ["water", "methane", "R134a"]

    Ts = [
        25,
        0,
        -1,
        -100,
        np.nan,
        [25, 30],
        [np.nan, 25],
        [np.nan, np.nan],
        [np.inf, np.nan],
        np.linspace(0, 10, 10),
        np.linspace(-10, 10, 10),
    ]

    Ps = [
        1,
        0,
        -1,
        -100,
        np.nan,
        [3, 4],
        [np.nan, 3],
        [np.nan, np.nan],
        [np.inf, np.nan],
        np.linspace(0, 10, 10),
        np.linspace(-10, 10, 10),
    ]

    for fluid_name in fluid_names:
        for T, P in zip(Ts, Ps, strict=False):
            fluid = Fluid(fluid_name, T=Q(T, "°C"), P=Q(P, "bar"))  # pyright: ignore[reportArgumentType, reportCallIssue]
            repr(fluid)

            for p in props:
                getattr(fluid, p)


def test_properties_HumidAir() -> None:
    props = HumidAir.ALL_PROPERTIES

    Ts = [
        25,
        0,
        -1,
        -100,
        np.nan,
        [25, 30],
        [np.nan, 25],
        [np.nan, np.nan],
        [np.inf, np.nan],
        np.linspace(0, 10, 10),
        np.linspace(-10, 10, 10),
    ]

    Ps = [
        1,
        0,
        -1,
        -100,
        np.nan,
        [3, 4],
        [np.nan, 3],
        [np.nan, np.nan],
        [np.inf, np.nan],
        np.linspace(0, 10, 10),
        np.linspace(-10, 10, 10),
    ]

    Rs = [
        0.5,
        0.1,
        -1,
        -100,
        np.nan,
        -0.5,
        0.00001,
        -0.0001,
        0.99999,
        1,
        1.00001,
        [0.3, 0.4],
        [np.nan, 0.3],
        [np.nan, np.nan],
        [np.inf, np.nan],
        np.linspace(0, 1, 10),
        np.linspace(-0.5, 0.5, 10),
    ]

    for T, P, R in zip(Ts, Ps, Rs, strict=False):
        ha = HumidAir(T=Q(T, "°C"), P=Q(P, "bar"), R=Q(R))  # pyright: ignore[reportCallIssue, reportArgumentType]
        repr(ha)

        for p in props:
            getattr(ha, p)


def test_magnitude_type() -> None:
    assert isinstance(Water(T=Q(25, "degC"), P=Q(25, "kPa")).H.m, float)


def test_polars_fluids() -> None:
    w_series = Water(P=Q(pl.Series([1, 2, 3]), "bar"), T=Q(pl.Series([150, 250, 350]), "degC"))
    assert_type(w_series.D, Q[ut.Density, pl.Series])

    w_series_const_T = Water(P=Q(pl.Series([1, 2, 3]), "bar"), T=Q(150, "degC"))
    assert_type(w_series_const_T.D, Q[ut.Density, pl.Series])

    assert pl.select(Water(P=Q(pl.lit(5), "bar"), T=Q(pl.lit(250), "degC")).D.m).item(0, 0) == approx(2.107798)

    w_expr = Water(P=Q(pl.lit(5), "bar"), T=Q(pl.col.T, "degC"))

    D = pl.DataFrame({"T": [150, 200, 250]}).with_columns(w_expr.D.m)["D"]

    assert D[0] == approx(917.020203)
    assert D[2] == approx(2.107798)

    w_expr_K = Water(P=Q(pl.lit(5), "bar"), T=Q(pl.col.T, "K"))

    D = pl.DataFrame({"T": [150, 200, 250]}).with_columns(w_expr_K.D.m)["D"]

    assert D.is_null().all()

    repr(Water(P=Q(pl.lit(5), "bar"), T=Q(50, "degC")))

    with pytest.raises(TypeError):
        Water(P=Q(pl.lit(5), "bar"), T=Q([1, 2, 3], "degC")).D  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError):
        Water(P=Q([1, 2, 3], "bar"), T=Q(pl.col.asd, "degC")).D  # pyright: ignore[reportArgumentType]
