# ruff: noqa: B018
# pyright: reportConstantRedefinition=false, reportPrivateUsage=false

import logging
from typing import Any, assert_type, cast

import numpy as np
import polars as pl
import pytest

from encomp import coolprop as encomp_coolprop

from .. import utypes as ut
from ..fluids import (
    CoolPropFluid,
    CProperty,
    Fluid,
    FluidState,
    HumidAir,
    HumidAirState,
    Water,
    _resolve_fluid_name,
    clear_expr_evaluation_cache,
)
from ..settings import SETTINGS
from ..units import Quantity as Q
from ..utypes import DT, Density, SpecificEntropy

# pytest.approx is loosely typed; expose it as an Any-typed alias
approx = cast(Any, pytest).approx


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def _approx_equal(q1: Q[DT, Any], q2: Q[DT, Any]) -> bool:
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
    water.HMOLAR_RESIDUAL.to("J/mol")
    water.GMOLAR_RESIDUAL.to("J/mol")
    water.SMOLAR_RESIDUAL.to("J/mol/K")

    HumidAir(T=Q(25, "degC"), P=Q(125, "kPa"), R=Q(0.2, "dimensionless"))

    Water(P=Q(1, "bar"), Q=Q(0.9, ""))
    Water(P=Q(1, "bar"), T=Q(0.9, "degC"))
    Water(T=Q(1, "bar"), Q=Q(0.9, ""))

    repr(Water(T=Q(np.nan, "degC"), Q=Q(0.9)))
    repr(Water(T=Q(np.inf, "degC"), Q=Q(0.9)))
    repr(Water(T=Q(-np.inf, "degC"), Q=Q(0.9)))

    repr(Water(T=Q([np.nan, np.nan], "degC"), Q=Q(0.9)))  # ty: ignore[invalid-argument-type]
    repr(Water(T=Q([np.inf, np.inf], "degC"), Q=Q(0.9)))  # ty: ignore[invalid-argument-type]
    repr(Water(T=Q([-np.inf, -np.inf], "degC"), Q=Q(0.9)))  # ty: ignore[invalid-argument-type]
    repr(Water(T=Q([-np.inf, np.inf], "degC"), Q=Q(0.9)))  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError):
        # cannot fix all of P, T, Q
        Water(P=Q(1, "bar"), T=Q(150, "degC"), Q=Q(0.4, ""))

    with pytest.raises(ValueError, match="same state input"):
        Fluid("water", D=Q(1, "kg/m³"), Dmass=Q(1, "kg/m³"))

    with pytest.raises(ValueError, match="same state input"):
        HumidAir(T=Q(25, "degC"), Tdb=Q(25, "degC"), P=Q(1, "bar"))

    with pytest.raises(ValueError, match="same state input"):
        HumidAir(T=Q(25, "degC"), W=Q(0.01, ""), HumRat=Q(0.01, ""))

    with pytest.raises(ValueError):
        # incorrect argument name (built dynamically so the static
        # Unpack[TypedDict] key check does not flag the intentional typo)
        Water(**{"T": Q(1, "bar"), "p": Q(9, "degC")})

    Fluid("water", T=Q([25, 95], "°C"), P=Q([1, 2], "bar")).H
    Fluid("water", T=Q([25, np.nan], "°C"), P=Q([1, 2], "bar")).H
    Fluid("water", T=Q([np.nan, np.nan], "°C"), P=Q([1, 2], "bar")).H
    Fluid("water", T=Q([np.nan, np.nan], "°C"), P=Q([np.nan, np.nan], "bar")).H
    Fluid("water", P=Q([1, 2], "bar"), T=Q(23, "°C")).H  # ty: ignore[invalid-argument-type]
    Fluid("water", P=Q([1], "bar"), T=Q(23, "°C")).H  # ty: ignore[invalid-argument-type]
    Fluid("water", T=Q([23, 25], "°C"), P=Q([1], "bar")).H
    Fluid("water", T=Q([23, 25], "°C"), P=Q(np.nan, "bar")).H  # ty: ignore[invalid-argument-type]
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

    ret = Water(P=Q([2, 3], "bar"), Q=Q(0.5)).D.m  # ty: ignore[invalid-argument-type]

    assert isinstance(ret, np.ndarray) and ret.size == 2

    ret = Water(P=Q([2], "bar"), Q=Q([0.5])).D.m

    assert isinstance(ret, np.ndarray) and ret.size == 1

    ret = Water(P=Q([2], "bar"), Q=Q(0.5)).D.m  # ty: ignore[invalid-argument-type]

    assert isinstance(ret, np.ndarray) and ret.size == 1

    ret = Water(Q=Q([0.5]), P=Q(2, "bar")).D.m  # ty: ignore[invalid-argument-type]

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

    assert isinstance(Fluid("water", P=Q([1], "bar"), T=Q(23, "°C")).H.m, np.ndarray)  # ty: ignore[invalid-argument-type]

    assert isinstance(Fluid("water", T=Q([23], "°C"), P=Q(1, "bar")).H.m, np.ndarray)  # ty: ignore[invalid-argument-type]

    # returns float
    assert isinstance(Fluid("water", T=Q(23, "°C"), P=Q(1, "bar")).H.m, float)

    with pytest.raises(ValueError):
        Fluid("water", T=Q([np.nan, np.nan], "°C"), P=Q([np.nan, np.nan, np.nan], "bar")).H

    with pytest.raises(ValueError):
        Fluid("water", T=Q([np.nan, np.nan], "°C"), P=Q([], "bar")).H


def test_fluid_return_units_are_normalized() -> None:
    water = Fluid("water", P=Q(2, "bar"), T=Q(25, "degC"))

    expected_units: dict[CProperty, str] = {
        "P": "kPa",
        "PCRIT": "kPa",
        "PMAX": "kPa",
        "PMIN": "kPa",
        "PTRIPLE": "kPa",
        "P_REDUCING": "kPa",
        "T": "degC",
        "TCRIT": "degC",
        "TMAX": "degC",
        "TMIN": "degC",
        "TTRIPLE": "degC",
        "T_REDUCING": "degC",
        "H": "kJ/kg",
        "U": "kJ/kg",
        "S": "kJ/kg/K",
        "C": "kJ/kg/K",
    }

    for prop, unit in expected_units.items():
        expected_unit = Q.get_unit(unit)
        assert water.get(prop).u == expected_unit
        assert getattr(water, prop).u == expected_unit

    assert water.get("p_critical").u == Q.get_unit("kPa")


def test_ignore_coolprop_warnings_gates_logger(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fluid = Fluid("water", P=Q(1.0, "bar"), T=Q(25.0, "degC"))

    monkeypatch.setattr(SETTINGS, "ignore_coolprop_warnings", True)
    with caplog.at_level(logging.WARNING, logger="encomp.fluids"):
        fluid.check_exception("D", ValueError("unexpected backend failure"))

    assert "CoolProp could not calculate" not in caplog.text

    caplog.clear()
    monkeypatch.setattr(SETTINGS, "ignore_coolprop_warnings", False)
    with caplog.at_level(logging.WARNING, logger="encomp.fluids"):
        fluid.check_exception("D", ValueError("unexpected backend failure"))

    assert 'CoolProp could not calculate "D"' in caplog.text


def test_humid_air_out_of_range_logs_warning(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    humid_air = HumidAir(T=Q(25, "degC"), P=Q(1, "bar"), R=Q(1.5, ""))

    monkeypatch.setattr(SETTINGS, "ignore_coolprop_warnings", False)
    with caplog.at_level(logging.WARNING, logger="encomp.fluids"):
        value = humid_air.W

    assert np.isnan(float(value.m))
    assert 'CoolProp could not calculate "W" for fluid "Humid air"' in caplog.text
    # The C HAPropsSI interface exposes details only through a process-global error
    # outbox. The native bridge deliberately does not read it, so concurrent failures
    # cannot attach another thread's message to this warning.
    assert "native evaluation returned no finite result" in caplog.text

    caplog.clear()
    monkeypatch.setattr(SETTINGS, "ignore_coolprop_warnings", True)
    with caplog.at_level(logging.WARNING, logger="encomp.fluids"):
        value = humid_air.W

    assert np.isnan(float(value.m))
    assert "CoolProp could not calculate" not in caplog.text


def test_incorrect_inputs() -> None:
    # the name is resolved eagerly (cached per name), so construction is where it fails
    with pytest.raises(ValueError, match="could not be initialized"):
        Fluid("this fluid name does not exist", P=Q(2, "bar"), T=Q(25, "°C"))

    p = np.zeros((5, 5))
    t = np.zeros(5)

    with pytest.raises(ValueError):
        Fluid("water", P=Q(cast(Any, p), "bar"), T=Q(t, "degC")).D

    p = np.zeros((5, 5))
    t = np.zeros(5 * 5)

    with pytest.raises(ValueError):
        Fluid("water", P=Q(cast(Any, p), "bar"), T=Q(t, "degC")).D

    with pytest.raises(ValueError):
        Fluid("water", P=Q(cast(Any, p), "bar"), T=Q(t, "degC"), H=Q(25, "kJ/kg"))  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError):
        Water(P=Q(cast(Any, p), "bar"), T=Q(t, "degC"), H=Q(25, "kJ/kg"))  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError):
        Water(P=Q(cast(Any, p), "bar"))

    with pytest.raises(ValueError, match=r"Invalid CoolProp state input.*VISCOSITY"):
        Fluid("water", **cast(Any, {"T": Q(25, "degC"), "VISCOSITY": Q(1, "cP")}))

    with pytest.raises(ValueError, match=r"Invalid CoolProp state input.*Visc"):
        HumidAir(**cast(Any, {"T": Q(25, "degC"), "P": Q(1, "bar"), "Visc": Q(1, "cP")}))

    with pytest.raises(AttributeError):
        Fluid("water", P=Q(2, "bar"), T=Q(25, "°C")).THIS_ATTRIBUTE_DOES_NOT_EXIST


def test_wrong_number_of_fixed_points() -> None:
    # a two-point fluid needs exactly two state inputs; a humid-air state needs three.
    # These cardinality checks run in __init__, before any CoolProp call.
    with pytest.raises(ValueError, match="Exactly two fixed points"):
        Fluid("water", **cast(Any, {"P": Q(1, "bar")}))

    with pytest.raises(ValueError, match="Exactly two fixed points"):
        Fluid("water", **cast(Any, {"P": Q(1, "bar"), "T": Q(300, "K"), "D": Q(1, "kg/m³")}))

    with pytest.raises(ValueError, match="Exactly two fixed points"):
        Water(**cast(Any, {"P": Q(1, "bar")}))

    with pytest.raises(ValueError, match="Exactly three fixed points"):
        HumidAir(**cast(Any, {"P": Q(1, "bar"), "T": Q(25, "degC")}))


def test_describe_rejects_unknown_property() -> None:
    # describe()/search() surface the property catalog; an unknown name is a ValueError
    with pytest.raises(ValueError, match="not a valid CoolProp property name"):
        Fluid.describe(cast(Any, "not_a_real_property"))


def test_invalid_fluid_name_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="could not be initialized"):
        Fluid(cast(Any, "Watr"), P=Q(2, "bar"), T=Q(25, "degC"))


def test_fluid_name_validation_is_cached() -> None:
    # a valid name is resolved by CoolProp once; every later construction is a cache hit
    _resolve_fluid_name.cache_clear()

    for _ in range(50):
        Fluid("Water", P=Q(2, "bar"), T=Q(25, "degC"))

    info = _resolve_fluid_name.cache_info()

    assert info.misses == 1
    assert info.hits == 49

    # an invalid name is NOT cached: functools.cache stores return values, not exceptions,
    # so a transient CoolProp failure can never be remembered as "this fluid is invalid"
    _resolve_fluid_name.cache_clear()

    for _ in range(3):
        with pytest.raises(ValueError, match="could not be initialized"):
            Fluid("Watr", P=Q(2, "bar"), T=Q(25, "degC"))

    assert _resolve_fluid_name.cache_info().currsize == 0


def test_invalid_fluid_state_repr_does_not_raise() -> None:
    # a valid name whose state cannot be fixed still reprs: the name resolved, only the
    # property evaluation fails, so the derived properties come back as NaN
    invalid = Fluid("water", P=Q(-2.0, "bar"), T=Q(25.0, "degC"))

    text = repr(invalid)

    assert text.startswith('<Fluid "water",')
    assert "D=nan" in text


def test_Water() -> None:
    water_single = Water(T=Q(25, "°C"), P=Q(5, "bar"))

    repr(water_single)
    steam_repr = repr(Water(P=Q(1, "bar"), T=Q(110, "degC")))
    assert "V=0.0 cP" not in steam_repr
    assert "V=0.013 cP" in steam_repr

    water_multi = Water(T=Q(np.linspace(25, 50), "°C"), P=Q(5, "bar"))  # ty: ignore[invalid-argument-type]

    repr(water_multi)

    water_mixed_phase = Water(T=Q(np.linspace(25, 500, 10), "°C"), P=Q(np.linspace(0.5, 10, 10), "bar"))

    repr(water_mixed_phase)

    with pytest.raises(ValueError, match="All inputs must have the same size"):
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

    ha = HumidAir(T=T, P=P, R=R)  # ty: ignore[invalid-argument-type]
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
            fluid = Fluid(fluid_name, T=Q(cast(Any, T), "°C"), P=Q(cast(Any, P), "bar"))
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
        ha = HumidAir(T=Q(cast(Any, T), "°C"), P=Q(cast(Any, P), "bar"), R=Q(cast(Any, R)))
        repr(ha)

        for p in props:
            getattr(ha, p)


def test_magnitude_type() -> None:
    assert isinstance(Water(T=Q(25, "degC"), P=Q(25, "kPa")).H.m, float)


def test_polars_fluids() -> None:
    w_series = Water(P=Q(pl.Series([1, 2, 3]), "bar"), T=Q(pl.Series([150, 250, 350]), "degC"))
    assert_type(w_series.D, Q[ut.Density, pl.Series])  # pyrefly: ignore[assert-type]

    w_series_const_T = Water(P=Q(pl.Series([1, 2, 3]), "bar"), T=Q(150, "degC"))  # ty: ignore[invalid-argument-type]
    assert_type(w_series_const_T.D, Q[ut.Density, pl.Series])  # pyrefly: ignore[assert-type]  # ty: ignore[type-assertion-failure]

    mixed_container = Water(
        P=Q(np.array([1.0, 2.0, 3.0]), "bar"),
        T=cast(Any, Q(pl.Series([150.0, 250.0, 350.0]), "degC")),
    )
    with pytest.raises(TypeError, match="Mixed vector magnitude containers"):
        mixed_container.D

    assert pl.select(Water(P=Q(pl.lit(5), "bar"), T=Q(pl.lit(250), "degC")).D.m).item(0, 0) == approx(2.107798)

    w_expr = Water(P=Q(pl.lit(5), "bar"), T=Q(pl.col.T, "degC"))

    D = pl.DataFrame({"T": [150, 200, 250]}).with_columns(w_expr.D.m)["D"]

    assert D[0] == approx(917.020203)
    assert D[2] == approx(2.107798)

    w_expr_K = Water(P=Q(pl.lit(5), "bar"), T=Q(pl.col.T, "K"))

    D = pl.DataFrame({"T": [150, 200, 250]}).with_columns(w_expr_K.D.m)["D"]

    assert D.is_null().all()

    repr(Water(P=Q(pl.lit(5), "bar"), T=Q(50, "degC")))  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        Water(P=Q(pl.lit(5), "bar"), T=cast(Any, Q([1, 2, 3], "degC"))).D

    with pytest.raises(TypeError):
        Water(P=cast(Any, Q([1, 2, 3], "bar")), T=Q(pl.col.asd, "degC")).D


def _count_water_h_evaluations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    # pl.Expr inputs are built into a native plugin expression; the expression cache
    # deduplicates identical requests before that call, so counting invocations
    # counts the distinct (cache-missing) expressions actually built.
    calls = [0]
    original_plugin_expr = CoolPropFluid._plugin_expr

    def counted_plugin_expr(
        self: CoolPropFluid[pl.Expr], output: str, points: tuple[tuple[str, pl.Expr], ...]
    ) -> pl.Expr:
        if getattr(self, "name", "") == "IF97::Water" and output == "H":
            calls[0] += 1

        return cast(pl.Expr, cast(Any, original_plugin_expr)(self, output, points))

    monkeypatch.setattr(CoolPropFluid, "_plugin_expr", counted_plugin_expr)

    return calls


def test_polars_fluids_expression_cache_col(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_expr_evaluation_cache()
    calls = _count_water_h_evaluations(monkeypatch)

    expressions = {
        f"H_{idx}": Water(P=Q(pl.lit(1.0), "bar"), T=Q(pl.col("T"), "degC")).H.m * (idx + 1) for idx in range(100)
    }

    df = pl.LazyFrame({"T": [150.0, 200.0, 250.0]}).select(**expressions).collect()

    assert len(df.columns) == 100
    assert calls[0] == 1


def test_polars_fluids_expression_cache_lit_series(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_expr_evaluation_cache()
    calls = _count_water_h_evaluations(monkeypatch)

    temperature_series = pl.Series("T", [150.0, 200.0, 250.0])

    expressions = {
        f"H_{idx}": Water(P=Q(pl.lit(1.0), "bar"), T=Q(pl.lit(temperature_series), "degC")).H.m * (idx + 1)
        for idx in range(100)
    }

    df = pl.LazyFrame({"idx": [0, 1, 2]}).select(**expressions).collect()

    assert len(df.columns) == 100
    assert calls[0] == 1


def test_polars_fluids_expression_cache_distinct_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_expr_evaluation_cache()
    calls = _count_water_h_evaluations(monkeypatch)

    expressions = {f"H_{idx}": Water(P=Q(pl.lit(1.0), "bar"), T=Q(pl.col("T") + idx, "degC")).H.m for idx in range(12)}

    df = pl.LazyFrame({"T": [150.0, 200.0, 250.0]}).select(**expressions).collect()

    assert len(df.columns) == 12
    assert calls[0] == 12


def test_state_typeddicts_match_plugin_inputs() -> None:
    # the constructor Unpack[TypedDict] key sets must stay in sync with the plugin's
    # canonical state-input property sets (single source of truth)
    assert frozenset(FluidState.__annotations__) == encomp_coolprop.FLUID_INPUTS
    assert frozenset(HumidAirState.__annotations__) == encomp_coolprop.HUMID_AIR_INPUTS


def test_polars_fluids_rust_backend() -> None:
    assert encomp_coolprop.self_check()

    P = np.full(7, 50e5)
    T = np.linspace(300.0, 500.0, 7)
    df = pl.DataFrame({"P": P, "T": T})

    # pl.Expr and eager NumPy inputs both use the native batch path and must match.
    clear_expr_evaluation_cache()
    w = Water[pl.Expr](P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"))
    rust = df.select(w.D.m.alias("D"), w.H.m.alias("H"), w.S.m.alias("S"), w.C.m.alias("C"))

    wn = Water(P=Q(P, "Pa"), T=Q(T, "K"))
    ref = {"D": wn.D.m, "H": wn.H.m, "S": wn.S.m, "C": wn.C.m}
    for col in ("D", "H", "S", "C"):
        assert rust[col].to_numpy() == approx(ref[col], rel=1e-4, nan_ok=True)

    # invalid inputs (T below the IF97 range) -> null, like the python path
    clear_expr_evaluation_cache()
    bad = pl.DataFrame({"T": [150.0, 200.0]}).select(Water(P=Q(pl.lit(5), "bar"), T=Q(pl.col("T"), "K")).D.m)["D"]
    assert bad.is_null().all()


def test_eager_plugin_matches_direct_scalar_bridge() -> None:
    # Every eager array size now uses the plugin. Compare it with the independent
    # direct-PyO3 scalar interface, including invalid rows and special configs.
    rng = np.random.default_rng(0)
    n = 24
    P = np.full(n, 50e5)
    T = 300.0 + 250.0 * rng.random(n)
    T[0] = np.nan  # non-finite input -> NaN
    P[1] = -1.0  # invalid (negative pressure) -> NaN
    T[2] = 50.0  # below the IF97 range -> NaN

    def compare_array_and_scalars(array: np.ndarray, scalars: np.ndarray) -> None:
        assert array.dtype == np.float64
        assert not np.isinf(array).any()
        assert np.array_equal(np.isnan(array), np.isnan(scalars))
        finite = np.isfinite(array)
        assert np.allclose(array[finite], scalars[finite], rtol=1e-9, atol=1e-12)

    array = cast(np.ndarray, Water(P=Q(P, "Pa"), T=Q(T, "K")).D.m)
    scalars = np.array([Water(P=Q(float(p), "Pa"), T=Q(float(t), "K")).D.m for p, t in zip(P, T, strict=True)])
    compare_array_and_scalars(array, scalars)

    array = cast(
        np.ndarray,
        Fluid("HEOS", composition={"CO2": 0.5, "O2": 0.5}, P=Q(P, "Pa"), T=Q(T, "K"))
        .assume_phase("supercritical_gas")
        .D.m,
    )
    scalars = np.array(
        [
            Fluid("HEOS", composition={"CO2": 0.5, "O2": 0.5}, P=Q(float(p), "Pa"), T=Q(float(t), "K"))
            .assume_phase("supercritical_gas")
            .D.m
            for p, t in zip(P, T, strict=True)
        ]
    )
    compare_array_and_scalars(array, scalars)

    Pa = np.full(n, 101325.0)
    Tdb = 290.0 + 20.0 * rng.random(n)
    R = 0.2 + 0.6 * rng.random(n)
    Tdb[0] = np.nan
    R[1] = 5.0  # relative humidity > 1 -> invalid
    array = cast(np.ndarray, HumidAir(P=Q(Pa, "Pa"), T=Q(Tdb, "K"), R=Q(R, "")).W.m)
    scalars = np.array(
        [
            HumidAir(P=Q(float(p), "Pa"), T=Q(float(t), "K"), R=Q(float(r), "")).W.m
            for p, t, r in zip(Pa, Tdb, R, strict=True)
        ]
    )
    compare_array_and_scalars(array, scalars)


def test_eager_plugin_threshold_is_deprecated_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("encomp.fluids.EAGER_PLUGIN_MIN_SIZE", 1000)
    calls = 0
    original_array = CoolPropFluid._evaluate_array

    def spy_rust(self: CoolPropFluid[Any], output: str, points: tuple[tuple[str, np.ndarray], ...]) -> np.ndarray:
        nonlocal calls
        calls += 1
        return cast(np.ndarray, cast(Any, original_array)(self, output, points))

    monkeypatch.setattr(CoolPropFluid, "_evaluate_array", spy_rust)
    Water(P=Q(np.full(2, 50e5), "Pa"), T=Q(np.full(2, 400.0), "K")).D.m
    monkeypatch.setattr("encomp.fluids.EAGER_PLUGIN_MIN_SIZE", 10**18)
    Water(P=Q(np.full(3, 50e5), "Pa"), T=Q(np.full(3, 400.0), "K")).D.m
    assert calls == 2


def test_polars_dtype_preservation() -> None:
    # pl.Float32() / pl.Float64() instances (the dtype= and == APIs accept either form)
    F32: pl.DataType = pl.Float32()
    F64: pl.DataType = pl.Float64()

    # --- lazy pl.Expr: output dtype follows the input COLUMN precision (Float32 only
    # if every non-scalar input is Float32, else Float64 = the supertype for mixed) ---
    def lazy_fluid(name: str, p_dt: pl.DataType, t_dt: pl.DataType) -> pl.Series:
        df = pl.DataFrame({"P": pl.Series([50e5] * 4, dtype=p_dt), "T": pl.Series([400.0] * 4, dtype=t_dt)})
        f = Fluid[pl.Expr](name, P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"))
        return df.select(f.D.m.alias("D"))["D"]

    for name in ("IF97::Water", "HEOS::Water"):
        assert lazy_fluid(name, F32, F32).dtype == F32
        assert lazy_fluid(name, F64, F64).dtype == F64
        assert lazy_fluid(name, F32, F64).dtype == F64  # mixed -> highest
        assert lazy_fluid(name, F64, F32).dtype == F64

    # scalar inputs are NEUTRAL: a Float64 literal / python float beside a Float32 column
    # must not force the result up to Float64
    df32 = pl.DataFrame({"T": pl.Series([400.0] * 4, dtype=F32)})
    lit_p = Water[pl.Expr](P=Q(pl.lit(50e5), "Pa"), T=Q(pl.col("T"), "K"))
    scalar_p = Water[pl.Expr](P=Q(5.0, "bar"), T=Q(pl.col("T"), "K"))
    assert df32.select(lit_p.D.m.alias("D"))["D"].dtype == F32
    assert df32.select(scalar_p.D.m.alias("D"))["D"].dtype == F32

    # humid air, lazy
    def lazy_ha(dt: pl.DataType) -> pl.Series:
        dfa = pl.DataFrame(
            {
                "P": pl.Series([101325.0] * 4, dtype=dt),
                "T": pl.Series([300.0] * 4, dtype=dt),
                "R": pl.Series([0.5] * 4, dtype=dt),
            }
        )
        ha = HumidAir[pl.Expr](P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"), R=Q(pl.col("R"), ""))
        return dfa.select(ha.W.m.alias("W"))["W"]

    assert lazy_ha(F32).dtype == F32
    assert lazy_ha(F64).dtype == F64

    # --- eager pl.Series: same rule ---
    def eager_fluid(p_dt: pl.DataType, t_dt: pl.DataType) -> pl.Series:
        w = Water[pl.Series](P=Q(pl.Series([50e5] * 4, dtype=p_dt), "Pa"), T=Q(pl.Series([400.0] * 4, dtype=t_dt), "K"))
        return w.D.m

    assert eager_fluid(F32, F32).dtype == F32
    assert eager_fluid(F64, F64).dtype == F64
    assert eager_fluid(F32, F64).dtype == F64  # mixed -> highest
    # a python-float scalar beside a Float32 series stays Float32
    scalar_eager = Water[pl.Series](P=Q(5.0, "bar"), T=Q(pl.Series([400.0] * 4, dtype=F32), "K"))
    assert scalar_eager.D.m.dtype == F32

    # --- numpy is always float64, regardless (no numpy float32 anywhere) ---
    np_water = Water[np.ndarray](P=Q(np.full(4, 50e5), "Pa"), T=Q(np.full(4, 400.0), "K"))
    assert np_water.D.m.dtype == np.float64

    # --- values are correct at Float32 (computed in f64, then cast) for both polars paths ---
    lazy_w = Water[pl.Expr](P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"))
    lazy_val = pl.DataFrame({"P": pl.Series([50e5], dtype=F32), "T": pl.Series([400.0], dtype=F32)}).select(
        lazy_w.D.m.alias("D")
    )["D"][0]
    eager_w = Water[pl.Series](P=Q(pl.Series([50e5], dtype=F32), "Pa"), T=Q(pl.Series([400.0], dtype=F32), "K"))
    eager_val = float(eager_w.D.m[0])
    assert lazy_val == approx(939.90625, rel=1e-4)
    assert eager_val == approx(939.90625, rel=1e-4)


def test_assume_phase() -> None:
    mix = "HEOS::CO2[0.5]&O2[0.5]"

    # chaining returns self
    f = Fluid(mix, P=Q(50.0, "bar"), T=Q(350.0, "degC"))
    assert f.assume_phase("gas") is f

    # in a genuinely single-phase region the assumed result matches auto-phase
    auto = Fluid(mix, P=Q(50.0, "bar"), T=Q(350.0, "degC")).D
    assumed = Fluid(mix, P=Q(50.0, "bar"), T=Q(350.0, "degC")).assume_phase("supercritical_gas").D
    assert assumed.u == auto.u
    assert float(assumed.m) == approx(float(auto.m), rel=1e-9)

    # array input
    T_arr = Q(np.linspace(300.0, 600.0, 5), "K")
    P_arr = Q(np.full(5, 50e5), "Pa")
    assumed_arr = Fluid(mix, P=P_arr, T=T_arr).assume_phase("supercritical_gas").D.m
    auto_arr = Fluid(mix, P=P_arr, T=T_arr).D.m
    assert np.allclose(assumed_arr, auto_arr, rtol=1e-9)

    # pl.Expr input flows through the low-level path (the .m expr is auto-named "D")
    fe = Fluid(mix, P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K")).assume_phase("supercritical_gas")
    res = pl.DataFrame({"P": [50e5], "T": [623.15]}).select(fe.D.m)  # 623.15 K = 350 degC
    assert res["D"][0] == approx(float(auto.m), rel=1e-4)

    # clearing restores automatic determination (checked behaviourally)
    f.assume_phase(None)
    assert float(f.D.m) == approx(float(auto.m), rel=1e-9)

    # unknown phase name is rejected (cast to Any to exercise the runtime guard)
    with pytest.raises(ValueError, match="unknown phase"):
        Fluid(mix, P=Q(50.0, "bar"), T=Q(350.0, "degC")).assume_phase(cast(Any, "plasma"))


def test_assume_phase_changes_value() -> None:
    # test_assume_phase only checks single-phase regions where assumed == auto, so a
    # regression that silently dropped assume_phase would pass it. Here a DISCRIMINATING
    # state (HEOS::Water at 1 bar, 110 C) is auto-phase GAS (steam, ~0.57 kg/m3); forcing
    # "liquid" must return the metastable subcooled-liquid root (~950 kg/m3). This proves
    # assume_phase actually takes effect, and that the direct scalar and batch paths
    # agree on the forced value. (An independent CoolProp
    # specify_phase reference for the same state lives in test_coolprop_plugin.)
    auto = Water(P=Q(1.0, "bar"), T=Q(110.0, "degC")).D.to("kg/m3")
    assert float(auto.m) < 10.0  # auto-phase water here is steam

    scalar = Fluid("HEOS::Water", P=Q(1.0, "bar"), T=Q(110.0, "degC")).assume_phase("liquid").D.to("kg/m3")
    assert float(scalar.m) > 900.0  # forcing liquid gives the subcooled-liquid root
    assert float(scalar.m) != approx(float(auto.m), rel=0.5)  # unmistakably not the auto (steam) value

    # same config, eager array -> plugin path; must match the direct scalar path
    n = 1200
    eager = (
        Fluid("HEOS::Water", P=Q(np.full(n, 1e5), "Pa"), T=Q(np.full(n, 383.15), "K"))
        .assume_phase("liquid")
        .D.to("kg/m3")
    )
    assert np.allclose(eager.m, float(scalar.m), rtol=1e-9)


def test_composition() -> None:
    # reference: same mixture/state with fractions baked into the name string
    ref = Fluid("HEOS::CO2[0.5]&O2[0.5]", P=Q(50.0, "bar"), T=Q(350.0, "degC")).assume_phase("supercritical_gas").D

    # scalar composition= matches the fixed-name reference exactly
    comp = (
        Fluid("HEOS", composition={"CO2": 0.5, "O2": 0.5}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))
        .assume_phase("supercritical_gas")
        .D
    )
    assert comp.u == ref.u
    assert float(comp.m) == approx(float(ref.m), rel=1e-9)

    # a fixed float composition evaluates against an N-element state
    fb = Fluid(
        "HEOS",
        composition={"CO2": 0.5, "O2": 0.5},
        P=Q(np.full(3, 50.0), "bar"),
        T=Q(np.full(3, 350.0), "degC"),
    ).assume_phase("supercritical_gas")
    assert isinstance(fb.D.m, np.ndarray) and fb.D.m.size == 3
    assert float(fb.D.m[0]) == approx(float(ref.m), rel=1e-9)

    # a fixed float composition also works with pl.Expr (lazy) state inputs, via the
    # rust plugin (the composition is constant, only P/T vary per row)
    fe = Fluid(
        "HEOS",
        composition={"CO2": 0.5, "O2": 0.5},
        P=Q(pl.col("P"), "bar"),
        T=Q(pl.col("T"), "degC"),
    ).assume_phase("supercritical_gas")
    res = pl.DataFrame({"P": [50.0], "T": [350.0]}).select(fe.D.m)
    assert res["D"][0] == approx(float(ref.m), rel=1e-4)

    # a composition that does not sum to 1 is an error (no silent renormalisation)
    with pytest.raises(ValueError, match="sum to 1"):
        Fluid("HEOS", composition={"CO2": 50.0, "O2": 50.0}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))

    # cannot set composition both in the name and via composition=
    with pytest.raises(ValueError, match="both"):
        Fluid("HEOS::CO2[0.5]&O2[0.5]", composition={"CO2": 0.5, "O2": 0.5}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))

    # species listed in the name must match the composition keys
    with pytest.raises(ValueError, match="do not match"):
        Fluid("HEOS::CO2&O2", composition={"CO2": 0.5, "N2": 0.5}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))

    # composition requires a mixture backend, not IF97
    with pytest.raises(ValueError, match="mixture backend"):
        Fluid("IF97", composition={"CO2": 0.5, "O2": 0.5}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))

    # at least two species are required
    with pytest.raises(ValueError, match="at least two species"):
        Fluid("HEOS", composition={"CO2": 1.0}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))


def test_assume_phase_if97_noop(caplog: pytest.LogCaptureFixture) -> None:
    # Water uses IF97, which is region-explicit and ignores an assumed phase:
    # assume_phase must be a no-op with a warning. The returned value must equal
    # the automatic-phase value.
    auto = Water(P=Q(50.0, "bar"), T=Q(150.0, "degC")).D
    w = Water(P=Q(50.0, "bar"), T=Q(150.0, "degC"))
    with caplog.at_level(logging.WARNING):
        assert w.assume_phase("gas") is w  # still chainable

    assert "ignores an assumed phase" in caplog.text
    assert float(w.D.m) == approx(float(auto.m), rel=1e-9)


def test_composition_non_unit_sum_raises() -> None:
    # fractions not summing to 1 are an error at construction (no silent renormalisation)
    with pytest.raises(ValueError, match="sum to 1"):
        Fluid("HEOS", composition={"CO2": 0.3, "O2": 0.3}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))

    # a small deviation (float rounding) is tolerated and normalised to exactly 1
    ok = Fluid("HEOS", composition={"CO2": 0.5, "O2": 0.4999999}, P=Q(50.0, "bar"), T=Q(350.0, "degC")).assume_phase(
        "supercritical_gas"
    )
    ref = Fluid("HEOS", composition={"CO2": 0.5, "O2": 0.5}, P=Q(50.0, "bar"), T=Q(350.0, "degC")).assume_phase(
        "supercritical_gas"
    )
    assert float(ok.D.m) == approx(float(ref.D.m), rel=1e-6)


def test_composition_rejects_non_float_fractions() -> None:
    # per-row varying composition was removed: only fixed float mole fractions are
    # accepted. array / pl.Series / pl.Expr fractions raise at construction; loop
    # over fixed compositions (one Fluid each) instead.
    with pytest.raises(TypeError, match="must be a float"):
        Fluid(
            "HEOS",
            composition=cast(Any, {"CO2": Q(np.array([0.5, 0.5]), ""), "O2": Q(np.array([0.5, 0.5]), "")}),
            P=Q(np.array([50.0, 50.0]), "bar"),
            T=Q(np.array([350.0, 360.0]), "degC"),
        )

    with pytest.raises(TypeError, match="must be a float"):
        Fluid(
            "HEOS",
            composition=cast(Any, {"CO2": Q(pl.col("x_CO2"), ""), "O2": Q(pl.col("x_O2"), "")}),
            P=Q(pl.col("P"), "bar"),
            T=Q(pl.col("T"), "degC"),
        )

    # bare (non-Quantity) array fractions are rejected too
    with pytest.raises(TypeError, match="must be a float"):
        Fluid(
            "HEOS",
            composition=cast(Any, {"CO2": np.array([0.5]), "O2": np.array([0.5])}),
            P=Q(50.0, "bar"),
            T=Q(350.0, "degC"),
        )

    # negative / non-finite fractions are rejected
    with pytest.raises(ValueError, match="non-negative"):
        Fluid("HEOS", composition={"CO2": -0.1, "O2": 1.1}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))

    with pytest.raises(ValueError, match="finite"):
        Fluid("HEOS", composition={"CO2": float("nan"), "O2": 1.0}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))

    # fractions that cannot be normalised (sum to zero) are rejected
    with pytest.raises(ValueError, match="sum to a positive value"):
        Fluid("HEOS", composition={"CO2": 0.0, "O2": 0.0}, P=Q(50.0, "bar"), T=Q(350.0, "degC"))


def test_incompressible_mixture_all_paths_agree() -> None:
    # regression: an INCOMP concentration mixture (INCOMP::MEG[0.5], ...) used to lose its
    # concentration identically in scalar, eager-array, and expression forms.
    assert encomp_coolprop.self_check()
    for name, temp in [("INCOMP::MEG[0.5]", 300.0), ("INCOMP::MITSW[0.035]", 290.0), ("INCOMP::AEG[0.4]", 300.0)]:
        pressure = np.full(6, 3e5)
        temperature = np.linspace(temp - 12.0, temp + 12.0, 6)
        ref = Fluid(name, P=Q(pressure, "Pa"), T=Q(temperature, "K")).D.m

        clear_expr_evaluation_cache()
        expr = (
            pl.DataFrame({"P": pressure, "T": temperature})
            .select(Fluid[pl.Expr](name, P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K")).D.m.alias("d"))["d"]
            .to_numpy()
        )
        assert np.asarray(expr, dtype=float) == approx(ref, rel=1e-4), f"expr {name}"

        clear_expr_evaluation_cache()
        big = Fluid(name, P=Q(np.full(1200, 3e5), "Pa"), T=Q(np.full(1200, temp), "K")).D.m
        small = float(Fluid(name, P=Q(3e5, "Pa"), T=Q(temp, "K")).D.m)
        assert float(np.asarray(big, dtype=float)[0]) == approx(small, rel=1e-4), f"array {name}"


def test_dimensional_property_missing_is_nan_never_zero() -> None:
    # A failed/undefined dimensional calc comes back MISSING, never a spurious 0.0: NaN for
    # float/numpy, NULL for polars (null is the single polars sentinel -- a pl.Series/pl.Expr
    # result must never contain NaN). Density is used because it is strictly positive, so 0 is
    # unambiguously missing (unlike enthalpy/entropy, whose reference points make ~0 legit).
    assert not Water(P=Q(1e5, "Pa"), T=Q(300.0, "K")).D.dimensionless

    temperature = np.array([300.0, 50.0, 400.0, 450.0])  # 50 K is below the IF97 range
    pressure = np.array([1e5, 1e5, -5.0, 2e5])  # negative pressure is invalid
    missing = np.array([False, True, True, False])

    def check_numpy(values: object, missing_mask: np.ndarray) -> None:
        # float / ndarray magnitude: missing -> NaN, present finite and non-zero
        arr = np.asarray(values, dtype=float)
        assert np.all(np.isnan(arr[missing_mask])), "missing rows must be NaN"
        present = arr[~missing_mask]
        assert np.all(np.isfinite(present)) and not np.any(present == 0.0), "present rows: non-zero, never 0.0"

    def check_polars(series: pl.Series, n_missing: int) -> None:
        # pl.Series / pl.Expr magnitude: missing -> NULL, NaN must never appear, present non-zero
        assert not series.is_nan().any(), "polars must never contain NaN -- missing is null"
        assert series.null_count() == n_missing, "missing rows must be null"
        present = series.drop_nulls().to_numpy()
        assert np.all(np.isfinite(present)) and not np.any(present == 0.0), "present rows: non-zero, never 0.0"

    reps = 300  # exercise the same native array path with a larger vector
    big_p, big_t, big_missing = np.tile(pressure, reps), np.tile(temperature, reps), int(missing.sum()) * reps

    # NumPy magnitudes of different lengths use the same native array path.
    check_numpy(Water(P=Q(pressure, "Pa"), T=Q(temperature, "K")).D.m, missing)
    check_numpy(Water(P=Q(big_p, "Pa"), T=Q(big_t, "K")).D.m, np.tile(missing, reps))

    # pl.Series magnitudes of different lengths use the same native array path.
    check_polars(
        Water[pl.Series](P=Q(pl.Series(pressure), "Pa"), T=Q(pl.Series(temperature), "K")).D.m, int(missing.sum())
    )
    check_polars(Water[pl.Series](P=Q(pl.Series(big_p), "Pa"), T=Q(pl.Series(big_t), "K")).D.m, big_missing)

    # lazy pl.Expr magnitude (plugin): missing -> null
    clear_expr_evaluation_cache()
    col = pl.DataFrame({"P": pressure, "T": temperature}).select(
        Water[pl.Expr](P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K")).D.m.alias("d")
    )["d"]
    check_polars(col, int(missing.sum()))

    assert np.isnan(float(Water(P=Q(1e5, "Pa"), T=Q(50.0, "K")).D.m))  # scalar: NaN, not 0.0

    # a property undefined for the state (surface tension, single-phase water) that CoolProp
    # raises on: NaN for float/numpy, but NULL (never NaN) for the polars magnitudes
    sp_p, sp_t = [100e5, 100e5], [300.0, 320.0]  # compressed liquid -> surface tension undefined
    assert np.isnan(float(Fluid("Water", P=Q(100e5, "Pa"), T=Q(300.0, "K")).get("I").m))
    check_numpy(Fluid("Water", P=Q(np.array(sp_p), "Pa"), T=Q(np.array(sp_t), "K")).get("I").m, np.array([True, True]))
    check_polars(Fluid[pl.Series]("Water", P=Q(pl.Series(sp_p), "Pa"), T=Q(pl.Series(sp_t), "K")).get("I").m, 2)
    clear_expr_evaluation_cache()
    st_col = pl.DataFrame({"P": sp_p, "T": sp_t}).select(
        Fluid[pl.Expr]("Water", P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K")).get("I").m.alias("i")
    )["i"]
    check_polars(st_col, 2)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_scalar_input_matches_vector_path(bad: float) -> None:
    # A non-finite input cannot fix a state. Handed one directly, CoolProp answers with
    # plausible finite values rather than failing: PHASE -> 0.0 ("Liquid"), Q -> the -1.0
    # single-phase sentinel, TCRIT -> a state-independent constant, and HAPropsSI echoes an
    # input back. Every property must be NaN, and the scalar answer must equal the 1-element
    # vector answer for the same state.
    scalar = Water(T=Q(bad, "degC"), P=Q(1.0, "bar"))
    vector = Water(T=Q(np.array([bad]), "degC"), P=Q(np.array([1.0]), "bar"))

    for prop in ("D", "Q", "TCRIT", "H", "PHASE"):
        scalar_value = float(cast(Any, scalar.get(cast(Any, prop))).m)
        vector_value = float(cast(Any, vector.get(cast(Any, prop))).m[0])

        assert np.isnan(scalar_value), f"scalar {prop} must be NaN for a {bad} input"
        assert np.isnan(vector_value), f"vector {prop} must be NaN for a {bad} input"

    assert scalar.phase == vector.phase == "N/A"

    # HAPropsSI echoes an input value straight back for some outputs, so mask the scalar path too
    humid_scalar = HumidAir(T=Q(bad, "degC"), P=Q(1.0, "bar"), R=Q(0.5))
    humid_vector = HumidAir(T=Q(np.array([bad]), "degC"), P=Q(np.array([1.0]), "bar"), R=Q(np.array([0.5])))

    assert np.isnan(float(humid_scalar.get("R").m))
    assert np.isnan(float(humid_vector.get("R").m[0]))


def test_phase_ignores_invalid_rows_and_is_length_independent() -> None:
    # an invalid row has no phase, and the answer must not depend on how many of them there are
    for n in (1, 2, 3):
        fluid = Water(T=Q(np.full(n, np.nan), "degC"), P=Q(np.full(n, 1.0), "bar"))
        assert fluid.phase == "N/A", f"all-invalid array of length {n}"

    # an empty array has no phase either
    assert Water(P=Q(np.array([]), "Pa"), T=Q(np.array([]), "K")).phase == "N/A"

    # invalid rows are not a phase: the determinate rows decide
    superheated = Water(T=Q(np.array([200.0, 250.0, np.nan]), "degC"), P=Q(np.array([1.0, 1.0, 1.0]), "bar"))
    assert np.isnan(float(cast(Any, superheated.PHASE.m)[2]))
    assert superheated.phase == "Gas"

    # genuine variation is still reported
    assert Water(T=Q(np.array([20.0, 250.0]), "degC"), P=Q(np.array([1.0, 1.0]), "bar")).phase == "Variable"


def test_new_typed_fluid_properties() -> None:
    water = Fluid("HEOS::Water", P=Q(1.0, "bar"), T=Q(25.0, "degC"))

    assert_type(water.RHOCRIT, Q[ut.Density, float])  # pyrefly: ignore[assert-type]
    assert_type(water.CVMASS, Q[ut.SpecificHeatCapacity, float])  # pyrefly: ignore[assert-type]
    assert_type(water.CPMOLAR, Q[ut.MolarSpecificHeatCapacity, float])  # pyrefly: ignore[assert-type]
    assert_type(water.CVMOLAR, Q[ut.MolarSpecificHeatCapacity, float])  # pyrefly: ignore[assert-type]
    assert_type(water.GAS_CONSTANT, Q[ut.MolarSpecificHeatCapacity, float])  # pyrefly: ignore[assert-type]
    assert_type(water.TAU, Q[ut.Dimensionless, float])  # pyrefly: ignore[assert-type]

    assert water.RHOCRIT.to("kg/m³").m == approx(322.0, rel=1e-3)
    # CoolProp carries its own value of R per equation of state, so compare loosely
    assert water.GAS_CONSTANT.to("J/mol/K").m == approx(8.3144598, rel=1e-4)

    # the typed accessors agree with the untyped __getattr__ path
    assert water.CVMASS.m == approx(water.__getattr__("Cvmass").m)
    assert water.CPMOLAR.m == approx(water.__getattr__("Cpmolar").m)

    # surface tension needs a new dimensionality, [force] / [length]
    saturated = Fluid("HEOS::Water", T=Q(25.0, "degC"), Q=Q(0.0))

    assert_type(saturated.I, Q[ut.SurfaceTension, float])  # pyrefly: ignore[assert-type]
    assert saturated.I.to("mN/m").m == approx(72.0, rel=1e-2)


def test_humid_air_constant_volume_heat_capacity_is_typed() -> None:
    humid_air = HumidAir(T=Q(25.0, "degC"), P=Q(1.0, "bar"), R=Q(0.5))

    # CV / CVha now match C / Cha instead of falling back to SpecificHeatCapacity
    assert_type(humid_air.CV, Q[ut.SpecificHeatPerDryAir, float])  # pyrefly: ignore[assert-type]
    assert_type(humid_air.CVha, Q[ut.SpecificHeatPerHumidAir, float])  # pyrefly: ignore[assert-type]

    assert type(humid_air.CV).__name__ == type(humid_air.C).__name__
    assert type(humid_air.CVha).__name__ == type(humid_air.Cha).__name__

    assert humid_air.CV.m < humid_air.C.m


def test_unknown_property_error_message_points_at_search() -> None:
    fluid = Fluid("water", P=Q(2.0, "bar"), T=Q(25.0, "degC"))

    with pytest.raises(AttributeError, match=r'use Fluid\.search\("\.\.\."\) to look one up'):
        fluid.NOT_A_PROPERTY

    # dunder probing (copy, pickle, inspect) still fails plainly
    with pytest.raises(AttributeError):
        fluid.__deepcopy__
