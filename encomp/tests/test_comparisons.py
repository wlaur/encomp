import math
import operator
from collections.abc import Callable
from typing import Any, assert_type

import numpy as np
import polars as pl
import pytest

from ..units import Quantity as Q
from ..utypes import Numpy1DBoolArray


def _assert_type(val: object, typ: type) -> None:
    from encomp.misc import isinstance_types

    if not isinstance_types(val, typ):
        raise TypeError(f"Type mismatch for {val}: {type(val)}, expected {typ}")


assert_type.__code__ = _assert_type.__code__


def test_comparisons() -> None:
    assert_type(Q(1, "kg") > Q(25, "g"), bool)
    assert_type(Q(1, "kg") >= Q(25, "g"), bool)
    assert_type(Q(1, "kg") <= Q(25, "g"), bool)
    assert_type(Q(1, "kg") < Q(25, "g"), bool)
    assert_type(Q(1, "kg") == Q(25, "g"), bool)
    assert_type(Q(1, "kg") != Q(25, "g"), bool)

    assert_type(Q([1], "kg") > Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") >= Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") <= Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") < Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") == Q(25, "g"), Numpy1DBoolArray)
    assert_type(Q([1], "kg") != Q(25, "g"), Numpy1DBoolArray)

    assert_type(Q(1, "kg") > Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") >= Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") <= Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") < Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") == Q([25], "g"), Numpy1DBoolArray)
    assert_type(Q(1, "kg") != Q([25], "g"), Numpy1DBoolArray)

    assert_type(Q(1, "kg") > Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") >= Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") <= Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") < Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") == Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(1, "kg") != Q(pl.Series([25]), "g"), pl.Series)

    assert_type(Q(pl.Series([1]), "kg") > Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") >= Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") <= Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") < Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") == Q(25, "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") != Q(25, "g"), pl.Series)

    assert_type(Q(pl.Series([1]), "kg") > Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") >= Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") <= Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") < Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") == Q(pl.Series([25]), "g"), pl.Series)
    assert_type(Q(pl.Series([1]), "kg") != Q(pl.Series([25]), "g"), pl.Series)

    assert (Q(pl.Series([0.1 + 0.2]), "m") == Q(pl.Series([0.3]), "m")).to_list() == [True]
    assert (Q(pl.Series([0.1 + 0.2]), "m") != Q(pl.Series([0.3]), "m")).to_list() == [False]
    assert (Q(pl.Series([0.1 + 0.2]), "m") == Q(0.3, "m")).to_list() == [True]
    assert (Q(0.3, "m") == Q(pl.Series([0.1 + 0.2]), "m")).to_list() == [True]

    assert_type(Q(1, "kg") > Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") >= Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") <= Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") < Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") == Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(1, "kg") != Q(pl.col.asd, "g"), pl.Expr)

    assert_type(Q(pl.col.asd, "kg") > Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") >= Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") <= Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") < Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") == Q(25, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") != Q(25, "g"), pl.Expr)

    assert_type(Q(pl.col.asd, "kg") > Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") >= Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") <= Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") < Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") == Q(pl.col.asd, "g"), pl.Expr)
    assert_type(Q(pl.col.asd, "kg") != Q(pl.col.asd, "g"), pl.Expr)

    # operands that are equal within (rtol, atol) must satisfy every relation consistently:
    # equal, not strictly ordered either way, and non-strictly ordered both ways
    near_high = Q(1.0 + 5e-10, "m")
    near_low = Q(1.0, "m")
    assert near_high == near_low
    assert not near_high > near_low
    assert not near_high < near_low
    assert near_high <= near_low
    assert near_high >= near_low
    assert near_low <= near_high
    assert near_low >= near_high

    # a difference larger than the tolerance still orders strictly
    high = Q(2.0, "m")
    assert high > near_low
    assert not high <= near_low
    assert near_low < high
    assert not near_low >= high


def test_tolerant_equality_is_required_by_lossy_unit_conversion() -> None:
    # an exact __eq__ would call these unequal, which is why the tolerance exists at all --
    # and why the ordering operators have to honour it too
    assert Q(1.0, "L").to("cm³").m != Q(1000.0, "cm³").m
    assert Q(1.0, "L") == Q(1000.0, "cm³")
    assert not Q(1.0, "L") > Q(1000.0, "cm³")
    assert Q(1.0, "L") <= Q(1000.0, "cm³")


def test_ordering_is_a_strict_partial_order() -> None:
    values = [0.0, 1e-12, 1.0, 1.0 + 5e-10, 1.0 + 2e-9, 2.0, 1e9]
    quantities = [Q(value, "m") for value in values]

    for a in quantities:
        assert not a < a  # irreflexive

        for b in quantities:
            assert not (a < b and b < a)  # asymmetric
            assert (a < b) == (b > a)
            assert (a <= b) == (a < b or a == b)
            assert (a >= b) == (a > b or a == b)
            assert (a == b) != (a != b)
            # exactly one of <, >, == holds
            assert sum([a < b, a > b, a == b]) == 1

            for c in quantities:
                if a < b and b < c:
                    assert a < c  # transitive


def test_values_within_tolerance_are_ties_and_sort_stably() -> None:
    # ties keep their input order (Python's sort is stable), so a list of physically equal
    # quantities is not reshuffled
    tied = [Q(1.0 + 5e-10, "m"), Q(1.0, "m"), Q(1.0 + 2e-10, "m")]
    assert [q.m for q in sorted(tied)] == [q.m for q in tied]

    # well-separated quantities sort exactly, across units
    mixed = [Q(1.0, "m"), Q(50, "cm"), Q(2000, "mm"), Q(0.001, "km")]
    assert [str(q) for q in sorted(mixed)] == ["50.0 cm", "1.0 m", "0.001 km", "2000.0 mm"]

    # closeness is not transitive, so a tolerance CHAIN (adjacent pairs within tolerance,
    # endpoints not) has no consistent total order -- sort on the raw magnitude when one is needed
    chain = [Q(1.0 + i * 9e-10, "m") for i in range(8)]
    assert chain[0] == chain[1] and chain[1] == chain[2]
    assert chain[0] != chain[-1]

    by_magnitude = sorted(chain, key=lambda q: q.to("m").m)
    assert [q.m for q in by_magnitude] == sorted(q.m for q in chain)


_INF = float("inf")
_NAN = float("nan")

# pairs that straddle the tolerance band, plus the non-finite corners. NaN appears only
# paired with itself: numpy and polars deliberately disagree on ordering NaN against a
# NUMBER (see test_numpy_and_polars_disagree_on_nan_ordering), but nan-vs-nan answers
# identically everywhere (never equal, never ordered)
_COMPARISON_PAIRS = [
    (1.0, 1.0),
    (0.0, 0.0),
    (1.0, 1.0 + 5e-10),
    (1.0, 1.0 + 1.001e-9),
    (1.0, 1.0 + 1.5e-9),
    (1.0, 2.0),
    (0.0, 1e-12),
    (0.0, 2e-12),
    (1e-13, 1.1e-12),
    (1e6, 1e6 + 1e-4),
    (_INF, _INF),
    (-_INF, -_INF),
    (1.0, _INF),
    (-_INF, _INF),
    (_NAN, _NAN),
]


_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _quantities(magnitude_type: str, a: float, b: float) -> tuple[Q[Any, Any], Q[Any, Any]]:
    match magnitude_type:
        case "float":
            return Q(a, "m"), Q(b, "m")
        case "ndarray":
            return Q(np.array([a]), "m"), Q(np.array([b]), "m")
        case "pl.Series":
            return Q(pl.Series([a]), "m"), Q(pl.Series([b]), "m")
        case _:
            return Q(pl.lit(a, dtype=pl.Float64), "m"), Q(pl.lit(b, dtype=pl.Float64), "m")


def _compare(magnitude_type: str, a: float, b: float, op: str) -> bool:
    lhs, rhs = _quantities(magnitude_type, a, b)
    result = _OPERATORS[op](lhs, rhs)

    if isinstance(result, pl.Expr):
        return bool(pl.select(result).item())
    if isinstance(result, pl.Series):
        return bool(result.to_list()[0])
    if isinstance(result, np.ndarray):
        return bool(result[0])

    return bool(result)


@pytest.mark.parametrize(("a", "b"), _COMPARISON_PAIRS)
@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_every_magnitude_type_compares_identically(a: float, b: float, op: str) -> None:
    # the same physical comparison must not depend on the magnitude container, so every
    # magnitude type evaluates the same tolerance predicate
    results = {mt: _compare(mt, a, b, op) for mt in ("float", "ndarray", "pl.Series", "pl.Expr")}

    assert len(set(results.values())) == 1, f"{a} {op} {b} disagrees across magnitude types: {results}"


@pytest.mark.parametrize(("a", "b"), _COMPARISON_PAIRS)
def test_equality_matches_math_isclose(a: float, b: float) -> None:
    # the tolerance predicate is exactly math.isclose / polars is_close
    assert _compare("float", a, b, "==") == math.isclose(a, b, rel_tol=Q.rtol, abs_tol=Q.atol)


def test_infinite_and_nan_magnitudes() -> None:
    nan = float("nan")

    assert Q(_INF, "m") == Q(_INF, "m")
    assert Q(-_INF, "m") == Q(-_INF, "m")
    assert Q(_INF, "m") != Q(-_INF, "m")

    # an infinite operand must not make every finite value "close" via an infinite tolerance
    assert Q(1.0, "m") != Q(_INF, "m")
    assert Q(1.0, "m") < Q(_INF, "m")

    # NaN is close to nothing, not even itself
    assert Q(nan, "m") != Q(nan, "m")
    assert not Q(nan, "m") >= Q(nan, "m")
    assert not Q(nan, "m") <= Q(nan, "m")


def test_comparing_numpy_and_polars_magnitudes_raises() -> None:
    # the numpy and polars worlds disagree on NaN ordering and on missing values, and there is no
    # principled answer for whether the result should be a bool ndarray or a pl.Series -- so the
    # typed API never pairs them and the runtime refuses instead of picking one silently
    array = Q(np.array([1.0]), "m")
    series = Q(pl.Series([1.0]), "m")
    expr = Q(pl.col("x"), "m")

    for lhs, rhs in ((array, series), (series, array)):
        for op in _OPERATORS.values():
            with pytest.raises(TypeError, match="convert one side first"):
                op(lhs, rhs)

    # when a pl.Expr is involved there is no data to convert on that side, so the
    # message points at evaluating the expression or lifting the other operand instead
    for lhs, rhs in ((array, expr), (expr, array)):
        for op in _OPERATORS.values():
            with pytest.raises(TypeError, match="lift the other side"):
                op(lhs, rhs)

    # the escape hatch is explicit conversion
    assert (array.astype("pl.Series") == series).to_list() == [True]


def test_comparing_series_and_expr_magnitudes_raises() -> None:
    # Series-vs-Expr is equally outside the typed API: polars' raw operator would lift the
    # Series into an Expr literal and compare EXACTLY, skipping the (rtol, atol) tolerance
    # every sanctioned path applies, with a length mismatch surfacing only at collect()
    # time as a ShapeError pointing into the plan -- so the runtime refuses this pair too
    series = Q(pl.Series([1.0]), "m")
    expr = Q(pl.col("x"), "m")

    for lhs, rhs in ((series, expr), (expr, series)):
        for op in _OPERATORS.values():
            with pytest.raises(TypeError, match="lift the other side"):
                op(lhs, rhs)

    # the escape hatch is an explicit literal lift, which states the intent and lands on
    # the sanctioned (tolerant) Expr-vs-Expr path
    df = pl.DataFrame({"x": [1.0 + 5e-10, 2.0]})
    lifted = Q(pl.lit(pl.Series([1.0, 2.0])), "m")
    assert df.select((expr == lifted).alias("r"))["r"].to_list() == [True, True]


def test_numpy_and_polars_disagree_on_nan_ordering() -> None:
    # numpy follows IEEE: every comparison against NaN is False. polars orders NaN as the largest
    # value. Neither is wrong, but they are not interchangeable -- which is why the two worlds are
    # kept apart above. (polars magnitudes use null, not NaN, as encomp's missing-value sentinel.)
    nan = float("nan")
    one = Q(1.0, "m")

    array = Q([nan], "m")
    assert_type(array > one, Numpy1DBoolArray)
    assert not (array > one).any()
    assert not (array <= one).any()

    series = Q(pl.Series([nan]), "m")
    assert_type(series > one, pl.Series)
    assert (series > one).to_list() == [True]
    assert (series <= one).to_list() == [False]

    # nan-vs-nan is the one NaN case the worlds AGREE on: polars' raw >= calls NaN equal
    # to NaN (total order), but the non-strict operators are derived as `strict or equal`
    # with encomp's tolerant __eq__ (NaN is never equal), so every container answers
    # False for all four orderings -- `ge == (gt or eq)` holds even here
    other_nan = Q(pl.Series([nan]), "m")
    assert (series >= other_nan).to_list() == [False]
    assert (series <= other_nan).to_list() == [False]
    assert (series > other_nan).to_list() == [False]
    assert (series == other_nan).to_list() == [False]

    # a null magnitude propagates as null in both directions
    with_null = Q(pl.Series([1.0, None]), "m")
    assert (with_null > one).to_list() == [False, None]
    assert (with_null <= one).to_list() == [True, None]
