import keyword
from typing import Any, cast

import numpy as np
import pytest

from ..sympy import Symbol, get_args, get_function, get_lambda_matrix, sp, symbols, to_identifier
from ..units import Quantity as Q


def test_sympy() -> None:
    x = cast(Symbol, sp.Symbol("x", positive=True))

    x._("n,text")
    x._("n,text,j")
    x.__("n,text")
    x.__("n,text,j")

    x._("n").append("text")
    x.__("n").append("text", where="sup")
    x.__("A").append("text", where="sub")


def test_to_identifier() -> None:
    x = cast(Symbol, sp.Symbol("x", positive=True))

    s = to_identifier(x._("subscript").__("superscript"))

    assert s.isidentifier()


def test_to_identifier_does_not_corrupt_plain_names() -> None:
    # a plain symbol whose name merely CONTAINS "text" or a keyword must not be mangled:
    # the \text collapse is anchored to the LaTeX token, keywords are suffixed not replaced
    assert to_identifier(cast(Any, sp.Symbol("context"))) == "context"
    assert to_identifier(cast(Any, sp.Symbol("flambda"))) == "flambda"
    assert to_identifier(cast(Any, sp.Symbol("m"))) == "m"

    # a bare Python keyword becomes a usable identifier (suffixed), not left as the keyword
    lam = to_identifier(cast(Any, sp.Symbol("lambda")))
    assert lam.isidentifier() and not keyword.iskeyword(lam)

    # the LaTeX \text{...} token is still collapsed, so "\text{m}" and "m" stay distinct
    assert to_identifier(cast(Any, sp.Symbol(r"\text{m}"))) != to_identifier(cast(Any, sp.Symbol("m")))


def test_get_args() -> None:
    x = cast(Symbol, sp.Symbol("x", positive=True))
    y = cast(Symbol, sp.Symbol("y", positive=True))

    # sympy is untyped, so its top-level functions resolve to unknown types
    sp_any = cast(Any, sp)
    e: sp.Basic = x**2 + sp_any.sin(sp_any.sqrt(y))

    assert set(get_args(e)) == {"x", "y"}


def test_decorate() -> None:
    n = cast(Symbol, sp.Symbol("n"))

    decorated = n.decorate(prefix=r"\sum", prefix_sub="2", suffix_sup="i", suffix=r"\ldots")
    assert str(decorated) == r"{\sum}_{2}n^{i}{\ldots}"
    n._("H_2O").__("out")


def test_sympy_to_Quantity_integration() -> None:
    x, y, z = symbols("x, y, z")

    # sympy does not type its arithmetic operators
    expr: Any = cast(Any, x) * y / z

    _subs = {x: Q(235, "yard"), y: Q(98, "K/m²"), z: Q(0.4, "minutes")}

    result = expr.subs(_subs)

    _result = _subs[x] * _subs[y] / _subs[z]

    assert isinstance(Q.from_expr(result), type(_result))

    a, b = symbols("a, b", nonzero=True)

    expr = cast(Any, a) / b

    result = expr.subs({a: Q(235, "m"), b: Q(98, "mile")})

    assert Q.from_expr(result).check(Q(0))


def test_Quantity_to_sympy_integration() -> None:
    x = sp.Symbol("x")

    _ = cast(Any, x) * Q(25, "kg")
    _ = cast(Any, x) * Q(25, "kg")

    _ = Q(25, "kg") * cast(Any, x)

    _ = cast(Any, x) + Q(2)
    _ = cast(Any, x) + Q(2, "m")


def test_get_lambda_matrix_arg_order() -> None:
    # the generated lambda signature MUST use the same parameter order as the returned list,
    # otherwise a caller binding the returned params positionally misaligns the values (the
    # collected args are a set, whose order is arbitrary / PYTHONHASHSEED-dependent)
    import inspect

    sp_any = cast(Any, sp)
    a, b, c, d, e = sp_any.symbols("a b c d e")
    M = sp_any.Matrix([[a - 10 * b], [100 * c - d], [e]])

    src, params = get_lambda_matrix(M)
    fcn = eval(src, {"np": np})  # generated numeric lambda, test-only

    assert list(inspect.signature(fcn).parameters) == params

    # binding the returned params positionally yields the correct matrix
    values = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}  # row 0 = a - 10b = 1
    result = fcn(*[values[p] for p in params])
    assert float(np.asarray(result).ravel()[0]) == 1.0


def test_get_lambda_rejects_colliding_identifiers() -> None:
    # to_identifier is a lossy per-symbol map, so distinct symbols CAN collapse to one
    # identifier (a bare keyword "lambda" -> "lambda_" collides with a symbol named "lambda_").
    # get_lambda / get_lambda_matrix must raise rather than emit a lambda that silently merges
    # the two symbols into one parameter.
    # get_lambda's return type uses a bare Callable (suppressed module-wide in sympy.py), so
    # reach it via cast to avoid a partially-unknown-type error in this strict-checked test
    from .. import sympy as _sympy

    get_lambda = cast("Any", _sympy).get_lambda
    sp_any = cast(Any, sp)
    lam, lam_ = sp_any.Symbol("lambda"), sp_any.Symbol("lambda_")

    with pytest.raises(ValueError, match="colliding identifiers"):
        get_lambda(lam + 1000 * lam_, to_str=True)

    with pytest.raises(ValueError, match="colliding identifiers"):
        get_lambda_matrix(sp_any.Matrix([[lam], [lam_]]))

    # a keyword symbol on its own is fine -- the guard only trips on an actual collision
    _, params = get_lambda(lam + 1, to_str=True)
    assert params == ["lambda_"]


def test_get_function() -> None:
    x, y, z = symbols("x, y, z")

    # sympy does not type its arithmetic operators
    expr: Any = 25 * cast(Any, x) * y / z

    fcn = get_function(expr, units=True)

    fcn(
        {
            x: Q(np.array([235, 335]), "yard"),
            y: Q(np.array([2, 5]), "m²"),
            z: Q(0.4, "m³/kg"),
        }
    )
