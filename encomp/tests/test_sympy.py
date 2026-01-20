import numpy as np

from ..sympy import get_args, get_function, sp, to_identifier  # pyright: ignore[reportUnknownVariableType]
from ..units import Quantity as Q


def test_sympy() -> None:
    x = sp.Symbol("x", positive=True)

    x._("n,text")  # pyright: ignore[reportUnknownMemberType]
    x._("n,text,j")  # pyright: ignore[reportUnknownMemberType]
    x.__("n,text")  # pyright: ignore[reportUnknownMemberType]
    x.__("n,text,j")  # pyright: ignore[reportUnknownMemberType]

    x._("n").append("text")  # pyright: ignore[reportUnknownMemberType]
    x.__("n").append("text", where="sup")  # pyright: ignore[reportUnknownMemberType]
    x.__("A").append("text", where="sub")  # pyright: ignore[reportUnknownMemberType]


def test_to_identifier() -> None:
    x = sp.Symbol("x", positive=True)

    s = to_identifier(x._("subscript").__("superscript"))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    assert s.isidentifier()


def test_get_args() -> None:
    x = sp.Symbol("x", positive=True)
    y = sp.Symbol("y", positive=True)

    e = x**2 + sp.sin(sp.sqrt(y))  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue, reportUnknownMemberType]

    assert set(get_args(e)) == {"x", "y"}  # pyright: ignore[reportUnknownArgumentType]


def test_decorate() -> None:
    n = sp.Symbol("n")

    n.decorate(prefix=r"\sum", prefix_sub="2", suffix_sup="i", suffix=r"\ldots")  # pyright: ignore[reportUnknownMemberType]
    n._("H_2O").__("out")  # pyright: ignore[reportUnknownMemberType]


def test_sympy_to_Quantity_integration() -> None:
    x, y, z = sp.symbols("x, y, z")  # pyright: ignore[reportUnknownMemberType]

    expr = x * y / z

    _subs = {x: Q(235, "yard"), y: Q(98, "K/m²"), z: Q(0.4, "minutes")}

    result = expr.subs(_subs)

    _result = _subs[x] * _subs[y] / _subs[z]

    assert isinstance(Q.from_expr(result), type(_result))

    a, b = sp.symbols("a, b", nonzero=True)  # pyright: ignore[reportUnknownMemberType]

    expr = a / b

    result = expr.subs({a: Q(235, "m"), b: Q(98, "mile")})

    assert Q.from_expr(result).check(Q(0))


def test_Quantity_to_sympy_integration() -> None:
    x = sp.Symbol("x")

    _ = x * Q(25, "kg")
    _ = x * Q(25, "kg")

    _ = Q(25, "kg") * x

    _ = x + Q(2)
    _ = x + Q(2, "m")


def test_get_function() -> None:
    x, y, z = sp.symbols("x, y, z")  # pyright: ignore[reportUnknownMemberType]

    expr = 25 * x * y / z

    fcn = get_function(expr, units=True)  # pyright: ignore[reportUnknownVariableType]

    fcn(
        {
            x: Q(np.array([235, 335]), "yard"),
            y: Q(np.array([2, 5]), "m²"),
            z: Q(0.4, "m³/kg"),
        }
    )
