"""
Introspection-based conformance tests for the dimensional-algebra overloads of
:py:class:`encomp.units.Quantity`.

Rather than hand-enumerating every supported operation (as :mod:`test_overloads`
does), these tests discover the overload set at runtime via :func:`typing.get_overloads`
and check three properties against the *runtime* unit registry, which is the single
source of truth for dimensional algebra:

1. **Soundness** - every statically declared overload result matches the dimensionality
   that the operation actually produces at runtime.
2. **Inverse closure** - for every supported product ``A * B = C`` the inverse
   divisions ``C / A = B`` and ``C / B = A`` are also covered (and vice versa).
   This is deliberately limited to *direct* inverses; transitive/compositional
   closure is intentionally out of scope and handled downstream via ``.asdim``.
3. **Magnitude-type regularity** - each concrete dimensional pair is covered by
   exactly the ``(MT, MT)``, ``(MT, float)`` and ``(float, MT_)`` magnitude variants.

If a new dimensional overload is added by hand but its inverse is forgotten, or its
declared result dimensionality is wrong, or a magnitude variant is missing, one of
these tests fails - without anyone having to remember to also update an enumerated
mirror of the overloads.
"""

import inspect
import sys
import typing
from typing import Any, NamedTuple

import pytest

from ..units import Quantity
from ..utypes import Dimensionality, Dimensionless, UnknownDimensionality, get_registered_units

# the binary operators whose overloads encode dimensional algebra
_DIMENSIONAL_OPS: dict[str, str] = {"__mul__": "*", "__truediv__": "/"}

# resolve the string annotations (units.py uses ``from __future__ import annotations``)
# against the module globals where Quantity is defined
_GLOBALNS: dict[str, Any] = vars(sys.modules[Quantity.__module__])

# representative unit for each registered dimensionality, used to build concrete
# quantities for the runtime soundness check
_UNIT: dict[str, str] = {name: units[0] for name, units in get_registered_units().items()}


class _Combo(NamedTuple):
    """A single concrete dimensional overload, e.g. ``MassFlow * Time -> Mass``."""

    op: str
    self_dim: type[Dimensionality]
    other_dim: type[Dimensionality]
    ret_dim: type[Dimensionality]
    # the magnitude type of each operand: None means the MT/MT_ type variable
    self_mt: type | None
    other_mt: type | None


def _concrete_dim(annotation: object) -> type[Dimensionality] | None:
    """Return the concrete :class:`Dimensionality` for a resolved annotation, or
    ``None`` for a type-variable slot, ``UnknownDimensionality``, ``Dimensionless``
    or a plain (non-Quantity) annotation."""

    dt = getattr(annotation, "_dimensionality_type", None)

    if not isinstance(dt, type) or not issubclass(dt, Dimensionality):
        return None

    if dt in (UnknownDimensionality, Dimensionless):
        return None

    return dt


def _magnitude_type(annotation: object) -> type | None:
    """Return the concrete magnitude type of a ``Quantity[...]`` annotation, or
    ``None`` when the magnitude slot is the MT/MT_ type variable."""

    mt = getattr(annotation, "_magnitude_type", None)

    return mt if isinstance(mt, type) else None


def _iter_concrete_overloads() -> list[_Combo]:
    """Discover all overloads of the dimensional operators where both operands and
    the result are concrete (distinct, registered) dimensionalities."""

    combos: list[_Combo] = []

    for opname in _DIMENSIONAL_OPS:
        for func in typing.get_overloads(getattr(Quantity, opname)):
            sig = inspect.signature(func, eval_str=True, globals=_GLOBALNS)

            if "other" not in sig.parameters:
                continue

            self_ann = sig.parameters["self"].annotation
            other_ann = sig.parameters["other"].annotation

            self_dim = _concrete_dim(self_ann)
            other_dim = _concrete_dim(other_ann)
            ret_dim = _concrete_dim(sig.return_annotation)

            if self_dim is None or other_dim is None or ret_dim is None:
                continue

            combos.append(
                _Combo(
                    op=opname,
                    self_dim=self_dim,
                    other_dim=other_dim,
                    ret_dim=ret_dim,
                    self_mt=_magnitude_type(self_ann),
                    other_mt=_magnitude_type(other_ann),
                )
            )

    return combos


_CONCRETE_OVERLOADS: list[_Combo] = _iter_concrete_overloads()

# the set of supported operations, keyed by (op, self_dim, other_dim), ignoring the
# magnitude-type variants
_SUPPORTED: dict[tuple[str, str, str], str] = {
    (c.op, c.self_dim.__name__, c.other_dim.__name__): c.ret_dim.__name__ for c in _CONCRETE_OVERLOADS
}


def test_concrete_overloads_discovered() -> None:
    # guard against the introspection silently returning nothing (e.g. if the
    # annotation-resolution mechanism breaks): there must be many concrete overloads
    assert len(_CONCRETE_OVERLOADS) > 50
    assert any(op == "__mul__" for op, _, _ in _SUPPORTED)
    assert any(op == "__truediv__" for op, _, _ in _SUPPORTED)


@pytest.mark.parametrize(
    "combo",
    sorted({(c.op, c.self_dim.__name__, c.other_dim.__name__, c.ret_dim.__name__) for c in _CONCRETE_OVERLOADS}),
    ids=lambda c: f"{c[1]}{_DIMENSIONAL_OPS[c[0]]}{c[2]}",
)
def test_overloads_are_runtime_sound(combo: tuple[str, str, str, str]) -> None:
    """Each declared overload result must match the dimensionality the operation
    produces at runtime."""

    opname, self_name, other_name, ret_name = combo

    qs = Quantity(1.0, _UNIT[self_name])
    qo = Quantity(1.0, _UNIT[other_name])

    result = qs * qo if opname == "__mul__" else qs / qo
    runtime_dim = result.dt.__name__

    assert runtime_dim == ret_name, (
        f"{self_name} {_DIMENSIONAL_OPS[opname]} {other_name}: overload declares "
        f"-> {ret_name}, but runtime produces -> {runtime_dim}"
    )


def test_dimensional_overloads_have_full_mt_triple() -> None:
    """Every concrete dimensional pair must be covered by exactly the three
    magnitude variants ``(MT, MT)``, ``(MT, float)`` and ``(float, MT_)``."""

    # None denotes the MT/MT_ type variable
    expected_variants: set[tuple[type | None, type | None]] = {(None, None), (None, float), (float, None)}

    by_pair: dict[tuple[str, str, str], set[tuple[type | None, type | None]]] = {}
    for c in _CONCRETE_OVERLOADS:
        key = (c.op, c.self_dim.__name__, c.other_dim.__name__)
        by_pair.setdefault(key, set()).add((c.self_mt, c.other_mt))

    irregular: list[str] = []
    for (opname, self_name, other_name), variants in sorted(by_pair.items()):
        if variants != expected_variants:
            missing = expected_variants - variants
            extra = variants - expected_variants
            irregular.append(
                f"{self_name} {_DIMENSIONAL_OPS[opname]} {other_name}: "
                f"missing={_format_variants(missing)} extra={_format_variants(extra)}"
            )

    assert not irregular, "dimensional pairs with an irregular magnitude-type triple:\n" + "\n".join(irregular)


def test_curated_algebra_is_closed_under_inverse() -> None:
    """For every supported product ``A * B = C``, the multiplications ``A * B`` and
    ``B * A`` and the inverse divisions ``C / A`` and ``C / B`` must all be covered."""

    # canonicalise every overload into a multiplication fact: operands {A, B} -> C
    facts: dict[frozenset[str], str] = {}
    for (opname, self_name, other_name), ret_name in _SUPPORTED.items():
        if opname == "__mul__":
            # A * B = C
            facts[frozenset((self_name, other_name))] = ret_name
        else:
            # A / B = C  <=>  B * C = A
            facts[frozenset((other_name, ret_name))] = self_name

    missing: list[str] = []
    for operands, product in sorted(facts.items(), key=lambda kv: sorted(kv[0])):
        members = sorted(operands)
        a, b = (members[0], members[0]) if len(members) == 1 else (members[0], members[1])

        # expected multiplications (commutative)
        expected_mul = {(a, b)} if a == b else {(a, b), (b, a)}
        for left, right in sorted(expected_mul):
            if ("__mul__", left, right) not in _SUPPORTED:
                missing.append(f"{left} * {right} = {product}")

        # expected inverse divisions
        expected_div = {(a, b)} if a == b else {(a, b), (b, a)}
        for denom, quotient in sorted(expected_div):
            if ("__truediv__", product, denom) not in _SUPPORTED:
                missing.append(f"{product} / {denom} = {quotient}")

    assert not missing, "curated dimensional algebra is not closed under inverse; missing overloads:\n" + "\n".join(
        sorted(set(missing))
    )


def _format_variants(variants: set[tuple[type | None, type | None]]) -> str:
    def name(t: type | None) -> str:
        return "MT" if t is None else t.__name__

    return "{" + ", ".join(f"({name(a)}, {name(b)})" for a, b in sorted(variants, key=str)) + "}"
