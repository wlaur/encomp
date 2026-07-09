"""Tests for ``encomp.coolprop.resolve_fluid_spec`` and the ``is_*`` name predicates.

``resolve_fluid_spec`` is the single place a fluid name + composition is parsed into the
``(backend, fluids, fractions)`` an ``AbstractState`` needs; both ``encomp.fluids.Fluid``
and ``encomp.coolprop.fluid`` delegate to it.
"""

from typing import cast

import pytest

from encomp import coolprop as cp


def test_bare_name_defaults_to_heos() -> None:
    assert cp.resolve_fluid_spec("Water") == ("HEOS", "Water", None)
    assert cp.resolve_fluid_spec("CarbonDioxide") == ("HEOS", "CarbonDioxide", None)


def test_backend_folded_into_the_name() -> None:
    assert cp.resolve_fluid_spec("IF97::Water") == ("IF97", "Water", None)
    assert cp.resolve_fluid_spec("HEOS::Water") == ("HEOS", "Water", None)
    assert cp.resolve_fluid_spec("BICUBIC&HEOS::Water") == ("BICUBIC&HEOS", "Water", None)


def test_water_name_constant_is_the_if97_fluid() -> None:
    assert cp.WATER_NAME == "IF97::Water"
    assert cp.resolve_fluid_spec(cp.WATER_NAME) == ("IF97", "Water", None)


def test_mixture_fractions_in_the_name_are_normalised() -> None:
    backend, fluids, fractions = cp.resolve_fluid_spec("HEOS::CO2[0.7]&O2[0.3]")

    assert (backend, fluids) == ("HEOS", "CO2&O2")
    assert fractions is not None
    assert fractions == pytest.approx([0.7, 0.3])


def test_mixture_via_composition_dict() -> None:
    backend, fluids, fractions = cp.resolve_fluid_spec("HEOS", composition={"CO2": 0.7, "O2": 0.3})

    assert (backend, fluids) == ("HEOS", "CO2&O2")
    assert fractions == pytest.approx([0.7, 0.3])

    # the species may also be spelled out in the name, as long as they agree
    assert cp.resolve_fluid_spec("HEOS::CO2&O2", composition={"CO2": 0.7, "O2": 0.3}) == (backend, fluids, fractions)


def test_composition_sum_tolerance() -> None:
    # a sum within COMPOSITION_SUM_TOLERANCE of 1 is normalised rather than rejected
    _, _, fractions = cp.resolve_fluid_spec("HEOS", composition={"CO2": 0.7, "O2": 0.305})
    assert fractions is not None
    assert sum(fractions) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="must sum to 1"):
        cp.resolve_fluid_spec("HEOS", composition={"CO2": 0.7, "O2": 0.9})


def test_incompressible_concentration_is_not_normalised() -> None:
    # a single species with a fraction is a concentration on the fluid's own basis, absolute
    backend, fluids, fractions = cp.resolve_fluid_spec("INCOMP::MEG[0.5]")

    assert (backend, fluids) == ("INCOMP", "MEG")
    assert fractions == pytest.approx([0.5])


@pytest.mark.parametrize(
    ("name", "composition", "match"),
    [
        ("HEOS", {"CO2": 1.0}, "at least two species"),
        ("HEOS", {}, "at least two species"),
        ("HEOS::CO2[0.5]&O2[0.5]", {"CO2": 0.5, "O2": 0.5}, "cannot set the composition both"),
        ("IF97", {"CO2": 0.5, "O2": 0.5}, "requires a mixture backend"),
        ("HEOS::CO2&N2", {"CO2": 0.5, "O2": 0.5}, "do not match"),
        ("HEOS", {"CO2": 0.0, "O2": 0.0}, "positive value"),
        ("HEOS", {"CO2": -0.5, "O2": 1.5}, "non-negative"),
        ("HEOS", {"CO2": float("nan"), "O2": 0.5}, "finite"),
    ],
)
def test_invalid_composition(name: str, composition: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cp.resolve_fluid_spec(name, composition=composition)


def test_composition_rejects_non_float_fractions() -> None:
    # bool is an int subclass, so it would otherwise slip through as 1.0
    with pytest.raises(TypeError, match="must be a float"):
        cp.resolve_fluid_spec("HEOS", composition=cast("cp.Composition", {"CO2": True, "O2": 0.5}))

    with pytest.raises(TypeError, match="must be a float"):
        cp.resolve_fluid_spec("HEOS", composition=cast("cp.Composition", {"CO2": "0.7", "O2": 0.3}))


def test_integer_fractions_are_coerced_to_float() -> None:
    _, _, fractions = cp.resolve_fluid_spec("HEOS", composition={"CO2": 1, "O2": 0})

    assert fractions is not None
    assert fractions == [1.0, 0.0]
    assert all(isinstance(fraction, float) for fraction in fractions)

    # they are still mole fractions, so they must sum to 1
    with pytest.raises(ValueError, match="must sum to 1"):
        cp.resolve_fluid_spec("HEOS", composition={"CO2": 1, "O2": 1})


def test_is_fluid_input() -> None:
    assert cp.is_fluid_input("P")
    assert cp.is_fluid_input("DMASS")
    assert not cp.is_fluid_input("VISCOSITY")  # a valid output, but not a state input
    assert not cp.is_fluid_input("nonsense")

    assert cp.FLUID_INPUTS <= cp.FLUID_PARAMS


def test_is_humid_air_input() -> None:
    assert cp.is_humid_air_input("T")
    assert cp.is_humid_air_input("RelHum")
    assert not cp.is_humid_air_input("Visc")  # output only
    assert not cp.is_humid_air_input("nonsense")

    assert cp.HUMID_AIR_INPUTS <= cp.HUMID_AIR_PARAMS


def test_phase_ignoring_backends() -> None:
    assert "IF97" in cp.PHASE_IGNORING_BACKENDS
    assert "HEOS" not in cp.PHASE_IGNORING_BACKENDS
