"""Locks the hand-maintained CoolProp name registries together.

The same name sets are encoded twice: as ``Literal`` types in ``encomp.coolprop``
(static typing, single source for the plugin) and as runtime maps in
``encomp.fluids`` (``PROPERTY_MAP`` / ``RETURN_UNITS`` / the ``FluidState`` /
``HumidAirState`` TypedDicts). A name added to one side but not the other fails
confusingly (or silently loses typing), so this module asserts they are identical
-- the same role ``test_overload_conformance`` plays for the units algebra.
"""

import typing
from collections.abc import Iterable

import CoolProp.CoolProp as _CoolProp

from .. import coolprop as cp
from ..fluids import CoolPropFluid, Fluid, FluidState, HumidAir, HumidAirState
from ..units import Quantity


def _assert_equal_sets(a: Iterable[str], b: Iterable[str], what: str) -> None:
    assert set(a) == set(b), (
        f"{what}: only in first: {sorted(set(a) - set(b))}, only in second: {sorted(set(b) - set(a))}"
    )


def test_fluid_property_map_matches_literal() -> None:
    _assert_equal_sets(
        CoolPropFluid.ALL_PROPERTIES,
        cp.FLUID_PARAMS,
        "fluids.CoolPropFluid.PROPERTY_MAP vs coolprop.FluidParam",
    )


def test_humid_air_property_map_matches_literal() -> None:
    _assert_equal_sets(
        HumidAir.ALL_PROPERTIES,
        cp.HUMID_AIR_PARAMS,
        "fluids.HumidAir.PROPERTY_MAP vs coolprop.HumidAirParam",
    )


def test_state_typed_dicts_match_input_literals() -> None:
    _assert_equal_sets(
        set(typing.get_type_hints(FluidState)),
        cp.FLUID_INPUTS,
        "fluids.FluidState keys vs coolprop.FluidInput",
    )
    _assert_equal_sets(
        set(typing.get_type_hints(HumidAirState)),
        cp.HUMID_AIR_INPUTS,
        "fluids.HumidAirState keys vs coolprop.HumidAirInput",
    )


def test_inputs_are_subsets_of_params() -> None:
    assert cp.FLUID_INPUTS <= cp.FLUID_PARAMS
    assert cp.HUMID_AIR_INPUTS <= cp.HUMID_AIR_PARAMS


def test_return_units_are_known_properties() -> None:
    assert set(CoolPropFluid.RETURN_UNITS) <= CoolPropFluid.ALL_PROPERTIES
    assert set(HumidAir.RETURN_UNITS) <= HumidAir.ALL_PROPERTIES


def test_property_map_units_parse() -> None:
    for cls in (CoolPropFluid, HumidAir):
        for names, (unit_str, _) in cls.PROPERTY_MAP.items():
            Quantity.get_unit(unit_str)  # raises if the unit string is invalid
            assert names, "empty name tuple in PROPERTY_MAP"


def test_typed_properties_are_known_names() -> None:
    # every typed @property on Fluid / HumidAir (except helpers) must be a name
    # from the class's own property map, so attribute access can never diverge
    helpers = {"phase"}
    for cls in (Fluid, HumidAir):
        for name, member in vars(cls).items():
            if isinstance(member, property) and not name.startswith("_") and name not in helpers:
                assert name in cls.ALL_PROPERTIES, f"{cls.__name__}.{name} is not in its PROPERTY_MAP"


def test_fluid_params_are_valid_per_coolprop() -> None:
    # CoolProp's parameter index is the runtime authority for the fluid namespace;
    # every FluidParam Literal must resolve (guards against typos in the Literal).
    # (HAPropsSI has no equivalent lookup, so the humid-air names are locked only
    # against fluids.HumidAir.PROPERTY_MAP above.)
    for name in sorted(cp.FLUID_PARAMS):
        _CoolProp.get_parameter_index(name)  # raises ValueError for unknown names
