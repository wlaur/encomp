"""Tests for the direct PyO3 side of encomp's single native CoolProp artifact."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import importlib.metadata
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import pytest
from pytest import approx, raises

from encomp import coolprop as cp
from encomp.coolprop._build_info import BUNDLED_COOLPROP_VERSION
from encomp.fluids import Fluid, HumidAir, Water
from encomp.units import Quantity as Q
from encomp.utypes import Numpy1DArray

from ._coolprop_golden import GOLDEN_CASES, GoldenCase


def _golden_result(
    case: GoldenCase, mode: Literal["expr", "numpy", "scalar", "series"]
) -> float | Numpy1DArray | pl.Series:
    frame_data: dict[str, list[float]] = {}
    points: dict[str, Any] = {}
    for name, value, unit in case.inputs:
        if mode == "scalar":
            magnitude: Any = value
        elif mode == "numpy":
            magnitude = np.array([value, value])
        elif mode == "series":
            magnitude = pl.Series(name, [value, value])
        else:
            frame_data[name] = [value, value]
            magnitude = pl.col(name)
        points[name] = Q(magnitude, unit)

    if case.kind == "water":
        state: Any = cast(Any, Water)(**points)
    elif case.kind == "humid_air":
        state = cast(Any, HumidAir)(**points)
    else:
        assert case.name is not None
        state = cast(Any, Fluid)(case.name, composition=case.composition, **points)
        if case.phase is not None:
            state.assume_phase(case.phase)

    result = state.get(case.output).m
    if mode == "expr":
        return pl.DataFrame(frame_data).select(result.alias("result"))["result"]
    return cast("float | Numpy1DArray | pl.Series", result)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("mode", ("scalar", "numpy", "series", "expr"))
def test_committed_golden_cases(case: GoldenCase, mode: Literal["expr", "numpy", "scalar", "series"]) -> None:
    result = _golden_result(case, mode)
    if mode == "scalar":
        assert isinstance(result, float)
        assert result == approx(case.expected, rel=5e-10)
        return

    if isinstance(result, pl.Series):
        assert result.dtype == pl.Float64
        values = result.to_numpy()
    else:
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        values = result
    assert values.shape == (2,)
    assert values == approx(np.full(2, case.expected), rel=5e-10)


def test_native_metadata_and_pair_resolution() -> None:
    native = cp._native()
    assert native.lib_version() == BUNDLED_COOLPROP_VERSION
    assert native.parameter_information("P", "units") == "Pa"
    assert native.parameter_index("D") == native.parameter_index("DMASS")

    for first, second in (("P", "T"), ("P", "H"), ("P", "Q"), ("D", "T"), ("H", "S")):
        pair, swap = native.resolve_input_pair(first, second)
        reverse_pair, reverse_swap = native.resolve_input_pair(second, first)
        assert pair > 0 and reverse_pair == pair
        assert swap is not reverse_swap

    with raises(ValueError, match="unsupported CoolProp input pair"):
        native.resolve_input_pair("P", "P")
    with raises(ValueError, match="unknown parameter"):
        native.parameter_index("NOT_A_PROPERTY")


def test_native_name_parsing_and_validation() -> None:
    native = cp._native()
    assert native.resolve_fluid_name("Water") == ("HEOS", "Water", None)
    assert native.resolve_fluid_name("IF97::Water") == ("IF97", "Water", None)
    assert native.resolve_fluid_name("HEOS::CO2[0.5]&O2[0.5]") == ("HEOS", "CO2&O2", [0.5, 0.5])
    assert native.resolve_fluid_name("INCOMP::MEG[0.5]") == ("INCOMP", "MEG", [0.5])
    assert native.resolve_fluid_name("INCOMP::EG-20%") == ("INCOMP", "EG", [0.2])
    with raises(ValueError, match="between 0% and 100%"):
        native.resolve_fluid_name("INCOMP::EG-120%")

    native.validate_fluid("IF97", "Water")
    native.validate_fluid("HEOS", "CO2&O2", [0.5, 0.5])
    with raises(ValueError, match="factory"):
        native.validate_fluid("HEOS", "DefinitelyNotAFluid")


def test_direct_scalar_and_batch_paths_agree() -> None:
    pressure = 5e6
    temperature = 400.0
    scalar = Water(P=Q(pressure, "Pa"), T=Q(temperature, "K")).D.m
    eager = Water(P=Q(np.array([pressure, pressure]), "Pa"), T=Q(np.array([temperature, temperature]), "K")).D.m
    assert scalar == approx(float(eager[0]), rel=1e-12)

    scalar_heos = Fluid("HEOS::CarbonDioxide", P=Q(pressure, "Pa"), T=Q(temperature, "K")).D.m
    eager_heos = Fluid(
        "HEOS::CarbonDioxide",
        P=Q(np.array([pressure, pressure]), "Pa"),
        T=Q(np.array([temperature, temperature]), "K"),
    ).D.m
    assert scalar_heos == approx(float(eager_heos[0]), rel=1e-12)

    scalar_ha = HumidAir(P=Q(101325.0, "Pa"), T=Q(300.0, "K"), R=Q(0.5, "")).W.m
    eager_ha = HumidAir(
        P=Q(np.array([101325.0, 101325.0]), "Pa"),
        T=Q(np.array([300.0, 300.0]), "K"),
        R=Q(np.array([0.5, 0.5]), ""),
    ).W.m
    assert scalar_ha == approx(float(eager_ha[0]), rel=1e-12)


def test_thread_local_cache_eviction_and_destruction() -> None:
    native = cp._native()
    native.clear_scalar_cache()
    _, freed_before = native.handle_counts()

    fluids = (
        "Water",
        "CarbonDioxide",
        "Nitrogen",
        "Oxygen",
        "Methane",
        "Ammonia",
        "R134a",
        "n-Propane",
        "Hydrogen",
        "Argon",
        "Ethane",
        "CarbonMonoxide",
        "Helium",
        "Neon",
        "Krypton",
        "Xenon",
        "SulfurHexafluoride",
    )
    for fluid in fluids:
        value = Fluid(f"HEOS::{fluid}", P=Q(101325.0, "Pa"), T=Q(300.0, "K")).D.m
        assert math.isfinite(value)

    size, capacity, _hits, misses, evictions = native.scalar_cache_info()
    assert size == capacity == 16
    assert misses == len(fluids)
    assert evictions == 1

    native.clear_scalar_cache()
    assert native.scalar_cache_info()[0] == 0
    _, freed_after = native.handle_counts()
    assert freed_after - freed_before >= len(fluids)


def test_failed_scalar_flash_keeps_cached_state() -> None:
    native = cp._native()
    native.clear_scalar_cache()

    assert math.isnan(Water(P=Q(-1.0, "Pa"), T=Q(300.0, "K")).D.m)
    assert math.isfinite(Water(P=Q(1e5, "Pa"), T=Q(300.0, "K")).D.m)

    size, _capacity, hits, misses, _evictions = native.scalar_cache_info()
    assert size == 1
    assert (hits, misses) == (1, 1)


def test_concurrent_scalar_success_and_failure_are_isolated() -> None:
    def evaluate(index: int) -> float:
        pressure = 5e6 if index % 5 else -1.0
        value = Fluid("HEOS::CarbonDioxide", P=Q(pressure, "Pa"), T=Q(300.0, "K")).D.m
        assert isinstance(value, float)
        return value

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(evaluate, range(200)))

    for index, value in enumerate(values):
        assert math.isnan(value) if index % 5 == 0 else math.isfinite(value)


def test_runtime_import_does_not_load_python_coolprop() -> None:
    code = (
        "import sys; import encomp.coolprop, encomp.fluids; "
        "assert not any(n == 'CoolProp' or n.startswith('CoolProp.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

    requirements = importlib.metadata.requires("encomp") or []
    assert not any(requirement.lower().startswith("coolprop") for requirement in requirements)
