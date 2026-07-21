#!/usr/bin/env python
"""Benchmark the direct scalar bridge and existing Polars batch path.

Run after a release build (``uv run maturin develop --release``). The optional
Python CoolProp oracle is reported only when the ``oracle`` dependency group is
installed; it is never needed by encomp itself.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import statistics
import time
import timeit
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from encomp import coolprop as cp
from encomp.fluids import Fluid, HumidAir
from encomp.units import Quantity as Q


def median_seconds(function: Callable[[], object], *, number: int, repeat: int) -> float:
    return statistics.median(timeit.repeat(function, number=number, repeat=repeat)) / number


def benchmark(*, array_size: int, repeat: int, quick: bool) -> dict[str, float | int | list[int]]:
    native = cp._native()  # pyright: ignore[reportPrivateUsage]
    pair = native.resolve_input_pair("P", "T")[0]
    density = native.parameter_index("DMASS")
    if97_config = native.prepare_fluid("IF97", "Water")
    heos_config = native.prepare_fluid("HEOS", "CarbonDioxide")
    humid_config = native.prepare_humid_air("W", "P", "T", "R")

    native.clear_scalar_cache()
    started_first_call = time.perf_counter_ns()
    native.fluid_scalar(if97_config, pair, 101325.0, 300.0, density)
    first_call_us = (time.perf_counter_ns() - started_first_call) / 1e3

    library = ctypes.CDLL(cp.lib_path())
    props_si: Any = library.PropsSI
    props_si.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_char_p,
    ]
    props_si.restype = ctypes.c_double
    ha_props_si: Any = library.HAPropsSI
    ha_props_si.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_char_p,
        ctypes.c_double,
        ctypes.c_char_p,
        ctypes.c_double,
    ]
    ha_props_si.restype = ctypes.c_double

    if97 = Fluid("IF97::Water", P=Q(101325.0, "Pa"), T=Q(300.0, "K"))
    heos = Fluid("HEOS::CarbonDioxide", P=Q(5e6, "Pa"), T=Q(300.0, "K"))
    humid = HumidAir(P=Q(101325.0, "Pa"), T=Q(300.0, "K"), R=Q(0.5, ""))
    _ = (if97.D.m, heos.D.m, humid.W.m)

    scalar_number = 2_000 if quick else 20_000
    fast_number = 10_000 if quick else 100_000
    results: dict[str, float | int | list[int]] = {
        "array_size": array_size,
        "native_if97_first_call_us": first_call_us,
        "native_if97_us": median_seconds(
            lambda: native.fluid_scalar(if97_config, pair, 101325.0, 300.0, density),
            number=fast_number,
            repeat=repeat,
        )
        * 1e6,
        "public_if97_us": median_seconds(lambda: if97.D.m, number=scalar_number, repeat=repeat) * 1e6,
        "native_heos_us": median_seconds(
            lambda: native.fluid_scalar(heos_config, pair, 5e6, 300.0, density),
            number=scalar_number,
            repeat=repeat,
        )
        * 1e6,
        "public_heos_us": median_seconds(lambda: heos.D.m, number=scalar_number, repeat=repeat) * 1e6,
        "native_humid_air_us": median_seconds(
            lambda: native.humid_air_scalar(humid_config, 101325.0, 300.0, 0.5),
            number=fast_number,
            repeat=repeat,
        )
        * 1e6,
        "public_humid_air_us": median_seconds(lambda: humid.W.m, number=scalar_number, repeat=repeat) * 1e6,
        "bundled_cabi_if97_us": median_seconds(
            lambda: props_si(b"DMASS", b"P", 101325.0, b"T", 300.0, b"IF97::Water"),
            number=fast_number,
            repeat=repeat,
        )
        * 1e6,
        "bundled_cabi_heos_us": median_seconds(
            lambda: props_si(b"DMASS", b"P", 5e6, b"T", 300.0, b"HEOS::CarbonDioxide"),
            number=scalar_number,
            repeat=repeat,
        )
        * 1e6,
        "bundled_cabi_humid_air_us": median_seconds(
            lambda: ha_props_si(b"W", b"P", 101325.0, b"T", 300.0, b"R", 0.5),
            number=fast_number,
            repeat=repeat,
        )
        * 1e6,
    }

    temperatures = tuple(float(value) for value in np.linspace(290.0, 310.0, 127))
    varying_index = 0

    def varying_heos() -> float:
        nonlocal varying_index
        value = native.fluid_scalar(heos_config, pair, 5e6, temperatures[varying_index % len(temperatures)], density)
        varying_index += 1
        return value

    results["native_heos_varying_state_us"] = (
        median_seconds(
            varying_heos,
            number=scalar_number,
            repeat=repeat,
        )
        * 1e6
    )

    def cold_if97() -> float:
        native.clear_scalar_cache()
        return native.fluid_scalar(if97_config, pair, 101325.0, 300.0, density)

    results["native_if97_cache_miss_us"] = (
        median_seconds(
            cold_if97,
            number=1,
            repeat=max(repeat, 11),
        )
        * 1e6
    )

    eviction_configs = [
        native.prepare_fluid("HEOS", fluid)
        for fluid in (
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
    ]
    for config in eviction_configs:
        native.fluid_scalar(config, pair, 101325.0, 300.0, density)
    native.clear_scalar_cache()

    def fill_and_evict() -> None:
        native.clear_scalar_cache()
        for config in eviction_configs:
            native.fluid_scalar(config, pair, 101325.0, 300.0, density)

    results["native_cache_fill_and_eviction_us_per_config"] = (
        median_seconds(fill_and_evict, number=1, repeat=repeat) / len(eviction_configs) * 1e6
    )

    pressure = np.full(array_size, 5e6)
    temperature = np.linspace(290.0, 310.0, array_size)
    eager = Fluid("HEOS::CarbonDioxide", P=Q(pressure, "Pa"), T=Q(temperature, "K"))
    _ = eager.D.m
    array_seconds = median_seconds(lambda: eager.D.m, number=1, repeat=repeat)
    results["heos_array_ms"] = array_seconds * 1e3
    results["heos_array_mrows_s"] = array_size / array_seconds / 1e6

    frame = pl.DataFrame({"P": pressure, "T": temperature})
    expressions = [cp.fluid(prop, "P", "T", name="HEOS::CarbonDioxide") for prop in ("DMASS", "HMASS", "SMASS")]
    frame.select(expressions)
    multi_seconds = median_seconds(lambda: frame.select(expressions), number=1, repeat=repeat)
    results["three_property_polars_ms"] = multi_seconds * 1e3
    results["three_property_polars_mprops_s"] = 3 * array_size / multi_seconds / 1e6
    results["scalar_cache_info"] = list(native.scalar_cache_info())

    try:
        oracle: Any = importlib.import_module("CoolProp.CoolProp")
    except ModuleNotFoundError:
        pass
    else:
        results["oracle_if97_us"] = (
            median_seconds(
                lambda: oracle.PropsSI("DMASS", "P", 101325.0, "T", 300.0, "IF97::Water"),
                number=fast_number,
                repeat=repeat,
            )
            * 1e6
        )
        results["oracle_heos_us"] = (
            median_seconds(
                lambda: oracle.PropsSI("DMASS", "P", 5e6, "T", 300.0, "HEOS::CarbonDioxide"),
                number=scalar_number,
                repeat=repeat,
            )
            * 1e6
        )
        results["oracle_humid_air_us"] = (
            median_seconds(
                lambda: oracle.HAPropsSI("W", "P", 101325.0, "T", 300.0, "R", 0.5),
                number=fast_number,
                repeat=repeat,
            )
            * 1e6
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-size", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    results = benchmark(array_size=args.array_size, repeat=args.repeat, quick=args.quick)
    results["elapsed_s"] = time.perf_counter() - started
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
