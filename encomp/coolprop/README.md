# encomp.coolprop

One bundled CoolProp implementation behind a dual-purpose native Rust artifact.
Its private PyO3 interface handles scalar evaluation, metadata, parsing, and
validation; its Polars plugin ABI handles every array and expression size.
Independent property nodes in one `collect()` run in parallel on the Polars
thread pool without holding the GIL. The Python `CoolProp` package is neither a
runtime dependency nor imported by `encomp`.

## Usage

```python
import polars as pl

from encomp import coolprop as cp

df = pl.DataFrame({"P": [50e5, 60e5], "T": [400.0, 450.0], "R": [0.4, 0.6]})  # Pa, K, -

df.select(
    cp.water("DMASS", "P", "T").alias("rho"),   # IF97 water/steam
    cp.water("HMASS", "P", "T").alias("h"),     # runs in parallel
    cp.humid_air("W", "P", "T", "R").alias("humidity_ratio"),
)
```

The API mirrors `encomp.fluids.Fluid`:

- `fluid(output, in1, in2, *, name, assume_phase=None,
  composition=None)` — each input names its property (a string, or an expression's
  output name, e.g. `pl.col("p").alias("P")`); both must be CoolProp state inputs (any
  pair: PT, PH, PQ, ...). `output` may be any property. `name` is required, as it is for
  `encomp.fluids.Fluid`. The fluid is `name` with the
  backend folded in (`name="HEOS::CarbonDioxide"`); a mixture is given by fractions in the
  name (`"HEOS::CO2[0.5]&O2[0.5]"`) or a `composition={species: mole fraction}` dict (mole
  fractions must sum to 1); an incompressible mixture instead carries a single concentration
  in the name (`"INCOMP::MEG[0.5]"`, on the fluid's own mass/volume basis). An assumed phase
  is `assume_phase="gas"` (skips the phase flash, HEOS/GERG only; a region-explicit backend
  such as IF97 ignores it and a warning is logged).
- `water(output, in1, in2)` — the IF97 water/steam shorthand, mirroring
  `encomp.fluids.Water`. Equivalent to `fluid(..., name="IF97::Water")`. It takes no
  `composition` (water is pure) and no `assume_phase` (IF97 would ignore it).
- `humid_air(output, in1, in2, in3)` — same naming rule for the three humid-air inputs.
- `FluidInput` / `HumidAirInput` (state inputs), `FluidParam` / `HumidAirParam` /
  `Backend` / `Phase` / `AssumedPhase` are `Literal`s; `CName` / `Composition` mirror
  `encomp.fluids`. The matching `frozenset`s and `is_*` `TypeIs` predicates are exported
  too, along with `resolve_fluid_spec` (name → backend/fluids/fractions).

## Performance regression gate

Apple M4 Pro, CPython 3.14.3, Polars 1.42.1, CoolProp 8.0.0; medians of seven
warmed runs. Lower is better. The baseline is the unmodified `v1.8.0` wheel.

| workload | v1.8.0 | native-only | change |
| --- | ---: | ---: | ---: |
| public IF97 scalar | 62.49 us | 59.68 us | -4.5% |
| public HEOS scalar | 97.36 us | 65.14 us | -33.1% |
| public humid-air scalar | 57.35 us | 55.30 us | -3.6% |
| HEOS array, 100k rows | 470.49 ms | 469.92 ms | -0.1% |
| three independent HEOS properties, 100k rows | 534.06 ms | 537.51 ms | +0.6% |

The direct bridge itself takes 0.20 us for a warm IF97 state, 4.88 us for a
varying-temperature cached HEOS state, and 2.05 us for humid air. The matching
high-level bundled C calls take 1.06 us, 31.05 us, and 2.33 us respectively;
the state-owning bridge is faster where it can reuse `AbstractState`. A cold
first IF97 call is 28.54 us, and a fill-plus-eviction pass across 17 HEOS
configurations averages 29.37 us/configuration. Reproduce with
`scripts/benchmark_coolprop.py`.

The matching macOS arm64 package check reduces the combined wheel download from
about 19.6 MB (`encomp` plus Python CoolProp) to about 9 MB for `encomp` alone
(roughly -54%). The encomp wheel itself is only slightly larger because the old
Python binding was a separate distribution; removing that separate wheel produces
the net saving. The bundled CoolProp distribution terms are shipped as
`encomp/coolprop/LICENSE.CoolProp`.

## Design

The direct scalar interface keeps a bounded LRU of `AbstractState` handles in
thread-local storage. A handle is mutated only by its owning thread and is freed
on eviction or thread shutdown. The plugin uses the batched C-API
(`AbstractState_update_and_1_out`): one call per chunk, with the handle-table lock
taken only for construction/destruction and the flash loop lock-free in C++.

Thread-safety (is it safe to evaluate concurrently?): the math is pure — a flash on
a per-handle `AbstractState` mutates no global state — safe to run concurrently
**iff** (a) each thread owns its handle (never shared; a state caches its last
flash), (b) handle create/destroy is synchronized (the global handle table is the
one shared structure — `coolprop.rs` guards `factory`/`free` with a narrow mutex,
never the hot path), and (c) global config isn't mutated during evaluation. The
PyO3 module stores no Python objects in global state and explicitly declares that
it does not require the GIL.

All `unsafe` is confined to `rust/src/coolprop.rs` (the FFI boundary); `lib.rs` has none.
Every `unsafe` block carries a `// SAFETY:` comment (rationale + error modes),
enforced by `clippy::undocumented_unsafe_blocks`. The safe wrapper uses an RAII
`State` (frees its handle on drop), length-checked slices, and stack-local error
buffers. libCoolProp is loaded at runtime via `libloading`, so the same plugin
works on macOS/Linux/Windows by shipping that platform's `.dylib`/`.so`/`.dll`.

Humid-air caveat: `humid_air` is correct and GIL-free, but CoolProp's humid-air
solver is internally iterative and more serialized than the fluid path.

## Caveats

- **Bound to the `polars-ffi` ABI, not the polars version.** The plugin is loaded over
  polars' plugin ABI (`polars-ffi`, currently **major 0**, minor 1). The host polars
  dispatches plugin calls on that ABI's *major* version and does not gate on the minor
  (`polars-plan/.../plugin.rs`: `if major == 0 { use version_0::* }`), so a plugin built
  against py-polars 1.42 keeps working on later 1.x releases — it is **not** pinned per
  minor (verified across 1.42.0 → 1.42.1). This matches how community plugins ship: one
  wheel per platform with a *minimum* polars, not a wheel per polars minor. A rebuild
  (bumping `pyo3-polars` to the new Rust polars) is needed only if polars bumps the ffi
  ABI **major** — rare and announced. `self_check()` evaluates a known value at first use,
  so a genuinely incompatible polars surfaces as a clear error, never a silent wrong result.
- **One version source**: `encomp/coolprop/_build_info.py` pins the bundled version and
  upstream commit. Parameter indexes and input-pair enum values are resolved at runtime
  from that bundled library; no numeric CoolProp enum is hardcoded in Python or Rust.
- **One property per node (no output batching).** Each `fluid(...)` / `humid_air(...)` is an
  independent plugin node, so selecting K properties of one state runs K flashes of it —
  Polars cannot reuse (CSE) the shared flash across opaque plugin nodes. Independent
  properties still parallelize, so this is about total work, not wall-clock.

## Build / install

This is part of `encomp`: the whole package (Python + the dual-purpose `_internal`
artifact + bundled `libCoolProp`) ships in one per-platform wheel. `libCoolProp` is
built from source once and bundled, so every build starts with
`scripts/build_libcoolprop.py`.

Dev (from the repo root, editable):

```bash
python scripts/build_libcoolprop.py   # builds CoolProp's shared lib into encomp/coolprop/
maturin develop --release             # builds the plugin + installs encomp editable
```

Cross-platform distribution wheels (the plugin + bundled `libCoolProp` are native;
`abi3` → one wheel per platform covers CPython >=3.13) are built with `cibuildwheel`
via `.github/workflows/release.yml`:

| target  | on this macOS host         | in CI                                  |
|---------|----------------------------|----------------------------------------|
| macOS   | yes (native)               | macos-14 (arm64)                       |
| Linux   | yes, via Docker (manylinux)| ubuntu (x86_64) + ubuntu-arm (aarch64) |
| Windows | no — needs Windows         | windows-latest (amd64)                 |

`cibuildwheel` runs `build_libcoolprop.py` per platform (its `before-all`), bundles
the lib, and emits PyPI-compatible tags. Lint/format: `cargo fmt`, `cargo clippy
--all-targets -- -D warnings` (wired into the repo's pre-commit).

Rust unit tests (the pure helpers: broadcast, output dtype, error-buffer decoding)
need `extension-module` off so the test binary can link libpython (see Cargo.toml):

```bash
PYO3_PYTHON=$(pwd)/.venv/bin/python cargo test --manifest-path rust/Cargo.toml --no-default-features
```

## Files

- `encomp/coolprop/__init__.py` — public Python API (`fluid`, `water`, `humid_air`).
- `rust/src/lib.rs` — the PyO3 scalar API plus `cp_evaluate` / `ha_evaluate` plugin expressions.
- `rust/src/coolprop.rs` — the CoolProp C-API bindings + thread-safety model (all `unsafe`).
- `scripts/build_libcoolprop.py` — builds + bundles CoolProp's shared library.
