# encomp-coolprop

Parallel CoolProp property evaluation as **native Polars expression plugins**
(Rust, `pyo3-polars`). Independent property nodes in one `collect()` run in
parallel on the Polars thread pool — GIL-free — instead of a `map_batches` Python
UDF that holds the GIL and serializes. Usable directly, or as the `"rust"`
backend of `encomp.fluids` (`settings.coolprop_backend`).

## Usage

```python
import polars as pl
import encomp_coolprop as cp

df = pl.DataFrame({"P": [50e5, 60e5], "T": [400.0, 450.0]})  # Pa, K

df.select(
    cp.fluid("DMASS", "P", "T", backend="IF97", fluids="Water").alias("rho"),
    cp.fluid("HMASS", "P", "T", backend="IF97", fluids="Water").alias("h"),   # parallel
    cp.humid_air("W", "P", "T", "R").alias("humidity_ratio"),
)
```

- `fluid(output, in1, in2, *, name1="P", name2="T", backend="HEOS", fluids="Water",
  phase=None, mole_fractions=None)` — any CoolProp input pair (any order, via
  `name1`/`name2`), mixtures (`fluids="CO2&O2"` + `mole_fractions`), and an assumed
  phase (`phase="phase_gas"`, skips the phase flash).
- `humid_air(output, in1, in2, in3, *, name1="P", name2="T", name3="R")`.
- `Backend` / `Phase` are `Literal`s; `BACKENDS` / `PHASES` / `FLUID_PARAMS` /
  `HUMID_AIR_PARAMS` are the matching runtime `frozenset`s (`get_args`).

## Performance (CoolProp 8.0, 14-thread pool)

```
CORRECTNESS  plugin vs raw PropsSI (IF97 water): 0.0e+00 exact (D/H/S/Cp)

SPEED (property D)            raw PropsSI | encomp map_batches | rust plugin | vs m_b
  N=1                            0.003 ms |          0.114 ms |    0.029 ms |  3.98x
  N=1,000,000                  177.6   ms |        184.1   ms |   86.6   ms |  2.12x

PARALLELISM  4 independent props, ONE collect(), 1M rows
  encomp map_batches : combined 823 ms,  1.00x   serial (GIL)
  rust plugin        : combined 178 ms,  2.57x   PARALLEL  (4.6x faster than encomp)

8 enthalpy calcs, 1M rows: 4.9x faster, ~6 cores vs 1, ~half the peak memory.
```

## Design

Removing the GIL is necessary but not sufficient: CoolProp's C-API takes a global
handle-table lock on every call, so per-row calls serialize even in pure Rust. The
plugin instead uses the **batched** C-API (`AbstractState_update_and_1_out`): one
call per chunk, the handle lock taken once at construction, then the flash loop
runs lock-free in C++ — so independent chunks/expressions parallelize.

Thread-safety (is it safe to evaluate concurrently?): the math is pure — a flash on
a per-handle `AbstractState` mutates no global state — safe to run concurrently
**iff** (a) each thread owns its handle (never shared; a state caches its last
flash), (b) handle create/destroy is synchronized (the global handle table is the
one shared structure — `coolprop.rs` guards `factory`/`free` with a narrow mutex,
never the hot path), and (c) global config isn't mutated during evaluation. Results
are bit-identical across 1/2/4/8 threads.

All `unsafe` is confined to `src/coolprop.rs` (the FFI boundary); `lib.rs` has none.
Every `unsafe` block carries a `// SAFETY:` comment (rationale + error modes),
enforced by `clippy::undocumented_unsafe_blocks`. The safe wrapper uses an RAII
`State` (frees its handle on drop), length-checked slices, and stack-local error
buffers. libCoolProp is loaded at runtime via `libloading`, so the same plugin
works on macOS/Linux/Windows by shipping that platform's `.dylib`/`.so`/`.dll`.

HumidAir caveat: `humid_air` is correct and GIL-free, but `HAPropsSI` is internally
more serialized than the Fluid path — it parallelizes only ~1.25x (vs ~2.5x) and is
much slower per call (iterative solver). Still better than the Python path.

## Caveats

- **Locked to a polars minor version** (`polars-ffi` ABI; built against Rust polars
  0.54.4 = py-polars 1.42.x). A polars upgrade needs a plugin rebuild — the
  `settings.coolprop_backend` rust/python runtime fallback makes that safe.
- **Version match**: CoolProp enum integers differ across versions; pin Python
  `coolprop==8.0.0` to match the bundled Rust lib. The plugin resolves parameter
  indices via CoolProp at runtime, never hardcoded.

## Build

```bash
maturin develop --release          # build + install into the active venv
# then place the platform libCoolProp next to the built _internal.* in the package
```

Production packaging = maturin + cibuildwheel, bundling `libCoolProp` 8.0 per
platform (`[tool.maturin] include`). Lint/format: `cargo fmt`, `cargo clippy
--all-targets -- -D warnings` (also wired into the repo's pre-commit).

## Files

- `python/encomp_coolprop/__init__.py` — public Python API (`fluid`, `humid_air`).
- `src/lib.rs` — the `cp_evaluate` / `ha_evaluate` plugin expressions.
- `src/coolprop.rs` — the CoolProp C-API bindings + thread-safety model (all `unsafe`).
