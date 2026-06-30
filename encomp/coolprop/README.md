# encomp.coolprop

Parallel CoolProp property evaluation as **native Polars expression plugins**
(Rust, `pyo3-polars`). Independent property nodes in one `collect()` run in
parallel on the Polars thread pool — GIL-free — instead of a `map_batches` Python
UDF that holds the GIL and serializes. Usable directly, or as the evaluation path
of `encomp.fluids` for `pl.Expr` inputs and large eager arrays (≥1000 elements);
small eager inputs (scalars, short arrays) use the Python CoolProp path.

## Usage

```python
import polars as pl
from encomp import coolprop as cp

df = pl.DataFrame({"P": [50e5, 60e5], "T": [400.0, 450.0]})  # Pa, K

df.select(
    cp.fluid("DMASS", "P", "T", fluid="Water").alias("rho"),
    cp.fluid("HMASS", "P", "T", fluid="Water").alias("h"),   # runs in parallel
    cp.humid_air("W", "P", "T", "R").alias("humidity_ratio"),
)
```

- `fluid(output, in1, in2, *, backend="IF97", fluid="Water", phase=None,
  mole_fractions=None)` — each input names its property (a string, or an
  expression's output name, e.g. `pl.col("p").alias("P")`); both must be CoolProp
  state inputs (any pair: PT, PH, PQ, ...). `output` may be any property. Supports
  mixtures (`fluid="CO2&O2"` + `mole_fractions`) and an assumed phase
  (`phase="phase_gas"`, skips the phase flash).
- `humid_air(output, in1, in2, in3)` — same naming rule for the three HAPropsSI inputs.
- `FluidInput` / `HumidAirInput` (state inputs), `FluidParam` / `HumidAirParam` /
  `Backend` / `Phase` are `Literal`s; the matching `frozenset`s and `is_*` `TypeIs`
  predicates are exported too.

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

All `unsafe` is confined to `rust/src/coolprop.rs` (the FFI boundary); `lib.rs` has none.
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
  0.54.4 = py-polars 1.42.x). A polars upgrade needs a plugin rebuild: there is no
  Python fallback, so until the plugin is rebuilt, `pl.Expr` (lazy) CoolProp
  evaluation fails (eager numpy / `pl.Series` inputs are unaffected).
- **Version match**: CoolProp enum integers differ across versions; pin Python
  `coolprop==8.0.0` to match the bundled Rust lib. The plugin resolves parameter
  indices via CoolProp at runtime, never hardcoded.

## Build / install

This is part of `encomp`: the whole package (Python + this compiled plugin +
bundled `libCoolProp`) ships in ONE per-platform wheel. `libCoolProp` is built from
source once and bundled, so every build starts with `scripts/build_libcoolprop.py`.

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
| macOS   | yes (native)               | macos-13 (x86_64), macos-14 (arm64)    |
| Linux   | yes, via Docker (manylinux)| ubuntu (x86_64) + ubuntu-arm (aarch64) |
| Windows | no — needs Windows         | windows-latest (amd64)                 |

`cibuildwheel` runs `build_libcoolprop.py` per platform (its `before-all`), bundles
the lib, and emits PyPI-compatible tags. Lint/format: `cargo fmt`, `cargo clippy
--all-targets -- -D warnings` (wired into the repo's pre-commit).

## Files

- `encomp/coolprop/__init__.py` — public Python API (`fluid`, `humid_air`).
- `rust/src/lib.rs` — the `cp_evaluate` / `ha_evaluate` plugin expressions (crate at repo root).
- `rust/src/coolprop.rs` — the CoolProp C-API bindings + thread-safety model (all `unsafe`).
- `scripts/build_libcoolprop.py` — builds + bundles CoolProp's shared library.
