# Working on encomp

encomp is a typed units/quantities library for process engineering: a pint-based
`Quantity[DT, MT]` generic over dimensionality (`DT`, e.g. `Pressure`) and magnitude type
(`MT`: `float`, 1-D `np.ndarray`, `pl.Series` or `pl.Expr`), plus CoolProp fluid
properties (`encomp.fluids`) with a native Rust Polars plugin for GIL-free parallel
evaluation. The [README](README.md) covers the user-facing API; this file covers the
conventions that are easy to get wrong when changing the code.

## Three type checkers, one verdict

CI gates on **pyright (strict), pyrefly and ty** — all three must pass with zero errors:

```bash
uv run pyright
uv run pyrefly check
uv run ty check --error-on-warning
```

Rules of engagement:

- **pyright is the primary checker and the source of truth.** When checkers disagree,
  the code stays as pyright wants it and the disagreeing checker is suppressed on that
  line. Never restructure working code just to appease pyrefly or ty — but do prefer a
  real code improvement over a suppression when one exists that all three accept.
- **Suppressions are per-checker and rule-specific.** Blanket `# type: ignore` is
  disabled for pyright (`enableTypeIgnoreComments = false`) and effectively banned
  everywhere. Each checker only sees its own directive, so they stack on one line:

  ```text
  def check(  # pyright: ignore[reportIncompatibleMethodOverride]  # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
  ```

- **Unused suppressions are errors in all three** (`reportUnnecessaryTypeIgnoreComment`,
  `unused-ignore`, `unused-ignore-comment`), so a fixed ignore must be deleted in the
  same change that fixes it.
- Typical reasons a ty-only ignore exists (pyright and pyrefly accept the same line):
  constrained-TypeVar solving on `MT`, TypeVar defaults materialized during `isinstance`
  narrowing, and `float`-vs-`int | float` promotion making a pyright-required `cast()`
  look redundant. Do not "fix" these by weakening annotations or removing casts —
  removing a cast that pyright strict needs to launder `Unknown` fails the pyright gate.
- `Q("24 kg")` (string magnitude) and `Q(1, Q(2, "m"))` (Quantity as unit) must be
  rejected *statically by all three checkers* and at runtime;
  `encomp/tests/test_static_typing.py` pins this. Don't add overloads that would let
  either slip through.

## Documentation is executed

Every ` ```python ` fence in every Markdown file in the repo — including this one — is
extracted by `encomp/tests/test_docs.py`, ruff-checked, type-checked by all three
checkers and **run in isolation**. There is no skip or no-run escape. A doc block must
carry its own imports and pass the same gates as source code:

```python
from encomp.units import Quantity as Q

pressure = Q(1.0, "bar")
assert pressure.to("kPa").m == 100.0
```

## Library invariants

- **Magnitude and unit are always separate arguments.** No string parsing of
  quantities, anywhere, including docs and tests.
- **No implicit physical identities.** A fluid name, density or similar physical
  identity is never defaulted — APIs require it explicitly (`cp.fluid(name=...)`;
  `cp.water()` exists for the common case). Don't add parameter defaults that encode
  physics.
- `bool` magnitudes are rejected at runtime (a bool is always a mistake), `int`
  magnitudes normalize to `float`, and only 1-D arrays are accepted.
- `Temperature` and `TemperatureDifference` are distinct dimensionalities with
  deliberate arithmetic (`T - T -> ΔT`, `T ± ΔT -> T`, `ΔT - T` is an error). The
  comparison and pickling overrides on `Quantity` intentionally narrow pint's
  signatures — that's the point of the library, not an LSP bug to fix.
- CoolProp evaluation is **one property per DAG node** by design; don't batch multiple
  output properties into one plugin call. All CoolProp paths hold the GIL — the Rust
  plugin (`rust/`) exists precisely to evaluate without it; don't add Python-side
  parallelism around CoolProp.

## Tooling

- `uv` with a committed lockfile (`uv sync --locked`); `ruff check` + `ruff format`
  (enforced by pre-commit and CI).
- Mixed Python + Rust project built with maturin. Development checkouts build the
  plugin locally: `python scripts/build_libcoolprop.py` then
  `uv run maturin develop --release` (see `encomp/coolprop/README.md`).
- Tests: `uv run pytest`. The suite ships in the wheel and must stay runnable from an
  installed package (checker-dependent tests skip when the tool is absent).
- CI (`.github/workflows/release.yml`): typecheck (ruff + all three checkers + doc
  execution), ReadTheDocs-parity docs build, dependency floors (lowest allowed versions
  on Python 3.13 — validate new floors on the newest Python too), Rust checks, wheels
  per OS, and tag-triggered PyPI publish.
- Release: bump the version in `pyproject.toml` *and* `uv.lock`, PR, then tag
  `v<version>`; the workflow publishes to PyPI and creates the GitHub Release. Delete
  merged branches.
