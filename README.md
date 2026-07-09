# encomp

[![PyPI](https://img.shields.io/pypi/v/encomp.svg)](https://pypi.org/project/encomp/)
[![Python versions](https://img.shields.io/pypi/pyversions/encomp.svg)](https://pypi.org/project/encomp/)
[![CI](https://github.com/wlaur/encomp/actions/workflows/release.yml/badge.svg)](https://github.com/wlaur/encomp/actions/workflows/release.yml)
[![Documentation](https://readthedocs.org/projects/encomp/badge/?version=latest)](https://encomp.readthedocs.io)
[![License](https://img.shields.io/pypi/l/encomp.svg)](https://github.com/wlaur/encomp/blob/main/LICENSE)

<img src="https://raw.githubusercontent.com/wlaur/encomp/v1.7.0/docs/img/logo.png" alt="encomp logo" width="150">

> General-purpose library for *en*gineering *comp*utations.

`encomp` is tested on Windows, Linux, and macOS (Apple Silicon), with Python 3.13 and 3.14.

## Overview

Every physical quantity in `encomp` carries a magnitude, a unit, and a dimensionality. The dimensionality is a type that both static checkers and the runtime understand.

- **`Quantity`** (`encomp.units`, `encomp.utypes`) extends [pint](https://pypi.org/project/Pint): each dimensionality (pressure, mass flow, density, ...) is a distinct subclass of `Quantity`. Dividing a `Mass` by a `Volume` gives a `Density`, inferred by static type checkers and verified at runtime. Magnitudes can be scalars, NumPy arrays, Polars `Series`, or Polars `Expr`.

- **Static unit checking.** Common dimensionalities and their `*` / `/` / `**` combinations are encoded as `__new__` overloads, so a type checker flags a `Temperature` passed where a `Power` is expected. `@typeguard.typechecked` extends the same checks to runtime.

- **`Fluid` / `Water` / `HumidAir`** (`encomp.fluids`) wrap [CoolProp](http://www.coolprop.org): fix a state with two points (three for humid air) and read any property as an attribute. Inputs and outputs are `Quantity` objects, so units are converted and validated automatically.

- **CoolProp as Polars expressions** (`encomp.coolprop`). Properties evaluate as native Polars expression plugins written in Rust, so independent properties in one `select` / `with_columns` run in parallel without holding the GIL. See [Parallel CoolProp evaluation with Polars](#parallel-coolprop-evaluation-with-polars).

- **Symbolic math with units** (`encomp.sympy`) extends [SymPy](https://pypi.org/project/sympy/): typeset sub- and superscripts, convert expressions or systems to NumPy functions, combine quantities directly with symbols.

- **Serialization and settings.** `Quantity` fields work as [Pydantic](https://pypi.org/project/pydantic/) model types (JSON round-trip, dimensionality validation). Library behavior is configured from an `.env` file.

The remaining modules (`encomp.gases`, `encomp.conversion`, `encomp.constants`, ...) implement process-engineering and thermodynamics computations.

## Versioning and stability

`encomp` uses semantic versioning for documented public APIs. Public APIs are the documented modules and objects in the API reference; private helpers, tests, notebooks, generated docs, and Rust internals may change in any release. The top-level `encomp` package intentionally exposes only `__version__`; import library APIs from their submodules.

`encomp.sympy` is legacy and soft-deprecated. It remains available for existing users, but new code should avoid depending on its `sympy.Symbol` monkey-patching and helper wrappers because the module is planned for removal in a future major release.

## Installation

```bash
pip install encomp
```

`encomp` ships as a single per-platform wheel that bundles the compiled Rust plugin and the CoolProp shared library, so supported platforms have nothing to build. Wheels are provided for Windows (x86_64), Linux (x86_64 and arm64), and macOS (Apple Silicon only). PyPI does not publish an sdist; unsupported platforms, including Intel Macs, need a build from the git repository; see [Tests](#tests).

## The `Quantity` class

`encomp.units.Quantity` extends `pint.Quantity`.
A quantity has a *magnitude* and a *unit*; each unit has a *dimensionality* (a combination of the base dimensions), and each dimensionality has multiple associated units.
The examples below abbreviate it as `Q` via `from encomp.units import Quantity as Q`; the library does not export a name `Q`.

```python
from encomp.units import Quantity as Q

# convert 1 bar to kPa
Q(1, "bar").to("kPa")

# list inputs are converted to np.ndarray
Q([1, 2, 3], "bar") * 2  # [2.0 4.0 6.0] bar

# without a unit, the quantity is dimensionless
assert Q(0.1) == Q(10, "%")
```

### `Quantity` type system

Each `Quantity` has a `Dimensionality` type parameter, determined at runtime from the unit.
Each dimensionality (for example *pressure*, *length*, *time*, *dimensionless*) is a subclass of `Quantity`.

For registered units (see `encomp.utypes.get_registered_units`), the dimensionality is also inferred statically, via overloads of `Quantity.__new__`.
The `*`, `/` and `**` operations between the default dimensionalities are overloaded as well.
When the dimensionality cannot be inferred, the static type falls back to `UnknownDimensionality`; the runtime dimensionality is always evaluated from the unit.

The dimensionality can also be given explicitly, using a subclass of `encomp.utypes.Dimensionality` as the type parameter.

Common dimensionalities are defined in the `encomp.utypes` module.
A newly created dimensionality gets a class name of the form `Dimensionality[...]` (for example `Quantity[Dimensionality[[mass] ** 2 / [length] ** 3]]`).
The second type parameter is the magnitude container. It defaults to `Numpy1DArray`, so annotate scalar quantities explicitly as `Quantity[Pressure, float]` (or `Q[Pressure, float]`).

```python
from typing import Any

from encomp.misc import isinstance_types
from encomp.units import ExpectedDimensionalityError
from encomp.units import Quantity as Q
from encomp.utypes import MassFlow, Volume

# the types are inferred by a static type checker

# the unit "kg" is registered as a Mass unit
m = Q(12, "kg")  # Quantity[Mass, float]

V = Q(25, "liter")  # Quantity[Volume, float]

# common / and * operations are encoded as overloads
rho = m / V  # Quantity[Density, float]

# the unit "kg/week" is not registered by default
# the individual units "kg" and "week" are registered, however
# the type checker does not know how to combine these units
m_ = Q(25, "kg/week")  # Quantity[UnknownDimensionality, float]

# at runtime, the dimensionality of m_ is evaluated to MassFlow;
# use isinstance_types for parameterized Quantity checks in type-checked code.
# always spell the magnitude parameter here: a bare Q[MassFlow] means
# Quantity[MassFlow, Numpy1DArray] to a type checker, which narrows m_ to Never
assert isinstance_types(m_, Q[MassFlow, Any])

# these operations (Mass**2 divided by Volume) are not explicitly defined as overloads
# at runtime, the type will be evaluated to
# Quantity[Dimensionality[[mass] ** 2 / [length] ** 3]]
x = m**2 / V  # Quantity[UnknownDimensionality, float]

# the unit name "meter cubed" is not defined using an overload
y = Q(15, "meter cubed").asdim(Volume)  # Quantity[Volume, float]

# if the explicit dimensionality does not match the unit,
# an error is raised at runtime
try:
    Q(15, "meter cubed").asdim(MassFlow)
except ExpectedDimensionalityError as e:
    print(f"Error: {e}")
    # Cannot convert 15.0 m³ to dimensionality
    # <class 'encomp.utypes.MassFlow'>, the dimensions do not match:
    # [length] ** 3 != [mass] / [time]
```

### Runtime type checking

The `Quantity` subtypes also restrict function and class attribute types at runtime.
The `typeguard.typechecked` decorator checks function inputs and outputs:

```python
from typing import Any, TypedDict, cast

from typeguard import TypeCheckError, typechecked

from encomp.units import Quantity as Q
from encomp.utypes import Length, Pressure, Temperature


@typechecked
def some_func(T: Q[Temperature, float]) -> tuple[Q[Length, float], Q[Pressure, float]]:
    return (T * Q(12.4, "m/K")).asdim(Length), Q(1, "bar")


some_func(Q(12, "K"))  # the dimensionalities check out

# a static type checker rejects some_func(Q(26, "kW")) before the code runs;
# typeguard catches the same error at runtime for values the checker cannot
# see through (simulated here by casting to Any)
try:
    some_func(cast(Any, Q(26, "kW")))
except TypeCheckError as e:
    print(f"Error: {e}")
    # argument "T" (encomp.units.Quantity[Power, float])
    # is not an instance of encomp.units.Quantity[Temperature, float]


class OutputDict(TypedDict):
    P: Q[Pressure, Any]
    T: Q[Temperature, Any]


@typechecked
def another_func(_s: Q[Length, Any]) -> OutputDict:
    # the value for the key "T" has the wrong dimensionality
    return {"T": cast(Any, Q(25, "m")), "P": Q(25, "kPa")}


try:
    another_func(Q(25, "m"))
except TypeCheckError as e:
    print(f"Error: {e}")
    # value of key 'T' of the return value (dict)
    # is not an instance of encomp.units.Quantity[Temperature]
```

To create a new dimensionality (for example temperature difference per mass flow rate), combine the `pint.UnitsContainer` objects stored in the `dimensions` class attribute.

```python
import contextlib

from encomp.units import DimensionalityError
from encomp.units import Quantity as Q
from encomp.utypes import Dimensionality, MassFlow, TemperatureDifference


# the class name TemperaturePerMassFlow must be globally unique
class TemperaturePerMassFlow(Dimensionality):
    dimensions = TemperatureDifference.dimensions / MassFlow.dimensions


# note the extra parentheses around (kg/s)
qty = Q(1, "delta_degC/(kg/s)").asdim(TemperaturePerMassFlow)

# raises an exception since liter is Length**3 and the Quantity expects Mass
with contextlib.suppress(DimensionalityError):
    Q(1, "delta_degC/(liter/hour)").asdim(TemperaturePerMassFlow)

# create a new subclass of Quantity with restricted input units
CustomCoolingCapacity = Q[TemperaturePerMassFlow, float]

# the pint library handles a wide range of input formats and unit names
# the prefix "delta_" can be omitted in this case
q1 = CustomCoolingCapacity(6, "°F per (lbs per week)")
q2 = Q(3, "delta_degF per (pound per fortnight)")

assert q1 == q2
assert type(q1) is type(q2)
```

## The `Fluid` class

`encomp.fluids.Fluid` wraps the *CoolProp* library.
Two input points (three for humid air) fix the state of the fluid; any other property is read as an attribute.
Inputs and outputs are `Quantity` objects.
Property names follow CoolProp; use the `.search()` method to find them.

```python
from encomp.fluids import Fluid
from encomp.units import Quantity as Q

air = Fluid("air", T=Q(25, "degC"), P=Q(2, "bar"))

# common fluid properties have type hints, and show up using autocomplete
density = air.D  # 2.338399526231983 kg/m³

air.search("density")
# ['DELTA, Delta: Reduced density (rho/rhoc) [dimensionless]',
#  'DMOLAR, Dmolar: Molar density [mol/m³]',
#  'D, DMASS, Dmass: Mass density [kg/m³]', ...

# any of the names are valid attributes (case-sensitive)
molar_density = air.Dmolar  # 80.73061937328056 mol/m³
```

The `Water` subclass (and `Fluid("IF97::Water")`) evaluates steam and water properties with *IAPWS-IF97* (Industrial Formulation 1997). For the *IAPWS-95* reference formulation, use the HEOS backend: `Fluid("HEOS::Water", ...)`; the bare name `Fluid("water", ...)` also resolves to HEOS.

```python
from encomp.fluids import Fluid, Water
from encomp.units import Quantity as Q

Fluid("water", P=Q(25, "bar"), T=Q(550, "°C"))
# <Fluid "water", P=2500 kPa, T=550.0 °C, D=6.7 kg/m³, V=0.031 cP>

# note that the CoolProp property "Q" (vapor quality) has the same name as the alias for the Quantity class
# the Water class has a slightly different string representation
Water(Q=Q(0.5), T=Q(170, "degC"))
# <Water (Two-phase), P=792 kPa, T=170.0 °C, D=8.2 kg/m³, V=nan cP>

Water(H=Q(2800, "kJ/kg"), S=Q(7300, "J/kg/K"))
# <Water (Gas), P=225 kPa, T=165.8 °C, D=1.1 kg/m³, V=0.015 cP>
```

Mixtures are given either by fractions folded into the name or by a `composition` dict of mole fractions.
Use `assume_phase` to skip CoolProp's phase-stability search when the phase is known; that search dominates the cost for the HEOS/GERG mixture backends.

```python
from encomp.fluids import Fluid
from encomp.units import Quantity as Q

# equivalent: fractions in the name, or a composition dict
Fluid("HEOS::CO2[0.7]&O2[0.3]", P=Q(10, "bar"), T=Q(300, "K"))
Fluid("HEOS", P=Q(10, "bar"), T=Q(300, "K"), composition={"CO2": 0.7, "O2": 0.3}).assume_phase("gas")
```

Incompressible fluid names such as `INCOMP::MEG[0.5]` and `INCOMP::MPG[0.5]` are aqueous ethylene-glycol and propylene-glycol solutions; the bracketed fraction is the concentration on CoolProp's documented basis, not the mole fraction used by `composition`.

The `HumidAir` class requires three input points (`R` means relative humidity):

```python
from encomp.fluids import HumidAir
from encomp.units import Quantity as Q

HumidAir(P=Q(1, "bar"), T=Q(100, "degC"), R=Q(0.5))
# <HumidAir, P=100 kPa, T=100.0 °C, R=0.50, Vda=2.2 m³/kg, Vha=1.3 m³/kg, M=0.017 cP>
```

## Parallel CoolProp evaluation with Polars

CoolProp property evaluation is exposed as native Polars expression plugins (Rust, over the CoolProp C-API). Independent property nodes in one `select` / `with_columns` / `collect()`, eager or lazy, are evaluated in parallel on the Polars thread pool without holding the GIL (a Python `map_batches` UDF re-acquires the GIL per batch and serializes).

`Fluid` properties accept `Quantity`-wrapped Polars expressions (`pl.Expr`) and return a `pl.Expr`:

```python
import polars as pl

from encomp.fluids import Water
from encomp.units import Quantity as Q

df = pl.DataFrame({"P": [50e5, 60e5], "T": [400.0, 450.0]})  # Pa, K
w: Water[pl.Expr] = Water(P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"))

# these independent CoolProp properties run in parallel across cores
df.select(w.D.m.alias("rho"), w.H.m.alias("h"), w.S.m.alias("s"))
```

`pl.Expr` (lazy) inputs are evaluated exclusively through the plugin (there is no `map_batches` fallback). Eager `float` / NumPy / `pl.Series` inputs use the Python CoolProp path, except arrays of at least `EAGER_PLUGIN_MIN_SIZE` (1000) elements, which also route through the plugin. The two paths are verified to agree on value, `NaN`/null handling, and dtype; results are bit-identical when the installed `coolprop` matches the bundled build (8.0.0).

The plugin is also usable directly on any Polars expression, independent of the `Fluid` class (the `encomp.coolprop` package):

```python
import polars as pl

from encomp import coolprop as cp

df = pl.DataFrame({"P": [1e5, 1e5], "T": [293.15, 313.15], "R": [0.4, 0.6]})  # Pa, K, -

df.select(
    cp.water("DMASS", "P", "T").alias("rho"),  # IF97 water/steam
    cp.water("HMASS", "P", "T").alias("h"),
    cp.humid_air("W", "P", "T", "R").alias("humidity_ratio"),
)
# mirrors encomp.fluids: any CoolProp input pair (in any order), the fluid via
# the required name='HEOS::CarbonDioxide' (cp.water is the IF97 water shorthand,
# as encomp.fluids.Water is for Fluid), mixtures via a
# composition={species: mole fraction} dict, and a fixed phase via assume_phase='gas'
```

### Implementation

The GIL is not the only serialization point: the CoolProp C-API takes a global handle-table lock on every call, so per-row calls serialize even in Rust. The plugin uses the batched C-API (`AbstractState_update_and_1_out`): one call per chunk, the handle lock taken once at construction, then the flash loop runs lock-free in C++. Independent chunks and independent property expressions parallelize.

Benchmarks (CoolProp 8.0, 14-thread pool):

| workload | vs `map_batches` | notes |
| --- | --- | --- |
| single property `D`, 1M rows | ~2.1x | also ~2x faster than vectorized `PropsSI` |
| 4 independent properties, one `collect()`, 1M rows | ~4.6x | `map_batches` is serial on the GIL; the plugin runs ~4 cores |
| 8 enthalpy evaluations, 1M rows | ~4.9x | ~6 cores vs 1, roughly half the peak memory |

Each `fluid(...)` / `humid_air(...)` is an independent plugin node, so selecting *K* properties of one state runs *K* flashes of it — Polars cannot reuse the shared flash across opaque plugin nodes. Independent properties still parallelize, so this is total work, not wall-clock. See `encomp/coolprop/README.md` for the design, thread-safety model, and caveats.

## Symbolic math

`encomp.sympy` is legacy and soft-deprecated; it is planned for removal in a future major release.
To load additional methods for the `sympy.Symbol` class, import SymPy via the `encomp.sympy` module. The `_` / `__` methods add typeset sub- and superscripts, and quantities combine directly with symbols:

```python
from typing import Any, cast

from encomp.sympy import sp
from encomp.units import Quantity as Q

n = sp.Symbol("n", integer=True)

# the _ / __ methods are added to sp.Symbol at runtime by encomp.sympy
cast(Any, n)._("H_2O").__("out")  # n_{\text{H}_2\text{O}}^{\text{out}}, keeps the integer assumption

x, y, z = sp.symbols("x, y, z")  # pyright: ignore[reportUnknownMemberType]
result_expr = (25 * x * y / z).subs({x: Q(235, "yard"), y: Q(2, "m²"), z: Q(0.4, "m³/kg")})
Q.from_expr(result_expr)  # ≈ 26860.5 kg
```

For array magnitudes, convert the expression to a NumPy-aware function with `encomp.sympy.get_function`.

## Tests

Development checkouts build the native CoolProp plugin locally. Install Rust, CMake, git, and a C++ compiler, then from the repository root run:

```bash
python scripts/build_libcoolprop.py
uv sync --all-extras --all-groups
uv run pytest
```

See `encomp/coolprop/README.md` for the plugin build details.

The test suite ships inside the wheel, so an installed `encomp` doubles as its own post-install smoke test (it needs only `pytest` and `hypothesis`):

```bash
pip install encomp pytest hypothesis
python -m pytest --pyargs encomp.tests
```

## Settings

The attributes of `encomp.settings.Settings` are overridden with a file named `.env` in the current working directory.
Attribute names are prefixed with `ENCOMP_`.
Settings are loaded when `encomp.settings` is imported; for runtime changes to quantity and unit rendering, use `encomp.units.set_quantity_format()`.

Because the `.env` file is resolved relative to the current working directory, a stray `.env` that sets an invalid `ENCOMP_*` value (for example `ENCOMP_UNITS` pointing at a missing file) makes `import encomp` fail with a `pydantic.ValidationError`, even in an unrelated project. Remove or correct the offending value; unrelated keys in the `.env` are ignored.

Importing `encomp.units` also registers a `typeguard` checker for `Quantity`, so `@typeguard.typechecked` and `encomp.misc.isinstance_types` compare dimensionality and magnitude type rather than falling back to a plain `isinstance`.

`import encomp` also installs `encomp.units.UNIT_REGISTRY` as pint's process-wide *application registry*. This is deliberate: every quantity in the process must come from that registry, or the dimensionality subclasses, the custom `[currency]` / `[normal]` dimensions and `on_redefinition="raise"` would silently not apply. The consequence is that another pint-based library in the same process gets encomp's registry (and its unit definitions, including the `Nm³` reinterpretation) after `import encomp`. Registry options that encomp pins — `force_ndarray`, `force_ndarray_like`, `autoconvert_offset_to_baseunit` — cannot be reassigned; a write that would change one is discarded and logs a warning.

## Documentation

The usage guide, example notebooks, and API reference are at [encomp.readthedocs.io](https://encomp.readthedocs.io).

Release notes for each version are published as [GitHub Releases](https://github.com/wlaur/encomp/releases).
