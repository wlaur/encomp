# encomp

> General-purpose library for *en*gineering *comp*utations, with focus on clean and consistent interfaces.

`encomp` is tested on Windows, Linux, and macOS, with Python 3.13 and 3.14.

## Highlights

`encomp` combines a few well-established scientific libraries behind a single, type-safe interface. Every physical quantity carries its magnitude, unit, and dimensionality. The dimensionality is a real type that static checkers and the runtime both understand.

- **A dimensional `Quantity` type** (`encomp.units`, `encomp.utypes`). Extends [pint](https://pypi.org/project/Pint) so that each dimensionality (pressure, mass flow, density, ...) is a distinct subclass of `Quantity`. Multiplying a `Mass` by a `Volume` gives a `Density` — inferred by a static type checker *and* verified at runtime. Magnitudes can be a scalar, a NumPy array, a Polars `Series`, or a Polars `Expr`.

- **A type system that catches unit errors before the code runs.** Common dimensionalities and their `*` / `/` / `**` combinations are encoded as `__new__` overloads, so type checkers flag a `Temperature` passed where a `Power` is expected. Decorate a function with `@typeguard.typechecked` to extend the same checks to runtime.

- **`Fluid` / `Water` / `HumidAir`** (`encomp.fluids`). A quantity-based wrapper over [CoolProp](http://www.coolprop.org): fix a state with two points (three for humid air) and read any property as an attribute. Inputs and outputs are `Quantity` objects, so units are converted and validated automatically.

- **Parallel CoolProp evaluation with Polars** (`encomp.coolprop`). CoolProp properties evaluate as **native Polars expression plugins** written in Rust, so independent properties in one `select` / `with_columns` run in parallel on the Polars thread pool without holding the GIL. This is several times faster than a `map_batches` Python UDF (which serializes on the GIL) and than vectorized `PropsSI`, at roughly half the peak memory. See [Parallel CoolProp evaluation with Polars](#parallel-coolprop-evaluation-with-polars).

- **Symbolic math that understands units** (`encomp.sympy`). Extends [Sympy](https://pypi.org/project/sympy/) with convenience methods for sub- and superscripts and for turning expressions (or whole systems) into NumPy-aware Python functions. Quantities and symbols combine directly.

- **Serialization and settings out of the box.** `Quantity` fields work as [Pydantic](https://pypi.org/project/pydantic/) model types (JSON round-trip, dimensionality validation), and library behavior is configured from an `.env` file.

The remaining modules (`encomp.gases`, `encomp.conversion`, `encomp.constants`, ...) implement calculations related to process engineering and thermodynamics.

## Installation

```bash
pip install encomp
```

`encomp` ships as a single per-platform wheel that bundles the compiled Rust plugin and the CoolProp shared library, so there is nothing to build. For a development checkout, see [Tests](#tests).

## The `Quantity` class

The fundamental building block of `encomp` is the `encomp.units.Quantity` class (shorthand alias `Q`), which is an extension of `pint.Quantity`.
This class is used to construct objects with a *magnitude* and *unit*.
Each unit also has a *dimensionality* (combination of the base dimensions), and each dimensionality will have multiple associated units.

```python
from encomp.units import Quantity as Q

# converts 1 bar to kPa, displays it in case it's the cell output
Q(1, "bar").to("kPa")

# list inputs are converted to np.ndarray
Q([1, 2, 3], "bar") * 2  # [2, 4, 6] bar

# in case no unit is specified, the quantity is dimensionless
Q(0.1) == Q(10, "%")
```

### `Quantity` type system

The `Quantity` object has an associated `Dimensionality` type parameter that is dynamically determined based on the unit.
Each dimensionality (for example *pressure*, *length*, *time*, *dimensionless*) is represented by a subclass of `Quantity`.

Common dimensionalities can be statically determined based on overload variants of the `Quantity.__new__` method (see `encomp.utypes.get_registered_units` for a list of units that support this).
Additionally, operations using `*`, `**` and `/` are also defined using overload variants for combinations of the default dimensionalities.

In case the dimensionality cannot be inferred, the type checker will use the dimensionality `Any`.
At runtime, the dimensionality will be evaluated based on the unit that was specified.

If necessary, the dimensionality of a quantity can be explicitly specified by providing a subclass of `encomp.utypes.Dimensionality` as type parameter.

Commonly used dimensionalities are defined in the `encomp.utypes` module.
When a new dimensionality is created, the classname will be `Dimensionality[...]` (for example `Quantity[Dimensionality[[mass] ** 2 / [length] ** 3]]`).

```python
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

# at runtime, the dimensionality of m_ will be evaluated to MassFlow
isinstance(m_, Q[MassFlow])  # True

# these operations (Mass**2 divided by Volume) are not explicitly defined as overloads
# at runtime, the type will be evaluated to
# Quantity[Dimensionality[[mass] ** 2 / [length] ** 3]]
x = m**2 / V  # Quantity[UnknownDimensionality, float]

# the unit name "meter cubed" is not defined using an overload
y = Q(15, "meter cubed").asdim(Volume)  # Quantity[Volume, float]

# in case the explicitly defined dimensionality does
# not match the unit, an error will be raised at runtime

y = Q(15, "meter cubed").asdim(MassFlow)
# ExpectedDimensionalityError: Quantity with unit "m³" has incorrect dimensionality
# [length] ** 3, expected [mass] / [time]
```

### Runtime type checking

The `Quantity` subtypes can be used to restrict function and class attribute types at runtime.
Use the `typeguard.typechecked` decorator to apply runtime typechecking to function inputs and outputs:

```python
from typing import Any, TypedDict

from typeguard import typechecked

from encomp.units import Quantity as Q
from encomp.utypes import Length, Pressure, Temperature


@typechecked
def some_func(T: Q[Temperature, float]) -> tuple[Q[Length, float], Q[Pressure, float]]:
    return (T * Q(12.4, "m/K")).asdim(Length), Q(1, "bar")


some_func(Q(12, "delta_degC"))  # the dimensionalities check out
some_func(Q(26, "kW"))  # raises an exception:
# TypeError: type of argument "T" must be Quantity[Temperature];
# got Quantity[Power] instead


class OutputDict(TypedDict):
    P: Q[Pressure, Any]
    T: Q[Temperature, Any]


@typechecked
def another_func(s: Q[Length, Any]) -> OutputDict:
    return {"T": Q(25, "m"), "P": Q(25, "kPa")}


another_func(Q(25, "m"))
# TypeError: type of dict item "T" for the return value must be
# encomp.units.Quantity[Temperature]; got encomp.units.Quantity[Length] instead
```

To create a new dimensionality (for example temperature difference per mass flow rate), combine the `pint.UnitsContainer` objects stored in the `dimensions` class attribute.

```python
from encomp.units import DimensionalityError
from encomp.units import Quantity as Q
from encomp.utypes import Dimensionality, MassFlow, TemperatureDifference, Volume


# the class name TemperaturePerMassFlow must be globally unique
class TemperaturePerMassFlow(Dimensionality):
    dimensions = TemperatureDifference.dimensions / MassFlow.dimensions


# note the extra parentheses around (kg/s)
qty = Q(1, "delta_degC/(kg/s)").asdim(TemperaturePerMassFlow)

# raises an exception since liter is Length**3 and the Quantity expects Mass
try:
    another_qty = Q(1, "delta_degC/(liter/hour)").asdim(TemperaturePerMassFlow)
except DimensionalityError:
    pass

# create a new subclass of Quantity with restricted input units
CustomCoolingCapacity = Q[TemperaturePerMassFlow, float]

# the pint library handles a wide range of input formats and unit names
# the prefix "delta_" can be omitted in this case
q1 = CustomCoolingCapacity(6, "°F per (lbs per week)")
q2 = Q("3 delta_degF per (pound per fortnight)")

assert q1 == q2
assert type(q1) is type(q2)
```

## The `Fluid` class

The class `encomp.fluids.Fluid` is a wrapper around the *CoolProp* library.
The class uses two input points (three for humid air) that fix the state of the fluid.
Other fluid parameters can be evaluated using attribute access.
The outputs and inputs are `Quantity` objects.
CoolProp property names and codes are used throughout.
Use the `.search()` method to find the correct name.

```python
from encomp.fluids import Fluid
from encomp.units import Quantity as Q

air = Fluid("air", T=Q(25, "degC"), P=Q(2, "bar"))

# common fluid properties have type hints, and show up using autocomplete
air.D  # 2.338399526231983 kilogram/meter3

air.search("density")
# ['DELTA, Delta: Reduced density (rho/rhoc) [dimensionless]',
#  'DMOLAR, Dmolar: Molar density [mol/m³]',
#  'D, DMASS, Dmass: Mass density [kg/m³]', ...

# any of the names are valid attributes (case-sensitive)
air.Dmolar  # 80.73061937328056 mole/meter3
```

The `Water` subclass (and `Fluid("IF97::Water")`) evaluates steam and water properties with the *IAPWS-IF97* (Industrial Formulation 1997, "IF97") by default — the fast industrial standard. For the higher-accuracy *IAPWS-95* reference formulation, use the HEOS backend explicitly with `Fluid("HEOS::Water", ...)`; the bare name `Fluid("water", ...)` also resolves to HEOS (IAPWS-95).

```python
from encomp.fluids import Fluid, Water
from encomp.units import Quantity as Q

Fluid("water", P=Q(25, "bar"), T=Q(550, "°C"))
# <Fluid "water", P=2500 kPa, T=550.0 °C, D=6.7 kg/m³, V=0.031 cP>

# note that the CoolProp property "Q" (vapor quality) has the same name as the class
# the Water class has a slightly different string representation
Water(Q=Q(0.5), T=Q(170, "degC"))
# <Water (Two-phase), P=792 kPa, T=170.0 °C, D=8.2 kg/m³, V=0.0 cP>

Water(H=Q(2800, "kJ/kg"), S=Q(7300, "J/kg/K"))
# <Water (Gas), P=225 kPa, T=165.8 °C, D=1.1 kg/m³, V=0.0 cP>
```

Mixtures are given either by fractions folded into the name or by a `composition` dict of mole fractions.
Pair a composition with `assume_phase` to skip CoolProp's phase-stability search when the phase is already known (a large speedup for the HEOS/GERG mixture backends).

```python
from encomp.fluids import Fluid

# equivalent: fractions in the name, or a composition dict
Fluid("HEOS::CO2[0.7]&O2[0.3]", P=Q(10, "bar"), T=Q(300, "K"))
Fluid("HEOS", P=Q(10, "bar"), T=Q(300, "K"), composition={"CO2": 0.7, "O2": 0.3}).assume_phase("gas")
```

The `HumidAir` class requires three input points (`R` means relative humidity):

```python
from encomp.fluids import HumidAir
from encomp.units import Quantity as Q

HumidAir(P=Q(1, "bar"), T=Q(100, "degC"), R=Q(0.5))
# <HumidAir, P=100 kPa, T=100.0 °C, R=0.50, Vda=2.2 m³/kg, Vha=1.3 m³/kg, M=0.017 cP>
```

## Parallel CoolProp evaluation with Polars

CoolProp property evaluation is exposed as **native Polars expression plugins** (Rust, over the CoolProp C-API). Independent property nodes in one `select` / `with_columns` / `collect()` — eager or lazy alike — are evaluated **in parallel on the Polars thread pool, without holding the GIL**. A Python `map_batches` UDF cannot do this: it re-acquires the GIL per batch and serializes.

`Fluid` properties accept `Quantity`-wrapped Polars expressions (`pl.Expr`) and return a `pl.Expr`:

```python
import polars as pl

from encomp.fluids import Water
from encomp.units import Quantity as Q

df = pl.DataFrame({"P": [50e5, 60e5], "T": [400.0, 450.0]})  # Pa, K
w = Water(P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"))

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
    cp.fluid("DMASS", "P", "T").alias("rho"),  # default: IF97 water
    cp.fluid("HMASS", "P", "T").alias("h"),
    cp.humid_air("W", "P", "T", "R").alias("humidity_ratio"),
)
# mirrors encomp.fluids: any CoolProp input pair (in any order), the fluid via
# name='HEOS::CarbonDioxide', mixtures via a composition={species: mole fraction}
# dict, and a fixed phase via assume_phase='gas'
```

### Why it is fast

Removing the GIL is necessary but not sufficient: the CoolProp C-API takes a global handle-table lock on every call, so naive per-row calls serialize even in pure Rust. The plugin instead uses the **batched** C-API (`AbstractState_update_and_1_out`): one call per chunk, the handle lock taken once at construction, then the flash loop runs lock-free in C++. Independent chunks and independent property expressions therefore parallelize.

Indicative numbers (CoolProp 8.0, 14-thread pool):

| workload | vs `map_batches` | notes |
| --- | --- | --- |
| single property `D`, 1,000,000 rows | ~2.1x | also ~2x faster than vectorized `PropsSI` |
| 4 independent properties, one `collect()`, 1M rows | ~4.6x | `map_batches` is serial on the GIL; the plugin runs ~4 cores |
| 8 enthalpy calculations, 1M rows | ~4.9x | ~6 cores vs 1, roughly half the peak memory |

Each `fluid(...)` / `humid_air(...)` is an independent plugin node, so selecting *K* properties of one state runs *K* flashes of it — Polars cannot reuse the shared flash across opaque plugin nodes. Independent properties still parallelize, so this is a statement about total work, not wall-clock. See `encomp/coolprop/README.md` for the full design, thread-safety model, and caveats.

## Symbolic math

To load additional methods for the `sympy.Symbol` class, import Sympy via the `encomp.sympy` module. The `_` / `__` methods add typeset sub- and superscripts, and quantities combine directly with symbols:

```python
from encomp.sympy import sp
from encomp.units import Quantity as Q

n = sp.Symbol("n", integer=True)
n._("H_2O").__("out")  # n_{\text{H}_2\text{O}}^{\text{out}}, keeps the integer assumption

x, y, z = sp.symbols("x, y, z")
result_expr = (25 * x * y / z).subs({x: Q(235, "yard"), y: Q(2, "m²"), z: Q(0.4, "m³/kg")})
Q.from_expr(result_expr)  # 26860.5 kg
```

For array magnitudes, convert the expression to a NumPy-aware function with `encomp.sympy.get_function`.

## Tests

First, make sure the development dependencies are installed with `uv sync --all-extras --all-groups`.
Run the tests with

```bash
pytest
```

## Settings

The attributes in the `encomp.settings.Settings` class can be modified with an `.env`-file.
Place a file named `.env` in the current working directory to override the default settings.
The attribute names are prefixed with `ENCOMP_`.

## Documentation

Full documentation, including the detailed usage guide and example notebooks, is at [encomp.readthedocs.io](https://encomp.readthedocs.io).
