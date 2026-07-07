# Usage

This guide covers the main `encomp` modules: units, fluids, and symbolic math.

## The Quantity class

A {py:class}`encomp.units.Quantity` stores the *magnitude*, *unit* and *dimensionality* of a physical quantity.
Each dimensionality is a separate subclass, so static type checkers catch dimensionality errors before the code runs.

:::{note}
`Q` is an alias for `Quantity`: `from encomp.units import Quantity as Q`
:::

Import the class, then create an instance representing an absolute pressure of 1 bar:

```python
from encomp.units import Quantity as Q

pressure = Q(1, "bar")
```

:::{warning}
`encomp` (and the underlying `pint` library) does not differentiate between absolute and gauge pressure.
:::

Convert the pressure to another unit:

```python
from encomp.units import Quantity as Q

pressure = Q(1, "bar")

# pressure_kpa is a new Quantity instance
pressure_kpa = pressure.to("kPa")
```

The unit definition file (`encomp/defs/units.txt`) lists the accepted unit names.
It is based on the `defaults_en.txt` file from `pint`, with minor modifications.

Quantities can also be constructed from unit registry attributes:

```python
from encomp.units import UNIT_REGISTRY
from encomp.units import Quantity as Q

# the registry attributes are typed for use as Quantity units, not for
# direct arithmetic, so these operations need pyrefly suppressions
d = 50 * UNIT_REGISTRY.m  # pyrefly: ignore[unsupported-operation]
v = d / UNIT_REGISTRY.s  # pyrefly: ignore[unsupported-operation]

mf = Q(25, UNIT_REGISTRY.kg / UNIT_REGISTRY.h)  # pyrefly: ignore[unsupported-operation]
```

### Quantity types

Each dimensionality is a unique subclass of {py:class}`encomp.units.Quantity`.
The base class itself cannot be instantiated, since it would have no dimensionality at all (a *dimensionless* quantity still has a dimensionality of *1*).

```python
from encomp.units import Quantity as Q

pressure = Q(1, "bar")
pressure_kpa = pressure.to("kPa")

type(pressure)  # <class 'encomp.units.Quantity[Pressure, float]'>

fraction = Q(5, "%")
type(fraction)  # <class 'encomp.units.Quantity[Dimensionless, float]'>

assert type(pressure) is type(pressure_kpa)

length = Q(1, "meter")
assert type(pressure) is not type(length)
```

To create a subclass of {py:class}`encomp.units.Quantity` with a certain dimensionality, provide a *type parameter* in square brackets.
The parameter must be a subclass of {py:class}`encomp.utypes.Dimensionality`, whose `dimensions` class attribute holds a `pint.unit.UnitsContainer` (a combination of the base dimensions).

:::{note}
The type parameter is the subclass itself, not an instance: `Q[Power]` works, `Q[Power()]` raises `TypeError`.
:::

Subclasses for common dimensionalities are defined in {py:mod}`encomp.utypes`.

```python
from encomp.units import Quantity as Q
from encomp.utypes import Dimensionality, Length, Power, Pressure

Q[Pressure, float]  # subclass with dimensionality pressure and magnitude float

pressure_dims = Pressure.dimensions  # <UnitsContainer({'[length]': -1, '[mass]': 1, '[time]': -2})>


# the class name PowerPerLength must be globally unique
class PowerPerLength(Dimensionality):
    dimensions = Power.dimensions / Length.dimensions


Q[PowerPerLength, float]  # new dimensionality
```

:::{note}
Dimensionality subclasses live in a single *process-wide*, name-keyed registry:

- Two subclasses with the same class name and the **same** dimensions are treated as
  one type -- the first definition wins and is silently reused (this keeps notebook
  cell re-runs and module reloads working).
- Two subclasses with the same class name but **different** dimensions raise
  `TypeError` at class-definition time -- also across independent packages that both
  define, say, `FuelPerAir`. Pick distinctive names for custom dimensionalities in
  library code.
:::

:::{important}
The typed constructor does not *validate* the dimensionality at runtime -- it *redirects*.
`Q[Length](1, "kg")` returns a `Quantity[Mass, float]`: the dimensionality of the created
object is always determined by the unit. This is by design (`pint` constructs arithmetic
results through `self.__class__(...)` with new dimensionalities, so the constructor must
accept them). Use {py:meth}`encomp.units.Quantity.check` for physical-dimensionality
checks. Semantic dimensionality enforcement happens in the static type checker and at
explicit runtime boundaries: `isinstance()` / {py:func}`encomp.misc.isinstance_types`,
`typeguard.typechecked` functions, Pydantic model fields (which raise
`pydantic.ValidationError`), and direct `.asdim()` calls (which raise
`ExpectedDimensionalityError` for a mismatch). Arithmetic also checks semantic
compatibility and may reject two quantities with the same physical dimensions.
:::

Check the dimensionality of a quantity with `isinstance()` or {py:meth}`encomp.units.Quantity.check`.
For parameterized types like `list[Quantity[Pressure]]`, use {py:func}`encomp.misc.isinstance_types` instead of `isinstance()`.

```python
from encomp.misc import isinstance_types
from encomp.units import Quantity as Q
from encomp.utypes import Length, Pressure, Temperature, TemperatureDifference

pressure = Q(1, "bar")

pressure.check(Length)  # False
pressure.check("meter")  # False

pressure.check(Pressure)  # True
pressure.check("psi")  # True
pressure.check("[pressure]")  # True

# check() compares physical dimensions only, not semantic sibling classes
Q(1, "degC").check(TemperatureDifference)  # True
Q(1, "delta_degC").check(Temperature)  # True

# alternative using isinstance()
# (parameterized isinstance is a runtime-only feature, hence the suppressions)

# pyrefly: ignore[invalid-argument]
isinstance(pressure, Q[Pressure])  # True
# pyrefly: ignore[invalid-argument]
isinstance(pressure, Q[Length])  # False

# complex types must use isinstance_types
# this function can also be used with simple types

isinstance_types([pressure, pressure], list[Q[Pressure]])  # True
isinstance_types({1: Q(2, "m"), 2: Q(25, "cm")}, dict[int, Q[Length]])  # True

# all Quantity[...] objects are subclasses of Quantity
isinstance_types(pressure, Q)  # True
```

For functions and methods, use the `typeguard.typechecked` decorator instead of explicit checks in the function body:

```python
from typeguard import typechecked

from encomp.units import Quantity as Q
from encomp.utypes import Length, Power, Pressure


@typechecked
def func(_p1: Q[Pressure]) -> tuple[Q[Length], Q[Power]]:
    return Q(1, "m"), Q(1, "kW")
```

`typeguard.TypeCheckError` is raised if the arguments or the return value have incorrect dimensionalities.

### Custom base dimensionalities

By default, the seven SI dimensionalities (and common combinations of these) are defined, along with some commonly used media (*water*, *air*, *fuel*).
Additionally, the *normal* dimensionality (used to represent normal volume) and *currency* are defined.

{py:func}`encomp.units.define_dimensionality` defines a new base dimensionality with a single unit of the same name.
If the dimensionality already exists, {py:class}`encomp.units.DimensionalityRedefinitionError` is raised.

```python
from encomp.units import Quantity as Q
from encomp.units import define_dimensionality

define_dimensionality("dry_air")
define_dimensionality("oxygen")

# the new dimensionality [dry_air] has a single unit: "dry_air"
m_air = Q(5, "kg * dry_air")
n_o2 = Q(2.4, "mol * oxygen")
M_O2 = Q(32, "g/mol")

# compute mass fraction
((n_o2 * M_O2) / m_air).to_base_units()  # 0.01536 oxygen/dry_air
```

### Quantities with vector magnitudes

Lists, Numpy arrays and Polars Series objects can also be used as magnitude.

```python
import numpy as np

from encomp.units import Quantity as Q

type(Q([1, 2, 3], "kg").m)  # numpy.ndarray

arr = np.linspace(0, 1)
Q(arr, "bar")
# [0.0 0.0204 0.0408 ... 0.9795 1.0] bar
```

### Quantities with expression magnitudes

Polars Expressions can be used as magnitude:

```python
import polars as pl

from encomp.units import Quantity as Q

type(Q(pl.lit(5), "kg").m)  # pl.Expr
```

A `Quantity` with a `pl.Expr` magnitude is a deferred plan, not data. Only unit algebra (arithmetic, comparison, `.to`, `abs`) is meaningful on it; reach the underlying Polars object with `.m` to compute inside a `select` / `with_columns`. This is also how the parallel CoolProp evaluation described below is driven.

### Combining quantities

The output of an operation on quantities is always consistent with the input dimensionalities.
Inconsistent or ambiguous operations raise descriptive errors.

Units do not always cancel out automatically.
Call {py:meth}`encomp.units.Quantity.to_base_units` to simplify to base SI units, {py:meth}`encomp.units.Quantity.to` when the target unit is known, or {py:meth}`encomp.units.Quantity.to_reduced_units` to cancel units without converting to base SI units.

```python
from encomp.units import Quantity as Q

(Q(5, "%") * Q(1, "meter")).to("mm")  # 50.0 mm
```

Temperature units need extra care.
A temperature *difference* in a degree scale is written with the prefix `delta_` (only needed when defining the difference directly).
Temperature ({py:class}`encomp.utypes.Temperature`) and temperature difference ({py:class}`encomp.utypes.TemperatureDifference`) are distinct dimensionalities and deliberately not interchangeable: a difference cannot silently be used as an absolute temperature.
Do not use {py:meth}`encomp.units.Quantity.check` to distinguish these two cases:
it compares physical dimensions, and both classes share `[temperature]`. Use
`isinstance()` / {py:func}`encomp.misc.isinstance_types`, typed function boundaries,
Pydantic fields, or arithmetic/conversion errors for the semantic distinction.

```python
from pint.errors import OffsetUnitCalculusError

from encomp.units import DimensionalityTypeError
from encomp.units import Quantity as Q

temp_diff = Q(5, "delta_degC")  # 5 Δ°C

# a temperature difference cannot be converted to an absolute temperature
try:
    temp_diff.to("degC")
except DimensionalityTypeError as e:
    print(f"Error: {e}")
    # Cannot convert Δ°C (dimensionality TemperatureDifference)
    # to °C (dimensionality Temperature)

Q(25, "degC") - Q(36, "degC")  # -11 Δ°C

# multiplying with an offset unit (°C) is ambiguous
try:
    Q(4.19, "kJ/kg/K") * Q(5, "°C")
except OffsetUnitCalculusError as e:
    print(f"Error: {e}")

# this is not the result we're after, °C is offset by 273.15 K
Q(4.19, "kJ/kg/K") * Q(5, "°C").to("K")  # 1165.4485 kJ/kg

Q(4.19, "kJ/kg/K") * Q(5, "delta_degC")  # 20.95 Δ°C·kJ/K/kg
Q(4.19, "kJ/kg/K") * Q(5, "K")  # 20.95 kJ/kg

# the units Δ°C and K don't cancel out automatically,
# use the to() method to convert to the desired output unit
(Q(4.19, "kJ/kg/K") * Q(5, "delta_degC")).to("kJ/kg")  # 20.95 kJ/kg
```

:::{note}
`pint.errors.OffsetUnitCalculusError` is raised when doing ambiguous unit conversions.
The environment variable `ENCOMP_AUTOCONVERT_OFFSET_TO_BASEUNIT` can be set to `True` to disable this error (this is not recommended).
:::

### Currency units

The dimensionality {py:class}`encomp.utypes.Currency` represents an arbitrary currency.
`SEK`, `EUR` and `USD` are defined by default.

:::{warning}
Do **not** use this system for currency *conversions*.
The scaling factors between the built-in currencies are fixed placeholders
(`10 SEK = 1 EUR = 1 USD`), **not** exchange rates -- converting a quantity from one
currency to another silently applies these fabricated factors.
Keep all quantities in a single currency, or refer to the
[pint documentation](https://pint.readthedocs.io/en/stable/advanced/currencies.html)
for how to implement a registry context that handles currency conversion correctly.
:::

```python
from encomp.units import Quantity as Q

mf = Q(25, "kg/s")
t = Q(365, "d")

price = Q(25, "EUR/ton")

yearly_cost = mf * t * price  # Quantity[Currency]

# SI prefixes can be used
print(yearly_cost.to("MEUR"))

# NOTE: this is only an approximation,
# uses the fixed placeholder scaling 10 SEK = 1 EUR
print(yearly_cost.to("MSEK"))

weekly_cost = Q(145, "GWh/year") * Q(1, "week") * Q(25, "EUR/MWh")

print(weekly_cost.to("MEUR"))
```

### Handling unit-related errors

Use `pint.errors.DimensionalityError` to catch all unit-related errors.
This error can also be imported from the {py:mod}`encomp.units` module.

```python
from encomp.units import DimensionalityError
from encomp.units import Quantity as Q
from encomp.utypes import Pressure

# alternatively, use pint.errors.DimensionalityError
# from pint.errors import DimensionalityError

try:
    # a static type checker rejects this addition as well
    Q(25, "bar") + Q(25, "m")  # pyrefly: ignore[unsupported-operation]
except DimensionalityError as e:
    print(f"Error: {e}")

try:
    Q[Pressure](25, "m")
except DimensionalityError as e:
    print(f"Error: {e}")

try:
    Q(15, "m").to("kg")
except DimensionalityError as e:
    print(f"Error: {e}")
```

### Integration with Pydantic

{py:class}`encomp.units.Quantity` (optionally with a dimensionality type parameter) works as a Pydantic field type.

```python
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from encomp.units import Quantity as Q
from encomp.utypes import Dimensionless, Length, Mass


class Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    # a can be any dimensionality
    a: Q

    m: Q[Mass]
    s: Q[Length]

    # float is converted to Quantity[Dimensionless]
    r: Q[Dimensionless, float] = Q(0.5)


model = Model(a=Q(25, "cSt"), m=Q(25, "kg"), s=Q(25, "cm"))

# Quantity fields round-trip through JSON, including the magnitude type
Model.model_validate_json(model.model_dump_json())

adapter = TypeAdapter(Q[Mass, float])
adapter.validate_json(adapter.dump_json(Q(2.0, "kg")))

try:
    Model(a=Q(25, "cSt"), m=Q(25, "m"), s=Q(25, "cm"))
except ValidationError as e:
    print(e.errors()[0]["type"])  # quantity_dimensionality
```

:::{note}
Pydantic model and {py:class}`pydantic.TypeAdapter` validation wraps quantity input errors
in {py:class}`pydantic.ValidationError`, using error types such as
`quantity_dimensionality`, `quantity_magnitude_type`, and `quantity_validation`. This lets
Pydantic attach field locations and collect multiple invalid fields in one exception.
:::

## The Fluid class

The {py:class}`encomp.fluids.Fluid` class represents a fluid at a fixed point.
The abstract base class {py:class}`encomp.fluids.CoolPropFluid` implements the CoolProp interface and documents the fluid and property names.
All inputs and outputs are {py:class}`encomp.units.Quantity` instances.

Pass the CoolProp fluid name and the fixed points (for example *P, T*) to the constructor.
Not every combination of input parameters can fix the state: with an invalid input pair, every property evaluates to `nan` and CoolProp emits a warning, but no exception is raised.
An invalid property *name*, on the other hand, raises `ValueError`.

```python
from typing import Any

from encomp.fluids import Fluid
from encomp.units import Quantity as Q

Fluid("toluene", T=Q(25, "°C"), P=Q(2, "bar"))
# <Fluid "toluene", P=200 kPa, T=25.0 °C, D=862.3 kg/m³, V=0.55 cP>

# PCRIT cannot be used to fix the state: it is an output-only property, not one
# of the FluidState inputs, so a static type checker rejects it at the call
# site (the inputs are routed through Any here to show the runtime behavior)
state: Any = {"D": Q(500, "kg/m³"), "PCRIT": Q(1, "bar")}

invalid_inputs = Fluid("water", **state)
# <Fluid "water", P=nan kPa, T=nan °C, D=nan kg/m³, V=nan cP>

# every property is nan (CoolProp emits a warning about the invalid input pair)
temperature = invalid_inputs.T  # nan °C
```

{py:class}`encomp.fluids.Water` omits the fluid name and uses `IAPWS-IF97` (Industrial Formulation 1997).
For the `IAPWS-95` reference formulation, use the HEOS backend: {py:class}`encomp.fluids.Fluid` with name `HEOS::Water` (the bare name `water` also resolves to HEOS).

The {py:class}`encomp.fluids.HumidAir` class has a different set of input and output properties.

```python
from encomp.fluids import HumidAir, Water
from encomp.units import Quantity as Q

# input units are converted to SI
Water(P=Q(30, "psi"), T=Q(250, "°F"))
# <Water (Liquid), P=207 kPa, T=121.1 °C, D=942.2 kg/m³, V=0.2 cP>

HumidAir(T=Q(25, "°C"), P=Q(2, "bar"), R=Q(25, "%"))
# <HumidAir, P=200 kPa, T=25.0 °C, R=0.25, Vda=0.4 m³/kg, Vha=0.4 m³/kg, M=0.018 cP>
```

Property names must match CoolProp's exactly.
An invalid state-input name is rejected statically at the call site, and the constructor raises `ValueError` at runtime:

```python
from typing import Any

from encomp.fluids import HumidAir
from encomp.units import Quantity as Q

# "Ps" is not a valid property name (the inputs are routed through Any here,
# since HumidAir(T=..., Ps=..., R=...) is also rejected statically)
state: Any = {"T": Q(25, "°C"), "Ps": Q(2, "bar"), "R": Q(25, "%")}

try:
    HumidAir(**state)
except ValueError as e:
    print(f"Error: {e}")
    # Invalid CoolProp property name: Ps
    # Valid names:
    # B, C, CV, CVha, Cha, Conductivity, D, DewPoint, Enthalpy, Entropy, H, Hda, Hha,
    # HumRat, K, M, Omega, P, P_w, R, RH, RelHum, S, Sda, Sha, T, T_db, T_dp, T_wb, Tdb,
    # Tdp, Twb, V, Vda, Vha, Visc, W, WetBulb, Y, Z, cp, cp_ha, cv_ha, k, mu, psi_w
```

Use the `search()` and `describe()` methods to get more information about the properties:

```python
from encomp.fluids import Fluid, HumidAir

HumidAir.search("bulb")
# ['B, Twb, T_wb, WetBulb: Wet-Bulb Temperature [K]',
#  'T, Tdb, T_db: Dry-Bulb Temperature [K]']

Fluid.describe("Z")
# 'Z: Compressibility factor [dimensionless]'
```

All property synonyms are valid instance attributes:

```python
from encomp.fluids import Water
from encomp.units import Quantity as Q

Water.describe("PCRIT")
# 'PCRIT, P_CRITICAL, Pcrit, p_critical, pcrit: Pressure at the critical point [Pa]'

water = Water(T=Q(25, "°C"), P=Q(1, "atm"))

critical = water.p_critical, water.PCRIT
# (22064000.0 <Unit('pascal')>, 22064000.0 <Unit('pascal')>)
```

:::{tip}
Common fluid properties are type hinted with the correct dimensionality and show up in IDE autocomplete.
:::

### Mixtures and assumed phase

A mixture is given either by fractions folded into the fluid name or by a `composition` dict of mole fractions (which must sum to 1). Both spellings resolve to the same state.

```python
from encomp.fluids import Fluid
from encomp.units import Quantity as Q

Fluid("HEOS::CO2[0.7]&O2[0.3]", P=Q(10, "bar"), T=Q(300, "K"))

Fluid("HEOS", P=Q(10, "bar"), T=Q(300, "K"), composition={"CO2": 0.7, "O2": 0.3})
```

For an incompressible mixture, the concentration is carried in the name instead, on the fluid's own basis (mass for glycols/brines, volume for the volume-specified antifreezes):

```python
from encomp.fluids import Fluid
from encomp.units import Quantity as Q

Fluid("INCOMP::MEG[0.5]", P=Q(1, "bar"), T=Q(20, "°C"))  # 50 % ethylene glycol
```

The {py:meth}`encomp.fluids.Fluid.assume_phase` method pins the phase, skipping CoolProp's phase-stability search, which dominates the cost for the HEOS/GERG mixture backends. It is a *speed* tool, not a validation tool: forcing a phase the fluid is not actually in returns `NaN` or a non-physical metastable root rather than raising.

```python
from encomp.fluids import Fluid
from encomp.units import Quantity as Q

# ~100-1000x faster for mixtures, when the phase is known
density = Fluid("HEOS::CO2[0.7]&O2[0.3]", P=Q(10, "bar"), T=Q(300, "K")).assume_phase("gas").D
```

`IF97` (the default backend for {py:class}`encomp.fluids.Water`) is region-explicit and ignores an assumed phase; the call is a no-op there and emits a warning. Use `Fluid("HEOS::Water", ...)` if you need an assumed phase for water.

### Using vector inputs

CoolProp evaluates vector inputs in a single backend call.
The inputs are {py:class}`encomp.units.Quantity` instances with one-dimensional Numpy arrays as magnitude, all of the same length (or a single scalar, which is repeated).

```python
import numpy as np

from encomp.fluids import Water
from encomp.units import Quantity as Q

Water(T=Q(np.linspace(25, 50, 10), "°C"), P=Q(np.linspace(25, 50, 10), "bar"))
# the repr shows only the head of each vector input
# <Water (Liquid), P=[2500 2778 3056 ...] kPa, T=[25.0 27.8 30.6 ...] °C,
# D=[998.1 997.5 996.8 ...] kg/m³, V=[0.9 0.8 0.8 ...] cP>

# different phases
phases = Water(T=Q(np.linspace(25, 500, 10), "°C"), P=Q(np.linspace(0.5, 10, 10), "bar")).PHASE
# <Quantity([0. 0. 5. 5. 5. 5. 5. 2. 2. 2.], 'dimensionless')>

phase_names = Water.PHASES
# {0.0: 'Liquid',
#  5.0: 'Gas',
#  6.0: 'Two-phase',
#  3.0: 'Supercritical liquid',
#  2.0: 'Supercritical gas',
#  1.0: 'Supercritical fluid',
#  8.0: 'Not imposed'}

# when one input is constant (float, int, single element array),
# it's repeated as an array
Water(T=Q(np.linspace(25, 500, 10), "°C"), P=Q(5, "bar"))
# <Water (Variable), P=[500 500 500 ...] kPa, T=[25.0 77.8 130.6 ...] °C,
# D=[997.2 973.4 934.5 ...] kg/m³, V=[0.9 0.4 0.2 ...] cP>
```

Missing or out-of-range results surface as `NaN` (for a numpy magnitude) or `null` (for a Polars magnitude), never as a zero or a raised exception, so a partly-invalid batch still returns the valid rows.

### Parallel evaluation with Polars

{py:class}`encomp.fluids.Fluid` properties also accept `Quantity`-wrapped Polars expressions (`pl.Expr`) and return a `pl.Expr`. Independent property nodes in one `select` / `with_columns` / `collect()` (eager or lazy) are evaluated in parallel by the `encomp.coolprop` plugin -- a native Rust extension over the CoolProp C-API that runs without holding the GIL.
`pl.Expr` (lazy) inputs are evaluated exclusively through this plugin (there is no `map_batches` fallback). Eager `float` / numpy / `pl.Series` inputs use the Python CoolProp path, except arrays of at least `EAGER_PLUGIN_MIN_SIZE` (1000) elements, which also route through the plugin (results are bit-identical when the installed `coolprop` matches the bundled build, 8.0.0).

```python
import polars as pl

from encomp.fluids import Water
from encomp.units import Quantity as Q

df = pl.DataFrame({"P": [50e5, 60e5], "T": [400.0, 450.0]})  # Pa, K
w: Water[pl.Expr] = Water(P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"))

# independent CoolProp properties evaluated in parallel across cores
df.select(w.D.m.alias("rho"), w.H.m.alias("h"), w.S.m.alias("s"))
```

Each property is a separate plugin node, so selecting *K* properties of one state (as above) runs *K* flashes of it -- Polars cannot reuse the shared flash across the opaque plugin nodes. They still evaluate in parallel, so this is total work, not wall-clock.

The plugin is also usable directly on any Polars expression, independent of the {py:class}`encomp.fluids.Fluid` class (the `encomp.coolprop` package):

```python
import polars as pl

from encomp import coolprop as cp

df = pl.DataFrame({"P": [1e5, 1e5], "T": [293.15, 313.15], "R": [0.4, 0.6]})  # Pa, K, -

df.select(
    cp.fluid("DMASS", "P", "T").alias("rho"),  # default: IF97 water
    cp.fluid("HMASS", "P", "T").alias("h"),
    cp.humid_air("W", "P", "T", "R").alias("humidity_ratio"),
)
```

The API mirrors {py:class}`encomp.fluids.Fluid`: any CoolProp input pair is supported (in any order), the fluid is given by `name` (with the backend folded in, e.g. `name="HEOS::CarbonDioxide"`), mixtures via a `composition={species: mole fraction}` dict, and a fixed phase via `assume_phase="gas"`. See the `encomp.coolprop` package README in the repository for the full design and thread-safety model.

## Sympy functionality

To load additional methods for the `sympy.Symbol` class, import Sympy via the {py:mod}`encomp.sympy` module.

### Typesetting

The following convenience methods are added to the `sp.Symbol` class:

- `sp.Symbol._()`: add subscript
- `sp.Symbol.__()`: add superscript
- `sp.Symbol.decorate()`: add sub- and superscript prefixes and suffixes ({py:meth}`encomp.sympy.Symbol.decorate`)

These methods return new `sp.Symbol` instances with the same assumptions (*positive*, *real*, *integer*, ...) as the original.

```python
from encomp.sympy import sp

n = sp.Symbol("n", integer=True)

# the _ method is added to sp.Symbol at runtime by encomp.sympy
n_test = n._("test")  # pyrefly: ignore[missing-attribute]
str(n_test)
# n_{\text{test}}

n_test.assumptions0["integer"]  # True
```

:::{tip}
The assumptions for an `sp.Symbol` instance are accessed with the attribute `assumptions0` (note the `0` at the end).
:::

The `_` and `__` methods typeset sub- and superscripts automatically:

- Single-letter lower case with math font: `n._("a")` → $n_a$
- Single-letter upper case with regular font: `n._("A")` → $n_{\text{A}}$
- Chemical formulas: `n._("H_2O")` → $n_{\text{H}_2\text{O}}$
- Strings with two or more characters with regular font: `n._("water")` → $n_{\text{water}}$
- Parts are split with `,`: `n._("outlet,A,i,H_2SO_4")` → $n_{\text{outlet},\text{A},i,\text{H}_2\text{SO}_4}$
- Combine sub- and superscript: `n._("a").__("in")` → $n_{a}^{\text{in}}$

The `decorate` method offers more control:

- `n.decorate(prefix="\sum", prefix_sub="2", suffix_sup="i", suffix="\ldots")` → ${\sum}_{2}n^{i}{\ldots}$

### Integration with quantities

Quantities can be substituted into Sympy expressions; the units are converted to Sympy symbols automatically.
The class method {py:meth}`encomp.units.Quantity.from_expr` converts an expression back to a quantity.

```python
from encomp.sympy import sp
from encomp.units import Quantity as Q

x, y, z = sp.symbols("x, y, z")

expr = 25 * x * y / z

result_expr = expr.subs({x: Q(235, "yard"), y: Q(2, "m²"), z: Q(0.4, "m³/kg")})

result_qty = Q.from_expr(result_expr)
# 26860.5 kg
```

{py:meth}`encomp.units.Quantity.from_expr` raises `KeyError` if residual symbols in the expression are not SI units.

:::{warning}
Sympy integration only works with the seven SI dimensionalities, not with dimensionalities defined via {py:func}`encomp.units.define_dimensionality`.
:::

{py:meth}`encomp.units.Quantity.from_expr` does not support Numpy array magnitudes.
Convert the expression to a function with {py:func}`encomp.sympy.get_function` instead:

```python
import numpy as np

from encomp.sympy import get_function, sp
from encomp.units import Quantity as Q

x, y, z = sp.symbols("x, y, z")

expr = 25 * x * y / z

# units=False by default, since this is faster to evaluate
fcn = get_function(expr, units=True)

result_qty = fcn(
    {
        x: Q(np.array([235, 335]), "yard"),
        y: Q([2, 5], "m²"),  # regular lists will be converted to array
        z: Q(0.4, "m³/kg"),
    }
)
# [26860.5 95726.25] kg
```

Quantity objects combine directly with Sympy symbols; the units are converted to their symbolic representations by the `Quantity._sympy_` method (the hook `sympy.sympify` looks for).

```python
from encomp.sympy import sp
from encomp.units import Quantity as Q

x, y, z = sp.symbols("x, y, z")

# the type of the left object determines the output

# output is a Quantity with a symbolic magnitude
Q(1) * x  # 1.0*x dimensionless
Q(10, "%") * x  # 10.0*x percent

# output is a sympy object
x * Q(1)  # 1.0*x
x * Q(10, "%")  # 0.1*x

# when the output is a sympy object,
# all derived units are expanded to the base SI units
x + y / Q(25, "kW")
# x + 4.0e-5*\text{s}**3*y/(\text{kg}*\text{m}**2)
```

:::{todo}
This behavior is not encoded in the type hints.
:::
