# Usage

This section contains instructions for incorporating `encomp` in your own tools.

## The Quantity class

The purpose of the {py:class}`encomp.units.Quantity` class is to store information about the *magnitude*, *dimensionality* and *units* of a physical quantity.
Each dimensionality is represented as a separate subclass.
This means that static type checkers can be used to catch dimensionality-related errors before the code is executed.

:::{note}
The shorthand `Q` is used as an alias for `Quantity`.
Import the class with `from encomp.units import Quantity as Q`
:::

To get started, import the class:

```python
from encomp.units import Quantity as Q
```

Create an instance representing an absolute pressure of 1 bar:

```python
pressure = Q(1, "bar")
```

:::{warning}
`encomp` (and the underlying `pint` library) does not differentiate between absolute and gauge pressure.
:::

Convert the pressure to another unit:

```python
# pressure_kPa is a new Quantity instance
pressure_kPa = pressure.to("kPa")
```

Refer to the unit definition file (`encomp/defs/units.txt`) for a list of accepted unit names.
This definition file is based on the `defaults_en.txt` file from `pint`, with some slight modifications.

Quantities can also be constructed by combining unit objects.
The unit registry contains a number of common units as attributes.

```python
from encomp.units import UNIT_REGISTRY

d = 50 * UNIT_REGISTRY.m
v = d / UNIT_REGISTRY.s

mf = Q(25, UNIT_REGISTRY.kg / UNIT_REGISTRY.h)
```

### Quantity types

The quantity class also contains information about *dimensionality*.
It is not possible to create an instance of the base class {py:class}`encomp.units.Quantity`, since this would not have any dimensionality at all (a *dimensionless* quantity still has a dimensionality of *1*).
Each new dimensionality is represented by a unique subclass of {py:class}`encomp.units.Quantity`.

```python
type(pressure)  # <class 'encomp.units.Quantity[encomp.utypes.Pressure]'>

fraction = Q(5, "%")
type(fraction)  # <class 'encomp.units.Quantity[encomp.utypes.Dimensionless]'>

assert type(pressure) is type(pressure_kPa)

length = Q(1, "meter")
assert type(pressure) is not type(length)
```

To create a subclass of {py:class}`encomp.units.Quantity` with a certain dimensionality, provide a dimensionality *type parameter* using square brackets.
All dimensionality type parameters must inherit from {py:class}`encomp.utypes.Dimensionality`.
The dimensions of a dimensionality (a combination of the base dimensions) is specified as a `pint.unit.UnitsContainer` instance (class attribute `dimensions`).

:::{note}
The dimensionality type parameters must be a *subclass* of {py:class}`encomp.utypes.Dimensionality` (not an instance of this subclass). `Q[Power]` creates a subclass of `Quantity` with dimensionality *power*, but `Q[Power()]` will raise a `TypeError`.
:::

The module {py:mod}`encomp.utypes` contains {py:class}`encomp.utypes.Dimensionality` subclasses for some common dimensionalities.

```python
from encomp.utypes import Dimensionality, Length, Power, Pressure

Q[Pressure, float]  # subclass with dimensionality pressure and magnitude float

Pressure.dimensions  # <UnitsContainer({'[length]': -1, '[mass]': 1, '[time]': -2})>


# the class name PowerPerLength must be globally unique
class PowerPerLength(Dimensionality):
    dimensions = Power.dimensions / Length.dimensions


Q[PowerPerLength, float]  # new dimensionality
```

The builtin `isinstance()` can be used to check dimensionalities of quantity objects.
Alternatively, the {py:meth}`encomp.units.Quantity.check` method can be used.
For more complex types, like `list[Quantity[Pressure]]`, the {py:func}`encomp.misc.isinstance_types` function must be used instead of `isinstance()`.

```python
pressure.check(Length)  # False
pressure.check("meter")  # False

pressure.check(Pressure)  # True
pressure.check("psi")  # True

# alternative using isinstance()

isinstance(pressure, Q[Pressure])  # True
isinstance(pressure, Q[Length])  # False

# complex types must use isinstance_types
# this function can also be used with simple types

from encomp.misc import isinstance_types

isinstance_types([pressure, pressure], list[Q[Pressure]])  # True
isinstance_types({1: Q(2, "m"), 2: Q(25, "cm")}, dict[int, Q[Length]])  # True

# all Quantity[...] objects are subclasses of Quantity
isinstance_types(pressure, Q)  # True
```

To check types for functions and methods, use the `typeguard.typechecked` decorator instead of writing explicit checks inside the function body:

```python
from typeguard import typechecked


@typechecked
def func(p1: Q[Pressure]) -> tuple[Q[Length], Q[Power]]:
    return Q(1, "m"), Q(1, "kW")
```

A `TypeError` will be raised if the function `func` is called with incorrect dimensionalities or if the return value has incorrect dimensionalities.

### Custom base dimensionalities

By default, the seven SI dimensionalities (and common combinations of these) are defined, along with some commonly used media (*water*, *air*, *fuel*).
Additionally, the *normal* dimensionality (used to represent normal volume) and *currency* are defined.

The function {py:func}`encomp.units.define_dimensionality` can be used to define a new base dimensionality.
In case the dimensionality already exists, {py:class}`encomp.units.DimensionalityRedefinitionError` is raised.
The new dimensionality will have a single unit with the same name as the dimensionality.

```python
from encomp.units import define_dimensionality

define_dimensionality("dry_air")
define_dimensionality("oxygen")

# the new dimensionality [dry_air] has a single unit: "dry_air"
m_air = Q(5, "kg * dry_air")
n_O2 = Q(2.4, "mol * oxygen")
M_O2 = Q(32, "g/mol")

# compute mass fraction
((n_O2 * M_O2) / m_air).to_base_units()  # 0.01536 oxygen/air
```

### Quantities with vector magnitudes

Lists, Numpy arrays and Polars Series objects can also be used as magnitude.

```python
type(Q([1, 2, 3], "kg").m)  # numpy.ndarray

import numpy as np

arr = np.linspace(0, 1)
Q(arr, "bar")
# [0.0 0.0204 0.0408 ... 0.9795 1.0] bar
```

### Quantities with expression magnitudes

Polars Expressions can be used as magnitude:

```python
import polars as pl

type(Q(pl.lit(5), "kg").m)  # pl.Expr
```

A `Quantity` with a `pl.Expr` magnitude is a deferred plan, not data. Only unit algebra (arithmetic, comparison, `.to`, `abs`) is meaningful on it; reach the underlying Polars object with `.m` to compute inside a `select` / `with_columns`. This is also how the parallel CoolProp evaluation described below is driven.

### Combining quantities

The output from operations on quantities will always be consistent with the input dimensionalities.
Descriptive errors are raised in case of inconsistent or ambiguous operations.

In some cases, units will not cancel out automatically.
Call {py:meth}`encomp.units.Quantity.to_base_units` to simplify the quantity to base SI units, or {py:meth}`encomp.units.Quantity.to` in case the desired unit is known.
The {py:meth}`encomp.units.Quantity.to_reduced_units` method can be used to cancel units without converting to base SI units.

```python
(Q(5, "%") * Q(1, "meter")).to("mm")  # 50.0 mm
```

Operations with temperature units can lead to unexpected results.
When using temperature degree scales, a temperature *difference* can be defined with the prefix `delta_`.
This is only required when defining the temperature difference directly.
Temperatures (absolute or degree units) and temperature differences have different dimensionalities ({py:class}`encomp.utypes.Temperature` and {py:class}`encomp.utypes.TemperatureDifference`), and are deliberately not interchangeable: this prevents a temperature difference from silently being used as an absolute temperature.

```python
dT = Q(5, "delta_degC")  # 5 Δ°C

# this is not allowed
dT.to("degC")
# DimensionalityTypeError: Cannot convert Δ°C (dimensionality TemperatureDifference)
# to °C (dimensionality Temperature)

Q(25, "degC") - Q(36, "degC")  # -11 Δ°C

Q(4.19, "kJ/kg/K") * Q(5, "°C")  # raises OffsetUnitCalculusError

# this is not the result we're after, °C is offset by 273.15 K
Q(4.19, "kJ/kg/K") * Q(5, "°C").to("K")  # 1165.4485 kJ/kg

Q(4.19, "kJ/kg/K") * Q(5, "delta_degC")  # 20.95 kJ Δ°C/(K kg)
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

Engineering calculations will often involve economic aspects.
To aid in this, the dimensionality {py:class}`encomp.utypes.Currency` can be used to represent an arbitrary currency.
By default, the currencies `SEK, EUR, USD` are defined.

```python
mf = Q(25, "kg/s")
t = Q(365, "d")

price = Q(25, "EUR/ton")

yearly_cost = mf * t * price  # Quantity[Currency]

# SI prefixes can be used
print(yearly_cost.to("MEUR"))

# NOTE: this is only an approximation,
# uses exchange rate 10 SEK = 1 EUR
print(yearly_cost.to("MSEK"))

weekly_cost = Q(145, "GWh/year") * Q(1, "week") * Q(25, "EUR/MWh")

print(weekly_cost.to("MEUR"))
```

:::{warning}
Do not use this system for currency *conversions*.
The scaling factors between the default currencies are approximations (`10 SEK = 1 EUR = 1 USD`).

Refer to the [pint documentation](https://pint.readthedocs.io/en/stable/currencies.html) for instructions on how to implement a registry context that handles currency conversion correctly.
:::

### Handling unit-related errors

Use `pint.errors.DimensionalityError` to catch all unit-related errors.
This error can also be imported from the {py:mod}`encomp.units` module.

```python
from encomp.units import DimensionalityError

# alternatively, use pint.errors.DimensionalityError
# from pint.errors import DimensionalityError

try:
    Q(25, "bar") + Q(25, "m")
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

Pydantic is used for runtime type validation of data models.
The {py:class}`encomp.units.Quantity` class (along with an optional dimensionality type parameter) can be used as a field type with Pydantic.
The field types are defined as type hints.
Pydantic models inherit from the `pydantic.BaseModel` class.

```python
from pydantic import BaseModel, ConfigDict

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


# in case the input dimensionalities do not match the type hint,
# a runtime error (pydantic.ValidationError) will be raised
model = Model(a=Q(25, "cSt"), m=Q(25, "kg"), s=Q(25, "cm"))

# Quantity fields round-trip through JSON, including the magnitude type
Model.model_validate_json(model.model_dump_json())
```

The `pydantic_settings.BaseSettings` class is used to read, convert and validate key-value pairs from an `.env`-file.

`.env`-file:

```text
any_quantity=1.215 kJ/kg/K
mass=24 kg
length=25 m
ratio=0.25
```

`.py`-file:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

from encomp.units import Quantity as Q
from encomp.utypes import Dimensionless, Length, Mass, Pressure


class Settings(BaseSettings):
    model_config = SettingsConfigDict(validate_assignment=True)

    any_quantity: Q
    mass: Q[Mass]
    length: Q[Length]

    ratio: Q[Dimensionless] = Q(0)
    pressure: Q[Pressure] = Q(1, "atm")


# parameters that are not explicitly passed here are read from the .env-file
s = Settings(ratio=0.75)

# raises pydantic.ValidationError (since validate_assignment is True)
s.mass = Q(25, "bar")
```

:::{note}
Vector quantities, for example `Q([25, 26], "kg")`, cannot be specified with an `.env`-file.
:::

## The Fluid class

The {py:class}`encomp.fluids.Fluid` class represents a fluid at a fixed point.
The abstract base class {py:class}`encomp.fluids.CoolPropFluid` implements an interface to CoolProp.
All inputs and outputs are {py:class}`encomp.units.Quantity` instances.

:::{note}
All input and output parameter names follow the conventions used in CoolProp.
:::

To create a new instance, pass the CoolProp fluid name and the fixed points (for example *P, T*) to the class constructor.
The documentation for the base class {py:class}`encomp.fluids.CoolPropFluid` contains a list of fluid and property names.
All combinations of input parameters are not valid -- in case of incorrect inputs, a `ValueError` is raised when evaluating an attribute (i.e. not when the instance is created).
The `__repr__` of the instance will show `N/A` instead of raising an error.

```python
from encomp.fluids import Fluid

Fluid("toluene", T=Q(25, "°C"), P=Q(2, "bar"))
# <Fluid "toluene", P=200 kPa, T=25.0 °C, D=862.3 kg/m³, V=0.55 cP>

# PCRIT cannot be used to fix the state
invalid_inputs = Fluid("water", D=Q(500, "kg/m³"), PCRIT=Q(1, "bar"))
# <Fluid "water", P=N/A, T=N/A, D=N/A, V=N/A>

# try to access the attribute "T" (temperature)
invalid_inputs.T
# ValueError: Input pair variable is invalid and output(s) are non-trivial; cannot do state update : PropsSI("T","D",500,"PCRIT",100000,"water")
```

When the class {py:class}`encomp.fluids.Water` is used, the fluid name can be omitted.
{py:class}`encomp.fluids.Water` uses the `IAPWS-IF97` (Industrial Formulation 1997) formulation by default.
For the higher-accuracy `IAPWS-95` reference formulation, use the HEOS backend: create an instance of {py:class}`encomp.fluids.Fluid` with name `HEOS::Water` (the bare name `water` also resolves to HEOS).

The {py:class}`encomp.fluids.HumidAir` class has a different set of input and output properties.

```python
from encomp.fluids import HumidAir, Water

# input units are converted to SI
Water(D=Q(12, "lbs / ft³"), T=Q(250, "°F"))
# <Water (Two-phase), P=206 kPa, T=121.1 °C, D=192.2 kg/m³, V=0.0 cP, Q=0.00>

HumidAir(T=Q(25, "°C"), P=Q(2, "bar"), R=Q(25, "%"))
# <HumidAir, P=200 kPa, T=25.0 °C, R=0.25, Vda=0.4 m³/kg, Vha=0.4 m³/kg, M=0.018 cP>
```

The exact names used by CoolProp must be used.
Note that these are different for {py:class}`encomp.fluids.HumidAir`.

```python
HumidAir(T=Q(25, "°C"), Ps=Q(2, "bar"), R=Q(25, "%"))
# ValueError: Invalid CoolProp property name: Ps
# Valid names:
# B, C, CV, CVha, Cha, Conductivity, D, DewPoint, Enthalpy, Entropy, H, Hda, Hha,
# HumRat, K, M, Omega, P, P_w, R, RH, RelHum, S, Sda, Sha, T, T_db, T_dp, T_wb, Tdb,
# Tdp, Twb, V, Vda, Vha, Visc, W, WetBulb, Y, Z, cp, cp_ha, cv_ha, k, mu, psi_w
```

Use the `search()` and `describe()` methods to get more information about the properties:

```python
HumidAir.search("bulb")
# ['B, Twb, T_wb, WetBulb: Wet-Bulb Temperature [K]',
#  'T, Tdb, T_db: Dry-Bulb Temperature [K]']

Fluid.describe("Z")
# 'Z: Compressibility factor [dimensionless]'
```

All property synonyms are valid instance attributes:

```python
Water.describe("PCRIT")
# 'PCRIT, P_CRITICAL, Pcrit, p_critical, pcrit: Pressure at the critical point [Pa]'

water = Water(T=Q(25, "°C"), P=Q(1, "atm"))

water.p_critical, water.PCRIT
# (22064000.0 <Unit('pascal')>, 22064000.0 <Unit('pascal')>)
```

:::{tip}
Common fluid properties are type hinted using the correct dimensionality.
These properties also show up in the autocomplete list when using an IDE.
:::

### Mixtures and assumed phase

A mixture is given either by fractions folded into the fluid name or by a `composition` dict of mole fractions (which must sum to 1). Both spellings resolve to the same state.

```python
from encomp.fluids import Fluid

Fluid("HEOS::CO2[0.7]&O2[0.3]", P=Q(10, "bar"), T=Q(300, "K"))

Fluid("HEOS", P=Q(10, "bar"), T=Q(300, "K"), composition={"CO2": 0.7, "O2": 0.3})
```

For an incompressible mixture, the concentration is carried in the name instead, on the fluid's own basis (mass for glycols/brines, volume for the volume-specified antifreezes):

```python
Fluid("INCOMP::MEG[0.5]", P=Q(1, "bar"), T=Q(20, "°C"))  # 50 % ethylene glycol
```

The {py:meth}`encomp.fluids.Fluid.assume_phase` method pins the phase, skipping CoolProp's phase-stability search. For the HEOS/GERG mixture backends that search dominates the cost, so assuming a phase you already know is a large speedup. It is a *speed* tool, not a validation tool: forcing a phase the fluid is not actually in returns `NaN` or a non-physical metastable root rather than raising.

```python
# ~100-1000x faster for mixtures, when the phase is known
Fluid("HEOS::CO2[0.7]&O2[0.3]", P=Q(10, "bar"), T=Q(300, "K")).assume_phase("gas").D
```

`IF97` (the default backend for {py:class}`encomp.fluids.Water`) is region-explicit and ignores an assumed phase; the call is a no-op there and emits a warning. Use `Fluid("HEOS::Water", ...)` if you need an assumed phase for water.

### Using vector inputs

The CoolProp library supports vector inputs, which means that multiple inputs can be evaluated at the same time (in a single call to the CoolProp backend).
The inputs must be instances of {py:class}`encomp.units.Quantity` with one-dimensional Numpy arrays as magnitude.
All inputs must have the same length (or a single scalar value).

```python
Water(T=Q(np.linspace(25, 50, 10), "°C"), P=Q(np.linspace(25, 50, 10), "bar"))
# <Water (Liquid), P=[2500 2778 3056 3333 3611 3889 4167 4444 4722 5000] kPa,
# T=[25.0 27.8 30.6 33.3 36.1 38.9 41.7 44.4 47.2 50.0] °C,
# D=[998.1 997.5 996.8 996.0 995.2 994.3 993.3 992.3 991.3 990.2] kg/m³,
# V=[0.9 0.8 0.8 0.7 0.7 0.7 0.6 0.6 0.6 0.5] cP>

# different phases
Water(T=Q(np.linspace(25, 500, 10), "°C"), P=Q(np.linspace(0.5, 10, 10), "bar")).PHASE
# array([0., 0., 5., 5., 5., 5., 5., 2., 2., 2.]) <Unit('dimensionless')>

Water.PHASES
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
# <Water (Variable), P=[500 500 500 500 500 500 500 500 500 500] kPa,
# T=[25.0 77.8 130.6 183.3 236.1 288.9 341.7 394.4 447.2 500.0] °C,
# D=[997.2 973.3 934.5 2.5 2.2 2.0 1.8 1.6 1.5 1.4] kg/m³,
# V=[0.9 0.4 0.2 0.0 0.0 0.0 0.0 0.0 0.0 0.0] cP>
```

Missing or out-of-range results surface as `NaN` (for a numpy magnitude) or `null` (for a Polars magnitude), never as a zero or a raised exception, so a partly-invalid batch still returns the valid rows.

### Parallel evaluation with Polars

{py:class}`encomp.fluids.Fluid` properties also accept `Quantity`-wrapped Polars expressions (`pl.Expr`) and return a `pl.Expr`. Independent property nodes in one `select` / `with_columns` / `collect()` (eager or lazy) are evaluated *in parallel* by the `encomp.coolprop` plugin -- a native Rust extension over the CoolProp C-API that runs without holding the GIL.
`pl.Expr` (lazy) inputs are evaluated exclusively through this plugin (there is no `map_batches` fallback). Eager `float` / numpy / `pl.Series` inputs use the Python CoolProp path, except arrays of at least `EAGER_PLUGIN_MIN_SIZE` (1000) elements, which also route through the plugin (faster and lower peak memory; results are bit-identical when the installed `coolprop` matches the bundled build, 8.0.0).

```python
import polars as pl

from encomp.fluids import Water
from encomp.units import Quantity as Q

df = pl.DataFrame({"P": [50e5, 60e5], "T": [400.0, 450.0]})  # Pa, K
w = Water(P=Q(pl.col("P"), "Pa"), T=Q(pl.col("T"), "K"))

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

```python
from encomp.sympy import sp
```

### Typesetting

The following convenience methods are added to the `sp.Symbol` class:

- `sp.Symbol._()`: add subscript
- `sp.Symbol.__()`: add superscript
- `sp.Symbol.decorate()`: add sub- and superscript prefixes and suffixes ({py:func}`encomp.sympy.decorate`)

These methods return new instances of `sp.Symbol` with the same assumptions (i.e. *positive*, *real*, *integer*, etc.) as the original instance.

```python
n = sp.Symbol("n", integer=True)

n_test = n._("test")
str(n_test)
# n_{\text{test}}

n_test.assumptions0["integer"]  # True
```

:::{tip}
The assumptions for an `sp.Symbol` instance are accessed with the attribute `assumptions0` (note the `0` at the end).
:::

The `_` and `__` methods will typeset the sub- and superscripts automatically:

- Single-letter lower case with math font: `n._("a")` → $n_a$
- Single-letter upper case with regular font: `n._("A")` → $n_{\text{A}}$
- Chemical formulas: `n._("H_2O")` → $n_{\text{H}_2\text{O}}$
- Strings with two or more characters with regular font: `n._("water")` → $n_{\text{water}}$
- Parts are split with `,`: `n._("outlet,A,i,H_2SO_4")` → $n_{\text{outlet},\text{A},i,\text{H}_2\text{SO}_4}$
- Combine sub- and superscript: `n._("a").__("in")` → $n_{a}^{\text{in}}$

The `decorate` method offers more control:

- `n.decorate(prefix="\sum", prefix_sub="2", suffix_sup="i", suffix="\ldots")` → ${\sum}_{2}n^{i}{\ldots}$

### Integration with quantities

Quantities can be used when evaluating Sympy expressions.
The units will be converted to Sympy symbols automatically.
The class method {py:meth}`encomp.units.Quantity.from_expr` is used to convert an expression back to a quantity.

```python
x, y, z = sp.symbols("x, y, z")

expr = 25 * x * y / z

result_expr = expr.subs({x: Q(235, "yard"), y: Q(2, "m²"), z: Q(0.4, "m³/kg")})

result_qty = Q.from_expr(result_expr)
# 26860.5 kg
```

{py:meth}`encomp.units.Quantity.from_expr` will raise `KeyError` in case residual symbols in the expression are not SI units.

:::{warning}
Sympy integration only works with the seven SI dimensionalities.
It does not work with user-defined dimensionalities (i.e. dimensionalities/units defined using {py:func}`encomp.units.define_dimensionality`).
:::

In case the magnitude of a quantity is a Numpy array, {py:meth}`encomp.units.Quantity.from_expr` does not work.
The expression must instead be converted to a function with {py:func}`encomp.sympy.get_function`:

```python
from encomp.sympy import get_function

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

Quantity objects can be directly combined with Sympy symbols.
The units are automatically converted to their symbolic representations (in the method {py:meth}`encomp.units.Quantity._sympy_`).

```python
x, y, z = sp.symbols("x, y, z")

# the type of the left object determines the output

# output is Quantity[Dimensionless]
Q(1) * x  # x dimensionless
Q(10, "%") * x  # 10*x percent

# output is a sympy object
x * Q(1)  # x
x * Q(10, "%")  # 0.1x

# when the output is a sympy object,
# all derived units are expanded to the base SI units
x + y / Q(25, "kW")
# x + 4.0e-5*\text{s}**3*y/(\text{kg}*\text{m}**2)
```

:::{todo}
This behavior is not encoded in the type hints.
:::
