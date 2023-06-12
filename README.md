# encomp

> General-purpose library for *en*gineering *comp*utations, with focus on clean and consistent interfaces.

Documentation at <https://encomp.readthedocs.io/en/latest/>

`encomp` is tested on Windows and Linux, with Python 3.11.

## Features

Main functionality of the `encomp` library:

- Handles physical quantities with magnitude(s), dimensionality and units

  - Modules `encomp.units`, `encomp.utypes`
  - Extends the [pint](https://pypi.org/project/Pint) library
  - Uses Python's type system to validate dimensionalities
  - Compatible with `mypy` and other type checkers
  - Integrates with Numpy arrays, Pandas series and Polars series and expressions
  - JSON serialization and decoding

- Implements a flexible interface to [CoolProp](http://www.coolprop.org)

  - Module `encomp.fluids`
  - Uses quantities for all inputs and outputs
  - Fluids are represented as class instances, the properties are class attributes

- Extends [Sympy](https://pypi.org/project/sympy/)

  - Module `encomp.sympy`
  - Adds convenience methods for creating symbols with sub- and superscripts
  - Additional functions to convert (algebraic) expressions and systems to Python code that supports Numpy arrays

The other modules implement calculations related to process engineering and thermodynamics.
The module `encomp.serialize` implements custom JSON serialization and decoding for classes used elsewhere in the library.

## Installation

Install with `pip`:

```bash
pip install encomp
```

This will install `encomp` along with its dependencies into the currently active Python environment.
To also install optional dependencies:

```bash
pip install encomp[optional]
```

## Getting started

To use `encomp` in a Jupyter Notebook (or REPL), star-import the `encomp.session` module:

```python
from encomp.session import *
```

This will import commonly used functions and classes.

### The `Quantity` class

The fundamental building block of `encomp` is the `encomp.units.Quantity` class (shorthand alias `Q`), which is an extension of `pint.Quantity`.
This class is used to construct objects with a *magnitude* and *unit*.
Each unit also has a *dimensionality* (combination of the base dimensions), and each dimensionality will have multiple associated units.

```python
from encomp.units import Quantity as Q

# converts 1 bar to kPa, displays it in case it's the cell output
Q(1, 'bar').to('kPa')

# a single string with one numerical value can also be given as input
Q('0.1 MPa').to('bar')

# list and tuple inputs are converted to np.ndarray
Q([1, 2, 3], 'bar') * 2 # [2, 4, 6] bar

# in case no unit is specified, the quantity is dimensionless
Q(0.1) == Q(10, '%')
```

#### `Quantity` type system

The `Quantity` object has an associated `Dimensionality` type parameter that is dynamically determined based on the unit.
Each dimensionality (for example *pressure*, *length*, *time*, *dimensionless*) is represented by a subclass of `Quantity`.

Common dimensionalities can be statically determined based on overload variants of the `Quantity.__new__` method (see `encomp.utypes.get_registered_units` for a list of units that support this).
Additionally, operations using `*`, `**` and `/` are also defined using overload variants for combinations of the default dimensionalities.

In case the dimensionality cannot be inferred, the type checker will use the dimensionality `Unknown`.
At runtime, the dimensionality will be evaluated based on the unit that was specified.
The `Unknown` dimensionality is also used for operations using `*`, `**` and `/` that are not explicitly defined as overload variants.

If necessary, the dimensionality of a quantity can be explicitly specified by providing a subclass of `encomp.utypes.Dimensionality` as type parameter.

Commonly used dimensionalities are defined in the `encomp.utypes` module.
When a new dimensionality is created, the classname will be `Dimensionality[...]` (for example `Quantity[Dimensionality[[mass] ** 2 / [length] ** 3]]`).

```python
from encomp.units import Quantity as Q
from encomp.utypes import Volume, MassFlow

# the types are inferred by a static type checker like mypy

# the unit "m" is registered as a Mass unit
m = Q(12, 'kg')  # Quantity[Mass]

V = Q(25, 'liter')  # Quantity[Volume]

# common / and * operations are encoded as overloads
rho = m / V  # Quantity[Density]

# the unit "kg/week" is not registered by default
# the individual units "kg" and "week" are registered, however
# the type checker does not know how to combine these units
m_ = Q(25, 'kg/week')  # Quantity[Unknown]

# at runtime, the dimensionality of m_ will be evaluated to MassFlow
isinstance(m_, Q[MassFlow])  # True

# these operations (Mass**2 divided by Volume) are not explicitly defined as overloads
# at runtime, the type will be evaluated to
# Quantity[Dimensionality[[mass] ** 2 / [length] ** 3]]
x = m**2 / V  # Quantity[Unknown]

# the unit name "meter cubed" is not defined using an overload,
# the type parameter Volume is instead used to infer the type
y = Q[Volume](15, 'meter cubed')  # Quantity[Volume]

# in case the explicitly defined dimensionality does
# not match the unit, an error will be raised at runtime

y = Q[MassFlow](15, 'meter cubed')
# ExpectedDimensionalityError: Quantity with unit "m³" has incorrect dimensionality
# [length] ** 3, expected [mass] / [time]
```

#### Runtime type checking

The `Quantity` subtypes can be used to restrict function and class attribute types at runtime.
Use the `typeguard.typechecked` decorator to apply runtime typechecking to function inputs and outputs:

```python
from typeguard import typechecked
from typing import TypedDict

from encomp.units import Quantity as Q
from encomp.utypes import Temperature, Length, Pressure

@typechecked
def some_func(T: Q[Temperature]) -> tuple[Q[Length], Q[Pressure]]:
    return T * Q(12.4, 'm/K'), Q(1, 'bar')

some_func(Q(12, 'delta_degC'))  # the dimensionalities check out
some_func(Q(26, 'kW'))  # raises an exception:
# TypeError: type of argument "T" must be Quantity[Temperature];
# got Quantity[Power] instead

class OutputDict(TypedDict):

    P: Q[Pressure]
    T: Q[Temperature]

@typechecked
def another_func(s: Q[Length]) -> OutputDict:
    return {
        'T': Q(25, 'm'),
        'P': Q(25, 'kPa')
    }

another_func(Q(25, 'm'))
# TypeError: type of dict item "T" for the return value must be
# encomp.units.Quantity[Temperature]; got encomp.units.Quantity[Length] instead
```

To create a new dimensionality (for example temperature difference per mass flow rate), combine the `pint.UnitsContainer` objects stored in the `dimensions` class attribute.

```python
from encomp.units import Quantity as Q
from encomp.units import DimensionalityError
from encomp.utypes import TemperatureDifference, MassFlow, Volume, Dimensionality

# the class name TemperaturePerMassFlow must be globally unique
class TemperaturePerMassFlow(Dimensionality):
    dimensions = TemperatureDifference.dimensions / MassFlow.dimensions

# note the extra parentheses around (kg/s)
qty = Q[TemperaturePerMassFlow](1, 'delta_degC/(kg/s)')

# raises an exception since liter is Length**3 and the Quantity expects Mass
try:
    another_qty = Q[TemperaturePerMassFlow](1, 'delta_degC/(liter/hour)')
except DimensionalityError:
    pass

# create a new subclass of Quantity with restricted input units

CustomCoolingCapacity = Q[TemperaturePerMassFlow]

# the pint library handles a wide range of input formats and unit names
# the prefix "delta_" can be omitted in this case
q1 = CustomCoolingCapacity(6, '°F per (lbs per week)')
q2 = Q('3 deltdegree_Fahrenheit per (pound per fortnight)')

assert q1 == q2
assert type(q1) is type(q2)
```

### The `Fluid` class

The class `encomp.fluids.Fluid` is a wrapper around the *CoolProp* library.
The class uses two input points (three for humid air) that fix the state of the fluid.
Other fluid parameters can be evaluated using attribute access.
The outputs and inputs are `Quantity` objects.
CoolProp property names and codes are used throughout.
Use the `.search()` method to find the correct name.

```python
from encomp.units import Quantity as Q
from encomp.fluids import Fluid

air = Fluid('air', T=Q(25, 'degC'), P=Q(2, 'bar'))

# common fluid properties have type hints, and show up using autocomplete
air.D # 2.338399526231983 kilogram/meter3

air.search('density')
# ['DELTA, Delta: Reduced density (rho/rhoc) [dimensionless]',
#  'DMOLAR, Dmolar: Molar density [mol/m³]',
#  'D, DMASS, Dmass: Mass density [kg/m³]', ...

# any of the names are valid attributes (case-sensitive)
air.Dmolar # 80.73061937328056 mole/meter3
```

The fluid name `'water'` (or the subclass `Water`) uses *IAPWS* to evaluate steam and water properties.

```python
from encomp.units import Quantity as Q
from encomp.fluids import Fluid, Water

Fluid('water', P=Q(25, 'bar'), T=Q(550, '°C'))
# <Fluid "water", P=2500 kPa, T=550.0 °C, D=6.7 kg/m³, V=0.031 cP>

# note that the CoolProp property "Q" (vapor quality) has the same name as the class
# the Water class has a slightly different string representation
Water(Q=Q(0.5), T=Q(170, 'degC'))
# <Water (Two-phase), P=792 kPa, T=170.0 °C, D=8.2 kg/m³, V=0.0 cP>

Water(H=Q(2800, 'kJ/kg'), S=Q(7300, 'J/kg/K'))
# <Water (Gas), P=225 kPa, T=165.8 °C, D=1.1 kg/m³, V=0.0 cP>
```

The `HumidAir` class requires three input points (`R` means relative humidity):

```python
from encomp.units import Quantity as Q
from encomp.fluids import HumidAir

HumidAir(P=Q(1, 'bar'), T=Q(100, 'degC'), R=Q(0.5))
# <HumidAir, P=100 kPa, T=100.0 °C, R=0.50, Vda=2.2 m³/kg, Vha=1.3 m³/kg, M=0.017 cP>
```

## Tests

First, make sure the development dependencies are installed with `pip install encomp[dev]` or `pip install encomp[full]`.
Run the tests with

```bash
pytest -W ignore --pyargs encomp -p no:mypy-testing
```

## Settings

The attributes in the `encomp.settings.Settings` class can be modified with an `.env`-file.
Place a file named `.env` in the current working directory to override the default settings.
The attribute names are prefixed with `ENCOMP_`.

## TODO

- Document the `Quantity[Dimensionality]` type system
- What is the license of this package?
  - For example, `pint` uses *3-Clause BSD License*, this should be compatible with `MIT`
  - Should this package include the `pint` license text somewhere?
    - Extending the `pint` package counts as modification
