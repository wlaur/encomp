# encomp

> General-purpose library for *en*gineering *comp*utations, with focus on clean and consistent interfaces.

Documentation at https://encomp.readthedocs.io/en/latest/

## Features

Main functionality of the `encomp` library:

- Handles physical quantities with magnitude(s), dimensionality and units

  - Modules `encomp.units`, `encomp.utypes`
  - Extends the [pint](https://pypi.org/project/Pint) library
  - Uses Python's type system to validate dimensionalities
  - Integrates with `np.ndarray` and `pd.Series`
  - Automatic JSON serialization and decoding

- Implements a flexible interface to [CoolProp](http://www.coolprop.org)

  - Module `encomp.fluids`
  - Uses quantities for all inputs and outputs
  - Fluids are represented as class instances, the properties are class attributes

- Extends [Sympy](https://pypi.org/project/sympy/)

  - Module `encomp.sympy`
  - Adds convenience methods for creating symbols with sub- and superscripts
  - Additional functions to convert (algebraic) expressions and systems to Python code that supports Numpy arrays

- Jupyter Notebook integration

  - Module `encomp.notebook`
  - Imports commonly used functions and classes
  - Defines custom Jupyter magics

The other modules implement calculations related to process engineering and thermodynamics.
The module `encomp.serialize` implements custom JSON serialization and decoding for classes used elsewhere in the library.

> This library is under work: all features are not yet implemented.

## Installation

Install with `pip`:

```
pip install encomp
```

This will install `encomp` along with its dependencies into the currently active Python environment.

> `CoolProp` is not installable with `pip` for Python 3.9. Install manually with `conda` for now:

```
conda install conda-forge::coolprop
```

## Getting started

To use `encomp` from a Jupyter Notebook, import the `encomp.notebook` module:

```python
# imports commonly used functions, registers Notebook magics
from encomp.notebook import *
```

This will import commonly used functions and classes.
It also registers the `%read` and `%%write` Jupyter magics for reading and writing custom objects from and to JSON.

### The `Quantity` class

The main part of `encomp` is the `encomp.units.Quantity` class (shorthand alias `Q`), which is an extension of `pint.Quantity`.
This class is used to construct objects with a _magnitude_ and _unit_.

Some examples:

```python
from encomp.units import Q

# converts 1 bar to kPa, displays it in case it's the cell output
Q(1, 'bar').to('kPa')

# a single string with one numerical value can also be given as input
Q('0.1 MPa').to('bar')

# list and tuple inputs are converted to np.ndarray
Q([1, 2, 3], 'bar') * 2 # [2, 4, 6] bar

# in case no unit is specified, the quantity is dimensionless
Q(0.1) == Q(10, '%')
```

The `Quantity` class can also be used to restrict function and class attribute types.
Each _dimensionality_ (for example _pressure_, _length_, _time_, _dimensionless_) is represented by a subclass of `Quantity`.
It is possible to use type annotations to restrict the dimensionalities of function parameters and return values.

In case the `ENCOMP_TYPE_CHECKING` environment variable is set to `True`, the `typeguard.typechecked` decorator is automatically applied to all functions and methods inside the main `encomp` library.
To use it on your own functions, apply the decorator explicitly:

```python
from typeguard import typechecked

# the full class name is used for annotations, for the sake of clarity
from encomp.units import Q, Quantity

@typechecked
def some_func(T: Quantity['Temperature']) -> tuple[Quantity['Length'], Quantity['Pressure']]:
    return T * Q(12.4, 'm/K'), Q(1, 'bar')

some_func(Q(12, 'delta_degC'))  # the dimensionalities check out
some_func(Q(26, 'kW'))  # raises an exception:
# TypeError: type of argument "T" must be Quantity[Temperature]; got Quantity[Power] instead
```

The dimensionality of a quantity can be specified with string values like `'Temperature'` or `pint.UnitsContainer` objects.
To create a new dimensionality (for example temperature difference per length), combine the `pint.UnitsContainer` objects defined in `encomp.utypes` using `*` and `/`.

```python
from encomp.units import Q, Quantity
from encomp.utypes import Temperature, Length, Volume

qty = Quantity[Temperature / Length](1, 'delta_degC / km')

# raises an exception since liter is Length**3 and the Quantity expects Length**2
another_qty = Quantity[Temperature / Length**2](1, 'delta_degC / liter')

# create a new subclass of Quantity with restricted input units
CustomCoolingCapacity = Quantity[Temperature / Volume]

# Quantity handles a wide range of input formats and unit names
assert CustomCoolingCapacity(3, '°F per yard³') == Q('3 degree_Fahrenheit per yard cubed')
```

### The `Fluid` class

The class `encomp.fluids.Fluid` is a wrapper around the _CoolProp_ library.
The class takes two input points (three for humid air) that fix the state of the fluid.
Other fluid parameters can be evaluated using attribute access.
The outputs and inputs are `Quantity` objects.
CoolProp property names and codes are used throughout.
Use the `.search()` method to find the correct name.

```python
from encomp.units import Q
from encomp.fluids import Fluid

air = Fluid('air', T=Q(25, 'degC'), P=Q(2, 'bar'))

air.D # 2.338399526231983 kilogram/meter3

air.search('density')
# ['DELTA, Delta: Reduced density (rho/rhoc) [dimensionless]',
#  'DMOLAR, Dmolar: Molar density [mol/m³]',
#  'D, DMASS, Dmass: Mass density [kg/m³]', ...

# any of the names are valid attributes (case-sensitive)
air.Dmolar # 80.73061937328056 mole/meter3
```

The fluid name `'water'` (or the alias class `Water`) uses _IAPWS_ to evaluate steam and water properties.

```python
from encomp.units import Q
from encomp.fluids import Fluid, Water

Fluid('water', P=Q(25, 'bar'), T=Q(550, 'C'))
# <Fluid "water", P=2500 kPa, T=550.0 °C, D=6.7 kg/m³, V=0.031 cP>

# note that the CoolProp property "Q" (vapor quality) has the same name as the class
# the Water class has a slightly different string representation
Water(Q=Q(0.5), T=Q(170, 'degC'))
# <Water (Two-phase), P=792 kPa, T=170.0 °C, D=8.2 kg/m³, V=0.0 cP>

Water(H=Q(2800, 'kJ/kg'), S=Q(7300, 'J/kg/K'))
# <Water (Gas), P=225 kPa, T=165.8 °C, D=1.1 kg/m³, V=0.0 cP>
```

The `HumidAir` class requires three input points (``R`` means relative humidity):

```python
from encomp.units import Q
from encomp.fluids import HumidAir

HumidAir(P=Q(1, 'bar'), T=Q(100, 'degC'), R=Q(0.5))
# <HumidAir, P=100 kPa, T=100.0 °C, R=0.50, Vda=2.2 m³/kg, Vha=1.3 m³/kg, M=0.017 cP>
```

## Settings

The attributes in the `encomp.settings.Settings` class can be modified with an `.env`-file.
Place a file named `.env` in the current working directory to override the default settings.
The attribute names are prefixed with `ENCOMP_`.
See the file `.env.example` in the base of this repository for examples.

## TODO

- Would be nice to see issues with dimensionality directly in the IDE
  - Might not be possible since the subclass `Quantity['Temperature']` is constructed at runtime
- Combine EPANET (`wntr`) for pressure / flow simulation with energy systems simulations (`omeof`)

Ensure compatibility with

- numpy
- pandas
- nbconvert (HTML and Latex/PDF output)

  - figure out how to typeset using SIUNITX
  - look into JupyterBook and similar projects

- http://www.thermocycle.net/
- https://github.com/topics/process-engineering
- https://github.com/oemof/tespy
- https://github.com/oemof
- https://python-control.readthedocs.io/en/0.9.0/index.html
- https://ruralwater.readthedocs.io/en/dev/readme.html
