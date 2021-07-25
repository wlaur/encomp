# encomp

> General-purpose library for *en*gineering *comp*utations, with focus on clean and consistent interfaces.

Package documentation at https://encomp.readthedocs.io/en/latest/

## Features

Main functionality of the ``encomp`` library:

* Handles physical quantities with magnitude(s), dimensionality and units

   * Modules ``encomp.units``, ``encomp.utypes``
   * Extends the ``pint`` library
   * Uses Python's type system to validate dimensionalities
   * Integrates with ``np.ndarray`` and ``pd.Series``
   * Automatic JSON serialization and decoding

* Implements a flexible interface to [CoolProp](http://www.coolprop.org)

   * Module ``encomp.fluids``
   * Uses quantities for all inputs and outputs
   * Fluids are represented as class instances, the properties are class attributes

* Extends Sympy

   * Modules ``encomp.sympy``, ``encomp.balances``
   * Adds convenience methods for creating symbols with sub- and superscripts
   * Additional functions to convert (algebraic) expressions and systems to Python code that supports Numpy arrays

* Jupyter Notebook integration

   * Module ``encomp.notebook``
   * Imports commonly used functions and classes
   * Defines custom Jupyter magics


The other modules implement calculations related to process engineering and thermodynamics.
The module ``encomp.serialize`` implements custom JSON serialization and decoding for classes used elsewhere in the library.

> This library is under work: all features are not yet implemented.


## Installation

Install with ``pip``:

```
pip install encomp
```

This will install ``encomp`` along with its dependencies into the currently active Python environment.

> ``CoolProp`` is not installable with ``pip`` for Python 3.9. Install manually with ``conda`` for now:

```
conda install conda-forge::coolprop
```


### Development environment

Install Miniconda or Anconda if not already installed.
Clone this repository, open a terminal and navigate to the root directory.
Setup a new environment using ``conda``:

```
conda env create -f environment.yml
```

This will install the necessary dependencies into a new ``conda`` environment named ``encomp-env``.
The dependencies (except for ``scipy`` and ``jupyter``) are installed with ``pip``.

Install ``encomp`` into the new environment:

```
conda activate encomp-env
pip install .
```


#### Removing the ``conda`` environment

To completely remove the ``conda`` environment for ``encomp``:

```
conda remove -y --name encomp-env --all
```


## Getting started

To use ``encomp`` from a Jupyter Notebook, import the ``encomp.notebook`` module:


```python
# imports commonly used functions, registers Notebook magics
from encomp.notebook import *
```

This will import commonly used functions and classes.
It also registers the ``%read`` and ``%%write`` Jupyter magics for reading and writing custom objects from and to JSON.

Some examples:

```python
# converts 1 bar to kPa, displays it in case it's the cell output
Q(1, 'bar').to('kPa')
```

```python
# creates an object that represents water at a certain temperature and pressure
Water(T=Q(25, 'degC'), P=Q(2, 'bar'))
```


### The ``Quantity`` class


The main part of ``encomp`` is the ``encomp.units.Quantity`` class (shorthand ``Q``), which is an extension of ``pint.Quantity``.
This class is used to construct objects with a *magnitude* and *unit*.
It can also be used to restrict function and class attribute types.
Each *dimensionality* (for example *pressure*, *length*, *time*) is represented by a subclass of ``Quantity``.

> Use type annotations to restrict the dimensionalities of function parameters and return values.

In case the ``ENCOMP_TYPE_CHECKING`` environment variable is set to ``True``, the ``typeguard.typechecked`` decorator is automatically applied to all functions and methods inside the main ``encomp`` library.
To use it on your own functions, apply the decorator explicitly:


```python
from typeguard import typechecked
from encomp.units import Q

@typechecked
def some_func(T: Q['Temperature']) -> tuple[Q['Length'], Q['Pressure']]:
    """
    Takes a temperature or temperature difference and
    returns length and pressure quantities.
    """

    return T * Q(12.4, 'm/K'), Q(1, 'bar')

some_func(Q(12, 'delta_degC'))  # the dimensionalities check out
some_func(Q(26, 'kW'))  # raises an exception
# TypeError: type of argument "T" must be Quantity[Temperature]; got Quantity[Power] instead
```

The dimensionality of a quantity can be specified with string values like ``'Temperature'`` or ``pint.UnitsContainer`` objects.
To create a new dimensionality (for example temperature difference per length), combine the ``pint.UnitsContainer`` objects defined in ``encomp.utypes`` using ``*`` and ``/``:


```python
from encomp.units import Q
from encomp.utypes import Temperature, Length

qty = Q[Temperature / Length](1, 'delta_degC / km')

# raises an exception since liter is Length**3 and the Quantity expects Length**2
another_qty = Q[Temperature / Length**2](1, 'delta_degC / liter')

# create a new subclass of Quantity with restricted input units
CustomQuantity = Q[Temperature / Length**3]

# pint handles a wide range of input formats
CustomQuantity(1, '°F per yard³')
```

## Settings

The attributes in the ``encomp.settings.Settings`` class can be modified with an ``.env``-file.
Place a file named ``.env`` in the current working directory to override the default settings.
The attribute names are prefixed with ``ENCOMP_``.
See the file ``.env.example`` in the base of this repository for examples.


## TODO

* Check compatibility with MyPy and other type checkers
   * Would be nice to see issues with dimensionality directly in the IDE
   * Might not be possible since the subclass ``Quantity['Temperature']`` is constructed at runtime
* Combine EPANET (``wntr``) for pressure / flow simulation with energy systems simulations (``omeof``)
* Make a web interface to draw circuits (using a JS node-graph editor) and visualize results.

Ensure compatibility with

* numpy
* pandas
* Excel (via ``df.to_excel``, both with ``openpyxl`` and ``xlsxwriter``
    * parse units from Excel (header name like "Pressure [bar]" etc...)
* nbconvert (HTML and Latex/PDF output)
    * figure out how to typeset using SIUNITX
    * look into JupyterBook and similar projects


* http://www.thermocycle.net/
* https://github.com/topics/process-engineering
* https://github.com/oemof/tespy
* https://github.com/oemof
* https://python-control.readthedocs.io/en/0.9.0/index.html
* https://ruralwater.readthedocs.io/en/dev/readme.html
