# encomp


> General-purpose library for *en*gineering *comp*utations, with focus on clean and consistent interfaces.

## Features

* Consistent interfaces to commonly used tools
    * ``pint`` for physical units
    * ``CoolProp`` for fluid properties
    * ``fluids`` and ``thermo`` for process engineering calculations
    * EPANET for piping network simulations
    * The rest of the Python scientific stack
* Strong type system that integrates physical units and their dimensionalities
    * Leverages the standard library ``typing`` module, ``pint`` and ``typeguard`` to ensure that inputs and outputs to functions and classes match the specified dimensionalities
    * Uses ``pydantic`` to create self-validating objects
* Seamless integration with Excel and JSON formats


> This library is under work: all features are not yet implemented.


## Installation

Open a terminal and navigate to the root directory of this repository.
First, setup an environment using ``conda``:

```
conda env create -f environment.yml
```

This will install the necessary dependencies into a new ``conda`` environment named ``encomp-env``.
Some dependencies are installed with ``pip``.


Then, install ``encomp`` into this environment:

```
conda activate encomp-env
pip install .
```


### Removing the ``conda`` environment

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

The ``api`` module contains all public functions and classes:

```python
from encomp.api import ...
```

### The ``Quantity`` class


The main part of ``encomp`` is the ``encomp.Quantity`` class, which is an extension of ``pint.Quantity``.
This class is used to construct objects with a *magnitude* and *unit*.
It can also be used to restrict function and class attribute types.
Each *dimensionality* (for example *pressure*, *length*, *time*) is represented by a subclass of ``Quantity``.

Use type annotations to restrict the dimensionalities of a function's parameters and return value.
The ``typeguard.typechecked`` decorator is automatically applied to all functions and methods inside the main ``encomp`` library.
To use it on your own functions, apply the decorator explicitly.


```python
from typeguard import typechecked
from encomp.api import Quantity

@typechecked
def some_func(T: Quantity['Temperature']) -> Quantity['Length']:
    return T * Quantity(12.4, 'm/K')

some_func(Q(12, 'delta_degC'))  # the dimensionalities check out
some_func(Q(26, 'kW'))  # raises an exception
# TypeError: type of argument "T" must be Quantity[Temperature]; got Quantity[Power] instead
```

The dimensionalities are listed the dict ``encomp.utypes._DIMENSIONALITIES``.
To create a new dimensionality (for example temperature difference per length), use the ``pint.UnitsContainer`` objects defined in ``encomp.utypes``.


## TODO


* Combine EPANET for pressure / flow simulation with energy systems simulations (``omeof``)
* Make a web interface to draw circuits (using a JS node-graph editor) and visualize results.

Ensure compatibility with

* numpy
* pandas
* Excel (via df.to_excel, both with ``openpyxl`` and ``xlsxwriter``
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
