# encomp


> General-purpose library for *en*gineering *comp*utations, with focus on clean and consistent interfaces.

## Getting started

Install ``encomp`` using ``pip``:


```
pip install encomp
```

To use ``encomp`` from a Jupyter Notebook, import the ``notebook`` module:


```python
# imports commonly used functions, registers jupyter magics
from encomp.notebook import *
```

The ``api`` module contains all public functions and classes:

```python
from encomp.api import ...
```


## Development

First, setup a development environment using ``conda``:

```
conda env create -f environment.yml
```


### Removing ``conda`` environments

To completely remove the ``conda`` environment for ``encomp``:

```
conda remove -y --name encomp-env --all
```



## TODO


> GOAL: combine EPANET for pressure / flow simulation with energy systems simulations (omeof). Make a web interface to draw circuits (using a JS node-graph editor) and visualize results.

Ensure compatibility with

* numpy
* pandas
* Excel (via df.to_excel, both with openpyxl and xlsxwriter. also read from CSV, XLS, XLSX, XLSM)
    * parse units from Excel (header name like "Pressure [bar]" etc...)
* nbconvert (HTML and Latex/PDF output)
    * figure out how to typeset using SIUNITX
* CoolProp



* http://www.thermocycle.net/
* https://github.com/topics/process-engineering
* https://github.com/oemof/tespy
* https://github.com/oemof
* https://python-control.readthedocs.io/en/0.9.0/index.html
* https://ruralwater.readthedocs.io/en/dev/readme.html
