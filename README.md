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
