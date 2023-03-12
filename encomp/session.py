import numpy as np
import pandas as pd
import polars as pl


from .settings import SETTINGS  # noqa
from .misc import grid_dimensions as _grid_dimensions  # noqa
from .sympy import sp  # noqa
from .units import Quantity, ureg  # noqa
from .units import ureg as u  # noqa
from .units import Quantity as Q  # noqa
from .utypes import *  # noqa
from .fluids import Fluid, Water, HumidAir  # noqa


q = Q(pl.col('asd'), 'bar')
q2 = Q([2.5, 35.2], 'bar')


q3 = Q([2, 4], 'bar')
q3 = Q(np.array([3, 4, 51,]), 'bar')
from typing import Any

aa = Q([1, 2, 3], 'kg')
aa._magnitude


dd = Q([1, 2, 4], 'asdss')
