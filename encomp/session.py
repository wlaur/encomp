import numpy as np  # noqa
import pandas as pd  # noqa
import polars as pl  # noqa

from .settings import SETTINGS  # noqa
from .misc import grid_dimensions as _grid_dimensions  # noqa
from .sympy import sp  # noqa
from .units import Quantity, ureg  # noqa
from .units import ureg as u  # noqa
from .units import Quantity as Q  # noqa
from .utypes import *  # noqa
from .fluids import Fluid, Water, HumidAir  # noqa


a = Q(np.array([2.5, 5.23]), 'kg')


b = a / Q(125, 'd') * 2 / 2

aa = b[0]
