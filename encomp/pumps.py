"""
Classes that represent and model the behavior of different types of pumps.
"""

from __future__ import annotations

from typing import Optional, Union, Annotated
import copy
from pydantic import validator, BaseModel
import numpy as np

from encomp.math import interpolate
from encomp.units import Quantity
from encomp.misc import isinstance_types


# TODO: find a suitable solution to type hint np.ndarray (certain shape etc.)
# maybe Annotated[numpy.typing.ArrayLike, shape] will work?
PumpData = Annotated[np.ndarray, '2D or 3D array']
Ratio = Annotated[float, 'Ratio, > 0']


class Pump(BaseModel):

    class Config:
        arbitrary_types_allowed = True


class CentrifugalPump(Pump):

    data: PumpData
    units: Union[tuple[str, str], tuple[str, str, str]]

    @validator('data')
    def check_data(cls, v):

        if v.shape[1] not in (2, 3):
            raise ValueError('Pump data must be an array with '
                             f'2 or 3 columns, passed shape: {v.shape}')

        return v

    @validator('units', pre=True)
    def check_units_type(cls, v):

        # pydantic will cast set to tuple, cannot do this
        # here since the order of the elements matters
        if isinstance(v, set):
            raise TypeError(
                f'Do not pass units as a set, use list or tuple instead: {v}')

        return v

    @validator('units')
    def check_units(cls, v):

        if not isinstance_types(Quantity(1, v[0]),
                                Union[Quantity['MassFlow'],
                                      Quantity['VolumeFlow']]):
            raise ValueError('First unit is mass or volume flow, '
                             f'passed "{v[0]}"')

        if not isinstance_types(Quantity(1, v[1]), Quantity['Pressure']):
            raise ValueError('Second unit is head (pressure), '
                             f'passed "{v[1]}"')

        if len(v) == 3 and not isinstance_types(Quantity(1, v[2]), Quantity['Power']):
            raise ValueError('Third unit is power, '
                             f'passed "{v[2]}"')

        return v

    def flow(self, head):
        pass

    def head(self, flow):
        pass

    def power(self, param):
        pass

    def transform(self, *,
                  frequency: Optional[Ratio] = None,
                  diameter: Optional[Ratio] = None) -> CentrifugalPump:
        """
        Applies affinity law transformations to the pump instance and returns
        a new instance. One or both of ``frequency`` and ``diameter`` can be
        specified, since the order of the transformations does not matter.

        Refer to https://en.wikipedia.org/wiki/Affinity_laws for more information.

        Parameters
        ----------
        frequency : Optional[Ratio]
            Ratio by which to change the rotation frequency (1 means unchanged), by default None
        diameter : Optional[Ratio]
            Ratio by which to change the impeller diameter (1 means unchanged), by default None

        Returns
        -------
        CentrifugalPump
            New instance with the rotation frequency and/or impeller diameter changed.
        """

        if frequency is None and diameter is None:
            return self

        # transform both frequency and diameter, order does not matter
        if frequency is not None and diameter is not None:
            return self.transform(frequency=frequency).transform(diameter=diameter)

        # [flow, head, power], power might be NaN (does not matter for this calculation)
        arr = self.data.copy()

        # constant impeller diameter, variable frequency
        if frequency is not None:
            exponents = {
                'flow': 1,
                'head': 2,
                'power': 3}

        # constant rotation frequency, variable impeller diameter
        elif diameter is not None:
            exponents = {
                'flow': 3,
                'head': 2,
                'power': 5}

        ratio = frequency or diameter or 1

        arr[:, 0] *= ratio**exponents['flow']
        arr[:, 1] *= ratio**exponents['head']
        arr[:, 2] *= ratio**exponents['power']

        transformed = self.copy()
        transformed.data = arr

        return transformed

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def grid_interpolation(x: np.ndarray,
                           arr: np.ndarray,
                           idx: int) -> np.ndarray:

        x = np.array(x)
        K = arr.shape[1]

        arr_interp = np.zeros((len(x), K))

        arr_interp[:, idx] += x

        for i in range(K):

            # this dimension should not be interpolated
            # since it is the basis for interpolation
            if i == idx:
                continue

            x_i, y_i = arr[:, idx], arr[:, i]
            interp_func = interpolate(x_i, y_i, fill_value='extrapolate')
            arr_interp[:, i] += interp_func(x)

        return arr_interp


class PositiveDisplacementPump(Pump):
    pass
