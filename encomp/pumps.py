"""
Classes that represent and model the behavior of different types of pumps.
"""


from typing import Optional
from typing_extensions import Annotated
import copy
from pydantic.dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from encomp.units import Quantity
from encomp.math import interpolate


# TODO: find a suitable solution to type hint np.ndarray (certain shape etc.)
# maybe Annotated[numpy.typing.ArrayLike, shape] will work?
PumpData = np.ndarray

Ratio = Annotated[float, 'Ratio, > 0']


@dataclass
class Pump:
    pass
    # d: Quantity['Temperature']


class CentrifugalPump(Pump):

    def flow(self, head):
        pass

    def head(self, flow):
        pass

    def power(self, param):
        pass

    def transform(self, *,
                  frequency: Optional[Ratio] = None,
                  diameter: Optional[Ratio] = None) -> 'CentrifugalPump':
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

        ratio = frequency or diameter

        arr[:, 0] *= ratio**exponents['flow']
        arr[:, 1] *= ratio**exponents['head']
        arr[:, 2] *= ratio**exponents['power']

        transformed = self.copy()
        transformed.data = arr

        return transformed

    def copy(self) -> 'CentrifugalPump':
        return copy.deepcopy(self)

    @staticmethod
    def grid_interpolation(x: npt.ArrayLike, arr: npt.ArrayLike, idx: int) -> np.ndarray:

        x = np.array(x)
        arr_interp = np.zeros((len(x), arr.shape[1]))

        arr_interp[:, idx] += x

        for i in range(arr.shape[1]):

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
