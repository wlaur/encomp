from typing import TYPE_CHECKING

import pytest
import numpy as np


from ..units import Quantity as Q

if not TYPE_CHECKING:
    def reveal_type(x): return x


@pytest.mark.mypy_testing
def test_quantity_magnitude_types() -> None:
    return

    p1 = Q(2)

    p1.m

    if isinstance(p1.m, np.ndarray):
        pass
    else:

        # autopep8: off

        # mypy will combine int | float -> float, since
        # all operations that support float will also support int
        reveal_type(p1.m) # R: builtins.float

        # autopep8: on

    p2 = Q([25, 35], 'km')

    reveal_type(p2.m)  # R: Union[builtins.float, numpy.ndarray[Any, Any]]
