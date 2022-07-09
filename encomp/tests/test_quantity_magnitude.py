import pytest
import numpy as np


from encomp.units import Quantity as Q


# it's important that the expected mypy output is a comment on the
# same line as the expression, disable autopep8 if necessary with
# autopep8: off
# ... some code above the line length limit
# autopep8: on


@pytest.mark.mypy_testing
def test_quantity_magnitude_types() -> None:

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
