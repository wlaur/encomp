import pytest

from encomp.units import DimensionalityError
from encomp.units import Quantity as Q
from encomp.utypes import (Dimensionless,
                           NormalVolumeFlow,
                           MassFlow,
                           Time,
                           Mass,
                           Temperature)



# it's important that the expected mypy output is a comment on the
# same line as the expression, disable autopep8 if necessary with
# autopep8: off
# ... some code above the line length limit
# autopep8: on


@pytest.mark.mypy_testing
def test_quantity_combined_types() -> None:
    pass