import numpy as np

from ..math import exponential


def test_exponential():
    fcn = exponential(1, 2, 2, 3, 0.1)

    x = np.linspace(-1, 1)

    y = fcn(x)

    assert len(y) == len(x)
