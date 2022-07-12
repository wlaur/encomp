import numpy as np
from typing import Union, Callable, Optional, Mapping, Any
from scipy.interpolate import interp1d
from sympy import geometry


def interpolate(
    x: Mapping[int, float],
    y: Mapping[int, float],
    fill_value: Union[str, tuple[float, float]] = 'nan'
) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """
    Wrapper around ``scipy.interpolate.interp1d``
    that returns function for interpolating :math:`x` vs :math:`x`.

    Example usage:

    .. code-block:: python

        x = np.linspace(0, 5, 25)
        y = np.random.rand(x.shape[0])
        interpolate(x, y)([0, 4, 4.1])  # interpolated values at x = 0, 4, 4,1

    Parameters
    ----------
    x : Mapping[int, float]
        Sequence of :math:`x`-values
    y : Mapping[int, float]
        Sequence of :math:`y`-values with same length as ``x``
    fill_value : str, optional
        What to do outside the interpolation range, by default 'nan'

        - 'limits': Values outside bounds are set to the upper and lower limit
        - 'nan': Return ``np.nan`` outside the interval
        - 'extrapolate': Extrapolate based on last and first numbers (linear extrapolation)
        - 'error': Raise ValueError when the function is called outside the range
        - (lower, upper): Explicit numerical values beyond the lower and upper limits

    Returns
    -------
    Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]
        An interpolation function based on the input dataset
    """

    kwargs: dict[str, Any] = {'bounds_error': False}

    if isinstance(fill_value, str):

        fill_value = fill_value.lower().strip()

        if fill_value not in ('nan', 'limits', 'extrapolate'):
            raise ValueError(
                f'Incorrect input for fill_value: {fill_value}, '
                'possible options are str "nan", "limits", "extrapolate" or float (lower, upper)')

    if fill_value == 'limits':
        kwargs['fill_value'] = (y[0], y[-1])

    elif fill_value == 'nan':
        kwargs['fill_value'] = (np.nan, np.nan)

    elif fill_value == 'error':
        kwargs['bounds_error'] = True

    else:
        kwargs['fill_value'] = fill_value

    return interp1d(x, y, **kwargs)


def polynomial(x: np.ndarray,
               y: np.ndarray,
               order: int = 2) -> np.poly1d:
    """
    Wrapper around ``np.poly1d`` and ``np.polyfit``
    that returns a polynomial function :math:`p(x)` based on a least-squares fit to
    data :math:`x` vs :math:`x`.

    Usage:

    .. code-block:: python

        x = np.linspace(0, 5, 25)a
        noise = np.random.rand(x.shape[0]) * 0.01
        y = x**2 + noise
        poly = polynomial(x, y)
        poly([0, 1, 2])  # values for p_2(x) at x = 0, 5, 7
        poly.coefficients  # polynomial coefficients, length order+1

    Parameters
    ----------
    x : np.ndarray
        Sequence of :math:`x`-values
    y : np.ndarray
        Sequence of :math:`y`-values with ame length as ``x``
    order : int, optional
        Order of the polynomial, by default 2

    Returns
    -------
    np.poly1d
        An polynomial function based on the input dataset
    """

    return np.poly1d(np.polyfit(x, y, order))


def exponential(x_start: float,
                x_end: float,
                y_start: float,
                y_end: float,
                k: float,
                eps: float = 1e-6) -> Callable[[float], float]:
    """
    Returns an exponential curve between the points

    :math:`(x_{\\text{start}}, y_{\\text{start}}) \\rightarrow (x_{\\text{end}}, y_{\\text{end}})`

    .. math::
        y(x) = A + B \\cdot \\exp{\\left(k \\cdot \\frac{x - x_{\\text{start}}}{x_{\\text{end}} - x_{\\text{start}}}\\right)}

    The parameter ``k`` is used to control the shape of the curve.

    Usage:

    .. code-block:: python

        x = np.linspace(0, 10)
        exp_func = exponential(0, 10, 20, 30, k=-5)
        plt.plot(x, exp_func(x))

    Parameters
    ----------
    x_start : float
        Start :math:`x`-coordinate
    x_end : float
        End :math:`x`-coordinate
    y_start : float
        Start :math:`y`-coordinate
    y_end : float
        End :math:`y`-coordinate
    k : float
        Curve shape factor

        - :math:`k < 0`: Largest increase at the start of the curve
        - :math:`k > 0`: Largest increase at end of the curve
        - :math:`k = 0`: Linear curve

        :math:`k` cannot be exactly 0 due to numerical accuracy, will be set to ``eps`` in that case.

    eps : float, optional
        Numerical accuracy, by default 1e-6

    Returns
    -------
    Callable
        An exponential function with single input and output
    """

    # avoid numerical problems around zero
    if abs(k - 0) < eps:
        k = eps

    return lambda x: (y_start + (y_end - y_start) /
                      (1 - np.exp(k)) *
                      (1 - np.exp(k * (x - x_start) / (x_end - x_start))))


def r_squared(y_pred: np.ndarray,
              y_data: np.ndarray) -> float:
    """
    Calculates the :math:`R^2`-value for predicted values ``y_pred``
    based on known data in ``y_data``.

    ``y_pred`` and ``y_data`` correspond to the same :math:`x`-values, and
    must have the same length.

    Parameters
    ----------
    y_pred : np.ndarray
        Sequence of estimated :math:`y`-values
    y_data : np.ndarray
        Sequence of known :math:`y`-values

    Returns
    -------
    float
        The :math:`R^2`-value for the specified inputs
    """

    residual = y_pred - y_data

    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_pred - np.mean(y_pred))**2)

    R2 = 1 - (ss_res / ss_tot)

    return R2


def circle_line_intersection(A: Union[tuple[float, float], geometry.Point2D],
                             B: Union[tuple[float, float], geometry.Point2D],
                             x0: float,
                             y0: float,
                             r: float) -> Optional[list[tuple[float, float]]]:
    """
    Finds the intersection point(s) between:

    * Circle with center :math:`x_0, y_0` and radius :math:`r`
    * Line between two points :math:`A \\rightarrow B`.

    Returns None in case the line does not intersect the circle.

    Parameters
    ----------
    A : Union[tuple[float], geometry.Point2D]
        start of the line
    B : Union[tuple[float], geometry.Point2D]
        end of the line
    x0 : float
        x-coordinate for circle center
    y0 : float
        y-coordinate for circle center
    r : float
        radius of circle

    Returns
    -------
    Optional[list[tuple[float, float]]]
        One or two intersection points between the circle and the line,
        or None in case the line and circle do no intersect.
    """

    if isinstance(A, tuple):
        A = geometry.Point2D(A[0], A[1])

    if isinstance(B, tuple):
        B = geometry.Point2D(B[0], B[1])

    circle = geometry.Circle((x0, y0), r)

    line = geometry.Segment2D(A, B)

    points = circle.intersection(line)

    if not points:
        return None

    return [(float(p[0]), float(p[1])) for p in points]

