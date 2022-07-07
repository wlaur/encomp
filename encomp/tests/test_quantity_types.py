import pytest

from encomp.units import DimensionalityError, ExpectedDimensionalityError
from encomp.units import Quantity as Q
from encomp.utypes import (Dimensionless,
                           NormalVolumeFlow,
                           MassFlow,
                           Time,
                           Length,
                           Mass,
                           Temperature)


# it's important that the expected mypy output is a comment on the
# same line as the expression, disable autopep8 if necessary with
# autopep8: off
# ... some code above the line length limit
# autopep8: on


@pytest.mark.mypy_testing
def test_quantity_reveal_type() -> None:

    m = Q[MassFlow](1, 'kg/s')

    reveal_type(m)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    T = Q[Temperature](1, 'degC')

    reveal_type(T)  # R: encomp.units.Quantity[encomp.utypes.Temperature]

    # if the dimensionality is omitted, the inferred type is Quantity[Unknown]
    # the type will be correct at runtime
    unknown = Q(25, 'bar/week')  # E: Need type annotation for "unknown"

    # TODO: mypy infers the dimensionality type as Any, but pylance
    # correctly identifies it as Unknown
    reveal_type(unknown)  # R: encomp.units.Quantity[Any]


@pytest.mark.mypy_testing
def test_quantity_construction() -> None:

    Q(25)
    Q(25, None)

    Q[Mass](25, Q[Mass](25, 'kg'))
    Q[Mass](25, Q[Mass](25, 'kg').u)

    with pytest.raises(ExpectedDimensionalityError):

        # autopep8: off

        Q[Mass](25, Q[Time](25, 'sec')) # E: Argument 2 to "Quantity" has incompatible type "Quantity[Time]"; expected "Union[Unit, UnitsContainer, str, Quantity[Mass], None]"

        # autopep8: on

    mf_ = Q(25, 'kg/s')
    mf = Q[MassFlow](25, 'kg/week')
    e = Q(25, 'kJ')

    reveal_type(mf_)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(mf)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(e)  # R: encomp.units.Quantity[encomp.utypes.Energy]


@pytest.mark.mypy_testing
def test_quantity_mul_types() -> None:

    m = Q[MassFlow](1, 'kg/s')
    n = Q[NormalVolumeFlow](1, 'Nm^3/h')

    d = Q[Dimensionless](1, '%')

    s_int = 5
    s_float = 0.1

    # multiplying 2 Quantities creates a new dimensionality
    # that is only known at runtime
    # the type checker will assign the dimensionality Unknown
    p1 = m * m
    p2 = m * n
    p3 = n * m
    p4 = n * n

    # scalar and Q[Dimensionless] multiplication preserves dimensionality

    k1 = m * s_int
    k2 = s_int * m

    reveal_type(k1)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(k2)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    k3 = m * s_float
    k4 = s_float * m

    reveal_type(k3)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(k4)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    k5 = m * d
    k6 = d * m

    reveal_type(k5)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(k6)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    class CustomDimensionless(Dimensionless):
        pass

    d_custom = Q[CustomDimensionless](0.5)

    k7 = m * d_custom
    k8 = d_custom * m

    reveal_type(k7)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(k8)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    # alternative way of specifying the dimensionality
    # NOTE: the _dt parameter is ignored at runtime
    d_custom_ = Q(0.6, _dt=CustomDimensionless)
    k9 = m * d_custom_
    k10 = d_custom * m

    # TODO: this does not work (Unknown instead of MassFlow)
    reveal_type(k9)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # ... but this does work?
    reveal_type(k10)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]


@pytest.mark.mypy_testing
def test_quantity_misc_types() -> None:

    # autopep8: off

    # don't mix the __class_getitem__ and _dt parameter
    Q[Temperature](25, 'degC', _dt=Mass) # E: Argument "_dt" to "Quantity" has incompatible type "Type[Mass]"; expected "Type[Temperature]"

    with pytest.raises(ExpectedDimensionalityError):

        _cls = Q[Temperature]

        # this raises a runtime error, since bar is not a Temperature unit
        a = _cls(25, 'bar')

        # the literal "bar" is correctly identified as a Pressure unit
        reveal_type(a)  # R: encomp.units.Quantity[encomp.utypes.Pressure]


        # this cannot be detected, however the variable b has type Q[Temperature]
        # this will raise an error at runtime
        b = Q[Mass](25, 'C')

        reveal_type(b) # R: encomp.units.Quantity[encomp.utypes.Temperature]

    # autopep8: on


@pytest.mark.mypy_testing
def test_quantity_div_types() -> None:

    m = Q[MassFlow](1, 'kg/s')
    n = Q[NormalVolumeFlow](1, 'Nm^3/h')

    d = Q[Dimensionless](1, '%')

    s_int = 5
    s_float = 0.1

    # dividing 2 different Quantities creates a new dimensionality
    # that is only known at runtime
    p1 = m / n
    p2 = n / m

    reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.Unknown]
    reveal_type(p2)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # scalar/dimensionless divided by Quantity also creates a new, unknown dimensionality
    p3 = s_int / m
    p4 = s_float / m
    p5 = d / m

    reveal_type(p3)  # R: encomp.units.Quantity[encomp.utypes.Unknown]
    reveal_type(p4)  # R: encomp.units.Quantity[encomp.utypes.Unknown]
    reveal_type(p5)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # quantity divided by scalar/dimensionless preserves dimensionality
    p6 = m / d
    p7 = m / s_int
    p8 = m / s_float

    reveal_type(p6)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(p7)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(p8)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    p9 = d / s_int
    p10 = d / s_float

    reveal_type(p9)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p10)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # dividing by the same dimensionality creates dimensionless

    p11 = m / m
    p12 = n / n
    p13 = d / d

    reveal_type(p11)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p12)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p13)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # in case the dimensionalities are Unknown, division will output Unknown
    unknown = 1 / m**2

    # autopep8: off

    reveal_type(unknown / unknown) # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # autopep8: on


@pytest.mark.mypy_testing
def test_quantity_floordiv_types() -> None:

    # floordiv is only implemented in case the dimensionalities match

    m = Q[MassFlow](1, 'kg/s')
    n = Q[NormalVolumeFlow](1, 'Nm^3/h')
    d = Q[Dimensionless](1, '%')

    s_int = 5
    s_float = 0.1

    p1 = m // m
    p2 = n // n
    p3 = d // d

    reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p2)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p3)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    p4 = s_int // d
    p5 = d // s_int
    p6 = s_float // d
    p7 = d // s_float

    reveal_type(p4)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p5)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p6)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p7)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    with pytest.raises(DimensionalityError):

        # autopep8: off

        p8 = m // n  # E: Unsupported operand types for // ("Quantity[MassFlow]" and "Quantity[NormalVolumeFlow]")

        # autopep8: on


@pytest.mark.mypy_testing
def test_quantity_pow_types() -> None:

    # pow is only implemented in case the dimensionalities match

    m = Q[MassFlow](1, 'kg/s')
    n = Q[NormalVolumeFlow](1, 'Nm^3/h')
    d = Q[Dimensionless](1, '%')

    s_int = 5
    s_float = 0.1

    p1 = s_int**d
    p2 = d**s_int
    p3 = s_float**d
    p4 = d**s_float

    reveal_type(p1)  # R: builtins.float
    reveal_type(p2)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p3)  # R: builtins.float
    reveal_type(p4)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    p5 = m**s_int
    p6 = m**s_float
    p7 = m**d

    reveal_type(p5)  # R: encomp.units.Quantity[encomp.utypes.Unknown]
    reveal_type(p6)  # R: encomp.units.Quantity[encomp.utypes.Unknown]
    reveal_type(p7)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    with pytest.raises(DimensionalityError):

        # autopep8: off

        # NOTE: the mypy test plugin cannot handle multiple errors on a single line

        m**n  # E: Unsupported operand types for ** ("Quantity[MassFlow]" and "Quantity[NormalVolumeFlow]")
        d**n  # E: Unsupported operand types for ** ("Quantity[Dimensionless]" and "Quantity[NormalVolumeFlow]")

        # autopep8: on

    unknown = 1 / m

    p8 = unknown**2.5

    reveal_type(p8)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


@pytest.mark.mypy_testing
def test_quantity_add_types() -> None:

    m = Q[MassFlow](1, 'kg/s')
    n = Q[NormalVolumeFlow](1, 'Nm^3/h')
    d = Q[Dimensionless](1, '%')

    s_int = 5
    s_float = 0.1

    p1 = m + m
    p2 = n + n
    p3 = d + d

    reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(p2)  # R: encomp.units.Quantity[encomp.utypes.NormalVolumeFlow]
    reveal_type(p3)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    with pytest.raises(DimensionalityError):

        # autopep8: off

        p4 = m + s_int # E: Unsupported operand types for + ("Quantity[MassFlow]" and "int")
        p5 = m + s_float # E: Unsupported operand types for + ("Quantity[MassFlow]" and "float")

        p4_ = s_int + m # E: Unsupported operand types for + ("int" and "Quantity[MassFlow]")
        p5_ = s_float + m # E: Unsupported operand types for + ("float" and "Quantity[MassFlow]")

        # autopep8: on

    p6 = d + s_int
    p7 = d + s_float
    p8 = s_int + d
    p9 = s_float + d

    reveal_type(p6)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p7)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p8)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p9)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # unknown dimensionalities cannot be added
    # TODO: is it possible to have an error instead of dimensionality Impossible?

    unknown = 1 / m

    p10 = unknown + unknown
    reveal_type(p10)  # R: encomp.units.Quantity[encomp.utypes.Impossible]


@pytest.mark.mypy_testing
def test_quantity_sub_types() -> None:

    m = Q[MassFlow](1, 'kg/s')
    n = Q[NormalVolumeFlow](1, 'Nm^3/h')
    d = Q[Dimensionless](1, '%')

    s_int = 5
    s_float = 0.1

    p1 = m - m
    p2 = n - n
    p3 = d - d

    reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(p2)  # R: encomp.units.Quantity[encomp.utypes.NormalVolumeFlow]
    reveal_type(p3)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    with pytest.raises(DimensionalityError):

        # autopep8: off

        p4 = m - s_int # E: Unsupported operand types for - ("Quantity[MassFlow]" and "int")
        p5 = m - s_float # E: Unsupported operand types for - ("Quantity[MassFlow]" and "float")

        p4_ = s_int - m # E: Unsupported operand types for - ("int" and "Quantity[MassFlow]")
        p5_ = s_float - m # E: Unsupported operand types for - ("float" and "Quantity[MassFlow]")

        # autopep8: on

    p6 = d - s_int
    p7 = d - s_float
    p8 = s_int - d
    p9 = s_float - d

    reveal_type(p6)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p7)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p8)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(p9)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # unknown dimensionalities cannot be subtracted
    # TODO: is it possible to have an error instead of dimensionality Impossible?

    unknown = 1 / m

    p10 = unknown - unknown
    reveal_type(p10)  # R: encomp.units.Quantity[encomp.utypes.Impossible]


@pytest.mark.mypy_testing
def test_quantity_comparison_types() -> None:

    m = Q[MassFlow](1, 'kg/s')
    n = Q[NormalVolumeFlow](1, 'Nm^3/h')
    d = Q[Dimensionless](1, '%')

    s_int = 5
    s_float = 0.1

    p1 = m > m
    p2 = m < m
    p3 = m >= m
    p4 = m <= m
    p5 = m == m

    reveal_type(p1)  # R: builtins.bool
    reveal_type(p2)  # R: builtins.bool
    reveal_type(p3)  # R: builtins.bool
    reveal_type(p4)  # R: builtins.bool
    reveal_type(p5)  # R: builtins.bool

    with pytest.raises(DimensionalityError):

        # autopep8: off

        p6 = m > n # E: Unsupported operand types for > ("Quantity[MassFlow]" and "Quantity[NormalVolumeFlow]")
        p7 = m > d # E: Unsupported operand types for > ("Quantity[MassFlow]" and "Quantity[Dimensionless]")
        p8 = m > s_int # E: Unsupported operand types for > ("Quantity[MassFlow]" and "int")
        p9 = m > s_float # E: Unsupported operand types for > ("Quantity[MassFlow]" and "float")
        p10 = s_int <= m  # E: Unsupported operand types for <= ("int" and "Quantity[MassFlow]")

        # autopep8: on

    # == is implemented for all types
    p11 = s_int == m
    p12 = m == s_float

    # dimensionless can be compared with scalars

    p13 = d < s_int
    p14 = d > s_float
    p15 = s_int <= d
    p16 = s_float >= d

    reveal_type(p13)  # R: builtins.bool
    reveal_type(p14)  # R: builtins.bool
    reveal_type(p15)  # R: builtins.bool
    reveal_type(p16)  # R: builtins.bool
