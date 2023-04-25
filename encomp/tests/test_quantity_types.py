import copy
from typing import TYPE_CHECKING

import pytest

from ..units import (
    DimensionalityError,
    DimensionalityTypeError,
    ExpectedDimensionalityError,
)
from ..units import (
    Quantity as Q,
)
from ..utypes import (
    Dimensionless,
    Length,
    LowerHeatingValue,
    Mass,
    MassFlow,
    NormalVolumeFlow,
    Temperature,
    Time,
)

if not TYPE_CHECKING:

    def reveal_type(x):
        return x


class Distance(Length):
    pass


@pytest.mark.mypy_testing
def test_quantity_reveal_type() -> None:
    return

    m = Q[MassFlow](1, "kg/s")

    reveal_type(m)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    T = Q[Temperature](1, "degC")

    reveal_type(T)  # R: encomp.units.Quantity[encomp.utypes.Temperature]

    # if the dimensionality is omitted, the inferred type is Quantity[Unknown]
    # the type will be correct at runtime
    unknown = Q(25, "bar/week")  # E: Need type annotation for "unknown"

    # TODO: mypy infers the dimensionality type as Any, but pyright
    # correctly identifies it as Unknown
    reveal_type(unknown)  # R: encomp.units.Quantity[Any]

    ms = Q([1, 2, 3], "kg/s")

    reveal_type(ms)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(ms[0])  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    ms_ = Q(ms)
    reveal_type(ms_)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]


@pytest.mark.mypy_testing
def test_quantity_asdim() -> None:
    return

    # fmt: off

    q = Q(15, "kJ/kg")

    # the string literal overload must be overridden when using Q[Dim]
    p = Q[LowerHeatingValue](15, str("kJ/kg"))
    s = Q(15, "kJ/kg").asdim(LowerHeatingValue)

    reveal_type(q)  # R: encomp.units.Quantity[encomp.utypes.EnergyPerMass]
    reveal_type(p)  # R: encomp.units.Quantity[encomp.utypes.LowerHeatingValue]
    reveal_type(s)  # R: encomp.units.Quantity[encomp.utypes.LowerHeatingValue]

    reveal_type(
        q.asdim(LowerHeatingValue)
    )  # R: encomp.units.Quantity[encomp.utypes.LowerHeatingValue]
    reveal_type(q.asdim(p))  # R: encomp.units.Quantity[encomp.utypes.LowerHeatingValue]

    # fmt: on


@pytest.mark.mypy_testing
def test_quantity_reveal_type_copy() -> None:
    return

    # fmt: off

    q = Q(25, "m")

    reveal_type(q)  # R: encomp.units.Quantity[encomp.utypes.Length]
    reveal_type(q.__copy__())  # R: encomp.units.Quantity[encomp.utypes.Length]
    reveal_type(q.__deepcopy__())  # R: encomp.units.Quantity[encomp.utypes.Length]

    reveal_type(copy.copy(q))  # R: encomp.units.Quantity[encomp.utypes.Length]
    reveal_type(copy.deepcopy(q))  # R: encomp.units.Quantity[encomp.utypes.Length]

    # fmt: on


@pytest.mark.mypy_testing
def test_quantity_reveal_custom_type() -> None:
    return

    # Distance is a custom dimensionality that is defined in this module
    # in case the subclass is defined inside this function,
    # mypy will identify the type as
    # encomp.units.Quantity[encomp.tests.test_quantity_types.Distance@XX]
    # where XX is the line number of the class definition inside the function
    d = Q[Distance](1, str("m"))

    # fmt: off

    reveal_type(
        d
    )  # R: encomp.units.Quantity[encomp.tests.test_quantity_types.Distance]
    reveal_type(
        d.to("km")
    )  # R: encomp.units.Quantity[encomp.tests.test_quantity_types.Distance]
    reveal_type(
        d.to_reduced_units()
    )  # R: encomp.units.Quantity[encomp.tests.test_quantity_types.Distance]
    reveal_type(
        d.to_base_units()
    )  # R: encomp.units.Quantity[encomp.tests.test_quantity_types.Distance]

    reveal_type(
        d * 2
    )  # R: encomp.units.Quantity[encomp.tests.test_quantity_types.Distance]
    reveal_type(
        2 * d
    )  # R: encomp.units.Quantity[encomp.tests.test_quantity_types.Distance]
    reveal_type(
        d + d
    )  # R: encomp.units.Quantity[encomp.tests.test_quantity_types.Distance]
    reveal_type(
        d - d
    )  # R: encomp.units.Quantity[encomp.tests.test_quantity_types.Distance]
    reveal_type(
        (d - d / 2) * 2.5
    )  # R: encomp.units.Quantity[encomp.tests.test_quantity_types.Distance]

    # fmt: off


@pytest.mark.mypy_testing
def test_quantity_construction() -> None:
    return

    # fmt: off

    reveal_type(Q(25))  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(Q(25, None))  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    reveal_type(Q(25, ""))  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(Q(25, "-"))  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(
        Q(25, "dimensionless")
    )  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    reveal_type(
        Q[Mass](25, Q[Mass](25, "kg"))
    )  # R: encomp.units.Quantity[encomp.utypes.Mass]
    reveal_type(
        Q[Mass](25, Q[Mass](25, "kg").u)
    )  # R: encomp.units.Quantity[encomp.utypes.Mass]

    with pytest.raises(ExpectedDimensionalityError):
        Q[Mass](
            25, Q[Time](25, "sec")
        )  # E: Argument 2 to "Quantity" has incompatible type "Quantity[Time]"; expected "Union[Unit, UnitsContainer, str, Quantity[Mass], None]"  # noqa: E501

    mf_ = Q(25, "kg/s")
    mf = Q[MassFlow](25, "kg/week")
    e = Q(25, "kJ")

    reveal_type(mf_)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(mf)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(e)  # R: encomp.units.Quantity[encomp.utypes.Energy]

    reveal_type(Q(25))  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(Q(Q(25)))  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(
        Q(Q(Q(Q(Q(Q(25))))))
    )  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    reveal_type(
        Q(Q(Q(Q(Q(Q(25, "%"))))))
    )  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    reveal_type(Q("25 m"))  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # cannot analyze string input when type checking
    reveal_type(Q("25"))  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # fmt: on


@pytest.mark.mypy_testing
def test_quantity_mul_types() -> None:
    return

    # m = Q[MassFlow](1, "kg/s")
    # n = Q[NormalVolumeFlow](1, "Nm^3/h")

    # d = Q[Dimensionless](1, "%")

    # s_int = 5
    # s_float = 0.1

    # # multiplying 2 Quantities creates a new dimensionality
    # # that is only known at runtime
    # # the type checker will assign the dimensionality Unknown
    # # p1 = m * m
    # # p2 = m * n
    # # p3 = n * m
    # # p4 = n * n

    # # scalar and Q[Dimensionless] multiplication preserves dimensionality

    # k1 = m * s_int
    # k2 = s_int * m

    # reveal_type(k1)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    # reveal_type(k2)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    # k3 = m * s_float
    # k4 = s_float * m

    # reveal_type(k3)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    # reveal_type(k4)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    # k5 = m * d
    # k6 = d * m

    # reveal_type(k5)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    # reveal_type(k6)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    # class _CustomDimensionless(Dimensionless):
    #     pass

    # d_custom = Q[_CustomDimensionless](0.5)

    # k7 = m * d_custom
    # k8 = d_custom * m

    # reveal_type(k7)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    # reveal_type(k8)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]

    # # alternative way of specifying the dimensionality
    # # NOTE: the _dt parameter is ignored at runtime, the subclass will not be validated
    # d_custom_ = Q(0.6, _dt=_CustomDimensionless)
    # k9 = m * d_custom_
    # k10 = d_custom * m

    # # TODO: this does not work (Unknown instead of MassFlow)
    # reveal_type(k9)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # # ... but this does work?
    # reveal_type(k10)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]


@pytest.mark.mypy_testing
def test_quantity_misc_types() -> None:
    return

    # fmt: off

    # don't mix the __class_getitem__ and _dt parameter
    Q[Temperature](
        25, "degC", _dt=Mass
    )  # E: Argument "_dt" to "Quantity" has incompatible type "Type[Mass]"; expected "Type[Temperature]"

    with pytest.raises(ExpectedDimensionalityError):
        _cls = Q[Temperature]

        # this raises a runtime error, since bar is not a Temperature unit
        a = _cls(25, "bar")

        # the literal "bar" is correctly identified as a Pressure unit
        reveal_type(a)  # R: encomp.units.Quantity[encomp.utypes.Pressure]

    with pytest.raises(ExpectedDimensionalityError):
        # this cannot be detected, however the variable b has type Q[Temperature]
        # this will raise an error at runtime
        b = Q[Mass](25, "C")

        reveal_type(b)  # R: encomp.units.Quantity[encomp.utypes.Temperature]

    # fmt: on


@pytest.mark.mypy_testing
def test_quantity_div_types() -> None:
    return

    # fmt: off

    m = Q[MassFlow](1, "kg/s")
    n = Q[NormalVolumeFlow](1, "Nm^3/h")

    d = Q[Dimensionless](1, "%")

    s_int = 5
    s_float = 0.1

    # dividing 2 different Quantities creates a new dimensionality
    p1 = m / n

    # this dimensionality is not defined as an overload
    p2 = n / m

    reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.MassPerNormalVolume]
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

    reveal_type(unknown / unknown)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    reveal_type(1.2 / Q(25, "s"))  # R: encomp.units.Quantity[encomp.utypes.Frequency]
    reveal_type(
        2 / Q(25, "kg/liter")
    )  # R: encomp.units.Quantity[encomp.utypes.SpecificVolume]
    reveal_type(
        0.2 / Q(25, "MWh/kg")
    )  # R: encomp.units.Quantity[encomp.utypes.MassPerEnergy]
    reveal_type(
        1 / (0.2 / Q(25, "MWh/kg"))
    )  # R: encomp.units.Quantity[encomp.utypes.EnergyPerMass]

    reveal_type(Q(1, "MJ/kg"))  # R: encomp.units.Quantity[encomp.utypes.EnergyPerMass]
    reveal_type(
        1 / Q(1, "MJ/kg")
    )  # R: encomp.units.Quantity[encomp.utypes.MassPerEnergy]
    reveal_type(
        1 / (1 / Q(1, "MJ/kg"))
    )  # R: encomp.units.Quantity[encomp.utypes.EnergyPerMass]

    reveal_type(1 / Q(25, "kW"))  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # fmt: on


@pytest.mark.mypy_testing
def test_quantity_floordiv_types() -> None:
    return

    # floordiv is only implemented in case the dimensionalities match

    m = Q[MassFlow](1, "kg/s")
    n = Q[NormalVolumeFlow](1, "Nm^3/h")
    d = Q[Dimensionless](1, "%")

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
        # fmt: off
        pass

        # p8 = (
        #     m // n
        # )  # E: Unsupported operand types for // ("Quantity[MassFlow]" and "Quantity[NormalVolumeFlow]")

        # fmt: on


@pytest.mark.mypy_testing
def test_quantity_pow_types() -> None:
    return

    # pow is only implemented in case the dimensionalities match

    m = Q[MassFlow](1, "kg/s")
    n = Q[NormalVolumeFlow](1, "Nm^3/h")
    d = Q[Dimensionless](1, "%")

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
        # fmt: off

        # NOTE: the mypy test plugin cannot handle multiple errors on a single line

        m**n  # E: Unsupported operand types for ** ("Quantity[MassFlow]" and "Quantity[NormalVolumeFlow]")

    with pytest.raises(DimensionalityError):
        d**n  # E: Unsupported operand types for ** ("Quantity[Dimensionless]" and "Quantity[NormalVolumeFlow]")

        # fmt: on

    unknown = 1 / m

    p8 = unknown**2.5

    reveal_type(p8)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


@pytest.mark.mypy_testing
def test_quantity_add_types() -> None:
    return

    # m = Q[MassFlow](1, "kg/s")
    # n = Q[NormalVolumeFlow](1, "Nm^3/h")
    # d = Q[Dimensionless](1, "%")

    # s_int = 5
    # s_float = 0.1

    # p1 = m + m
    # p2 = n + n
    # p3 = d + d

    # reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    # reveal_type(p2)  # R: encomp.units.Quantity[encomp.utypes.NormalVolumeFlow]
    # reveal_type(p3)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # with pytest.raises(DimensionalityError):
    #     # fmt: off

    #     p4 = (
    #         m + s_int
    #     )  # E: Unsupported operand types for + ("Quantity[MassFlow]" and "int")

    # with pytest.raises(DimensionalityError):
    #     p5 = (
    #         m + s_float
    #     )  # E: Unsupported operand types for + ("Quantity[MassFlow]" and "float")

    # with pytest.raises(DimensionalityError):
    #     p4_ = (
    #         s_int + m
    #     )  # E: Unsupported operand types for + ("int" and "Quantity[MassFlow]")

    # with pytest.raises(DimensionalityError):
    #     p5_ = (
    #         s_float + m
    #     )  # E: Unsupported operand types for + ("float" and "Quantity[MassFlow]")

    #     # fmt: on

    # p6 = d + s_int
    # p7 = d + s_float
    # p8 = s_int + d
    # p9 = s_float + d

    # reveal_type(p6)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    # reveal_type(p7)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    # reveal_type(p8)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    # reveal_type(p9)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # unknown = 1 / m

    # p10 = unknown + unknown
    # reveal_type(p10)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


@pytest.mark.mypy_testing
def test_quantity_sub_types() -> None:
    return

    # m = Q[MassFlow](1, "kg/s")
    # n = Q[NormalVolumeFlow](1, "Nm^3/h")
    # d = Q[Dimensionless](1, "%")

    # s_int = 5
    # s_float = 0.1

    # p1 = m - m
    # p2 = n - n
    # p3 = d - d

    # reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    # reveal_type(p2)  # R: encomp.units.Quantity[encomp.utypes.NormalVolumeFlow]
    # reveal_type(p3)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # with pytest.raises(DimensionalityError):
    #     # fmt: off

    #     p4 = (
    #         m - s_int
    #     )  # E: Unsupported operand types for - ("Quantity[MassFlow]" and "int")

    # with pytest.raises(DimensionalityError):
    #     p5 = (
    #         m - s_float
    #     )  # E: Unsupported operand types for - ("Quantity[MassFlow]" and "float")

    # with pytest.raises(DimensionalityError):
    #     p4_ = (
    #         s_int - m
    #     )  # E: Unsupported operand types for - ("int" and "Quantity[MassFlow]")

    # with pytest.raises(DimensionalityError):
    #     p5_ = (
    #         s_float - m
    #     )  # E: Unsupported operand types for - ("float" and "Quantity[MassFlow]")

    #     # fmt: on

    # p6 = d - s_int
    # p7 = d - s_float
    # p8 = s_int - d
    # p9 = s_float - d

    # reveal_type(p6)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    # reveal_type(p7)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    # reveal_type(p8)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]
    # reveal_type(p9)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # unknown = 1 / m

    # p10 = unknown - unknown
    # reveal_type(p10)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


@pytest.mark.mypy_testing
def test_quantity_comparison_types() -> None:
    return

    # m = Q[MassFlow](1, "kg/s")
    # n = Q[NormalVolumeFlow](1, "Nm^3/h")
    # d = Q[Dimensionless](1, "%")

    # s_int = 5
    # s_float = 0.1

    # p1 = m > m
    # p2 = m < m
    # p3 = m >= m
    # p4 = m <= m
    # p5 = m == m

    # reveal_type(p1)  # R: builtins.bool
    # reveal_type(p2)  # R: builtins.bool
    # reveal_type(p3)  # R: builtins.bool
    # reveal_type(p4)  # R: builtins.bool
    # reveal_type(p5)  # R: builtins.bool

    # with pytest.raises(DimensionalityError):
    #     # fmt: off

    #     p6 = (
    #         m > n
    #     )  # E: Unsupported operand types for > ("Quantity[MassFlow]" and "Quantity[NormalVolumeFlow]")

    # with pytest.raises(DimensionalityError):
    #     p7 = (
    #         m > d
    #     )  # E: Unsupported operand types for > ("Quantity[MassFlow]" and "Quantity[Dimensionless]")

    # with pytest.raises(DimensionalityError):
    #     p8 = (
    #         m > s_int
    #     )  # E: Unsupported operand types for > ("Quantity[MassFlow]" and "int")

    # with pytest.raises(DimensionalityError):
    #     p9 = (
    #         m > s_float
    #     )  # E: Unsupported operand types for > ("Quantity[MassFlow]" and "float")

    # with pytest.raises(DimensionalityError):
    #     p10 = (
    #         s_int <= m
    #     )  # E: Unsupported operand types for <= ("int" and "Quantity[MassFlow]")

    #     # fmt: on

    # # == is implemented for all types
    # p11 = s_int == m
    # p12 = m == s_float

    # # dimensionless can be compared with scalars

    # p13 = d < s_int
    # p14 = d > s_float
    # p15 = s_int <= d
    # p16 = s_float >= d

    # reveal_type(p13)  # R: builtins.bool
    # reveal_type(p14)  # R: builtins.bool
    # reveal_type(p15)  # R: builtins.bool
    # reveal_type(p16)  # R: builtins.bool


@pytest.mark.mypy_testing
def test_quantity_currency_types() -> None:
    return

    # fmt: off

    p1 = Q(1, "SEK")
    p2 = Q(1, "kEUR")
    p3 = Q(1, "MUSD")

    reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.Currency]
    reveal_type(p2)  # R: encomp.units.Quantity[encomp.utypes.Currency]
    reveal_type(p3)  # R: encomp.units.Quantity[encomp.utypes.Currency]

    # NOTE: exchange rates are not implemented here, the defintions use
    # approximate values (10 SEK = 1 EUR = 1 USD)

    s = p1 + p2 - p3

    reveal_type(s)  # R: encomp.units.Quantity[encomp.utypes.Currency]

    # prefixes k and M can be used (other SI prefixes work but make no sense)
    reveal_type(p1.to("MSEK"))  # R: encomp.units.Quantity[encomp.utypes.Currency]

    p4 = Q(25, "SEK/MWh")
    reveal_type(p4)  # R: encomp.units.Quantity[encomp.utypes.CurrencyPerEnergy]

    p5 = Q(25, "EUR/t")
    reveal_type(p5)  # R: encomp.units.Quantity[encomp.utypes.CurrencyPerMass]

    r = (p4 * Q(25, "kWh")).to("MSEK")

    reveal_type(r)  # R: encomp.units.Quantity[encomp.utypes.Currency]

    t = p1 / Q(256, "L")

    reveal_type(t)  # R: encomp.units.Quantity[encomp.utypes.CurrencyPerVolume]

    p6 = Q(25, "EUR/h")
    reveal_type(p6)  # R: encomp.units.Quantity[encomp.utypes.CurrencyPerTime]

    k = p6 * Q(25, "d")
    reveal_type(k)  # R: encomp.units.Quantity[encomp.utypes.Currency]

    m = Q(25, "MSEK") / Q(300, "d")
    reveal_type(m)  # R: encomp.units.Quantity[encomp.utypes.CurrencyPerTime]

    r = (Q(25, "kg/s") * Q(2, "week")) * Q(25, "EUR/ton")

    reveal_type(r.to("MEUR"))  # R: encomp.units.Quantity[encomp.utypes.Currency]

    weekly_cost = Q(145, "GWh/year") * Q(1, "week") * Q(25, "EUR/MWh")
    reveal_type(weekly_cost)  # R: encomp.units.Quantity[encomp.utypes.Currency]

    # fmt: off


@pytest.mark.mypy_testing
def test_quantity_misc_operators() -> None:
    return

    q = Q(25, "kg/s")

    # TODO: why are types not inferred in round() and abs()

    # __pos__, __neg__ operators
    reveal_type(-q)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type(+q)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]


@pytest.mark.mypy_testing
def test_quantity_temperature_types() -> None:
    return

    T1 = Q(15, "°C")
    T2 = Q(25, "°C")

    dT = Q(1, "delta_°C")

    # fmt: off

    reveal_type(
        T1 - T2
    )  # R: encomp.units.Quantity[encomp.utypes.TemperatureDifference]
    reveal_type(
        T2 - T1
    )  # R: encomp.units.Quantity[encomp.utypes.TemperatureDifference]
    reveal_type(
        (T2 - T1) / 2
    )  # R: encomp.units.Quantity[encomp.utypes.TemperatureDifference]

    reveal_type(
        (T2 - T1) / (T2 - T1)
    )  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    reveal_type(T1 + dT)  # R: encomp.units.Quantity[encomp.utypes.Temperature]
    reveal_type(T1 + (T2 - T1))  # R: encomp.units.Quantity[encomp.utypes.Temperature]

    with pytest.raises(DimensionalityTypeError):
        (
            dT + T1
        )  # E: Unsupported operand types for + ("Quantity[TemperatureDifference]" and "Quantity[Temperature]")

    with pytest.raises(DimensionalityTypeError):
        (
            T1 - T2
        ) - T2  # E: Unsupported operand types for - ("Quantity[TemperatureDifference]" and "Quantity[Temperature]")

    # fmt: on
