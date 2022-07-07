import pytest

from encomp.units import convert_volume_mass
from encomp.units import Quantity as Q
from encomp.utypes import (Dimensionless,
                           Time,
                           Length,
                           MassFlow,
                           Mass,
                           VolumeFlow,
                           Volume,
                           Energy,
                           NormalVolumeFlow,
                           Power,
                           Temperature)


# it's important that the expected mypy output is a comment on the
# same line as the expression, disable autopep8 if necessary with
# autopep8: off
# ... some code above the line length limit
# autopep8: on


@pytest.mark.mypy_testing
def test_quantity_to_types() -> None:

    q = Q[Temperature](25, 'degC')
    s = Q[Power](1, 'kW/(cm/m)')

    q_K = q.to('K')

    reveal_type(q_K)  # R: encomp.units.Quantity[encomp.utypes.Temperature]

    q_base = q.to_base_units()

    reveal_type(q_base)  # R: encomp.units.Quantity[encomp.utypes.Temperature]

    s_red = s.to_reduced_units()

    reveal_type(s_red)  # R: encomp.units.Quantity[encomp.utypes.Power]

    s_base = s.to_base_units()

    reveal_type(s_base)  # R: encomp.units.Quantity[encomp.utypes.Power]

    # to_reduced_units does not do the same thing as to_base_units
    assert s_base.m != s_red.m


@pytest.mark.mypy_testing
def test_quantity_combined_types() -> None:

    m = Q[MassFlow](25, 'kg/s')
    d = Q[Dimensionless](25, '%')

    p1 = ((m / m) * d)**2 / (1 - d / 2)
    reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]


@pytest.mark.mypy_testing
def test_quantity_custom_mul_div_types() -> None:

    s = Q[Length](1, 'm')

    # autopep8: off

    reveal_type(s * s)  # R: encomp.units.Quantity[encomp.utypes.Area]

    reveal_type(s * s * s)  # R: encomp.units.Quantity[encomp.utypes.Volume]
    reveal_type((s * s) * s)  # R: encomp.units.Quantity[encomp.utypes.Volume]
    reveal_type(s * (s * s))  # R: encomp.units.Quantity[encomp.utypes.Volume]


    reveal_type(s * (s * s) / s) # R: encomp.units.Quantity[encomp.utypes.Area]
    reveal_type(s * (s * s) / s / s) # R: encomp.units.Quantity[encomp.utypes.Length]
    reveal_type(s * s * s / s / s / s) # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    mf = Q[MassFlow](1, 'kg/s')
    vf = Q[VolumeFlow](1, 'liter/s')
    nvf = Q[NormalVolumeFlow](1, 'Nm3/s')
    v = Q[Volume](1, 'liter')
    m = Q[Mass](25, 'kg')

    t = Q[Time](1, 's')

    reveal_type(mf * t)  # R: encomp.units.Quantity[encomp.utypes.Mass]
    reveal_type(t * vf)  # R: encomp.units.Quantity[encomp.utypes.Volume]
    reveal_type(nvf * t)  # R: encomp.units.Quantity[encomp.utypes.NormalVolume]


    reveal_type((mf * t) / t)  # R: encomp.units.Quantity[encomp.utypes.MassFlow]
    reveal_type((t * vf) / t)  # R: encomp.units.Quantity[encomp.utypes.VolumeFlow]
    reveal_type((nvf * t) / t)  # R: encomp.units.Quantity[encomp.utypes.NormalVolumeFlow]


    reveal_type(m / v)  # R: encomp.units.Quantity[encomp.utypes.Density]
    reveal_type(v / m)  # R: encomp.units.Quantity[encomp.utypes.SpecificVolume]

    # autopep8: on

    e = Q[Energy](25, 'MJ')

    reveal_type(e / t)  # R: encomp.units.Quantity[encomp.utypes.Power]
    reveal_type(t * (e / t))  # R: encomp.units.Quantity[encomp.utypes.Energy]
    reveal_type(e / t * t)  # R: encomp.units.Quantity[encomp.utypes.Energy]

    u = e * t
    reveal_type(u / t)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


@pytest.mark.mypy_testing
def test_quantity_custom_pow_types() -> None:

    s = Q[Length](1, 'm')

    # NOTE: this only works with int 1, 2, 3, it's not possible to
    # represent literal floats like "Literal[1.0]"
    reveal_type(s**1)  # R: encomp.units.Quantity[encomp.utypes.Length]
    reveal_type(s**2)  # R: encomp.units.Quantity[encomp.utypes.Area]
    reveal_type(s**3)  # R: encomp.units.Quantity[encomp.utypes.Volume]

    # this works with pylance but not mypy for some reason...
    exp = 2
    reveal_type(s**exp)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # this does not work, only int literals can be detected by mypy
    exp_ = int(3.0)
    reveal_type(s**exp_)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


@pytest.mark.mypy_testing
def test_convert_mass_flow_types() -> None:

    mf = Q[MassFlow](25, 'kg/s')

    vf = convert_volume_mass(mf)

    reveal_type(vf)  # R: encomp.units.Quantity[encomp.utypes.VolumeFlow]

    v = Q(25, 'liter')

    m = convert_volume_mass(v, Q(25, 'g/liter'))

    reveal_type(m)  # R: encomp.units.Quantity[encomp.utypes.Mass]

    p1 = Q(25, 'liter/week')  # E: Need type annotation for "p1"

    unknown_output = convert_volume_mass(p1)

    # mypy does not identify this as Union[...], but pylance does
    reveal_type(unknown_output)  # R: encomp.units.Quantity[Any]
