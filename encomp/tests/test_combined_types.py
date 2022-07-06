import pytest

from encomp.units import convert_volume_mass
from encomp.units import Quantity as Q
from encomp.utypes import (Dimensionless,
                           Time,
                           MassFlow,
                           Volume,
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
    t = Q[Time](25, 's')
    d = Q[Dimensionless](25, '%')

    p1 = ((m / m) * d)**2 / (1 - d / 2)
    reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    # p2 = m * t

    # reveal_type(p2)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]



@pytest.mark.mypy_testing
def test_convert_mass_flow_types() -> None:

    mf = Q[MassFlow](25, 'kg/s')

    vf = convert_volume_mass(mf)

    reveal_type(vf)  # R: encomp.units.Quantity[encomp.utypes.VolumeFlow]

    v = Q[Volume](25, 'liter')

    m = convert_volume_mass(v, Q(25, 'g/liter'))

    reveal_type(m)  # R: encomp.units.Quantity[encomp.utypes.Mass]

    vf = Q(25, 'liter/day')  # E: Need type annotation for "vf"

    # TODO: output is incorrect, should be Union[...]
    unknown_output = convert_volume_mass(vf)
