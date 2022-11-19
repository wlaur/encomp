from typing import TYPE_CHECKING

import pytest

from ..units import Quantity as Q
from ..conversion import convert_volume_mass
from ..utypes import (Dimensionless,
                      Time,
                      Length,
                      MassFlow,
                      Mass,
                      VolumeFlow,
                      Volume,
                      Energy,
                      NormalVolumeFlow,
                      ThermalConductivity,
                      Power,
                      Temperature)

if not TYPE_CHECKING:
    def reveal_type(x): return x


@pytest.mark.mypy_testing
def test_quantity_to_types() -> None:
    return

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
    return

    m = Q[MassFlow](25, 'kg/s')
    d = Q[Dimensionless](25, '%')

    p1 = ((m / m) * d)**2 / (1 - d / 2)
    reveal_type(p1)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]


@pytest.mark.mypy_testing
def test_quantity_custom_mul_div_types() -> None:
    return

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

    # Energy * Time is an Unknown dimensionality
    u = e * t
    reveal_type(u)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # Unknown divided by anything is also Unknown
    reveal_type(u / t)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


@pytest.mark.mypy_testing
def test_quantity_custom_pow_types() -> None:
    return

    s = Q[Length](1, 'm')

    # NOTE: this only works with int 1, 2, 3, it's not possible to
    # represent literal floats like "Literal[1.0]"
    reveal_type(s**1)  # R: encomp.units.Quantity[encomp.utypes.Length]
    reveal_type(s**2)  # R: encomp.units.Quantity[encomp.utypes.Area]
    reveal_type(s**3)  # R: encomp.units.Quantity[encomp.utypes.Volume]

    # this works with pyright but not mypy for some reason...
    exp = 2
    reveal_type(s**exp)  # R: encomp.units.Quantity[encomp.utypes.Unknown]

    # this does not work, only int literals can be detected by mypy
    exp_ = int(3.0)
    reveal_type(s**exp_)  # R: encomp.units.Quantity[encomp.utypes.Unknown]


@pytest.mark.mypy_testing
def test_convert_mass_flow_types() -> None:
    return

    mf = Q[MassFlow](25, 'kg/s')

    vf = convert_volume_mass(mf)

    reveal_type(vf)  # R: encomp.units.Quantity[encomp.utypes.VolumeFlow]

    v = Q(25, 'liter')

    m = convert_volume_mass(v, Q(25, 'g/liter'))

    reveal_type(m)  # R: encomp.units.Quantity[encomp.utypes.Mass]

    p1 = Q(25, 'liter/week')  # E: Need type annotation for "p1"

    unknown_output = convert_volume_mass(p1)

    # mypy does not identify this as Union[...], but pyright does
    reveal_type(unknown_output)  # R: encomp.units.Quantity[Any]


@pytest.mark.mypy_testing
def test_various_mul_div_types() -> None:
    return

    # these signatures are autogenerated

    # autopep8: off

    reveal_type(Q(25, 'cSt') * Q(25, 'm'))  # R: encomp.units.Quantity[encomp.utypes.VolumeFlow]

    reveal_type(Q[ThermalConductivity](25, 'W/m/K') * Q(25, 'm'))  # R: encomp.units.Quantity[encomp.utypes.PowerPerTemperature]

    reveal_type(Q(25, 'MWh') / Q(25, 'kg'))  # R: encomp.units.Quantity[encomp.utypes.EnergyPerMass]
    reveal_type((Q(2, 'd') * Q(25, 'kW')) / (Q(25, 'kg/s') * Q(2, 'w')))  # R: encomp.units.Quantity[encomp.utypes.EnergyPerMass]

    # autopep8: on
