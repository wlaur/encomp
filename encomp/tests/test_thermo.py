from ..misc import isinstance_types
from ..thermo import heat_balance
from ..units import Quantity as Q
from ..utypes import Energy, MassFlow, Power, TemperatureDifference


def test_heat_balance() -> None:
    assert isinstance_types(heat_balance(Q(2, "kg/s"), Q(2, "kJ/s").asdim(Power)), Q[TemperatureDifference])
    assert isinstance_types(heat_balance(Q(2, "K"), Q(2, "kJ/s").asdim(Power)), Q[MassFlow])
    assert isinstance_types(heat_balance(Q(2, "kg"), Q(2, "delta_degF")), Q[Energy])
    assert isinstance_types(heat_balance(Q(2, "kg/s"), Q(2, "delta_degF")), Q[Power])
