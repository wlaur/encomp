"""
Classes that represent and model the behavior of different types of heat exchangers.
"""

from pydantic.dataclasses import dataclass


@dataclass
class HeatExchanger:
    pass


class ShellAndTubeHeatExchanger(HeatExchanger):
    pass


class PlateHeatExchanger(HeatExchanger):
    pass


class Condenser(HeatExchanger):
    pass
