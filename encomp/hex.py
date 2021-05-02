"""
Classes that represent and model the behavior of different types of heat exchangers.
"""
from typing import Optional
from encomp.units import Quantity
from encomp.utypes import Area, Temperature
from encomp.thermo import HeatTransferCoefficient


class HeatExchanger:

    def __init__(self,
                 U: Quantity[HeatTransferCoefficient],
                 A: Quantity[Area],
                 T_sec: Quantity[Temperature],
                 name: Optional[str] = None):
        """
        Class that represents a basic heat exchanger. Uses a constant value for
        the *overall heat transfer coefficient* :math:`U` and the *heat transfer
        area* :math:`A`. Flow, temperature difference and pressure drop on the
        *primary* side is considered, and the secondary side is not modeled at all.

        Heat transfer :math:`\\dot{Q}` (energy per time, W) is calculated with

        .. math::
            \\dot{Q} = U A \\Delta T_{m}


        Parameters
        ----------
        U : Quantity[HeatTransferCoefficient]
            **Overall** heat transfer coefficient (constant)
        A : Quantity[Area]
            Effective heat transfer area (constant)
        T_sec : Quantity[Temperature]
            Constant average temperature of the secondary fluid.
            The average temperature difference is
            :math:`\\Delta T_{m} = T_{\\mathrm{primary}} - T_{\\mathrm{secondary}}`.
        name : str
            Name of this heat exchanger
        """

        self.U = U.to('W/m²/delta_degC')
        self.A = A.to('m²')
        self.T_sec = T_sec.to('degC')
        self.name = name


class ShellAndTubeHeatExchanger(HeatExchanger):
    pass


class PlateHeatExchanger(HeatExchanger):
    pass


class Condenser(HeatExchanger):
    pass
