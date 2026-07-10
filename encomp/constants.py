"""Physical constants and reference conditions used by encomp helpers."""

from typing import Any

from .units import Quantity
from .utypes import Pressure, Temperature

__all__ = ["CONSTANTS", "Constants"]


class Constants:
    """Namespace of constants exposed through the :data:`CONSTANTS` singleton.

    Every attribute returns a fresh :class:`~encomp.units.Quantity`. A ``Quantity`` is
    mutable -- :meth:`~encomp.units.Quantity.ito` converts in place and the ``m`` setter
    replaces the magnitude -- so handing out one shared instance would let a caller alter
    the constant for the whole process.
    """

    @property
    def R(self) -> Quantity[Any, float]:
        """Molar gas constant, exact by the 2019 SI definition."""

        return Quantity(8.31446261815324, "kg*m²/K/mol/s²")

    @property
    def SIGMA(self) -> Quantity[Any, float]:
        """Stefan-Boltzmann constant."""

        return Quantity(5.670374419e-8, "W/m**2/K**4")

    @property
    def normal_conditions_pressure(self) -> Quantity[Pressure, float]:
        """Normal-condition pressure, 1 atm."""

        return Quantity(1, "atm")

    @property
    def normal_conditions_temperature(self) -> Quantity[Temperature, float]:
        """Normal-condition temperature, 0 degC."""

        return Quantity(0, "degC").to("K")

    @property
    def standard_conditions_pressure(self) -> Quantity[Pressure, float]:
        """Standard-condition pressure, 1 atm."""

        return Quantity(1, "atm")

    @property
    def standard_conditions_temperature(self) -> Quantity[Temperature, float]:
        """Standard-condition temperature, 15 degC."""

        return Quantity(15, "degC").to("K")


CONSTANTS = Constants()
"""Singleton :class:`Constants` instance; every attribute access returns a fresh
:class:`~encomp.units.Quantity`."""
