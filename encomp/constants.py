"""Physical constants and reference conditions used by encomp helpers."""

from dataclasses import dataclass, field
from typing import Any

from .units import Quantity as Q
from .utypes import Pressure, Temperature

__all__ = ["CONSTANTS", "Constants"]


@dataclass(frozen=True, slots=True)
class Constants:
    """Namespace of constants exposed through the :data:`CONSTANTS` singleton."""

    R: Q[Any, float] = field(default_factory=lambda: Q(8.31446261815324, "kg*m²/K/mol/s²"))
    """Molar gas constant, exact by the 2019 SI definition."""

    SIGMA: Q[Any, float] = field(default_factory=lambda: Q(5.670374419e-8, "W/m**2/K**4"))
    """Stefan-Boltzmann constant."""

    normal_conditions_pressure: Q[Pressure, float] = field(default_factory=lambda: Q(1, "atm"))
    """Normal-condition pressure, 1 atm."""

    normal_conditions_temperature: Q[Temperature, float] = field(default_factory=lambda: Q(0, "degC").to("K"))
    """Normal-condition temperature, 0 degC."""

    standard_conditions_pressure: Q[Pressure, float] = field(default_factory=lambda: Q(1, "atm"))
    """Standard-condition pressure, 1 atm."""

    standard_conditions_temperature: Q[Temperature, float] = field(default_factory=lambda: Q(15, "degC").to("K"))
    """Standard-condition temperature, 15 degC."""


CONSTANTS = Constants()
