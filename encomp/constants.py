from .units import Quantity as Q


# A namespace of physical constants, accessed via the CONSTANTS singleton (CONSTANTS.R, ...).
# NOT a dataclass: the members have no field annotations, so @dataclass would generate an
# empty __init__ (Constants(R=...) would fail) and add nothing -- they are plain class-level
# constants by design.
class Constants:
    R = Q(8.3144598, "kg*m²/K/mol/s²")
    SIGMA = Q(5.670374419e-8, "W/m**2/K**4")

    normal_conditions_pressure = Q(1, "atm")
    normal_conditions_temperature = Q(0, "°C").to("K")

    standard_conditions_pressure = Q(1, "atm")
    standard_conditions_temperature = Q(15, "degC").to("K")


CONSTANTS = Constants()
