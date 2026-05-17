"""
Functions relating to thermodynamics.
"""

from typing import Any, cast

from .misc import isinstance_types
from .units import Quantity
from .utypes import (
    Energy,
    Mass,
    MassFlow,
    Power,
    SpecificHeatCapacity,
    Temperature,
    TemperatureDifference,
)

DEFAULT_CP = Quantity(4.18, "kJ/kg/K").asdim(SpecificHeatCapacity)

type HeatBalanceResult = (
    Quantity[Mass, Any]
    | Quantity[MassFlow, Any]
    | Quantity[Energy, Any]
    | Quantity[Power, Any]
    | Quantity[TemperatureDifference, Any]
)


def heat_balance(
    *args: HeatBalanceResult | Quantity[Temperature, Any],
    cp: Quantity[SpecificHeatCapacity, float] = DEFAULT_CP,
) -> HeatBalanceResult:
    """
    Solves the heat balance equation

    .. math::
        \\dot{Q}_h = C_p \\cdot \\dot{m} \\cdot \\Delta T

    for the 3:rd unknown variable.

    Parameters
    ----------
    args : Quantity
        The two known variables in the heat balance equation:
        mass, mass flow, energy, power or temperature difference.
        Temperature input is interpreted as temperature difference.
    cp : Quantity[SpecificHeatCapacity], optional
        Heat capacity, by default 4.18 kg/kJ/K (water)

    Returns
    -------
    Quantity
        The third unknown variable
    """

    # this function might be too general to be
    # expressed succinctly using type annotations

    if len(args) != 2:
        raise ValueError("Must pass exactly two parameters out of dT, Q_h and m")

    params = {
        "m": (Quantity[Mass] | Quantity[MassFlow], ("kg", "kg/s")),
        "dT": (
            Quantity[TemperatureDifference] | Quantity[Temperature],
            ("delta_degC",),
        ),
        "Q_h": (Quantity[Energy] | Quantity[Power], ("kJ", "kW")),
    }

    vals: dict[str, Quantity[Any, Any]] = {}
    units = {a: b[1] for a, b in params.items()}

    for a in args:
        for param_name, tp in params.items():
            if isinstance(a, cast(type, tp[0])):
                if param_name == "dT" and not cast(Any, a)._ok_for_muldiv():
                    raise ValueError(
                        f"Cannot pass temperature difference using degree unit {a.u}, convert to delta_deg"
                    )

                vals[param_name] = a

    if "dT" in vals:
        vals["dT"] = vals["dT"].to("delta_degC")

    # whether the calculation is per unit time or amount of mass / energy
    per_time = any(isinstance_types(a, Quantity[MassFlow]) or isinstance_types(a, Quantity[Power]) for a in args)

    unit_idx = 1 if per_time else 0

    if "Q_h" not in vals:
        ret = cp * vals["m"] * vals["dT"]
        unit = units["Q_h"][unit_idx]

    elif "m" not in vals:
        ret = vals["Q_h"] / (cp * vals["dT"])
        unit = units["m"][unit_idx]

    elif "dT" not in vals:
        ret = vals["Q_h"] / (cp * vals["m"])
        unit = units["dT"][0]

        if not (ret.check(Temperature) or ret.check(TemperatureDifference)):
            raise ValueError(f"Both units must be per unit time in case one of them is: {vals}")

    else:
        raise ValueError(f"Incorrect input to heat_balance: {vals}")

    ret = ret.to(unit)

    # the resolved dimensionality depends on which inputs were supplied;
    # this is validated at runtime above but cannot be expressed statically
    return cast(HeatBalanceResult, ret)
