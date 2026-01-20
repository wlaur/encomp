"""
Functions relating to thermodynamics.
"""

from typing import Any

import numpy as np

from .constants import CONSTANTS
from .misc import isinstance_types
from .units import Quantity
from .utypes import (
    Energy,
    HeatTransferCoefficient,
    Length,
    Mass,
    MassFlow,
    Power,
    SpecificHeatCapacity,
    Temperature,
    TemperatureDifference,
    ThermalConductivity,
)

SIGMA = CONSTANTS.SIGMA
DEFAULT_CP = Quantity(4.18, "kJ/kg/K").asdim(SpecificHeatCapacity)


def heat_balance(
    *args: Quantity[Mass, Any]
    | Quantity[MassFlow, Any]
    | Quantity[Energy, Any]
    | Quantity[Power, Any]
    | Quantity[TemperatureDifference, Any]
    | Quantity[Temperature, Any],
    cp: Quantity[SpecificHeatCapacity, float] = DEFAULT_CP,
) -> (
    Quantity[Mass, Any]
    | Quantity[MassFlow, Any]
    | Quantity[Energy, Any]
    | Quantity[Power, Any]
    | Quantity[TemperatureDifference, Any]
):
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
            if isinstance(a, tp[0]):
                if param_name == "dT" and not a._ok_for_muldiv():  # pyright: ignore[reportPrivateUsage]
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

    return ret  # pyright: ignore[reportReturnType]


def intermediate_temperatures(
    T_b: Quantity[Temperature, Any],
    T_s: Quantity[Temperature, Any],
    k: Quantity[ThermalConductivity, Any],
    d: Quantity[Length, Any],
    h_in: Quantity[HeatTransferCoefficient, Any],
    h_out: Quantity[HeatTransferCoefficient, Any],
    epsilon: float,
    tol: float = 1e-6,
) -> tuple[Quantity[Temperature, float], Quantity[Temperature, float]]:
    """
    Solves a nonlinear system of equations to find intermediate
    temperatures of a barrier with the following modes of heat transfer:

    * inner convection
    * conduction through the barrier
    * outer convection
    * outer radiation

    Parameters
    ----------
    T_b : Quantity[Temperature]
        Bulk temperature inside the barrier
    T_s : Quantity[Temperature]
        Bulk temperature outside the barrier (surroundings)
    k : Quantity[ThermalConductivity]
        The thermal conductivity of the barrier material.
        Supply the combined value in case there are multiple layers
    d : Quantity[Length]
        Total thickness of the barrier

        .. note::
            In case ``d`` is set to 0, it will be reset to ``tol`` to
            avoid division by zero.

    h_in : Quantity[HeatTransferCoefficient]
        The convective heat transfer coefficient at the inner barrier wall
    h_out : Quantity[HeatTransferCoefficient]
        The convective heat transfer coefficient at the outer barrier wall
    epsilon : float
        The emissivity of the outside surface,
        used to account for radiative heat transfer
    tol : float, optional
        Numerical accuracy for the conduction layer:
        ``d`` is set to this if 0 is passed, by default 1e-6

    Returns
    -------
    tuple[Quantity[Temperature, float], Quantity[Temperature, float]]
        The intermediate temperatures :math:`T_1` and :math:`T_2`:
        the surface temperatures of the inside and outside of the barrier
    """

    from scipy.optimize import fsolve

    # convert input to numerical values with correct unit
    T_s_val = T_s.to("K").m
    T_b_val = T_b.to("K").m
    k_val = k.to("W/m/K").m
    d_val: float | np.ndarray = d.to("m").m
    h_in_val = h_in.to("W/m²/K").m
    h_out_val = h_out.to("W/m²/K").m

    if abs(d_val - 0) < tol:
        d_val = tol

    # system of coupled equations: heat transfer rate through all layers is identical
    # inner convection == conduction == (outer convection + radiation)

    def fun(x: tuple[np.ndarray, np.ndarray]) -> list[np.ndarray]:
        T1, T2 = x

        eq1 = k_val / d_val * (T1 - T2) - h_out_val * (T2 - T_s_val) - epsilon * SIGMA.m * (T2**4 - T_s_val**4)

        eq2 = k_val / d_val * (T1 - T2) - h_in_val * (T_b_val - T1)

        return [eq1, eq2]

    # use the boundary temperatures as initial guesses
    _ret = fsolve(fun, [T_b_val, T_s_val])  # pyright: ignore[reportArgumentType]

    T1_val: float = _ret[0]
    T2_val: float = _ret[1]

    T1 = Quantity(T1_val, "K").to("degC")
    T2 = Quantity(T2_val, "K").to("degC")

    return T1, T2
