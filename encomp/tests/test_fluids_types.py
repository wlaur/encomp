import pytest


from encomp.fluids import Water, HumidAir
from encomp.units import Quantity as Q


# it's important that the expected mypy output is a comment on the
# same line as the expression, disable autopep8 if necessary with
# autopep8: off
# ... some code above the line length limit
# autopep8: on


@pytest.mark.mypy_testing
def test_fluids_properties_types() -> None:

    w = Water(
        P=Q(25, 'bar'),
        T=Q(250, 'degC')
    )

    # autopep8: off

    # TODO: these types are revealed as Overload(def ...) by mypy, but correctly by pylance

    # reveal_type(w.P)  # R: encomp.units.Quantity[encomp.utypes.Pressure]
    # reveal_type(w.T)  # R: encomp.units.Quantity[encomp.utypes.Temperature]
    # reveal_type(w.H)  # R: encomp.units.Quantity[encomp.utypes.SpecificEnthalpy]
    # reveal_type(w.S)  # R: encomp.units.Quantity[encomp.utypes.SpecificEntropy]
    # reveal_type(w.D)  # R: encomp.units.Quantity[encomp.utypes.Density]
    # reveal_type(w.V)  # R: encomp.units.Quantity[encomp.utypes.DynamicViscosity]
    # reveal_type(w.Q)  # R: encomp.units.Quantity[encomp.utypes.Dimensionless]

    ha = HumidAir(
        P=Q(25, 'bar'),
        T=Q(250, 'degC'),
        R=Q(25, '%')
    )

    # reveal_type(w.P)  # R: encomp.units.Quantity[encomp.utypes.Pressure]
    # reveal_type(w.T)  # R: encomp.units.Quantity[encomp.utypes.Temperature]
    # reveal_type(w.D)  # R: encomp.units.Quantity[encomp.utypes.Density]
    # reveal_type(w.V)  # R: encomp.units.Quantity[encomp.utypes.DynamicViscosity]

    # autopep8: on


@pytest.mark.mypy_testing
def test_water_init_hints() -> None:

    # valid init calls
    Water(P=Q(25, 'bar'), T=Q(25, 'degC'))
    Water(T=Q(25, 'degC'), P=Q(25, 'bar'))
    Water(P=Q(25, 'bar'), Q=Q(50, '%'))
    Water(T=Q(25, 'C'),  Q=Q(50, '%'))
    Water(Q=Q(50, '%'), P=Q(25, 'kPa'))

    # TODO: implement more overload variants,
    # or try to allow any other inputs in case they are not P/T/Q
    # Water(H=Q(2800, 'kJ/kg'), S=Q(7300, 'J/kg/K'))

    # autopep8: off

    # TODO: mypy also outputs a list of possible overload variants, how to
    # catch these in a comment for the pytest mypy plugin?

    # Water(P=25, T=25)  # E: No overload variant of "Water" matches argument types "int", "int"
    # Water(P=25, T=Q(25, 'C'))  # E: No overload variant of "Water" matches argument types "int", "Quantity[Temperature]"

    # Water(P=Q(25, 'm'), T=Q(25, 'C'))  # E: Argument "P" to "Water" has incompatible type "Quantity[Length]"; expected "Quantity[Pressure]"

    # with pytest.raises(ValueError):

    #     Water(P=Q(25, 'bar'), T=Q(25, 'C'),  Q=Q(50, '%'))  # E: No overload variant of "Water" matches argument types "Quantity[Pressure]", "Quantity[Temperature]", "Quantity[Dimensionless]"

    # # should maybe have a better error message if the kwarg names are incorrect
    # Water(p=Q(25, 'bar'), t=Q(25, 'degC'))  # E: No overload variant of "Water" matches argument types "Quantity[Pressure]", "Quantity[Temperature]"

    # # only keyword args are permitted (not positional)
    # # this error message is also a bit misleading

    # Water(Q(12, 'bar'), Q(25, 'C'))  # E: No overload variant of "Water" matches argument types "Quantity[Pressure]", "Quantity[Temperature]"

    # autopep8: on
