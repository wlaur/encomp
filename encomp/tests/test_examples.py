import numpy as np
import pytest
from pytest import approx  # pyright: ignore[reportUnknownVariableType]

from ..misc import isinstance_types
from ..units import DimensionalityError, ExpectedDimensionalityError, define_dimensionality
from ..units import Quantity as Q
from ..utypes import (
    Currency,
    Dimensionality,
    Dimensionless,
    Length,
    Mass,
    MassFlow,
    Power,
    Pressure,
    TemperatureDifference,
    Volume,
)


class TestREADMEExamples:
    """Test examples from README.md"""

    def test_basic_quantity_creation(self) -> None:
        """Test basic quantity creation and conversion"""
        # converts 1 bar to kPa
        result = Q(1, "bar").to("kPa")
        assert result.m == 100.0
        assert str(result.u) == "kPa"

        # Note: single string parsing like Q('0.1 MPa') is not supported
        # Use two arguments instead
        result = Q(0.1, "MPa").to("bar")
        assert result.m == approx(1.0)

        # list and tuple inputs are converted to np.ndarray
        result = Q([1, 2, 3], "bar") * 2
        assert isinstance(result.m, np.ndarray)
        assert np.array_equal(result.m, [2, 4, 6])

        # dimensionless quantities
        assert Q(0.1) == Q(10, "%")

    def test_quantity_type_system(self) -> None:
        """Test the Quantity type system with dimensionalities"""
        # the unit "kg" is registered as a Mass unit
        m = Q(12, "kg")
        assert isinstance_types(m, Q[Mass])

        V = Q(25, "liter")
        assert isinstance_types(V, Q[Volume])

        # common / and * operations
        m / V  # pyright: ignore[reportUnusedExpression]
        # Quantity[Density]

        # the unit "kg/week" is not registered by default
        m_ = Q(25, "kg/week")

        # at runtime, the dimensionality of m_ will be evaluated to MassFlow
        assert isinstance_types(m_, Q[MassFlow])

        # these operations (Mass**2 divided by Volume) are not explicitly defined
        m**2 / V  # pyright: ignore[reportUnusedExpression]

        # the unit name "meter cubed" is not defined using an overload,
        # the type parameter Volume is used to infer the type
        y = Q(15, "meter cubed").asdim(Volume)
        assert isinstance_types(y, Q[Volume])

    def test_runtime_type_checking(self) -> None:
        """Test runtime type checking with typeguard"""
        from typeguard import TypeCheckError, typechecked

        @typechecked
        def some_func(T: Q[TemperatureDifference, float]) -> tuple[Q[Length, float], Q[Pressure, float]]:
            return (T * Q(12.4, "m/K")).asdim(Length), Q(1, "bar")

        # the dimensionalities check out
        result = some_func(Q(12, "delta_degC"))
        assert isinstance_types(result[0], Q[Length])
        assert isinstance_types(result[1], Q[Pressure])

        # raises an exception with wrong dimensionality
        with pytest.raises(TypeCheckError, match="is not an instance"):
            some_func(Q(26, "kW"))  # pyright: ignore[reportArgumentType] # Power, not TemperatureDifference

    def test_custom_dimensionality(self) -> None:
        """Test creating custom dimensionalities"""

        # the class name TemperaturePerMassFlow must be globally unique
        class TemperaturePerMassFlowTest(Dimensionality):
            dimensions = TemperatureDifference.dimensions / MassFlow.dimensions

        # note the extra parentheses around (kg/s)
        qty = Q(1, "delta_degC/(kg/s)").asdim(TemperaturePerMassFlowTest)
        assert isinstance_types(qty, Q[TemperaturePerMassFlowTest])

        with pytest.raises(ExpectedDimensionalityError):
            _ = Q(1, "delta_degC/(liter/hour)").asdim(TemperaturePerMassFlowTest)

        # create a new subclass of Quantity with restricted input units
        CustomCoolingCapacity = Q[TemperaturePerMassFlowTest, float]

        q1 = CustomCoolingCapacity(6, "delta_degF per (lbs per week)")
        q2 = Q(3, "delta_degF per (pound per fortnight)")

        assert q1 == q2
        # Both are instances of the same dimensionality
        assert isinstance_types(q1, Q[TemperaturePerMassFlowTest])
        assert isinstance_types(q2, Q[TemperaturePerMassFlowTest])

    def test_fluid_class(self) -> None:
        """Test Fluid class examples"""
        from ..fluids import Fluid

        air = Fluid("air", T=Q(25, "degC"), P=Q(2, "bar"))

        # common fluid properties
        assert air.D.check("kg/m³")
        assert air.D.m > 0

        # search for properties
        results = air.search("density")
        assert len(results) > 0

        # any of the names are valid attributes
        assert air.Dmolar.check("mol/m³")

    def test_water_fluid(self) -> None:
        """Test Water fluid examples"""
        from ..fluids import Fluid, Water

        water = Fluid("water", P=Q(25, "bar"), T=Q(550, "°C"))
        assert water.D.check("kg/m³")

        # Water class with vapor quality
        water_twophase = Water(Q=Q(0.5), T=Q(170, "degC"))
        assert water_twophase.D.check("kg/m³")

        water_gas = Water(H=Q(2800, "kJ/kg"), S=Q(7300, "J/kg/K"))
        assert water_gas.D.check("kg/m³")

    def test_humid_air(self) -> None:
        """Test HumidAir examples"""
        from ..fluids import HumidAir

        ha = HumidAir(P=Q(1, "bar"), T=Q(100, "degC"), R=Q(0.5))
        assert ha.P.check("Pa")
        assert ha.T.check("K")


class TestUsageExamples:
    """Test examples from docs/usage.rst"""

    def test_quantity_types(self) -> None:
        """Test quantity types and dimensionalities"""
        pressure = Q(1, "bar")
        assert isinstance_types(pressure, Q[Pressure])

        fraction = Q(5, "%")
        assert isinstance_types(fraction, Q[Dimensionless])

        pressure_kPa = pressure.to("kPa")
        assert type(pressure) is type(pressure_kPa)

        length = Q(1, "meter")
        assert type(pressure) is not type(length)

    def test_custom_power_per_length(self) -> None:
        """Test custom dimensionality PowerPerLength"""

        class PowerPerLengthTest(Dimensionality):
            dimensions = Power.dimensions / Length.dimensions

        qty = Q(100, "W/m").asdim(PowerPerLengthTest)
        assert isinstance_types(qty, Q[PowerPerLengthTest])

    def test_isinstance_checks(self) -> None:
        """Test isinstance and check methods"""
        pressure = Q(1, "bar")

        assert pressure.check(Pressure)
        assert pressure.check("psi")
        assert not pressure.check(Length)
        assert not pressure.check("meter")

        # alternative using isinstance()
        assert isinstance_types(pressure, Q[Pressure])
        assert not isinstance_types(pressure, Q[Length])

        assert isinstance_types([pressure, pressure], list[Q[Pressure]])
        assert isinstance_types({1: Q(2, "m"), 2: Q(25, "cm")}, dict[int, Q[Length]])

        # all Quantity[...] objects are subclasses of Quantity
        assert isinstance_types(pressure, Q)

    def test_custom_base_dimensionality(self) -> None:
        """Test defining custom base dimensionalities"""
        # Use unique names to avoid conflicts with other tests
        define_dimensionality("dry_air_ex")
        define_dimensionality("oxygen_ex")

        m_air = Q(5, "kg * dry_air_ex")
        n_O2 = Q(2.4, "mol * oxygen_ex")
        M_O2 = Q(32, "g/mol")

        # compute mass fraction
        result = ((n_O2 * M_O2) / m_air).to_base_units()
        assert result.m == approx(0.01536)

    def test_vector_magnitudes(self) -> None:
        """Test quantities with vector magnitudes"""
        # lists and tuples are converted to array
        assert isinstance(Q([1, 2, 3], "kg").m, np.ndarray)

        # Note: sets are not supported as magnitude
        # Use lists or arrays instead

        arr = np.linspace(0, 1, 50)
        qty = Q(arr, "bar")
        assert isinstance(qty.m, np.ndarray)  # default linspace size

    def test_combining_quantities(self) -> None:
        """Test combining quantities with operations"""
        result = (Q(5, "%") * Q(1, "meter")).to("mm")
        assert result.m == approx(50.0)

    def test_temperature_differences(self) -> None:
        """Test temperature and temperature difference operations"""
        dT = Q(5, "delta_degC")

        # cannot convert temperature difference to absolute temperature
        with pytest.raises(DimensionalityError):
            dT.to("degC")

        # subtracting temperatures gives temperature difference
        result = Q(25, "degC") - Q(36, "degC")
        assert result.check("delta_degC")
        assert result.m == approx(-11)

        # operations with temperature differences
        result = Q(4.19, "kJ/kg/K") * Q(5, "delta_degC")
        # convert to desired unit
        result_final = result.to("kJ/kg")
        assert result_final.m == approx(20.95)

    def test_currency_units(self) -> None:
        """Test currency units"""
        mf = Q(25, "kg/s")
        t = Q(365, "d")
        price = Q(25, "EUR/ton")

        yearly_cost = mf * t * price
        assert isinstance_types(yearly_cost, Q[Currency])

        # SI prefixes can be used
        result_meur = yearly_cost.to("MEUR")
        assert result_meur.m > 0

    def test_unit_errors(self) -> None:
        """Test handling unit-related errors"""
        # Adding incompatible units
        with pytest.raises(DimensionalityError):
            _ = Q(25, "bar") + Q(25, "m")  # pyright: ignore[reportUnknownVariableType, reportOperatorIssue]

        # Converting to incompatible unit
        with pytest.raises(DimensionalityError):
            Q(15, "m").to("kg")

    def test_pydantic_integration(self) -> None:
        """Test Pydantic integration"""
        from pydantic import BaseModel

        from ..units import ExpectedDimensionalityError

        class Model(BaseModel):
            a: Q
            m: Q[Mass]
            s: Q[Length]
            r: Q[Dimensionless, float] = Q(0.5)

        # Valid model creation
        model = Model(a=Q(25, "cSt"), m=Q(25, "kg"), s=Q(25, "cm"))  # pyright: ignore[reportArgumentType]
        assert isinstance_types(model.a, Q)
        assert isinstance_types(model.m, Q[Mass])
        assert isinstance_types(model.s, Q[Length])

        # Invalid dimensionality raises ExpectedDimensionalityError
        with pytest.raises(ExpectedDimensionalityError):
            Model(a=Q(25, "kg"), m=Q(25, "m"), s=Q(25, "cm"))  # pyright: ignore[reportArgumentType]

    def test_fluid_search_describe(self) -> None:
        """Test Fluid search and describe methods"""
        from ..fluids import Fluid, HumidAir, Water

        # Search for properties
        results = HumidAir.search("bulb")
        assert len(results) > 0
        assert any("WetBulb" in r for r in results)

        # Describe properties
        desc = Fluid.describe("Z")
        assert "Compressibility" in desc

        # Property synonyms
        desc_pcrit = Water.describe("PCRIT")
        assert "critical" in desc_pcrit.lower()

        water = Water(T=Q(25, "°C"), P=Q(1, "atm"))
        assert water.p_critical.m == water.PCRIT.m

    def test_water_vector_inputs(self) -> None:
        """Test Water with vector inputs"""
        from ..fluids import Water

        water = Water(T=Q(np.linspace(25, 50, 10), "°C"), P=Q(np.linspace(25, 50, 10), "bar"))
        assert isinstance(water.D.m, np.ndarray)
        assert len(water.D.m) == 10

        # Different phases
        water_multi = Water(T=Q(np.linspace(25, 500, 10), "°C"), P=Q(np.linspace(0.5, 10, 10), "bar"))
        phases = water_multi.PHASE
        assert isinstance(phases.m, np.ndarray)

        # Constant input
        water_const_p = Water(T=Q(np.linspace(25, 500, 10), "°C"), P=Q(5, "bar"))
        assert isinstance(water_const_p.D.m, np.ndarray)


class TestIntegrationExamples:
    """Test examples for integration with other libraries"""

    def test_fluids_integration(self) -> None:
        """Test fluids library integration"""
        pytest.importorskip("fluids")

        D = Q(25, "cm")
        rhop = Q(800, "kg/m3")
        rho = Q(700, "kg/m3")
        mu = Q(10, "cP")
        t = Q(25, "s")
        V = Q(25, "m/s")

        from fluids.units import (
            integrate_drag_sphere,  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue]
        )

        result = integrate_drag_sphere(D, rhop, rho, mu, t, V=V)  # pyright: ignore[reportUnknownVariableType]
        assert result.check("m/s")  # pyright: ignore[reportUnknownMemberType]
        assert result.m > 0  # pyright: ignore[reportUnknownMemberType]

    def test_fluids_reynolds(self) -> None:
        """Test fluids Reynolds number"""
        pytest.importorskip("fluids")

        from fluids.units import Reynolds  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue]

        # Correct units
        result = Reynolds(V=Q(1, "m/s"), D=Q(15, "cm"), nu=Q(1e-6, "m^2/s"))  # pyright: ignore[reportUnknownVariableType]
        assert result.check("dimensionless")  # pyright: ignore[reportUnknownMemberType]

        # Incorrect units should raise error
        with pytest.raises((DimensionalityError, ValueError)):
            Reynolds(V=Q(1, "m/s"), D=Q(15, "cm"), nu=Q(12, "cP"))
