"""
Classes and functions relating to fluid properties.
Uses CoolProp as backend.
"""

from typing import Annotated
import numpy as np

try:
    from CoolProp.CoolProp import PropsSI
    from CoolProp.HumidAirProp import HAPropsSI

except ImportError:

    import warnings
    warnings.warn('CoolProp package not installed, install with conda:'
                  '\nconda install conda-forge::coolprop')

from encomp.structures import flatten
from encomp.units import Quantity, Unit

# type alias for a CoolProp property name
CProperty = Annotated[str, 'CoolProp property name']
CName = Annotated[str, 'CoolProp fluid name']


class CoolPropFluid:

    PHASES: dict[float, str] = {
        0.0: 'Liquid',
        5.0: 'Gas',
        6.0: 'Two-phase',
        3.0: 'Supercritical liquid',  # P > P_crit
        2.0: 'Supercritical gas',     # T > T_crit
        1.0: 'Supercritical fluid',   # P > P_crit and T > T_crit
        8.0: 'Not imposed'
    }

    # unit and description for properties in function PropsSI
    # (name1, name2, ...): (unit, description)
    # names are case-sensitive
    PROPERTY_MAP: dict[tuple[str, ...], tuple[str, str]] = {
        ('DELTA', 'Delta'):                             ('dimensionless', 'Reduced density (rho/rhoc)'),
        ('DMOLAR', 'Dmolar'):                           ('mol/m³', 'Molar density'),
        ('D', 'DMASS', 'Dmass'):                        ('kg/m³', 'Mass density'),
        ('HMOLAR', 'Hmolar'):                           ('J/mol', 'Molar specific enthalpy'),
        ('H', 'HMASS', 'Hmass'):                        ('J/kg', 'Mass specific enthalpy'),
        ('P', ):                                        ('Pa', 'Pressure'),
        ('Q', ):                                        ('dimensionless', 'Mass vapor quality'),
        ('SMOLAR', 'Smolar'):                           ('J/mol/K', 'Molar specific entropy'),
        ('S', 'SMASS', 'Smass'):                        ('J/kg/K', 'Mass specific entropy'),
        ('TAU', 'Tau'):                                 ('dimensionless', 'Reciprocal reduced temperature (Tc/T)'),
        ('T', ):                                        ('K', 'Temperature'),
        ('UMOLAR', 'Umolar'):                           ('J/mol', 'Molar specific internal energy'),
        ('U', 'UMASS', 'Umass'):                        ('J/kg', 'Mass specific internal energy'),
        ('A', 'SPEED_OF_SOUND', 'speed_of_sound'):      ('m/s', 'Speed of sound'),
        ('CONDUCTIVITY', 'L', 'conductivity'):          ('W/m/K', 'Thermal conductivity'),
        ('CP0MASS', 'Cp0mass'):                         ('J/kg/K', 'Ideal gas mass specific constant pressure specific heat'),
        ('CP0MOLAR', 'Cp0molar'):                       ('J/mol/K', 'Ideal gas molar specific constant pressure specific heat'),
        ('CPMOLAR', 'Cpmolar'):                         ('J/mol/K', 'Molar specific constant pressure specific heat'),
        ('CVMASS', 'Cvmass', 'O'):                      ('J/kg/K', 'Mass specific constant volume specific heat'),
        ('CVMOLAR', 'Cvmolar'):                         ('J/mol/K', 'Molar specific constant volume specific heat'),
        ('C', 'CPMASS', 'Cpmass'):                      ('J/kg/K', 'Mass specific constant pressure specific heat'),
        ('DIPOLE_MOMENT', 'dipole_moment'):             ('C*m', 'Dipole moment'),
        ('GAS_CONSTANT', 'gas_constant'):               ('J/mol/K', 'Molar gas constant'),
        ('GMOLAR_RESIDUAL', 'Gmolar_residual'):         ('J/mol/K', 'Residual molar Gibbs energy'),
        ('GMOLAR', 'Gmolar'):                           ('J/mol', 'Molar specific Gibbs energy'),
        ('G', 'GAMES', 'Gmass'):                        ('J/kg', 'Mass specific Gibbs energy'),
        ('HELMHOLTZMASS', 'Helmholtzmass'):             ('J/kg', 'Mass specific Helmholtz energy'),
        ('HELMHOLTZMOLAR', 'Helmholtzmolar'):           ('J/mol', 'Molar specific Helmholtz energy'),
        ('HMOLAR_RESIDUAL', 'Hmolar_residual'):         ('J/mol/K', 'Residual molar enthalpy'),
        ('ISENTROPIC_EXPANSION_COEFFICIENT',
         'isentropic_expansion_coefficient'):           ('dimensionless', 'Isentropic expansion coefficient'),
        ('ISOBARIC_EXPANSION_COEFFICIENT',
         'isobaric_expansion_coefficient'):             ('1/K', 'Isobaric expansion coefficient'),
        ('ISOTHERMAL_COMPRESSIBILITY',
         'isothermal_compressibility'):                 ('1/Pa', 'Isothermal compressibility'),
        ('I', 'SURFACE_TENSION', 'surface_tension'):    ('N/m', 'Surface tension'),
        ('M', 'MOLARMASS', 'MOLAR_MASS', 'MOLEMASS',
         'molar_mass', 'molarmass', 'molemass'):        ('kg/mol', 'Molar mass'),
        ('PCRIT', 'P_CRITICAL', 'Pcrit',
         'p_critical', 'pcrit'):                        ('Pa', 'Pressure at the critical point'),
        ('PHASE', 'Phase'):                             ('dimensionless', 'Phase index as a float'),
        ('PMAX', 'P_MAX', 'P_max', 'pmax'):             ('Pa', 'Maximum pressure limit'),
        ('PMIN', 'P_MIN', 'P_min', 'pmin'):             ('Pa', 'Minimum pressure limit'),
        ('PRANDTL', 'Prandtl'):                         ('dimensionless', 'Prandtl number'),
        ('PTRIPLE', 'P_TRIPLE', 'p_triple', 'ptriple'): ('Pa', 'Pressure at the triple point (pure only)'),
        ('P_REDUCING', 'p_reducing'):                   ('Pa', 'Pressure at the reducing point'),
        ('RHOCRIT', 'RHOMASS_CRITICAL',
         'rhocrit', 'rhomass_critical'):                ('kg/m³', 'Mass density at critical point'),
        ('RHOMASS_REDUCING', 'rhomass_reducing'):       ('kg/m³', 'Mass density at reducing point'),
        ('RHOMOLAR_CRITICAL', 'rhomolar_critical'):     ('mol/m³', 'Molar density at critical point'),
        ('RHOMOLAR_REDUCING', 'rhomolar_reducing'):     ('mol/m³', 'Molar density at reducing point'),
        ('SMOLAR_RESIDUAL', 'Smolar_residual'):         ('J/mol/K', 'Residual molar entropy'),
        ('TCRIT', 'T_CRITICAL',
         'T_critical', 'Tcrit'):                        ('K', 'Temperature at the critical point'),
        ('TMAX', 'T_MAX', 'T_max', 'Tmax'):             ('K', 'Maximum temperature limit'),
        ('TMIN', 'T_MIN', 'T_min', 'Tmin'):             ('K', 'Minimum temperature limit'),
        ('TTRIPLE', 'T_TRIPLE', 'T_triple', 'Ttriple'): ('K', 'Temperature at the triple point'),
        ('T_FREEZE', 'T_freeze'):                       ('K', 'Freezing temperature for incompressible solutions'),
        ('T_REDUCING', 'T_reducing'):                   ('K', 'Temperature at the reducing point'),
        ('V', 'VISCOSITY', 'viscosity'):                ('Pa*s', 'Viscosity'),
        ('Z', ):                                        ('dimensionless', 'Compressibility factor')
    }

    ALL_PROPERTIES: set[str] = set(flatten(PROPERTY_MAP))
    REPR_PROPERTIES: tuple[tuple[str, str]] = (('P', '.0f'), ('T', '.1f'),
                                               ('D', '.1f'), ('V', '.2g'))

    # preferred return units
    # key is the first name in the tuple used in PROPERTY_MAP
    RETURN_UNITS: dict[str, str] = {
        'P': 'kPa',
        'T': '°C',
        'TMAX': '°C',
        'TMIN': '°C',
        'TTRIPLE': '°C',
        'T_FREEZE': '°C',
        'V': 'cP',
        'H': 'kJ/kg',
        'C': 'kJ/kg/K'
    }

    # numerical accuracy, determines if return values are zero
    _EPS: float = 1e-9

    # skip checking for zero for these properties
    _SKIP_ZERO_CHECK: tuple[str, ...] = ('PHASE', )

    def __init__(self, name: CName):
        """
        Base class that represents a fluid (pure or mixture, gas or liquid).
        Uses *CoolProp* as backend to determine fluid properties.

        This class should not be used directly, since it does not contain a fixed
        point to determine fluid properties (temperature, pressure, enthalpy, entropy, ...).

        Fluid names for pure fluids are not case-sensitive, but the mixture names are.
        The following fluid names are recognized by CoolProp:

        **Pure**

        .. code-block::

            1-Butene,Acetone,Air,Ammonia,Argon,Benzene,CarbonDioxide,CarbonMonoxide,
            CarbonylSulfide,CycloHexane,CycloPropane,Cyclopentane,D4,D5,D6,Deuterium,
            Dichloroethane,DiethylEther,DimethylCarbonate,DimethylEther,Ethane,
            Ethanol,EthylBenzene,Ethylene,EthyleneOxide,Fluorine,HFE143m,HeavyWater,
            Helium,Hydrogen,HydrogenChloride,HydrogenSulfide,IsoButane,IsoButene,
            Isohexane,Isopentane,Krypton,MD2M,MD3M,MD4M,MDM,MM,Methane,Methanol,
            MethylLinoleate,MethylLinolenate,MethylOleate,MethylPalmitate,MethylStearate,
            Neon,Neopentane,Nitrogen,NitrousOxide,Novec649,OrthoDeuterium,OrthoHydrogen,
            Oxygen,ParaDeuterium,ParaHydrogen,Propylene,Propyne,R11,R113,R114,R115,
            R116,R12,R123,R1233zd(E),R1234yf,R1234ze(E),R1234ze(Z),R124,R1243zf,
            R125,R13,R134a,R13I1,R14,R141b,R142b,R143a,R152A,R161,R21,R218,R22,R227EA,
            R23,R236EA,R236FA,R245ca,R245fa,R32,R365MFC,R40,R404A,R407C,R41,R410A,
            R507A,RC318,SES36,SulfurDioxide,SulfurHexafluoride,Toluene,Water,Xenon,
            cis-2-Butene,m-Xylene,n-Butane,n-Decane,n-Dodecane,n-Heptane,n-Hexane,
            n-Nonane,n-Octane,n-Pentane,n-Propane,n-Undecane,o-Xylene,p-Xylene,trans-2-Butene

        **Incompressible pure**

        .. code-block::

            INCOMP::AS10,INCOMP::AS20,INCOMP::AS30,INCOMP::AS40,INCOMP::AS55,INCOMP::DEB,
            INCOMP::DSF,INCOMP::DowJ,INCOMP::DowJ2,INCOMP::DowQ,INCOMP::DowQ2,INCOMP::HC10,
            INCOMP::HC20,INCOMP::HC30,INCOMP::HC40,INCOMP::HC50,INCOMP::HCB,INCOMP::HCM,
            INCOMP::HFE,INCOMP::HFE2,INCOMP::HY20,INCOMP::HY30,INCOMP::HY40,INCOMP::HY45,
            INCOMP::HY50,INCOMP::NBS,INCOMP::NaK,INCOMP::PBB,INCOMP::PCL,INCOMP::PCR,
            INCOMP::PGLT,INCOMP::PHE,INCOMP::PHR,INCOMP::PLR,INCOMP::PMR,INCOMP::PMS1,
            INCOMP::PMS2,INCOMP::PNF,INCOMP::PNF2,INCOMP::S800,INCOMP::SAB,INCOMP::T66,
            INCOMP::T72,INCOMP::TCO,INCOMP::TD12,INCOMP::TVP1,INCOMP::TVP1869,INCOMP::TX22,
            INCOMP::TY10,INCOMP::TY15,INCOMP::TY20,INCOMP::TY24,INCOMP::Water,INCOMP::XLT,
            INCOMP::XLT2,INCOMP::ZS10,INCOMP::ZS25,INCOMP::ZS40,INCOMP::ZS45,INCOMP::ZS55

        **Incompressible mixtures**

        .. code-block::

            INCOMP::FRE,INCOMP::IceEA,INCOMP::IceNA,INCOMP::IcePG,INCOMP::LiBr,INCOMP::MAM,
            INCOMP::MAM2,INCOMP::MCA,INCOMP::MCA2,INCOMP::MEA,INCOMP::MEA2,INCOMP::MEG,
            INCOMP::MEG2,INCOMP::MGL,INCOMP::MGL2,INCOMP::MITSW,INCOMP::MKA,INCOMP::MKA2,
            INCOMP::MKC,INCOMP::MKC2,INCOMP::MKF,INCOMP::MLI,INCOMP::MMA,INCOMP::MMA2,
            INCOMP::MMG,INCOMP::MMG2,INCOMP::MNA,INCOMP::MNA2,INCOMP::MPG,INCOMP::MPG2,
            INCOMP::VCA,INCOMP::VKC,INCOMP::VMA,INCOMP::VMG,INCOMP::VNA,INCOMP::AEG,
            INCOMP::AKF,INCOMP::AL,INCOMP::AN,INCOMP::APG,INCOMP::GKN,INCOMP::PK2,
            INCOMP::PKL,INCOMP::ZAC,INCOMP::ZFC,INCOMP::ZLC,INCOMP::ZM,INCOMP::ZMC

        **Mixtures**

        .. code-block::

            AIR.MIX,AMARILLO.MIX,Air.mix,Amarillo.mix,EKOFISK.MIX,Ekofisk.mix,GULFCOAST.MIX,
            GULFCOASTGAS(NIST1).MIX,GulfCoast.mix,GulfCoastGas(NIST1).mix,HIGHCO2.MIX,
            HIGHN2.MIX,HighCO2.mix,HighN2.mix,NATURALGASSAMPLE.MIX,NaturalGasSample.mix,
            R401A.MIX,R401A.mix,R401B.MIX,R401B.mix,R401C.MIX,R401C.mix,R402A.MIX,R402A.mix,
            R402B.MIX,R402B.mix,R403A.MIX,R403A.mix,R403B.MIX,R403B.mix,R404A.MIX,R404A.mix,
            R405A.MIX,R405A.mix,R406A.MIX,R406A.mix,R407A.MIX,R407A.mix,R407B.MIX,R407B.mix,
            R407C.MIX,R407C.mix,R407D.MIX,R407D.mix,R407E.MIX,R407E.mix,R407F.MIX,R407F.mix,
            R408A.MIX,R408A.mix,R409A.MIX,R409A.mix,R409B.MIX,R409B.mix,R410A.MIX,R410A.mix,
            R410B.MIX,R410B.mix,R411A.MIX,R411A.mix,R411B.MIX,R411B.mix,R412A.MIX,R412A.mix,
            R413A.MIX,R413A.mix,R414A.MIX,R414A.mix,R414B.MIX,R414B.mix,R415A.MIX,R415A.mix,
            R415B.MIX,R415B.mix,R416A.MIX,R416A.mix,R417A.MIX,R417A.mix,R417B.MIX,R417B.mix,
            R417C.MIX,R417C.mix,R418A.MIX,R418A.mix,R419A.MIX,R419A.mix,R419B.MIX,R419B.mix,
            R420A.MIX,R420A.mix,R421A.MIX,R421A.mix,R421B.MIX,R421B.mix,R422A.MIX,R422A.mix,
            R422B.MIX,R422B.mix,R422C.MIX,R422C.mix,R422D.MIX,R422D.mix,R422E.MIX,R422E.mix,
            R423A.MIX,R423A.mix,R424A.MIX,R424A.mix,R425A.MIX,R425A.mix,R426A.MIX,R426A.mix,
            R427A.MIX,R427A.mix,R428A.MIX,R428A.mix,R429A.MIX,R429A.mix,R430A.MIX,R430A.mix,
            R431A.MIX,R431A.mix,R432A.MIX,R432A.mix,R433A.MIX,R433A.mix,R433B.MIX,R433B.mix,
            R433C.MIX,R433C.mix,R434A.MIX,R434A.mix,R435A.MIX,R435A.mix,R436A.MIX,R436A.mix,
            R436B.MIX,R436B.mix,R437A.MIX,R437A.mix,R438A.MIX,R438A.mix,R439A.MIX,R439A.mix,
            R440A.MIX,R440A.mix,R441A.MIX,R441A.mix,R442A.MIX,R442A.mix,R443A.MIX,R443A.mix,
            R444A.MIX,R444A.mix,R444B.MIX,R444B.mix,R445A.MIX,R445A.mix,R446A.MIX,R446A.mix,
            R447A.MIX,R447A.mix,R448A.MIX,R448A.mix,R449A.MIX,R449A.mix,R449B.MIX,R449B.mix,
            R450A.MIX,R450A.mix,R451A.MIX,R451A.mix,R451B.MIX,R451B.mix,R452A.MIX,R452A.mix,
            R453A.MIX,R453A.mix,R454A.MIX,R454A.mix,R454B.MIX,R454B.mix,R500.MIX,R500.mix,
            R501.MIX,R501.mix,R502.MIX,R502.mix,R503.MIX,R503.mix,R504.MIX,R504.mix,R507A.MIX,
            R507A.mix,R508A.MIX,R508A.mix,R508B.MIX,R508B.mix,R509A.MIX,R509A.mix,R510A.MIX,
            R510A.mix,R511A.MIX,R511A.mix,R512A.MIX,R512A.mix,R513A.MIX,R513A.mix,
            TYPICALNATURALGAS.MIX,TypicalNaturalGas.mix


        Refer to the CoolProp documentation for more information:

        * http://www.coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids
        * http://www.coolprop.org/fluid_properties/Mixtures.html#binary-pairs
        * http://www.coolprop.org/fluid_properties/Incompressibles.html#the-different-fluids
        * http://www.coolprop.org/coolprop/HighLevelAPI.html#table-of-inputs-outputs-to-hapropssi
        * http://www.coolprop.org/fluid_properties/HumidAir.html

        The names ``Water`` and ``HEOS::Water`` uses the formulation defined by IAPWS-95.
        Use the name ``IF97::Water`` to instead use the slightly faster
        (but less accurate) IAPWS-97 formulation.
        In most cases, the difference between IAPWS-95 and IAPWS-97 is negligible.
        Read CoolProp's `introduction
        <http://www.coolprop.org/fluid_properties/IF97.html>`_ about water properties for more information.


        Parameters
        ----------
        name : CName
            The name of the fluid, same name as is used by CoolProp.
            Include the ``INCOMP::`` prefix and potential mixing ratio for incompressible mixtures.

            Examples:

                * ``INCOMP::MITSW[0.05]``: seawater with 5 mass-percent salt.
                * ``INCOMP::MPG[0.5]``: 50 % ethylene glycol
                * ``INCOMP::T66``: Therminol 66 (https://www.therminol.com/product/71093438)
        """

        self.name = name

    @classmethod
    def get_prop_key(cls, prop: CProperty) -> tuple[str, ...]:

        if prop not in cls.ALL_PROPERTIES:
            raise ValueError(
                f'Property "{prop}" is not a valid CoolProp property name')

        for names in cls.PROPERTY_MAP:
            if prop in names:
                return names

    @classmethod
    def get_coolprop_unit(cls, prop: CProperty) -> Unit:

        key = cls.get_prop_key(prop)

        if key in cls.PROPERTY_MAP:
            unit_str = cls.PROPERTY_MAP[key][0]
            return Quantity.get_unit(unit_str)

        raise ValueError(f'Key {key} does not exist')

    @classmethod
    def describe(cls, prop: CProperty) -> str:

        key = cls.get_prop_key(prop)

        if key in cls.PROPERTY_MAP:

            unit_str, description = cls.PROPERTY_MAP[key]
            unit = Quantity.get_unit(unit_str)
            unit_repr = f'{unit:~P}'

            if not unit_repr:
                unit_repr = 'dimensionless'

            return f'{", ".join(key)}: {description} [{unit_repr}]'

        raise ValueError(f'Key {key} does not exist')

    @classmethod
    def search(cls, inp: str) -> list[str]:
        """
        Returns a list of CoolProp properties that matches the search input.


        Parameters
        ----------
        inp : str
            Input search string

        Returns
        -------
        list[str]
            list of CoolProp properties (with descriptions) that matches the search string.
        """

        matches = []

        for key in cls.PROPERTY_MAP:
            description = cls.describe(key[0])
            if inp.lower() in description.lower():
                matches.append(description)

        return matches

    def get(self,
            output: CProperty,
            point_1: tuple[CProperty, Quantity],
            point_2: tuple[CProperty, Quantity]) -> Quantity:
        """
        Wraps the function ``CoolProp.CoolProp.PropsSI``, handles input
        and output with :py:class:`encomp.units.Quantity` objects.

        Parameters
        ----------
        output : CProperty
            Name of the output property
        point_1 : tuple[str, Quantity]
            First fixed state variable: name and value of the property
        point_2 : tuple[str, Quantity]
            Second fixed state variable: name and value of the property

        Returns
        -------
        Quantity
            Value (and unit) of the output property
        """

        prop_1, qty_1 = point_1
        prop_2, qty_2 = point_2

        unit_1 = self.get_coolprop_unit(prop_1)
        unit_2 = self.get_coolprop_unit(prop_2)
        unit_output = self.get_coolprop_unit(output)

        val_1 = qty_1.to(unit_1).m
        val_2 = qty_2.to(unit_2).m

        val = PropsSI(output,
                      prop_1, val_1,
                      prop_2, val_2,
                      self.name)

        qty = Quantity(val, unit_output)

        # value with dimensions cannot be zero
        # CoolProp uses 0.0 for missing data, change this to NaN
        # the values are not exactly 0, use the _EPS class attribute to check this
        # skip this check for some properties
        if (not isinstance(qty.m, np.ndarray) and output not in self._SKIP_ZERO_CHECK and
                not qty.dimensionless and val < self._EPS):
            qty = Quantity(float('NaN'), unit_output)

        key = self.get_prop_key(output)

        if key[0] in self.RETURN_UNITS:
            ret_unit = self.RETURN_UNITS[key[0]]
            qty = qty.to(ret_unit)

        return qty


class Fluid(CoolPropFluid):

    def __init__(self, name: CName, **kwargs: Quantity):
        """
        Represents a fluid at a fixed state, for example at a
        specific temperature and pressure.

        Parameters
        ----------
        name : CName
            Name of the fluid
        kwargs: Quantity
            Values for the two fixed points. The name of the keyword argument is the
            CoolProp property name.
        """

        super().__init__(name)

        if len(kwargs) != 2:
            raise ValueError(
                f'Exactly two fixed points are required, passed {list(kwargs)}')

        kwargs = list(kwargs.items())

        self.point_1 = kwargs[0]
        self.point_2 = kwargs[1]

    def get(self, output: CProperty) -> Quantity:
        """
        Uses the constant fixed points to call
        :py:meth:`encomp.fluids.CoolPropFluid.get`.

        Parameters
        ----------
        output : CProperty
            Name of the output property

        Returns
        -------
        Quantity
            Value (and unit) of the output property
        """

        return super().get(output,
                           self.point_1,
                           self.point_2)

    @property
    def phase(self) -> str:
        return self.PHASES[self.get('PHASE')]

    def __getattr__(self, attr):

        # this is called in case attr does not exist
        return self.get(attr)

    def __repr__(self) -> str:

        props = []

        for p, fmt in self.REPR_PROPERTIES:

            # CoolProp might not have a backend for all props
            try:
                props.append(f'{p}={self.get(p):{fmt}}')
            except Exception:
                props.append(f'{p}=N/A')

        props_str = ', '.join(props)

        s = f'<{self.__class__.__name__} "{self.name}", {props_str}>'

        return s


class HumidAir(Fluid):

    # unit and description for properties in function HAPropsSI
    PROPERTY_MAP: dict[tuple[str, ...], tuple[str, str]] = {

        ('B', 'Twb', 'T_wb', 'WetBulb'):  ('K', 'Wet-Bulb Temperature'),
        ('C', 'cp'):                      ('J/kg/K', 'Mixture specific heat per unit dry air'),
        ('Cha', 'cp_ha'):                 ('J/kg/K', 'Mixture specific heat per unit humid air'),
        ('CV', ):                         ('J/kg/K', 'Mixture specific heat at constant volume per unit dry air'),
        ('CVha', 'cv_ha'):                ('J/kg/K', 'Mixture specific heat at constant volume per unit humid air'),
        ('D', 'Tdp', 'DewPoint', 'T_dp'): ('K', 'Dew-Point Temperature'),
        ('H', 'Hda', 'Enthalpy'):         ('J/kg', 'Mixture enthalpy per dry air'),
        ('Hha', ):                        ('J/kg', 'Mixture enthalpy per humid air'),
        ('K', 'k', 'Conductivity'):       ('W/m/K', 'Mixture thermal conductivity'),
        ('M', 'Visc', 'mu'):              ('Pa*s', 'Mixture viscosity'),
        ('psi_w', 'Y'):                   ('dimensionless', 'Water mole fraction'),
        ('P', ):                          ('Pa', 'Pressure'),
        ('P_w', ):                        ('Pa', 'Partial pressure of water vapor'),
        ('R', 'RH', 'RelHum'):            ('dimensionless', 'Relative humidity in [0, 1]'),
        ('S', 'Sda', 'Entropy'):          ('J/kg/K', 'Mixture entropy per unit dry air'),
        ('Sha', ):                        ('J/kg/K', 'Mixture entropy per unit humid air'),
        ('T', 'Tdb', 'T_db'):             ('K', 'Dry-Bulb Temperature'),
        ('V', 'Vda'):                     ('m³/kg', 'Mixture volume per unit dry air'),
        ('Vha', ):                        ('m³/kg', 'Mixture volume per unit humid air'),
        ('W', 'Omega', 'HumRat'):         ('dimensionless', 'Humidity Rat mass water per mass dry air'),
        ('Z', ):                          ('dimensionless', 'Compressibility factor')
    }

    ALL_PROPERTIES: set[str] = set(flatten(PROPERTY_MAP))

    # HAPropsSI has different parameter names
    # density is not defined, need to use either Vda (volume per dry air)
    # or Vha (per humid air)
    RETURN_UNITS: dict[str, str] = {
        'P': 'kPa',
        'P_w': 'kPa',
        'M': 'cP',
        'T': '°C',
        'D': '°C',
        'B': '°C',
    }

    REPR_PROPERTIES: tuple[tuple[str, str]] = (('P', '.0f'), ('T', '.1f'),
                                               ('R', '.2f'), ('Vda', '.1f'),
                                               ('Vha', '.1f'), ('M', '.2g'))

    def __init__(self, **kwargs: Quantity):
        """
        Interface to the CoolProp function for humid air, ``CoolProp.CoolProp.HAPropsSI``.
        Needs three fixed points instead of two.

        Parameters
        ----------
        kwargs: Quantity
            Values for the three fixed points. The name of the keyword argument is the
            CoolProp property name.
        """

        self.name = 'Humid air'

        if len(kwargs) != 3:
            raise ValueError(
                f'Exactly three fixed points are required, passed {list(kwargs)}')

        kwargs = list(kwargs.items())

        self.point_1 = kwargs[0]
        self.point_2 = kwargs[1]
        self.point_3 = kwargs[2]

    def get(self, output: CProperty) -> Quantity:
        """
        Uses the constant fixed points to call ``CoolProp.CoolProp.HAPropsSI``.

        Parameters
        ----------
        output : CProperty
            Name of the output property

        Returns
        -------
        Quantity
            Value (and unit) of the output property
        """

        prop_1, qty_1 = self.point_1
        prop_2, qty_2 = self.point_2
        prop_3, qty_3 = self.point_3

        unit_1 = self.get_coolprop_unit(prop_1)
        unit_2 = self.get_coolprop_unit(prop_2)
        unit_3 = self.get_coolprop_unit(prop_3)

        unit_output = self.get_coolprop_unit(output)

        val_1 = qty_1.to(unit_1).m
        val_2 = qty_2.to(unit_2).m
        val_3 = qty_3.to(unit_3).m

        val = HAPropsSI(output,
                        prop_1, val_1,
                        prop_2, val_2,
                        prop_3, val_3)

        qty = Quantity(val, unit_output)

        if (not isinstance(qty.m, np.ndarray) and output not in self._SKIP_ZERO_CHECK and
                not qty.dimensionless and val < self._EPS):
            qty = Quantity(float('NaN'), unit_output)

        key = self.get_prop_key(output)

        if key[0] in self.RETURN_UNITS:
            ret_unit = self.RETURN_UNITS[key[0]]
            qty = qty.to(ret_unit)

        return qty

    def __getattr__(self, attr):

        # this is called in case attr does not exist
        return self.get(attr)

    def __repr__(self) -> str:

        props_str = ', '.join(f'{p}={self.get(p):{fmt}}'
                              for p, fmt in self.REPR_PROPERTIES)

        s = f'<{self.__class__.__name__}, {props_str}>'

        return s


class Water(Fluid):

    REPR_PROPERTIES: tuple[tuple[str, str]] = (('P', '.0f'), ('T', '.1f'),
                                               ('D', '.1f'), ('V', '.1f'))

    def __init__(self, **kwargs: Quantity):
        """
        Convenience class to access water and steam properties via CoolProp.

        Parameters
        ----------
        kwargs: Quantity
            Values for the two fixed points. The name of the keyword argument is the
            CoolProp property name.
        """

        self.name = 'Water'

        if len(kwargs) != 2:

            if set(kwargs) == {'P', 'T', 'Q'}:

                raise ValueError(
                    'Cannot set both P, T and vapor quality Q. Remove one of P, T to '
                    'get properties of saturated steam.')

            raise ValueError(
                f'Exactly two fixed points are required, passed {list(kwargs)}')

        kwargs = list(kwargs.items())

        self.point_1 = kwargs[0]
        self.point_2 = kwargs[1]

    def __repr__(self) -> str:

        repr_properties = self.REPR_PROPERTIES

        # show the vapor quality in the repr in case of saturated steam
        vapor_quality = self.Q

        if vapor_quality > 0 and vapor_quality < 1:
            repr_properties = list(repr_properties) + [('Q', '.2f')]

        props_str = ', '.join(f'{p}={self.get(p):{fmt}}'
                              for p, fmt in repr_properties)

        s = f'<{self.__class__.__name__} ({self.phase}), {props_str}>'

        return s
