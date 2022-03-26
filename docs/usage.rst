Usage
=====

This section contains instructions for incorporating ``encomp`` in your own tools.


The Quantity class
------------------

The purpose of the :py:class:`encomp.units.Quantity` class is to store information about the *magnitude*, *dimensionality* and *units* of a physical quantity.


.. note::
    The shorthand ``Q`` is used as an alias for ``Quantity``

To get started, import the class:


.. code-block:: python

    from encomp.units import Q

Create an instance representing an absolute pressure of 1 bar:

.. code-block:: python

    pressure = Q(1, 'bar')

.. warning::
    ``encomp`` (and the underlying ``pint`` library) does not differentiate between absolute and gauge pressure.

Convert the pressure to another unit:

.. code-block:: python

    pressure_kPa = pressure.to('kPa')

Refer to the default ``pint`` unit definition file (`defaults_en.txt <https://github.com/hgrecco/pint/blob/master/pint/default_en.txt>`_) for a list of accepted unit names.


Quantity types
~~~~~~~~~~~~~~

The quantity class also contains information about its *dimensionality*.
It is not possible to create an instance of the base class :py:class:`encomp.units.Quantity`, since this would not have any dimensionality at all (a *dimensionless* quantity still has a dimensionality of 1).
A unique subclass of :py:class:`encomp.units.Quantity` is created for each new dimensionality.
The subclasses are cached and reused, which means that the ``is`` operator can be used to check for equality.

.. code-block:: python

    type(pressure) # <class 'encomp.units.Quantity[Pressure]'>

    fraction = Q(5, '%')
    type(fraction) # <class 'encomp.units.Quantity[Dimensionless]'>

    assert type(pressure) is type(pressure_kPa)

    length = Q(1, 'meter')
    assert type(pressure) is not type(length)


To create a subclass of :py:class:`encomp.units.Quantity` with a certain dimensionality, enclose the string name of the dimensionality (or a ``UnitsContainer`` object) in square brackets:


.. code-block:: python

    Q['Pressure'] # subclass with dimensionality pressure

    from encomp.utypes import Length, Power
    Q[Power / Length] # use UnitsContainer objects to combine dimensionalities

The builtin ``isinstance()`` can be used to check dimensionalities of quantity objects, but it's better to use the :py:meth:`encomp.units.Quantity.check` method.
This method takes a string unit or a ``UnitsContainer`` object as input.
The module :py:mod:`encomp.utypes` contains ``UnitsContainer`` objects for the most common dimensionalities.


.. code-block:: python

    from encomp.utypes import Pressure, Length

    pressure.check(Length) # False
    pressure.check('meter') # False

    pressure.check(Pressure) # True
    pressure.check('psi') # True

    # alternative using isinstance()

    isinstance(pressure, Q['Pressure']) # True
    isinstance(pressure, Q[Pressure]) # True

    isinstance(pressure, Q['Length']) # False
    isinstance(pressure, Q[Length]) # False


To check more complex types, use the function :py:func:`encomp.misc.isinstance_types`.
This function calls ``typeguard.check_types`` in case ``isinstance()`` cannot be used.

.. code-block:: python

    from encomp.misc import isinstance_types

    lst = [Q(1, 'bar'), Q(2, 'bar')]

    isinstance_types(lst, list[Q['Pressure']]) # True
    isinstance_types(lst, list[Q['Length']]) # False

    tup = (Q(25, 'm/s'), Q(1, 'kg'))
    isinstance_types(tup, tuple[Q['Velocity'], Q['Mass']]) # True
    isinstance_types(tup, tuple[Q['Velocity'], Q['Power']]) # False


To check types for functions and methods, use the ``@typeguard.typechecked`` decorator instead of writing explicit checks inside the function body:


.. code-block:: python

    from typeguard import typechecked

    @typechecked
    def func(p1: Q['Pressure']) -> tuple[Q['Length'], Q['Power']]:
        return Q(1, 'm'), Q(1, 'kW')

A ``TypeError`` will be raised in case the function ``func`` is called with incorrect units or if the return value(s) have incorrect units.


Custom dimensionalities
~~~~~~~~~~~~~~~~~~~~~~~

By default, the seven SI dimensionalities (and combinations of these) are defined, along with some commonly used media (water, air, fuel).
Additionally, the ``normal`` dimensionality is defined (used to represent normal volume).

The function :py:func:`encomp.units.define_dimensionality` can be used to define a new, custom dimensionality.
In case the dimensionality already exists, ``DimensionalityRedefinitionError`` is raised.


.. code-block:: python

    from encomp.units import define_dimensionality

    define_dimensionality('dry_air')
    define_dimensionality('oxygen')

    m_air = Q(5, 'kg * dry_air')
    n_O2 = Q(2.4, 'mol * oxygen')
    M_O2 = Q(32, 'g/mol')

    # compute mass fraction
    ((n_O2 * M_O2) / m_air).to_base_units() # 0.01536 oxygen/air


Using multiple magnitudes
~~~~~~~~~~~~~~~~~~~~~~~~~


Lists, tuples, sets, Numpy arrays and Pandas Series objects can also be used as magnitude.
In case a tuple or list is given as magnitude when creating a quantity, it will be converted to a Numpy array.


.. code-block:: python

    # lists and tuples are converted to array
    type(Q([1, 2, 3], 'kg').m) # numpy.ndarray
    type(Q((1, 2, 3), 'kg').m) # numpy.ndarray

    # set is not converted, since Numpy has no corresponding type
    type(Q({1, 2, 3}, 'kg').m) # set

    import numpy as np

    arr = np.linspace(0, 1)
    Q(arr, 'bar')
    # [0.0 0.02040816326530612 0.04081632653061224 ... 0.9795918367346939 1.0] bar



Series objects are also converted to ``ndarray`` before constructing the quantity, which means that all metadata is removed.


.. code-block:: python

    import pandas as pd

    s = pd.Series(arr, name='series_name')

    pressure_ = Q(s, 'bar') # Series is converted to np.ndarray


Combining quantities
~~~~~~~~~~~~~~~~~~~~

The result from operations on quantities will always match the input dimensionalities.
Descriptive errors are raised in case of inconsistent or ambiguous operations.
In some cases, units will not cancel out automatically.
Call ``to_base_units()`` to simplify the quantity to base SI units, or ``to()`` in case the desired unit is known.

.. code-block:: python

    (Q(5, '%') * Q(1, 'meter')).to('mm') # 50.0 mm

Operations with temperature units can lead to unexpected results.
When using degree units, a temperature *difference* can be defined with the prefix ``delta_``.
This is only required when defining the temperature difference directly.


.. code-block:: python

    dT = Q(5, 'delta_degC') # 5 Δ°C
    dT.to('degC') # -268.15 °C, same as converting 5 K to °C

    # 5°C is converted to 278.15 K before multiplying
    Q(4.19, 'kJ/kg/K') * Q(5, '°C') # 1165.4485 kJ/kg

    # the degree step for °C is equal to 1 K
    Q(4.19, 'kJ/kg/K') * Q(5, 'delta_degC') # 20.95 kJ Δ°C/(K kg)
    Q(4.19, 'kJ/kg/K') * Q(5, 'K') # 20.95 kJ/kg

    # the units Δ°C and K don't cancel out automatically
    (Q(4.19, 'kJ/kg/K') * Q(5, 'delta_degC')).to('kJ/kg') # 20.95 kJ/kg

.. tip::

    To raise an error (for example ``pint.errors.OffsetUnitCalculusError``) when doing ambiguous unit conversions, set the environment variable ``ENCOMP_AUTOCONVERT_OFFSET_TO_BASEUNIT`` to ``0``.
    See :py:class:`encomp.settings.Settings` for instructions on how to set global configuration parameters.





The Fluid class
---------------

The :py:class:`encomp.fluids.Fluid` class represents a fluid at a fixed point.
The parent class :py:class:`encomp.fluids.CoolPropFluid` implements an interface to CoolProp.
All inputs and outputs are :py:class:`encomp.units.Quantity` instances.

To create a new instance, pass the CoolProp fluid name and the fixed point to the class constructor.
The documentation for the parent class :py:class:`encomp.fluids.CoolPropFluid` contains a list of fluid and property names.
All combinations of input parameters are not valid -- in case of incorrect inputs a ``ValueError`` is raised when evaluating the attribute(s).
The ``__repr__`` of the instance will show ``N/A`` instead of raising an error.


.. code-block:: python

    from encomp.fluids import Fluid

    Fluid('toluene', T=Q(25, '°C'), P=Q(2, 'bar'))
    # <Fluid "toluene", P=200 kPa, T=25.0 °C, D=862.3 kg/m³, V=0.55 cP>

    # PCRIT cannot be used to fix the state
    invalid_inputs = Fluid('water', D=Q(500, 'kg/m³'), PCRIT=Q(1, 'bar'))
    # <Fluid "water", P=N/A, T=N/A, D=N/A, V=N/A>

    # try to access the attribute "T" (temperature)
    invalid_inputs.T
    # ValueError: Input pair variable is invalid and output(s) are non-trivial; cannot do state update : PropsSI("T","D",500,"PCRIT",100000,"water")


If the convenience class :py:class:`encomp.fluids.Water` is used, the fluid name can be omitted.
:py:class:`encomp.fluids.Water` uses ``IAPWS-95``.
To use ``IAPWS-97`` instead, create an instance of :py:class:`encomp.fluids.Fluid` with name ``IF97::Water``.
The :py:class:`encomp.fluids.HumidAir` class has a different set of input and output properties.

.. code-block:: python

    from encomp.fluids import Water, HumidAir

    # input units are converted to SI
    Water(D=Q(12, 'lbs / ft³'), T=Q(250, '°F'))
    # <Water (Two-phase), P=206 kPa, T=121.1 °C, D=192.2 kg/m³, V=0.0 cP, Q=0.00>

    HumidAir(T=Q(25, 'C'), P=Q(2, 'bar'), R=Q(25, '%'))
    # <HumidAir, P=200 kPa, T=25.0 °C, R=0.25, Vda=0.4 m³/kg, Vha=0.4 m³/kg, M=0.018 cP>



The exact names used by CoolProp must be used.
Note that these are different for humid air.

.. code-block:: python

    HumidAir(T=Q(25, 'C'), Ps=Q(2, 'bar'), R=Q(25, '%'))
    # ValueError: Invalid CoolProp property name: Ps
    # Valid names:
    # B, C, CV, CVha, Cha, Conductivity, D, DewPoint, Enthalpy, Entropy, H, Hda, Hha,
    # HumRat, K, M, Omega, P, P_w, R, RH, RelHum, S, Sda, Sha, T, T_db, T_dp, T_wb, Tdb,
    # Tdp, Twb, V, Vda, Vha, Visc, W, WetBulb, Y, Z, cp, cp_ha, cv_ha, k, mu, psi_w


Use the ``search()`` and ``describe()`` methods to get more information about the properties:


.. code-block:: python

    HumidAir.search('bulb')
    # ['B, Twb, T_wb, WetBulb: Wet-Bulb Temperature [K]',
    #  'T, Tdb, T_db: Dry-Bulb Temperature [K]']

    Fluid.describe('Z')
    # 'Z: Compressibility factor [dimensionless]'


All property synonyms are valid instance attributes:


.. code-block:: python

    Water.describe('PCRIT')
    # 'PCRIT, P_CRITICAL, Pcrit, p_critical, pcrit: Pressure at the critical point [Pa]'

    water = Water(T=Q(25, '°C'), P=Q(1, 'atm'))

    water.p_critical, water.PCRIT
    # (22064000.0 <Unit('pascal')>, 22064000.0 <Unit('pascal')>)


.. note::

    The instance attributes don't show up when calling ``dir(fluid_instance)``, since
    they are only evaluated as needed (using the :py:meth:`encomp.fluids.CoolPropFluid.get` method).



Using multiple inputs
~~~~~~~~~~~~~~~~~~~~~

The CoolProp library supports vector inputs, which means that multiple inputs can be evaluated at the same time.
The inputs must be instances of :py:class:`encomp.units.Quantity` with one-dimensional Numpy arrays as magnitude.
All inputs must be the same length (or a single value).


.. code-block:: python

    Water(T=Q(np.linspace(25, 50, 10), '°C'),
          P=Q(np.linspace(25, 50, 10), 'bar'))
    # <Water (Liquid), P=[2500 2778 3056 3333 3611 3889 4167 4444 4722 5000] kPa,
    # T=[25.0 27.8 30.6 33.3 36.1 38.9 41.7 44.4 47.2 50.0] °C,
    # D=[998.1 997.5 996.8 996.0 995.2 994.3 993.3 992.3 991.3 990.2] kg/m³,
    # V=[0.9 0.8 0.8 0.7 0.7 0.7 0.6 0.6 0.6 0.5] cP>

    # different phases
    Water(T=Q(np.linspace(25, 500, 10), '°C'),
          P=Q(np.linspace(0.5, 10, 10), 'bar')).PHASE
    # [0.0 0.0 5.0 5.0 5.0 5.0 5.0 2.0 2.0 2.0]

    Water.PHASES
    # {0.0: 'Liquid',
    #  5.0: 'Gas',
    #  6.0: 'Two-phase',
    #  3.0: 'Supercritical liquid',
    #  2.0: 'Supercritical gas',
    #  1.0: 'Supercritical fluid',
    #  8.0: 'Not imposed'}

    # if one input is constant, it's converted to an array
    Water(T=Q(np.linspace(25, 500, 10), '°C'),
          P=Q(5, 'bar'))
    # <Water (Variable), P=[500 500 500 500 500 500 500 500 500 500] kPa,
    # T=[25.0 77.8 130.6 183.3 236.1 288.9 341.7 394.4 447.2 500.0] °C,
    # D=[997.2 973.3 934.5 2.5 2.2 2.0 1.8 1.6 1.5 1.4] kg/m³,
    # V=[0.9 0.4 0.2 0.0 0.0 0.0 0.0 0.0 0.0 0.0] cP>




Sympy functionality
-------------------

To load additional methods for the ``sympy.Symbol`` class, import Sympy via the :py:mod:`encomp.sympy` module.


.. code-block:: python

    from encomp.sympy import sp


Typesetting
~~~~~~~~~~~

The following convenience methods are added to the ``sp.Symbol`` class:

* ``sp.Symbol._()``: add subscript
* ``sp.Symbol.__()``: add superscript
* ``sp.Symbol.decorate()``: add sub- and superscript prefixes and suffixes (:py:func:`encomp.sympy.decorate`)

These methods return new instances of ``sp.Symbol`` with the same assumptions (i.e. *positive*, *real*, *integer*, etc...) as the original instance.


.. code-block:: python

    n = sp.Symbol('n', integer=True)

    n_test = n._('test')
    str(n_test)
    # n_{\\text{test}}

    n_test.assumptions0['integer'] # True

.. tip::

    The assumptions for a ``sp.Symbol`` are accessed with the attribute ``assumptions0`` (note the ``0`` at the end).


The ``_`` and ``__`` methods will typeset the sub- and superscripts automatically:

* Single-letter lower case with math font: ``n._('a')`` → :math:`n_a`
* Single-letter upper case with regular font: ``n._('A')`` → :math:`n_{\text{A}}`
* Chemical formulas: ``n._('H_2O')`` → :math:`n_{\text{H}_2\text{O}}`
* Strings with two or more characters with regular font: ``n._('water')`` → :math:`n_{\text{water}}`
* Parts are split with ``,``: ``n._('outlet,A,i,H_2SO_4')`` → :math:`n_{\text{outlet},\text{A},i,\text{H}_2\text{SO}_4}`
* Combine sub- and superscript: ``n._('a').__('in')`` → :math:`n_{a}^{\text{in}}`


The ``decorate`` method offers more control:

* ``n.decorate(prefix='\sum', prefix_sub='2', suffix_sup='i', suffix='\ldots')`` → :math:`{\sum}_{2}n^{i}{\ldots}`


Integration with quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantities can be used when evaluating Sympy expressions.
The units will be converted to Sympy symbols automatically.
The class method :py:meth:`encomp.units.Quantity.from_expr` is used to convert an expression back to a quantity.


.. code-block:: python

    x, y, z = sp.symbols('x, y, z')

    expr = 25 * x * y / z

    result_expr = expr.subs({
        x: Q(235, 'yard'),
        y: Q(2, 'm²'),
        z: Q(0.4, 'm³/kg')
    })

    result_qty = Q.from_expr(result_expr)
    # 26860.5 kg


:py:meth:`encomp.units.Quantity.from_expr` will raise ``KeyError`` in case there is a symbol in the expression that is not an SI unit.
This does not work with user-defined dimensionalities.


In case the magnitude of a quantity is a Numpy array, :py:meth:`encomp.units.Quantity.from_expr` does not work.
The expression must instead be converted to a function with :py:func:`encomp.sympy.get_function`:


.. code-block:: python

    from encomp.sympy import get_function

    x, y, z = sp.symbols('x, y, z')

    expr = 25 * x * y / z

    # units=False by default, since this is faster to evaluate
    fcn = get_function(expr, units=True)

    result_qty = fcn({
        x: Q(np.array([235, 335]), 'yard'),
        y: Q([2, 5], 'm²'), # regular lists will be converted to array
        z: Q(0.4, 'm³/kg')
    })
    # [26860.5 95726.25] kg

