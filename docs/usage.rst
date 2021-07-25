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



Using multiple magnitudes
~~~~~~~~~~~~~~~~~~~~~~~~~

Numpy arrays and Pandas series objects can also be used as magnitude.
Series objects are converted to ``ndarray`` before constructing the quantity, which means that all metadata is removed.


.. code-block:: python

    import numpy as np
    import pandas as pd

    arr = np.linspace(0, 1)
    s = pd.Series(arr, name='series_name')

    pressure = Q(arr, 'bar')
    pressure_ = Q(s, 'bar') # Series is converted to np.ndarray


Combining quantities
~~~~~~~~~~~~~~~~~~~~

The result from operations on quantities will always match the input dimensionalities.
Descriptive errors are raised in case of inconsistent or ambiguous operations.
In some cases, units will not cancel out automatically.
Call ``to_base_units()`` to simplify the quantity to base SI units, or ``to()`` in case the desired unit is known.

.. code-block:: python

    (Q(5, '%') * Q(1, 'meter')).to('mm') # 50.0 mm

Operations with temperature units can easily lead to ambiguous results.
When using degree units, a temperature *difference* can be defined with the prefix ``delta_``.
This is only required when defining the temperature difference directly.


.. code-block:: python

    dT = Q(5, 'delta_degC') # 5 Δ°C
    dT.to('degC') # -268.15 °C, same as converting 5 K to °C

    # 5°C is converted to 278.15 K before multiplying
    Q(4.19, 'kJ/kg/K') * Q(5, '°C') # 1165.4485 kJ/kg

    # the degree step for °C is equal to 1 K
    Q(4.19, 'kJ/kg/K') * Q(5, 'delta_degC') # 20.95 kJ Δ°C/(K kg)
    Q(4.19, 'kJ/kg/K') * Q(5, 'K') # 20.95 kJ Δ°C/(K kg)

    # the units Δ°C and K don't cancel out automatically
    (Q(4.19, 'kJ/kg/K') * Q(5, 'K')).to('kJ/kg') # 20.95 kJ/kg

.. tip::

    To raise an error (for example ``pint.errors.OffsetUnitCalculusError``) when doing ambiguous unit conversions, set the environment variable ``ENCOMP_AUTOCONVERT_OFFSET_TO_BASEUNIT`` to ``0``.
    See :py:class:`encomp.settings.Settings` for instructions on how to set global configuration parameters.





The Fluid class
---------------

The :py:class:`encomp.fluids.Fluid` class represents a fluid at a fixed point.
All inputs and outputs are :py:class:`encomp.units.Quantity` instances.
