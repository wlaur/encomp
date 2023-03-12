
encomp documentation
====================

.. image:: img/logo.svg
   :width: 150px
   :align: center


Overview
---------

.. centered:: General-purpose library for *en*\gineering *com*\putations, with focus on clean and consistent interfaces.


Main functionality of the ``encomp`` library:

* Handles physical quantities with magnitude(s), dimensionality and units
   * Modules :py:mod:`encomp.units`, :py:mod:`encomp.utypes`
   * Extends the `pint <https://pypi.org/project/Pint/>`_ library
   * Uses Python's type system to validate dimensionalities
   * Compatible with ``mypy`` and other type checkers
   * Integrates with with Numpy arrays, Pandas series and Polars series and expressions
   * JSON serialization and decoding
* Implements a flexible interface to `CoolProp <http://www.coolprop.org>`_
   * Module :py:mod:`encomp.fluids`
   * Uses quantities for all inputs and outputs (including dimensionless quantities)
   * Fluids are represented as class instances, the properties are class attributes
* Extends `Sympy <https://pypi.org/project/sympy/>`_
   * Module :py:mod:`encomp.sympy`
   * Adds convenience methods for creating symbols with sub- and superscripts
   * Additional functions to convert (algebraic) expressions and systems to Python code that supports Numpy arrays# add mock imports to avoid running modules:
# autodoc_mock_imports = ['encomp.notebook', 'encomp.magics']



The other modules implement calculations related to process engineering and thermodynamics.
The module :py:mod:`encomp.serialize` implements custom JSON serialization and decoding for classes used elsewhere in the library.


.. tip::

   This library can be used as a starting point when developing your own engineering calculations.
   For instance, using the :py:class:`encomp.units.Quantity` class and decorating functions with ``@typeguard.typechecked`` will eliminate all unit-related errors in your calculations.

   ``encomp`` also serves as an overview of commonly used Python libraries for engineering and science.



Contents
--------

.. toctree::
   :maxdepth: 3

   setup
   usage
   examples

   source/encomp
