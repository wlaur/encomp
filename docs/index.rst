
encomp documentation
====================

.. image:: img/logo.svg
   :width: 150px
   :align: center


Overview
---------

.. centered:: General-purpose library for *en*\gineering *com*\putations, with focus on clean and consistent interfaces.


Main functionality of the ``encomp`` library:

* Handles physical quantities with a magnitude, unit and dimensionality

   * Module ``encomp.units``, ``encomp.utypes``
   * Extends the ``pint`` library
   * Uses Python's type system to validate dimensionalities
   * Integrates with ``np.ndarray`` and ``pd.Series``
   * Automatic JSON serialization and decoding

* Implements a flexible interface to CoolProp

   * Module ``encomp.fluids``
   * Uses quantities for all inputs and outputs
   * Fluids are represented as class instances, the properties are class attributes

* Extends Sympy

   * Module ``encomp.sympy``, ``encomp.balances``
   * Adds convenience methods for creating symbols with sub- and superscripts
   * Additional functions to convert (algebraic) expressions and systems to Python code that supports Numpy arrays

* Jupyter Notebook integration

   * Module ``encomp.notebook``
   * Imports commonly used functions and classes
   * Defines custom Jupyter magics


The other modules implement calculations related to process engineering and thermodynamics.
The module ``encomp.serialize`` implements custom JSON serialization and decoding for classes used elsewhere in the library.


.. tip::

   This library should be used as a starting point when developing your own engineering calculations.
   For instance, using the ``encomp.units.Quantity`` class and decorating functions with ``@typeguard.typechecked`` will eliminate all unit-related errors in your calculations.


   ``encomp`` also serves as an overview of commonly used Python libraries for engineering and science.



Contents
--------

.. toctree::
   :maxdepth: 3

   examples

   source/modules
