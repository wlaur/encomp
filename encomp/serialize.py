"""
Functions related to serializing and decoding (deserializing)
various objects to and from JSON.

The general structure for custom object classes is

    .. code-block:: python

        from typing import Union
        import json

        from encomp.serialize import serialize, decode

        class CustomClass:

            def __init__(self, A=None, B=None):
                self.A = A
                self.B = B

            @classmethod
            def from_dict(cls, d: dict) -> 'CustomClass':
                return cls(**d)

            def to_json(self) -> dict | str:
                d = {'A': self.A, 'B': self.B}

                # return JSON string or dict
                # return json.dumps(d)
                return d

            @property
            def json(self) -> dict:
                return self.to_json()

        obj = CustomClass()

        json_repr = serialize(obj)
        # {'type': 'CustomClass', 'data': {'A': 1, 'B': 2}}

        # construct a new object from the serialized representation
        obj_ = decode(json_repr, custom=CustomClass)

The custom class must contain a method named ``to_json()`` or a property named
``json`` that returns a JSON-serializable ``dict`` (or a string representation).


To decode a serialized custom object, a method named ``from_dict()`` must be defined.
Additionally, the (uninitialized) class implementation (or a list of classes) must be passed to the
:py:func:`encomp.serialize.decode` function with the parameter ``custom``.
The class name is used as the ``type`` key when serializing.
This same name must be used when passing the class implementation for decoding.

.. todo::

    Handle timestamps, datetime objects etc., also as pandas index.

"""

import inspect
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sympy as sp
from uncertainties import ufloat
from uncertainties.core import AffineScalarFunc

from .misc import isinstance_types
from .units import Quantity, Unit
from .utypes import Dimensionality

# type alias for objects that can be serialized using json.dumps()
JSONBase = dict | list | float | int | str | bool | None

JSON = JSONBase | list[JSONBase] | dict[str, JSONBase]


def is_serializable(x: Any) -> bool:
    """
    Checks if x is serializable to JSON,
    tries to execute ``json.dumps(x)``.

    Parameters
    ----------
    x : Any
        Object to check

    Returns
    -------
    bool
        Whether the object is serializable
    """

    try:
        json.dumps(x)
        return True

    except Exception:
        return False


def serialize(obj: Any) -> JSON:
    """
    Converts the input to a serializable object that can be rendered as JSON.
    Assumes that all non-iterable objects are serializable,
    raises an exception otherwise.
    All tuples are converted to lists, tuples cannot be used as dict keys.
    Uses recursion to handle nested structures, this will raise RecursionError in
    case a dict contains cyclical references.

    Custom objects are serialized with :py:func:`encomp.serialize.custom_serializer`.

    Parameters
    ----------
    obj : Any
        Input object

    Returns
    -------
    JSON
        Serializable representation of the input that
        preserves the nested structure.
    """

    if isinstance(obj, str):
        return obj

    # tuple is converted to list, no way to
    # convert back to tuple when decoding
    # tuples should not be used with objects that are serialized
    if isinstance(obj, tuple):
        obj = list(obj)

    # only allow dict or list containers
    if isinstance(obj, list):
        lst = []

        for n in obj:
            lst.append(serialize(n))

        return lst

    if isinstance(obj, dict):
        dct = {}

        for key in obj:
            if isinstance(key, tuple):
                raise TypeError(
                    "Tuples cannot be used as dictionary keys when "
                    f"serializing to JSON: {key}"
                )

            dct[key] = serialize(obj[key])

        return dct

    return custom_serializer(obj)


def custom_serializer(obj: Any) -> JSON:
    """
    Serializes objects that are not JSON-serializable by default.
    Fallback is ``obj.__dict__``, if this does not exist
    the input object is returned unchanged.
    Numpy arrays and pandas DataFrames are handled separately.

    Custom classes can use a method named ``to_json()`` or
    a property named ``json`` that returns a ``dict`` or
    a string representation of a ``dict``.
    The class name is stored in the ``type`` attribute in the output JSON.

    If these attributes are not found, ``__dict__`` or ``str()`` is used.

    Parameters
    ----------
    obj : Any
        Input object

    Returns
    -------
    JSON
        Serializable representation of the input that
        preserves the nested structure.
    """

    if isinstance(obj, Path):
        return {"type": "Path", "data": str(obj.absolute())}

    if isinstance(obj, Quantity):
        return {
            "type": "Quantity",
            "dimensionality": obj._dimensionality_type.__name__,
            "data": [serialize(obj.m), str(obj.u._units)],
        }

    if isinstance(obj, pd.Series):
        return {
            "type": "Series",
            "data": obj.to_json(
                orient="split", default_handler=custom_serializer  # type: ignore
            ),
        }

    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "data": obj.to_json(
                orient="split", default_handler=custom_serializer  # type: ignore
            ),
        }

    if isinstance(obj, np.ndarray):
        return {"type": "ndarray", "data": [serialize(x) for x in obj.tolist()]}

    if isinstance(obj, Decimal):
        return {"type": "Decimal", "data": str(obj)}

    if isinstance(obj, AffineScalarFunc):
        return {"type": "AffineScalarFunc", "data": [obj.nominal_value, obj.std_dev]}

    if isinstance(obj, sp.Basic):
        return {"type": "Sympy", "data": sp.srepr(obj)}

    # method named "to_json" or @property named "json"
    if hasattr(obj, "to_json") or hasattr(obj, "json"):
        # this method can return a dict or a string
        # JSON representation
        if hasattr(obj, "json"):
            json_repr = getattr(obj, "json")
        else:
            json_repr = obj.to_json()

        # make sure the object's dict representation is serializable
        # if this is a string this is not necessary
        if not isinstance(json_repr, str):
            json_repr = serialize(json_repr)

        obj_type = obj.__class__.__name__

        # the custom class must implement classmethod
        # from_dict, which takes json_repr as input and
        # returns a class instance
        return {"type": obj_type, "data": json_repr}

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    if is_serializable(obj):
        return obj

    # fallback, this cannot be deserialized
    return str(obj)


def decode(inp: JSON, custom: type | list[type] | None = None) -> Any:
    """
    Decodes objects that were serialized
    with :py:func:`encomp.serialize.custom_serializer`.
    Custom classes can be constructed from an optional
    class constructor parameter.

    .. warning::

        Dictionary keys are always strings in JSON.
        This function cannot determine if the key ``"2.0"``
        was originally a string or float, all keys in
        the returned object are strings.

    Parameters
    ----------
    inp : JSON
        Serialized representation of an object
    custom : type | list[type] | None
        Potential custom class implementation(s). The class name
        is stored as the ``type`` key in the input JSON (if the object
        was serialized with :py:func:`encomp.serialize.custom_serializer`).

        .. note::
            The exact class names that were used for serializing
            must be used when decoding. The ``obj.__class__.__name__``
            attribute is used, make sure to use unique class names.

    Returns
    -------
    Any
        The decoded object
    """

    if custom is None:
        custom = []

    if not isinstance(custom, list):
        custom = [custom]

    # nested list (cannot be tuple, since JSON does not support tuples)
    if isinstance(inp, list):
        return [decode(n, custom=custom) for n in inp]

    if isinstance(inp, dict):
        # serialized pint.Quantity with array as magnitude
        # NOTE: this is for compatibility with units that are
        # serialized directly from pint (i.e. without the encomp.units extension)
        if {"_units", "_magnitude"} <= set(inp):
            # decode custom np.array objects
            # not necessary to pass on the custom kwarg,
            # the Quantity magnitude cannot be a custom class
            m = decode(inp["_magnitude"])
            units: str = inp["_units"]  # type: ignore
            return Quantity(m, units)

        # check if this dict is output from custom_serializer
        # it might also be a regular dict that just happens to
        # have the key "type", in this case it will be decoded normally
        if "type" in inp:
            if inp["type"] == "Quantity":
                # optional key with the name of the dimensionality class
                dimensionality_name = inp.get("dimensionality")
                val, unit = inp["data"]

                if unit is None:
                    unit = ""

                dimensionality: type[Dimensionality] | None = None

                for d in Dimensionality._registry:
                    if d.__name__ == dimensionality_name:
                        dimensionality = d
                        break

                val = decode(val)

                # check if this list has types that matches a serialized Quantity
                if isinstance_types(unit, Unit | str):
                    if dimensionality is None:
                        return Quantity(val, unit)

                    else:
                        return Quantity[dimensionality](val, unit)  # type: ignore

            if inp["type"] == "Path":
                return Path(inp["data"])  # type: ignore

            if inp["type"] == "Series":
                return pd.read_json(inp["data"], typ="series", orient="split")

            if inp["type"] == "DataFrame":
                return pd.read_json(inp["data"], typ="frame", orient="split")

            if inp["type"] == "ndarray":
                # not necessary to pass on the custom kwarg
                data = [decode(x) for x in inp["data"]]  # type: ignore
                return np.array(data)

            if inp["type"] == "Decimal":
                return Decimal(inp["data"])  # type: ignore

            if inp["type"] == "AffineScalarFunc":
                return ufloat(*inp["data"])

            if inp["type"] == "Sympy":
                return sp.sympify(inp["data"])

            # load custom classes, based on list of classes
            custom_dict = {n.__name__: n for n in custom}

            if inp["type"] in custom_dict:
                custom_class = custom_dict[inp["type"]]  # type: ignore

                if not hasattr(custom_class, "from_dict"):
                    # want to raise an error here
                    # implementation is incorrect if this method is missing
                    raise AttributeError(
                        f"Custom class {custom_class} "
                        'must contain a classmethod named "from_dict" '
                        "that takes a dict as input and returns a class instance"
                    )

                d = inp["data"]  # type: ignore

                # in case the to_json() method returns a string,
                # the string must be loaded into a dict
                # it's not possible to have a non-JSON string here,
                # a custom object cannot serialize to a single string value
                if isinstance(d, str):
                    d = json.loads(d)

                # in case the dict representation contains objects
                # that must be decoded
                d = decode(d, custom=custom)  # type: ignore

                return custom_class.from_dict(d)

        # it's not possible to have any custom object as key,
        # JSON allows only str (float and int are converted to str)
        # not possible to determine if string "1" was originally an int,
        # so this function will not convert numeric str to int
        # to avoid issues with this, avoid using integer / float keys
        # need to pass on the custom class list here
        return {a: decode(b, custom=custom) for a, b in inp.items()}

    # basic types should be returned as-is
    return inp


def save(names: dict[str, Any], path: str | Path = "variables.json") -> None:
    """
    Saves variables from a Jupyter Notebook session as JSON.

    Parameters
    ----------
    names : dict[str, Any]
        Names and their corresponding objects, for example output from ``locals()``
    path : str | Path
        File path or name of a JSON file, by default 'variables.json'
    """

    skip = [
        "In",
        "Out",
        "ipython",
        "get_ipython",
        "exit",
        "quit",
        "getsizeof",
        "SNS_BLUE",
        "SNS_PALETTE",
    ]

    names = {
        a: b
        for a, b in names.items()
        if not a.startswith("_")
        and not inspect.ismodule(b)
        and not isinstance(b, Callable)
        and a not in skip  # type: ignore
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(names, default=serialize, indent=4))


def load(path: str | Path = "variables.json") -> dict[str, Any]:
    """
    Load variables from a JSON file.

    Parameters
    ----------
    path : str | Path
        File path or name of a JSON file, by default 'variables.json'

    Returns
    -------
    dict[str, Any]
        Dictionary with names and their corresponding objects
    """

    with open(path, "r", encoding="utf-8") as f:
        names = json.loads(f.read())

    names_decoded = {}

    for key, serialized in names.items():
        names_decoded[key] = decode(serialized)

    return names_decoded
