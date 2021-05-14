"""
Functions related to serializing and decoding (deserializing)
objects to/from JSON.
"""

from typing import Any, Union, Callable
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sp
from decimal import Decimal
from uncertainties import ufloat
from uncertainties.core import AffineScalarFunc

from encomp.units import Quantity, Magnitude, Unit
from encomp.misc import isinstance_types

# type alias for objects that can be serialized using json.dumps()
JSONBase = Union[dict,
                 list,
                 float,
                 int,
                 str,
                 bool,
                 None]

JSON = Union[JSONBase,
             list[JSONBase],
             dict[str, JSONBase]]  # don't use int/float dict keys


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

    if isinstance(obj, tuple):
        obj = list(obj)

    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

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
                raise TypeError('Tuples cannot be used as dictionary keys when '
                                f'serializing to JSON: {key}')

            dct[key] = serialize(obj[key])

        return dct

    return custom_serializer(obj)


def decode(inp: JSON) -> Any:
    """
    Decodes objects that were serialized
    with :py:func:`encomp.serialize.custom_serializer`.

    .. note::
        Dictionary keys are always str in JSON, this function
        cannot determine if the key "1" or "2.0" was originally a string or integer.
        String keys can be converted to float/int afterwards if necessary.

    Parameters
    ----------
    inp : JSON
        Serialized representation of some object

    Returns
    -------
    Any
        The actual object
    """

    # nested list (cannot be tuple)
    if isinstance(inp, list):
        return [decode(n) for n in inp]

    if isinstance(inp, dict):

        # serialized pint.Quantity with array as magnitude
        if {'_units', '_magnitude'} <= set(inp):

            m = decode(inp['_magnitude'])
            return Quantity(m, inp['_units'])

        # check if this dict is output from custom_serializer
        # it might also be a regular dict that just happens to
        # have the key "type", in this case it will be decoded normally
        if 'type' in inp:

            if inp['type'] == 'Quantity':

                val, unit = inp['data']

                # decode custom np.array objects
                val = decode(val)

                # check if this list has types that matches a serialized Quantity
                if (isinstance_types(val, Magnitude) and
                        isinstance_types(unit, Union[Unit, str])):

                    if unit == '':
                        unit = 'dimensionless'

                    try:
                        return Quantity(val, unit)

                    except Exception:
                        pass

            if inp['type'] == 'Path':
                return Path(inp['data'])

            if inp['type'] == 'DataFrame':

                df_dict = decode(json.loads(inp['data']))
                return pd.DataFrame.from_dict(df_dict)

            if inp['type'] == 'ndarray':

                data = [decode(x) for x in inp['data']]
                return np.array(data)

            if inp['type'] == 'Decimal':
                return Decimal(inp['data'])

            if inp['type'] == 'AffineScalarFunc':
                return ufloat(*inp['data'])

            if inp['type'] == 'Sympy':
                return sp.sympify(inp['data'])

        # not possible to have a custom object as key,
        # JSON allows only str (float and int are converted to str)
        # not possible to determine if string "1" was originally an int,
        # so this function will not convert numeric str to int
        # to avoid issues with this, avoid using integer / float keys
        return {a: decode(b)
                for a, b in inp.items()}

    return inp


def custom_serializer(obj: Any) -> JSON:
    """
    Serializes objects that are not JSON-serializable by default.
    Fallback is ``obj.__dict__``, if this does not exist
    return the object unchanged.

    The general structure for custom objects is

    .. code-block::python

        class CustomClass:

            def __init__(self, A=None, B=None):
                self.A = A
                self.B = B

            @classmethod
            def from_dict(cls, d):
                return cls(**d)

            def to_json(self):
                return {'A': self.A, 'B': self.B}

        obj = CustomClass()

        custom_serializer(obj)
        # {'type': 'CustomClass', 'data': {'A': 1, 'B': 2}}

    Numpy arrays and pandas DataFrames are handled separately.

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
        return {
            'type': 'Path',
            'data': str(obj.absolute())
        }

    if isinstance(obj, Quantity):

        return {
            'type': 'Quantity',
            'data': [serialize(obj.m), str(obj.u)]
        }

    if isinstance(obj, pd.DataFrame):

        return {
            'type': 'DataFrame',
            'data': obj.to_json(default_handler=custom_serializer)
        }

    if isinstance(obj, np.ndarray):

        return {
            'type': 'ndarray',
            'data': [serialize(x) for x in obj.tolist()]
        }

    if isinstance(obj, Decimal):

        return {
            'type': 'Decimal',
            'data': str(obj)
        }

    if isinstance(obj, AffineScalarFunc):

        return {
            'type': 'AffineScalarFunc',
            'data': [obj.nominal_value, obj.std_dev]
        }

    if isinstance(obj, sp.Basic):

        return {
            'type': 'Sympy',
            'data': sp.srepr(obj)
        }

    if hasattr(obj, 'to_json'):
        return obj.to_json()

    if hasattr(obj, 'json'):
        return obj.json

    if hasattr(obj, '__dict__'):
        return obj.__dict__

    if is_serializable(obj):
        return obj

    # fallback, this cannot be deserialized
    return str(obj)


def save(names: dict[str, Any],
         path: Union[str, Path] = 'variables.json') -> None:
    """
    Saves variables from a Jupyter Notebook session as JSON.

    Parameters
    ----------
    names : dict[str, Any]
        Names and their corresponding objects, for example output from ``locals()``
    path : Union[str, Path]
        File path or name of a JSON file, by default 'variables.json'
    """

    skip = [
        'In',
        'Out',
        'ipython',
        'get_ipython',
        'exit',
        'quit',
        'getsizeof',
        'SNS_BLUE',
        'SNS_PALETTE'
    ]

    names = {
        a: b for a, b in names.items() if
        not a.startswith('_') and
        not inspect.ismodule(b) and
        not isinstance(b, Callable) and
        a not in skip
    }

    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(names,
                           default=serialize,
                           indent=4))


def load(path: Union[str, Path] = 'variables.json') -> dict[str, Any]:
    """
    Load variables from a JSON file.

    Parameters
    ----------
    path : Union[str, Path]
        File path or name of a JSON file, by default 'variables.json'

    Returns
    -------
    dict[str, Any]
        Dictionary with names and their corresponding objects
    """

    with open(path, 'r', encoding='utf-8') as f:
        names = json.loads(f.read())

    names_decoded = {}

    for key, serialized in names.items():
        names_decoded[key] = decode(serialized)

    return names_decoded
