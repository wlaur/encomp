"""
Functions related to serializing and deserializing custom
objects to/from JSON.
"""

from typing import Any, Union, Dict, List
import json
from pathlib import Path
import numpy as np
import pandas as pd

from encomp.units import Q, Magnitude, Unit, check_quantity


JSON = Union[
    Dict[str, Any],
    List[dict, Any],
    float, int, str, bool, None
]


def is_serializable(x: Any) -> bool:
    """
    Checks if x is serializable to JSON:
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


def make_serializable(obj: Any) -> JSON:
    """
    Converts the input to a serializable object that can be rendered as JSON.
    Assumes that all non-iterable objects are serializable,
    raises an exception otherwise.
    All tuples are converted to lists, tuples cannot be used as dict keys.
    Uses recursion to handle nested structures, this will raise RecursionError in
    case a dict contains cyclical references.

    Custom objects are serialized with :py:func:`encomp.serialize.custom_json_serializer`.

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
            lst.append(make_serializable(n))

        return lst

    if isinstance(obj, dict):

        dct = {}

        for key in obj:

            if isinstance(key, tuple):
                raise TypeError('Tuples cannot be used as dictionary keys when '
                                f'serializing to JSON: {key}')

            dct[key] = make_serializable(obj[key])

        return dct

    return custom_json_serializer(obj)


def custom_json_serializer(obj: Any) -> JSON:
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

        custom_json_serializer(obj)
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

    if isinstance(obj, pd.DataFrame):

        return {
            'type': 'DataFrame',
            'data': obj.to_json(default_handler=custom_json_serializer)
        }

    if isinstance(obj, np.ndarray):

        return {
            'type': 'ndarray',
            'data': obj.tolist()
        }

    if hasattr(obj, 'to_json'):
        return obj.to_json()

    if hasattr(obj, 'json'):
        return obj.json

    if hasattr(obj, '__dict__'):
        return obj.__dict__

    if is_serializable(obj):
        return obj

    # fallback, this can usually not be deserialized
    return str(obj)


def custom_json_decoder(inp: JSON) -> Any:
    """
    Decodes objects that were serialized
    with :py:func:`encomp.serialize.custom_json_serializer`

    Parameters
    ----------
    inp : JSON
        Serialized representation of some object

    Returns
    -------
    Any
        The actual object
    """

    if isinstance(inp, list) and len(inp) == 2:

        # might be a Quantity with a float/int/list/array magnitude
        if isinstance(inp[1], Unit):

            unit = inp[1]

            if not unit:
                unit = 'dimensionless'

            try:
                m = custom_json_decoder(inp[0])
                return Quantity(m, unit)
            except Exception:
                pass

        # if this is a nested list with length 2
        if len(inp) < len(list(flatten(inp))):
            return [custom_json_decoder(n) for n in inp]

    # nested list (cannot be tuple)
    if isinstance(inp, list):
        return [custom_json_decoder(n) for n in inp]

    if isinstance(inp, dict):

        # serialized pint.Quantity with array as magnitude
        if '_units' in inp and '_magnitude' in inp:
            m = custom_json_decoder(inp['_magnitude'])
            return Quantity(m, inp['_units'])

        # check if a dict is a serialized dataframe or other custom object
        if 'type' in inp:

            if inp['type'] == 'Path':
                return Path(inp['data']).absolute()

            if inp['type'] == 'DataFrame':

                df_dict = custom_json_decoder(json.loads(inp['data']))
                df = pd.DataFrame.from_dict(df_dict)

                return df

            if inp['type'] == 'ndarray':
                return np.array(inp['data'])

        # not possible to have a custom object as key,
        # JSON allows only str (float and int are converted to str)
        return {a: custom_json_decoder(b)
                for a, b in inp.items()}

    return inp
