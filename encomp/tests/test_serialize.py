import json
import numpy as np
import pandas as pd
from decimal import Decimal
from uncertainties import ufloat
from dataclasses import dataclass

from encomp.units import Q
from encomp.serialize import serialize, decode, is_serializable


def test_serialize():

    P1 = Q(1, 'bar')

    ds = [
        {'key': P1},
        {'key': list(np.linspace(P1, P1 * 20))}  # np.array does not support ==
    ]

    for d in ds:
        s = serialize(d)

        assert is_serializable(s)

        json_str = json.dumps(s)
        d_ = decode(json.loads(json_str))

        assert d == d_

    d = Q(np.linspace(0, 1), 'kg/s')

    s = serialize(d)

    assert is_serializable(s)

    json_str = json.dumps(s)
    d_ = decode(json.loads(json_str))

    assert np.array_equal(d_.m, d.m)
    assert d_.dimensionality == d.dimensionality

    qty = Q(Decimal('1.123'), 'kg') * 100
    s = serialize(qty)
    json_str = json.dumps(s)
    d_ = decode(json.loads(json_str))

    qty = Q([Decimal('1.123'), Decimal('1.125')], 'kg') * 100
    s = serialize(qty)
    json_str = json.dumps(s)
    d_ = decode(json.loads(json_str))

    qty = Q(np.array([Decimal('1.123'), Decimal('1.125')]), 'kg') * 100
    s = serialize(qty)
    json_str = json.dumps(s)
    d_ = decode(json.loads(json_str))

    qty = Q(np.zeros((5, 5)), 'kg')
    s = serialize(qty)
    json_str = json.dumps(s)
    d_ = decode(json.loads(json_str))

    x = ufloat(1, 0.1)

    qty = Q([x * 2] * 5, 'kg')
    s = serialize(qty)
    json_str = json.dumps(s)
    d_ = decode(json.loads(json_str))

    qty = Q({1, 2}, 'kg')
    s = serialize(qty)
    json_str = json.dumps(s)
    d_ = decode(json.loads(json_str))

    # pd.Series is converted to array in Quantity.__new__, the name is stripped
    qty = Q(pd.Series(np.array([1, 2, 3]), name='test'), 'kg')
    s = serialize(qty)
    json_str = json.dumps(s)
    d_ = decode(json.loads(json_str))


def test_custom_object():

    @dataclass
    class CustomOtherClass:

        s: pd.Series
        df: pd.DataFrame

        @classmethod
        def from_dict(cls, d):
            return cls(**d)

        def to_json(self):

            # str repr of JSON
            return json.dumps(serialize(self.__dict__))

    @dataclass
    class CustomClass:

        arr: np.ndarray
        name: str

        nested: CustomOtherClass
        nested_list: list[CustomOtherClass]

        @classmethod
        def from_dict(cls, d):
            return cls(**d)

        def to_json(self):

            # dict repr of JSON, does not have
            # to be serializable
            return self.__dict__

    s = pd.Series(np.random.rand(10), name='test')
    df = pd.DataFrame(np.random.rand(5, 5), columns=['A', 2, '3', '4', '5'])

    b = CustomOtherClass(s=s, df=df)

    a = CustomClass(arr=np.random.rand(19, 2), name='asd',
                    nested=b,
                    nested_list=[b, b, b])

    json_dict = serialize(a)

    a_decoded = decode(json_dict, custom=[CustomClass, CustomOtherClass])

    # float str repr will only include ~10 decimals
    assert (a_decoded.nested.df - df).max().max() < 1e-10
    assert isinstance(a_decoded.nested, CustomOtherClass)
    assert isinstance(a_decoded.nested_list[0], CustomOtherClass)
    assert isinstance(a_decoded.nested_list[-1], CustomOtherClass)
