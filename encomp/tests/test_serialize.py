import json
import numpy as np
from decimal import Decimal
from uncertainties import ufloat

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
