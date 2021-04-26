import json
import numpy as np

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
