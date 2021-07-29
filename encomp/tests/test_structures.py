from encomp.structures import flatten


def test_flatten():

    nested = [(1, 2, 3), 3, [6, 7]]

    flat = list(flatten(nested))

    assert len(flat) == 6
