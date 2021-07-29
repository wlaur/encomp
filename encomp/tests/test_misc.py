from encomp.misc import name_assignments


def test_name_assignments():

    s = 'a = 5; b = [1, 2, 3]'

    assignments = name_assignments(s)

    assert len(assignments) == 2
