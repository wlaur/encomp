from encomp.settings import SETTINGS


def test_SETTINGS():

    assert isinstance(SETTINGS.type_checking, bool)
