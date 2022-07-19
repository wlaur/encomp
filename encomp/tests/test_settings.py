from encomp.settings import SETTINGS


def test_SETTINGS():

    assert isinstance(SETTINGS.autoconvert_offset_to_baseunit, bool)
