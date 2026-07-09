from typing import Any, cast

import pytest
from pydantic import ValidationError

from ..settings import PACKAGE_ROOT, PINT_FORMATTING_SPECIFIERS, SETTINGS, Settings

# `_env_file` is a pydantic-settings init kwarg that its typed signature does not expose.
# Passing None ignores any .env in the current directory, so these tests do not depend on it.
_Settings: Any = Settings


def test_SETTINGS() -> None:
    assert isinstance(SETTINGS, Settings)
    assert isinstance(SETTINGS.autoconvert_offset_to_baseunit, bool)


def test_package_root_contains_the_definition_files() -> None:
    assert (PACKAGE_ROOT / "defs" / "units.txt").is_file()
    assert (PACKAGE_ROOT / "defs" / "constants.txt").is_file()


def test_formatting_specifiers() -> None:
    assert PINT_FORMATTING_SPECIFIERS == ("~P", "~L", "~H", "~Lx")
    assert SETTINGS.default_unit_format in PINT_FORMATTING_SPECIFIERS


def test_defaults() -> None:
    settings = _Settings(_env_file=None)

    assert settings.units == PACKAGE_ROOT / "defs/units.txt"
    assert settings.typeset_symbol_scripts
    assert settings.ignore_ndarray_unit_stripped_warning
    assert settings.ignore_coolprop_warnings
    assert not settings.autoconvert_offset_to_baseunit
    assert settings.default_unit_format == "~P"


def test_env_vars_override_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # names carry the ENCOMP_ prefix and are case-insensitive
    monkeypatch.setenv("ENCOMP_DEFAULT_UNIT_FORMAT", "~L")
    monkeypatch.setenv("encomp_ignore_coolprop_warnings", "false")

    settings = _Settings(_env_file=None)

    assert settings.default_unit_format == "~L"
    assert not settings.ignore_coolprop_warnings


def test_unknown_env_vars_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENCOMP_NOT_A_SETTING", "1")

    _Settings(_env_file=None)


def test_invalid_values_raise() -> None:
    with pytest.raises(ValidationError):
        Settings(default_unit_format=cast(Any, "bogus"))

    with pytest.raises(ValidationError):
        # FilePath requires the file to exist
        Settings(units=cast(Any, "/does/not/exist.txt"))
