"""
Contains settings used elsewhere in the library.
"""

from pathlib import Path
from typing import Literal, get_args

from pydantic import FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).parent.resolve()

PintFormattingSpecifier = Literal["~P", "~L", "~H", "~Lx"]
PINT_FORMATTING_SPECIFIERS = get_args(PintFormattingSpecifier)

__all__ = ["PACKAGE_ROOT", "PINT_FORMATTING_SPECIFIERS", "SETTINGS", "PintFormattingSpecifier", "Settings"]


class Settings(BaseSettings):
    """
    Settings class.

    Use an ``.env``-file in the current working directory to override the
    defaults; ``pydantic-settings`` loads it automatically when this module is
    first imported (no manual loading needed).

    The variables in the ``.env``-file have the same names (not case-sensitive)
    as the attributes of this class, with the additional prefix ``ENCOMP_``.
    In case of invalid values in the ``.env``-file or environment variables,
    a ``ValidationError`` is raised.

    .. note::

        Names that are defined as global environment variables (either on the system
        or user level) take precedence over names in the ``.env``-file.
        The global environment variables are loaded even if no ``.env``-file was found.

    .. warning::

        Because the ``.env``-file is resolved relative to the *current working
        directory*, a stray ``.env`` containing an invalid ``ENCOMP_*`` value
        (e.g. ``ENCOMP_UNITS`` pointing to a missing file) makes ``import encomp``
        fail with a ``ValidationError`` -- even in an unrelated project.

    * ``UNITS``: path to a file with unit definitions for ``pint``
    * ``TYPESET_SYMBOL_SCRIPTS``: whether to typeset SymPy symbol sub- and superscripts
    * ``IGNORE_NDARRAY_UNIT_STRIPPED_WARNING``: whether to suppress the ``pint`` warning
      when converting Quantity to NumPy array.
    * ``IGNORE_COOLPROP_WARNINGS``: whether to suppress warnings
      from the CoolProp backend
    * ``AUTOCONVERT_OFFSET_TO_BASEUNIT``: whether to automatically convert
      offset units in computations.
      If this is False, °C must be converted to K before multiplication (for example)
    * ``DEFAULT_UNIT_FORMAT``: default unit format for ``Quantity`` objects:
      one of ``~P`` (compact), ``~L`` (Latex), ``~H`` (HTML),
      ``~Lx`` (Latex with SIUNITX package)

    .. note::
        All names are case-insensitive.

    Environment-backed settings are loaded once when this module is imported.
    Some values are consumed during registry initialization, so assigning new
    values to :data:`SETTINGS` is not a general runtime configuration API. Use
    :func:`encomp.units.set_quantity_format` to change quantity/unit rendering
    for the current process.

    """

    units: FilePath = PACKAGE_ROOT / "defs/units.txt"

    typeset_symbol_scripts: bool = True
    ignore_ndarray_unit_stripped_warning: bool = True
    ignore_coolprop_warnings: bool = True

    autoconvert_offset_to_baseunit: bool = False
    default_unit_format: PintFormattingSpecifier = "~P"

    model_config = SettingsConfigDict(
        env_prefix="ENCOMP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


SETTINGS = Settings()
"""Singleton :class:`Settings` instance, initialized the first time the library loads.
Some consumers read values during their own import/initialization, so attribute
assignment on this instance is not a general runtime-configuration API."""
