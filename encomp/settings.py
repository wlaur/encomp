"""
Contains settings used elsewhere in the library.
"""

from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict

# find the first file named ".env" in the current directory or a parent directory
load_dotenv(dotenv_path=find_dotenv(filename=".env"))

ENCOMP_BASE = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    """
    Settings class.

    Use an ``.env``-file to override the defaults.
    The ``.env``-file is located using ``dotenv.find_dotenv(filename='.env')``, this will find a
    file in the directory of the running Python process or in a parent directory.

    The variables in the ``.env``-file have the same names (not case-sensitive)
    as the attributes of this class, with the additional prefix ``ENCOMP_``.
    In case of invalid values in the ``.env``-file or environment variables,
    a ``ValidationError`` is raised.

    .. note::

        Names that are defined as global environment variables (either on the system
        or user level) take precedence over names in the ``.env``-file.
        The global environment variables are loaded even if no ``.env``-file was found.

    * ``UNITS``: path to a file with unit definitions for ``pint``
    * ``TYPESET_SYMBOL_SCRIPTS``: whether to typeset Sympy symbol sub- and superscripts
    * ``IGNORE_NDARRAY_UNIT_STRIPPED_WARNING``: whether to suppress the ``pint`` warning
      when converting Quantity to Numpy array.
    * ``IGNORE_COOLPROP_WARNINGS``: whether to suppress warnings from the CoolProp backend
    * ``AUTOCONVERT_OFFSET_TO_BASEUNIT``: whether to automatically convert offset units in calculations.
      If this is False, °C must be converted to K before multiplication (for example)
    * ``DEFAULT_UNIT_FORMAT``: default unit format for ``Quantity`` objects:
      one of ``~P`` (compact), ``~L`` (Latex), ``~H`` (HTML), ``~Lx`` (Latex with SIUNITX package)

    .. note::
        All names are case-insensitive.

    """

    units: FilePath = ENCOMP_BASE / "defs/units.txt"

    typeset_symbol_scripts: bool = True
    ignore_ndarray_unit_stripped_warning: bool = True
    ignore_coolprop_warnings: bool = True

    autoconvert_offset_to_baseunit: bool = False
    default_unit_format: Literal["~P", "~L", "~H", "~Lx"] = "~P"

    model_config = SettingsConfigDict(
        env_prefix="ENCOMP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# the settings object is initialized the first time the library loads
# settings can be changed during runtime by setting attributes on this instance
SETTINGS = Settings()
