"""
Contains settings used elsewhere in the library.
"""

from typing import Literal
from pathlib import Path
from pydantic import BaseSettings, DirectoryPath, FilePath
from dotenv import load_dotenv, find_dotenv

# find the first file named ".env" in the current directory or a parent directory
load_dotenv(dotenv_path=find_dotenv(filename='.env'))

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

    * ``DATA_DIRECTORY``: path to a directory with auxiliary data
    * ``UNITS``: path to a file with unit definitions for ``pint``
    * ``ADDITIONAL_UNITS``: path to a file with additional unit definitions for ``pint``
    * ``TYPESET_SYMBOL_SCRIPTS``: whether to typeset Sympy symbol sub- and superscripts
    * ``IGNORE_NDARRAY_UNIT_STRIPPED_WARNING``: whether to suppress the ``pint`` warning
      when converting Quantity to Numpy array.
    * ``MATPLOTLIB_NOTEBOOK_FORMAT``: figure format for Matplotlib figures in Jupyter Notebooks
    * ``AUTOCONVERT_OFFSET_TO_BASEUNIT``: whether to automatically convert offset units in calculations. If this is False, °C must be converted to K before multiplication (for example)
    * ``DEFAULT_UNIT_FORMAT``: default unit format for ``Quantity`` objects: one of ``~P`` (compact), ``~L`` (Latex), ``~H`` (HTML), ``~Lx`` (Latex with SIUNITX package)

    .. note::
        All names are case-insensitive.

    """

    data_directory: DirectoryPath = ENCOMP_BASE / 'data'

    units: FilePath = data_directory / 'units.txt'
    additional_units: FilePath = data_directory / 'additional-units.txt'

    typeset_symbol_scripts: bool = True
    ignore_ndarray_unit_stripped_warning: bool = True

    matplotlib_notebook_format: Literal['retina', 'png', 'svg'] = 'retina'
    autoconvert_offset_to_baseunit: bool = False
    default_unit_format: Literal['~P', '~L', '~H', '~Lx'] = '~P'

    class Config:
        env_prefix = 'ENCOMP_'
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False


# the settings object is initialized the first time the library loads
# settings can be changed during runtime by setting attributes on this instance
SETTINGS = Settings()
