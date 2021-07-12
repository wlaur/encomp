"""
Contains settings used elsewhere in the library.
"""

from typing import Literal
from pathlib import Path
from pydantic import BaseSettings
from dotenv import load_dotenv, find_dotenv

load_dotenv(dotenv_path=find_dotenv())

ENCOMP_BASE = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    """
    Settings class.

    Use an ``.env`` file to override the defaults.
    The ``.env``-file is located using ``dotenv.find_dotenv()``, this will find a
    file in the directory of the running Python process.

    .. tip::
        The variables in the ``.env``-file have the same names (not case-sensitive)
        as the attributes of this class. Names that are defined
        as global environment variables take precedence over names in the ``.env``-file.

    * ``DATA_DIRECTORY``: path to a directory with auxiliary data
    * ``ADDITIONAL_UNITS``: path to a file with additional unit definitions for ``pint``
    * ``TYPE_CHECKING``: whether to check parameter and return value types of the core
      library function. This does not impact user-defined functions, the
      ``typeguard.typechecked`` decorator must be used explicitly
    * ``TYPESET_SYMBOL_SCRIPTS``: whether to typeset Sympy symbol sub- and superscripts
    * ``IGNORE_NDARRAY_UNIT_STRIPPED_WARNING``: whether to suppress the ``pint`` warning
      when converting Quantity to Numpy array.
    * ``MATPLOTLIB_NOTEBOOK_FORMAT``: figure format for Matplotlib figures in Jupyter Notebooks
    """

    data_directory: Path = ENCOMP_BASE / 'data'
    additional_units: Path = data_directory / 'additional-units.txt'

    type_checking: bool = False
    typeset_symbol_scripts: bool = True
    ignore_ndarray_unit_stripped_warning: bool = True

    matplotlib_notebook_format: Literal['retina', 'png', 'svg'] = 'retina'

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


SETTINGS = Settings()
