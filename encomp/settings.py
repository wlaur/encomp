"""
Contains settings used elsewhere in the library.
"""

from typing import Literal
from pathlib import Path
from pydantic import BaseSettings


ENCOMP_BASE = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    """
    Settings class.

    .. todo::
        * How to use environment variables etc...?
        * Maybe use the conventional way of type checking instead (what is this?)
    """

    data_directory: Path = ENCOMP_BASE / 'data'
    additional_units: Path = data_directory / 'additional-units.txt'
    type_checking: bool = False

    # whether to typeset sub- and superscripts for sympy symbols
    typeset_symbol_scripts: bool = True

    ignore_ndarray_unit_stripped_warning: bool = True

    matplotlib_notebook_format: Literal['retina', 'png', 'svg'] = 'retina'


SETTINGS = Settings()
