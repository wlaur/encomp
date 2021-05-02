"""
Contains settings used elsewhere in the library.
"""

from pathlib import Path
from pydantic.dataclasses import dataclass


ENCOMP_BASE = Path(__file__).parent.resolve()


@dataclass
class Settings:
    """
    Settings class.

    .. todo::
        * How to use environment variables etc...?
        * Maybe use the conventional way of type checking instead (what is this?)
    """

    data_directory: Path = ENCOMP_BASE / 'data'
    additional_units: Path = data_directory / 'additional-units.txt'
    type_checking: bool = True


SETTINGS = Settings()
