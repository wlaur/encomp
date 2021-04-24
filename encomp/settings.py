from pathlib import Path
from pydantic import BaseSettings


ENCOMP_BASE = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    """
    Settings class.

    .. todo::
        How to use environment variables etc...?
    """

    data_directory: Path = ENCOMP_BASE / 'data'
    additional_units: Path = data_directory / 'additional-units.txt'
    dimensionality_checking: bool = True


SETTINGS = Settings()
