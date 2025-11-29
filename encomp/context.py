import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .units import UNIT_REGISTRY, set_quantity_format


@contextmanager
def working_dir(path: Path | str) -> Iterator[None]:
    """
    Context manager that changes the working directory.
    The working directory is changed back after the context
    manager exits.

    Parameters
    ----------
    path : Path | str
        The new working directory
    """

    cwd = Path.cwd()

    try:
        os.chdir(path)
        yield

    finally:
        os.chdir(cwd)


@contextmanager
def temp_dir() -> Iterator[None]:
    """
    Context manager that changes the current working directory
    to a temporary directory. The temporary directory is deleted
    after the context manager exits.
    """

    cwd = Path.cwd()

    t_dir = tempfile.TemporaryDirectory()

    try:
        os.chdir(t_dir.name)
        yield

    finally:
        os.chdir(cwd)
        t_dir.cleanup()


@contextmanager
def silence_stdout() -> Iterator[None]:
    """
    Context manager that redirects ``stdout`` to ``os.devnull``.
    This is used suppress functions that print to ``stdout``.
    """

    old_target = sys.stdout

    try:
        with Path(os.devnull).open("w") as new_target:
            sys.stdout = new_target
            yield
    finally:
        sys.stdout = old_target


@contextmanager
def quantity_format(fmt: str = "compact") -> Iterator[None]:
    """
    Context manager version of :py:func:`encomp.units.set_quantity_format`
    that resets to the previous value afterwards.

    Parameters
    ----------
    fmt : str
        Unit format string: one of ``'~P', '~L', '~H', '~Lx'``.
        Also accepts aliases: ``'compact': '~P'`` and ``'siunitx': '~Lx'``.
    """

    default = UNIT_REGISTRY.formatter.default_format or "~P"

    set_quantity_format(fmt)

    try:
        yield
    finally:
        set_quantity_format(default)
