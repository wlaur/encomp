"""Context managers for process-local filesystem and formatting changes.

Two unrelated groups share this module:

* *process utilities* -- :func:`working_dir`, :func:`temp_dir` and :func:`silence_stdout`
  temporarily change the working directory or redirect ``stdout``; they have nothing to do
  with units.
* *quantity formatting* -- :func:`quantity_format` scopes the process-wide default format
  used when rendering quantities and units.
"""

import os
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from .units import UNIT_REGISTRY, set_quantity_format

__all__ = ["quantity_format", "silence_stdout", "temp_dir", "working_dir"]


@contextmanager
def working_dir(path: Path | str) -> Generator[None]:
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
def temp_dir() -> Generator[Path]:
    """
    Context manager that changes the current working directory
    to a temporary directory. The temporary directory is deleted
    after the context manager exits.
    """

    cwd = Path.cwd()

    t_dir = tempfile.TemporaryDirectory()
    path = Path(t_dir.name).resolve()

    try:
        os.chdir(path)
        yield path

    finally:
        os.chdir(cwd)
        t_dir.cleanup()


@contextmanager
def silence_stdout() -> Generator[None]:
    """
    Context manager that redirects ``stdout`` to ``os.devnull``.
    This is used to suppress functions that print to ``stdout``.
    """

    old_target = sys.stdout

    try:
        with Path(os.devnull).open("w") as new_target:
            sys.stdout = new_target
            yield
    finally:
        sys.stdout = old_target


@contextmanager
def quantity_format(fmt: str = "compact") -> Generator[None]:
    """
    Context manager version of :py:func:`encomp.units.set_quantity_format`
    that resets to the previous value afterwards.

    Parameters
    ----------
    fmt : str
        Unit format string: one of ``'~P', '~L', '~H', '~Lx'``.
        Also accepts the aliases ``'compact': '~P'``, ``'normal': '~P'``
        and ``'siunitx': '~Lx'``.
    """

    default = UNIT_REGISTRY.formatter.default_format or "~P"

    set_quantity_format(fmt)

    try:
        yield
    finally:
        set_quantity_format(default)
