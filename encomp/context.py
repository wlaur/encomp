"""
Various context managers.
"""

import sys
import os
from pathlib import Path
from contextlib import contextmanager
import tempfile
from typing import Union


@contextmanager
def working_dir(path: Union[Path, str]) -> None:
    """
    Context manager that changes the working directory.
    The working directory is changed back after the context
    manager exits.

    Parameters
    ----------
    path : Union[Path, str]
        The new working directory
    """

    cwd = os.getcwd()

    try:
        os.chdir(path)
        yield

    finally:
        os.chdir(cwd)


@contextmanager
def temp_dir() -> None:
    """
    Context manager that changes the current working directory
    to a temporary directory. The temporary directory is deleted
    after the context manager exits.
    """

    cwd = os.getcwd()

    t_dir = tempfile.TemporaryDirectory()

    try:
        os.chdir(t_dir.name)
        yield

    finally:
        os.chdir(cwd)
        t_dir.cleanup()


@contextmanager
def silence_stdout() -> None:
    """
    Context manager that redirects ``stdout`` to ``os.devnull``.
    This is used suppress functions that print to ``stdout``.
    """

    old_target = sys.stdout

    try:
        with open(os.devnull, 'w') as new_target:
            sys.stdout = new_target
            yield
    finally:
        sys.stdout = old_target
