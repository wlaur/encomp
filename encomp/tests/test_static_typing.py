"""Some ``Quantity`` constructor inputs are deliberately unsupported and must fail the
type checkers, not just the runtime.

* the magnitude and unit are always passed separately (``Q(24, "kg")``, never
  ``Q("24 kg")``), so ``str`` is excluded from the magnitude types;
* a ``Quantity`` is not a unit: ``Q(1, Q(2, "m"))`` silently dropped the ``2``, so the
  fallback overload excludes it from the unit types.

The runtime rejects both (``_validate_magnitude`` / ``__new__``); these tests additionally
pin the *static* rejection through the fallback ``Quantity.__new__`` overload, under both
pyright and pyrefly. A valid twin snippet is checked as a control, so a failure of the
invalid snippet cannot be explained away by import-resolution problems.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from ..units import Quantity as Q


def _tool(name: str) -> str | None:
    candidate = Path(sys.executable).with_name(name)
    if candidate.exists():
        return str(candidate)
    return shutil.which(name)


_PYREFLY = _tool("pyrefly")
_PYRIGHT = _tool("pyright")

_VALID = """
from encomp.units import Quantity as Q

q = Q(24, "kg")
"""

_INVALID = """
from encomp.units import Quantity as Q

q1 = Q("24 kg")
q2 = Q("24", "kg")
"""

_VALID_UNIT = """
from encomp.units import Quantity as Q

q = Q(1, Q(2, "m").u)
"""

_INVALID_UNIT = """
from encomp.units import Quantity as Q

q = Q(1, Q(2, "m"))
"""


def _repo_root() -> Path | None:
    # the checkers need the project's pyproject.toml for their configuration; an installed
    # wheel does not ship it, so the checker tests below skip rather than fail there
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    return None


_ROOT = _repo_root()

_NO_CONFIG = "encomp pyproject.toml not on disk (installed wheel) -- source-tree check only"


def _check(cmd: list[str], code: str, tmp_path: Path) -> int:
    assert _ROOT is not None  # narrowed by the skipif on every caller
    snippet = tmp_path / "snippet.py"
    snippet.write_text(code)
    proc = subprocess.run(
        [*cmd, str(snippet)],
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    return proc.returncode


@pytest.mark.skipif(_PYREFLY is None, reason="pyrefly not installed")
@pytest.mark.skipif(_ROOT is None, reason=_NO_CONFIG)
def test_string_input_rejected_by_pyrefly(tmp_path: Path) -> None:
    assert _PYREFLY is not None and _ROOT is not None  # narrowed by the skipifs above
    # --config is required: without it, pyrefly silently skips files outside any
    # configured project (reporting "0 errors" without analyzing)
    cmd = [_PYREFLY, "check", "--config", str(_ROOT / "pyproject.toml")]
    assert _check(cmd, _VALID, tmp_path) == 0, "control snippet must type-check"
    assert _check(cmd, _INVALID, tmp_path) != 0, "string inputs must not type-check"


@pytest.mark.skipif(_PYRIGHT is None, reason="pyright not installed")
@pytest.mark.skipif(_ROOT is None, reason=_NO_CONFIG)
def test_string_input_rejected_by_pyright(tmp_path: Path) -> None:
    assert _PYRIGHT is not None  # narrowed by the skipif above
    assert _check([_PYRIGHT], _VALID, tmp_path) == 0, "control snippet must type-check"
    assert _check([_PYRIGHT], _INVALID, tmp_path) != 0, "string inputs must not type-check"


def test_string_input_rejected_at_runtime() -> None:
    with pytest.raises(ValueError):
        Q(cast(Any, "24 kg"))

    with pytest.raises(ValueError):
        Q(cast(Any, "24"), "kg")


@pytest.mark.skipif(_PYREFLY is None, reason="pyrefly not installed")
@pytest.mark.skipif(_ROOT is None, reason=_NO_CONFIG)
def test_quantity_unit_rejected_by_pyrefly(tmp_path: Path) -> None:
    assert _PYREFLY is not None and _ROOT is not None  # narrowed by the skipifs above
    cmd = [_PYREFLY, "check", "--config", str(_ROOT / "pyproject.toml")]
    assert _check(cmd, _VALID_UNIT, tmp_path) == 0, "control snippet must type-check"
    assert _check(cmd, _INVALID_UNIT, tmp_path) != 0, "a Quantity unit must not type-check"


@pytest.mark.skipif(_PYRIGHT is None, reason="pyright not installed")
@pytest.mark.skipif(_ROOT is None, reason=_NO_CONFIG)
def test_quantity_unit_rejected_by_pyright(tmp_path: Path) -> None:
    assert _PYRIGHT is not None  # narrowed by the skipif above
    assert _check([_PYRIGHT], _VALID_UNIT, tmp_path) == 0, "control snippet must type-check"
    assert _check([_PYRIGHT], _INVALID_UNIT, tmp_path) != 0, "a Quantity unit must not type-check"


def test_quantity_unit_rejected_at_runtime() -> None:
    with pytest.raises(TypeError, match="got Quantity"):
        Q(1, cast(Any, Q(2, "m")))
