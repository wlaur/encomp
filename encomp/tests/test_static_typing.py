"""String inputs to ``Quantity`` are deliberately unsupported: the magnitude and unit
are always passed separately (``Q(24, "kg")``, never ``Q("24 kg")``).

The runtime already rejects strings (``_validate_magnitude``); these tests additionally
pin the *static* rejection: the fallback ``Quantity.__new__`` overload enumerates the
supported magnitude types and excludes ``str``, so ``Q("24 kg")`` must fail both pyright
and pyrefly. A valid twin snippet is checked as a control, so a failure of the invalid
snippet cannot be explained away by import-resolution problems.
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

_ROOT = Path(__file__).resolve().parents[2]


def _check(cmd: list[str], code: str, tmp_path: Path) -> int:
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
def test_string_input_rejected_by_pyrefly(tmp_path: Path) -> None:
    assert _PYREFLY is not None  # narrowed by the skipif above
    # --config is required: without it, pyrefly silently skips files outside any
    # configured project (reporting "0 errors" without analyzing)
    cmd = [_PYREFLY, "check", "--config", str(_ROOT / "pyproject.toml")]
    assert _check(cmd, _VALID, tmp_path) == 0, "control snippet must type-check"
    assert _check(cmd, _INVALID, tmp_path) != 0, "string inputs must not type-check"


@pytest.mark.skipif(_PYRIGHT is None, reason="pyright not installed")
def test_string_input_rejected_by_pyright(tmp_path: Path) -> None:
    assert _PYRIGHT is not None  # narrowed by the skipif above
    assert _check([_PYRIGHT], _VALID, tmp_path) == 0, "control snippet must type-check"
    assert _check([_PYRIGHT], _INVALID, tmp_path) != 0, "string inputs must not type-check"


def test_string_input_rejected_at_runtime() -> None:
    with pytest.raises(ValueError):
        Q(cast(Any, "24 kg"))

    with pytest.raises(ValueError):
        Q(cast(Any, "24"), "kg")
