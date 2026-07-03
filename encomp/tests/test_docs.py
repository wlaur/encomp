"""Every ``python`` code block in the docs must be self-contained: it carries its own
imports, is clean under ``ruff check``, type-checks under ``pyrefly``, and runs without
error in isolation.

This guards the documentation against drift -- a renamed API, a missing import, or a
snippet that only worked as a continuation of an earlier block all fail here. The block
is the unit: each fenced block is extracted verbatim, then linted, type-checked, and
executed on its own, so a block that relies on a name defined in a *previous* block is
a failure.

Discovery is dynamic: EVERY ``*.md`` under the repo (minus build/vendor/scratch dirs) is
scanned for ```` ```python ```` fences, so adding a block anywhere is automatically covered
-- no list to maintain.

Checks per block:
- ``ruff check`` under the repo's full ruff config (run with cwd at the repo root), so each
  block is held to the same ruleset as the source (imports, naming, bugbear, ...) with no
  per-block exceptions -- ``F401`` (unused import) and ``F821`` (undefined name) both gate.
- ``pyrefly check`` against the repo config (``--config`` is passed explicitly: without
  it, pyrefly silently skips files outside a configured project and reports "0 errors"
  without analyzing anything). A type error in an example is a real defect to fix --
  ``Q(1, "bar")`` infers ``Quantity[Pressure, float]``, so a correct block is clean.
  Blocks that demonstrate runtime-only dynamic behavior (parameterized ``isinstance``,
  the sympy ``_``/``__`` methods added at import time, deliberately-invalid operations
  shown inside ``try``/``except``) carry explicit ``# pyrefly: ignore[...]`` comments;
  the repo config elevates ``unused-ignore`` to an error, so a suppression that stops
  matching real errors fails this test instead of lingering.
- execution as a real script: the block is written to a file and run in a fresh
  interpreter with a temporary working directory, exactly as a reader would run it.
  A subprocess (not in-process ``exec``) is required for the ``typeguard`` examples --
  ``@typechecked`` instruments by re-reading the module source, which does not exist for
  exec'd strings -- and the temporary cwd lets blocks create scratch files (e.g. the
  ``.env`` example) without touching the repo.

There is deliberately no opt-out: a snippet that cannot run as written (for example one
that demonstrates an exception) shows the failure with ``try: ... except SomeError:``
instead, so it still executes.

The whole module skips when the repo docs are not on disk (e.g. running from an installed
wheel via ``pytest --pyargs encomp.tests``); it is a source-tree / CI check. Individual
tool checks skip if ``ruff`` / ``pyrefly`` are not installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "docs").is_dir():
            return parent
    return None


def _tool(name: str) -> str | None:
    candidate = Path(sys.executable).with_name(name)
    if candidate.exists():
        return str(candidate)
    return shutil.which(name)


ROOT = _repo_root()
_RUFF = _tool("ruff")
_PYREFLY = _tool("pyrefly")

pytestmark = pytest.mark.skipif(
    ROOT is None, reason="docs sources not on disk (installed wheel) -- source-tree check only"
)

_FENCE = re.compile(r"^```python\s*$(.*?)^```", re.MULTILINE | re.DOTALL)
# directories that never hold published docs: build output, deps, caches, VCS, and the
# repo's `temp/` scratch area
_SKIP_DIRS = {
    ".venv",
    "_build",
    "node_modules",
    ".git",
    "site-packages",
    "target",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    "temp",
}


def _doc_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if not _SKIP_DIRS.intersection(p.relative_to(root).parts))


def _blocks() -> list[tuple[str, str]]:
    """(id, code) for every ```python block, id = ``<relpath>:<line>``."""
    if ROOT is None:
        return []
    cases: list[tuple[str, str]] = []
    for md in _doc_files(ROOT):
        text = md.read_text()
        for m in _FENCE.finditer(text):
            code: str = m.group(1).lstrip("\n")
            line = text[: m.start()].count("\n") + 1
            cases.append((f"{md.relative_to(ROOT)}:{line}", code))
    return cases


_CASES = _blocks()
_IDS = [case_id for case_id, _ in _CASES]
_CODES = [code for _, code in _CASES]


def test_docs_have_blocks() -> None:
    # tripwire: if extraction silently finds nothing, the checks below vacuously pass
    assert len(_CASES) > 20, f"expected many doc code blocks, found {len(_CASES)}"


@pytest.mark.skipif(_RUFF is None, reason="ruff not installed")
@pytest.mark.parametrize("code", _CODES, ids=_IDS)
def test_doc_block_lints(code: str, tmp_path: Path) -> None:
    assert _RUFF is not None  # narrowed by the skipif above
    block = tmp_path / "block.py"
    block.write_text(code)
    proc = subprocess.run(
        [_RUFF, "check", "--no-cache", str(block)],
        capture_output=True,
        text=True,
        cwd=ROOT,  # apply the repo's ruff config to every block, regardless of invocation cwd
    )
    assert proc.returncode == 0, f"ruff check failed:\n{proc.stdout}{proc.stderr}"


@pytest.mark.skipif(_PYREFLY is None, reason="pyrefly not installed")
@pytest.mark.parametrize("code", _CODES, ids=_IDS)
def test_doc_block_typechecks(code: str, tmp_path: Path) -> None:
    assert _PYREFLY is not None  # narrowed by the skipif above
    assert ROOT is not None  # narrowed by the module-level skipif
    block = tmp_path / "block.py"
    block.write_text(code)
    # --config is required: without it, pyrefly silently checks NOTHING for a file
    # outside any configured project (it reports "0 errors" without analyzing), so
    # the block must be checked against the repo config explicitly
    proc = subprocess.run(
        [_PYREFLY, "check", "--config", str(ROOT / "pyproject.toml"), str(block)],
        capture_output=True,
        text=True,
        cwd=ROOT,  # resolve `encomp` via the project venv
    )
    assert proc.returncode == 0, f"pyrefly check failed:\n{proc.stdout}{proc.stderr}"


@pytest.mark.parametrize("code", _CODES, ids=_IDS)
def test_doc_block_runs(code: str, tmp_path: Path) -> None:
    block = tmp_path / "block.py"
    block.write_text(code)
    proc = subprocess.run(
        [sys.executable, str(block)],
        capture_output=True,
        text=True,
        cwd=tmp_path,  # scratch files a block creates (e.g. the .env example) land in tmp
    )
    assert proc.returncode == 0, f"doc block failed:\n--- stdout:\n{proc.stdout}\n--- stderr:\n{proc.stderr}"
