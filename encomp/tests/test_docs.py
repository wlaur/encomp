"""Every ``python`` code block in the docs must be self-contained: it carries its own
imports, is clean under ``ruff check``, type-checks under ``pyright``, ``pyrefly`` and
``ty``, and runs without error in isolation.

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
- ``pyright`` and ``pyrefly`` against the repo config. A type error in an example is a
  real defect to fix -- ``Q(1, "bar")`` infers ``Quantity[Pressure, float]``, so a
  correct block is clean. Blocks that demonstrate runtime-only dynamic behavior
  (parameterized ``isinstance``, the sympy ``_``/``__`` methods added at import time,
  deliberately-invalid operations shown inside ``try``/``except``) carry explicit
  checker comments. The pyright test disables ``reportUnusedExpression`` for snippets
  because docs naturally include REPL-style expressions followed by comments showing
  their value.
- execution as a real script: the block is written to a file and run in a fresh
  interpreter with a temporary working directory, exactly as a reader would run it.
  A subprocess (not in-process ``exec``) is required for the ``typeguard`` examples --
  ``@typechecked`` instruments by re-reading the module source, which does not exist for
  exec'd strings -- and the temporary cwd lets blocks create scratch files (e.g. the
  ``.env`` example) without touching the repo. Native ``encomp.coolprop`` plugin
  examples execute too; source-tree CI builds the compiled extension before running
  this module, and wheel CI runs the plugin self-check and the full test suite against
  the built wheels.

There is deliberately no opt-out: a snippet that cannot run as written (for example one
that demonstrates an exception) shows the failure with ``try: ... except SomeError:``
instead, so it still executes.

The whole module skips when the repo docs are not on disk (e.g. running from an installed
wheel via ``pytest --pyargs encomp.tests``); it is a source-tree / CI check. Individual
tool checks skip if ``ruff`` / ``pyright`` are not installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import cast

import pytest


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.is_file() or not (parent / "docs").is_dir():
            continue
        try:
            project = tomllib.loads(pyproject.read_text()).get("project", {})
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if isinstance(project, dict) and cast("dict[str, object]", project).get("name") == "encomp":
            return parent
    return None


def _tool(name: str) -> str | None:
    candidate = Path(sys.executable).with_name(name)
    if candidate.exists():
        return str(candidate)
    return shutil.which(name)


ROOT = _repo_root()
_RUFF = _tool("ruff")
_PYRIGHT = _tool("pyright")
_PYREFLY = _tool("pyrefly")
_TY = _tool("ty")

pytestmark = pytest.mark.skipif(
    ROOT is None, reason="docs sources not on disk (installed wheel) -- source-tree check only"
)

_FENCE = re.compile(r"^```python\s*$(.*?)^```", re.MULTILINE | re.DOTALL)
_PYRIGHT_PREFIX = "# pyright: reportUnusedExpression=false\n"
# directories that never hold published docs: build output, deps, caches, VCS, and the
# repo's `temp/` scratch area
_SKIP_DIRS = {
    ".venv",
    "_build",
    "_coolprop_build",
    "node_modules",
    ".git",
    "site-packages",
    "target",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    "temp",
}
_SKIP_FILES = {"REVIEW.md"}


def _doc_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.md")
        if p.name not in _SKIP_FILES and not _SKIP_DIRS.intersection(p.relative_to(root).parts)
    )


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


def test_repo_root_rejects_unrelated_user_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "user-project"
    installed_test = project / ".venv" / "lib" / "python3.13" / "site-packages" / "encomp" / "tests" / "test_docs.py"
    installed_test.parent.mkdir(parents=True)
    (project / "docs").mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "user-project"\n')

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "__file__", str(installed_test))
    assert _repo_root() is None


def test_doc_block_discovery_covers_public_docs() -> None:
    """Public Markdown docs must remain in the checked block set."""
    covered = {case_id.split(":", 1)[0] for case_id in _IDS}
    assert {"README.md", "docs/usage.md"} <= covered
    assert not any(case_id.startswith("_coolprop_build/") for case_id in _IDS)


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


@pytest.mark.skipif(_PYRIGHT is None, reason="pyright not installed")
@pytest.mark.parametrize("code", _CODES, ids=_IDS)
def test_doc_block_typechecks(code: str, tmp_path: Path) -> None:
    assert _PYRIGHT is not None  # narrowed by the skipif above
    assert ROOT is not None  # narrowed by the module-level skipif
    block = tmp_path / "block.py"
    block.write_text(_PYRIGHT_PREFIX + code)
    proc = subprocess.run(
        [_PYRIGHT, "--project", str(ROOT / "pyproject.toml"), str(block)],
        capture_output=True,
        text=True,
        cwd=ROOT,  # resolve `encomp` via the project venv
    )
    assert proc.returncode == 0, f"pyright failed:\n{proc.stdout}{proc.stderr}"


@pytest.mark.skipif(_PYREFLY is None, reason="pyrefly not installed")
@pytest.mark.parametrize("code", _CODES, ids=_IDS)
def test_doc_block_typechecks_pyrefly(code: str, tmp_path: Path) -> None:
    assert _PYREFLY is not None  # narrowed by the skipif above
    assert ROOT is not None  # narrowed by the module-level skipif
    block = tmp_path / "block.py"
    block.write_text(code)
    proc = subprocess.run(
        [_PYREFLY, "check", "--config", str(ROOT / "pyproject.toml"), str(block)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, f"pyrefly failed:\n{proc.stdout}{proc.stderr}"


@pytest.mark.skipif(_TY is None, reason="ty not installed")
@pytest.mark.parametrize("code", _CODES, ids=_IDS)
def test_doc_block_typechecks_ty(code: str, tmp_path: Path) -> None:
    assert _TY is not None  # narrowed by the skipif above
    assert ROOT is not None  # narrowed by the module-level skipif
    block = tmp_path / "block.py"
    block.write_text(code)
    proc = subprocess.run(
        # --project resolves the repo's [tool.ty] config (including its
        # error-on-warning gate) and venv even though the block lives in tmp_path
        [_TY, "check", "--project", str(ROOT), str(block)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, f"ty failed:\n{proc.stdout}{proc.stderr}"


@pytest.mark.parametrize("code", _CODES, ids=_IDS)
def test_doc_block_runs(code: str, tmp_path: Path) -> None:
    assert ROOT is not None  # narrowed by the module-level skipif
    block = tmp_path / "block.py"
    block.write_text(code)
    env = {
        **dict(os.environ),
        "PYTHONPATH": f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep),
    }
    proc = subprocess.run(
        [sys.executable, str(block)],
        capture_output=True,
        text=True,
        cwd=tmp_path,  # scratch files a block creates (e.g. the .env example) land in tmp
        env=env,
    )
    assert proc.returncode == 0, f"doc block failed:\n--- stdout:\n{proc.stdout}\n--- stderr:\n{proc.stderr}"
