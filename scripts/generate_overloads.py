#!/usr/bin/env python
"""Regenerate Quantity's dimensional overloads; --check reports stale blocks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if regeneration would change units.py")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from encomp._dimensional_algebra import replace_generated_blocks

    path = root / "encomp" / "units.py"
    source = path.read_text()
    generated = replace_generated_blocks(source)
    if arguments.check:
        if generated != source:
            print("Dimensional overloads are stale; run uv run python scripts/generate_overloads.py", file=sys.stderr)
            return 1
        return 0
    if generated != source:
        path.write_text(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
