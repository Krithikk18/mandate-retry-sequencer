"""Machine-readable output for the scripts that produce documented numbers.

Every figure in README.md and CONTEXT.md is supposed to come from a
reproducible run. ``scripts/verify_docs.py`` is what enforces that, and it needs
the numbers rather than the prose around them - so each producing script takes
an optional ``--json PATH`` and writes what it just measured to that file.

Three rules, because the point of the file is that it cannot drift from the
report beside it:

1. **The JSON is written from the same objects the report prints.** Never
   recompute a figure for the file; take it off the tally the printed table was
   rendered from. A second computation is a second opinion, and a verifier that
   checks the second opinion checks nothing.
2. **``--json`` never changes stdout.** The human report is emitted exactly as
   it was before the flag existed, so a run with the flag and a run without it
   differ only in whether a file appears.
3. **Field names are part of the contract.** ``verify_docs`` addresses them by
   dotted path, so renaming one breaks a documented claim loudly - which is the
   intended behaviour, not an inconvenience.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Bumped when the shape of a payload changes incompatibly, so a stale cache
# from an older layout is rejected rather than silently mis-read.
SCHEMA_VERSION = 1


def add_json_argument(parser: argparse.ArgumentParser) -> None:
    """Give a producing script its ``--json PATH`` flag."""
    parser.add_argument(
        "--json",
        dest="json_path",
        metavar="PATH",
        default=None,
        help="also write the measured figures to PATH as JSON, for verify_docs",
    )


def emit(path: str | Path | None, script: str, seeds: int, n: int, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path``. A no-op when no path was given."""
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "script": script,
                "seeds": seeds,
                "n": n,
                **payload,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
