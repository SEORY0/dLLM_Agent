#!/usr/bin/env -S uv run --script
# ruff: noqa: EXE001
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pydantic>=2.11,<3",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run tools/analyze_openexr_one_day.py CAMPAIGN_ROOT
# 3. Or make executable and run:
#      chmod +x tools/analyze_openexr_one_day.py && ./tools/analyze_openexr_one_day.py CAMPAIGN_ROOT
# ─────────────────

"""Audit and analyze one frozen OpenEXR one-day campaign."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, repository_root.as_posix())
    sys.path.insert(0, (repository_root / "src").as_posix())

from dllm_agent.experiments.openexr_one_day.analysis import AnalysisError, analyze_campaign


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the fail-closed analyzer and return its process status."""
    values = tuple(sys.argv[1:] if arguments is None else arguments)
    if len(values) != 1:
        _ = sys.stderr.write("analysis failed: expected exactly one CAMPAIGN_ROOT\n")
        return 2
    try:
        result = analyze_campaign(Path(values[0]))
    except AnalysisError as error:
        _ = sys.stderr.write(f"analysis failed: {error}\n")
        return 2
    _ = sys.stdout.write(f"evidence_tree_sha256={result.evidence_tree_sha256}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
