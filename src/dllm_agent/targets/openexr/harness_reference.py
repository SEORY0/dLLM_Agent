"""Load the human-authored reference OpenEXR harness, parsed for diff comparison.

Task 8's diversity check (and any later comparison of a synthesized harness against the
baseline) needs the pinned ``tools/openexr_runtime/poc_probe.cc`` source and its detected
regions; this wraps ``harness_synthesis.regions.statement_regions`` around it rather than
re-implementing parsing here (spec §7 lists this file alongside the search-tree package,
not inside it, because it is target-specific — a ``targets/openexr`` concern — while region
detection itself is target-agnostic).
"""

from dataclasses import dataclass
from pathlib import Path

from dllm_agent.harness_synthesis.regions import SourceRegion, statement_regions


@dataclass(frozen=True, slots=True)
class ReferenceHarness:
    """The pinned human-authored harness source and its detected regions."""

    source: str
    regions: tuple[SourceRegion, ...]


def load_reference_harness(path: Path) -> ReferenceHarness:
    """Read and parse the reference harness at ``path`` (e.g. ``poc_probe.cc``)."""
    source = path.read_text()
    return ReferenceHarness(source=source, regions=statement_regions(source))
