"""Statement-boundary detection for ``LLVMFuzzerTestOneInput`` harness bodies.

Regex/paren-matching per line, not an AST or full C++ parser: spec §5 rejects
tree-sitter/Clang for the MVP because the target harnesses are simple enough that
per-line detection is expected to hold in practice. This is unverified against C
macros or string literals that contain a semicolon or brace — see
``docs/superpowers/specs/2026-09-15-trajectory-guided-harness-synthesis.md`` §10.
"""

from collections.abc import Sequence
from dataclasses import dataclass

_ENTRY_POINT = "LLVMFuzzerTestOneInput"
_COMMENT_PREFIXES = ("//", "/*")


@dataclass(frozen=True, slots=True)
class SourceRegion:
    """One remaskable source line: a statement or an ``#include``.

    ``char_start``/``char_end`` are offsets into the harness source this region was
    detected from, excluding the line's trailing newline (``source[start:end]``
    reproduces ``text`` exactly, including original indentation).
    """

    line_number: int
    text: str
    char_start: int
    char_end: int


def statement_regions(source: str) -> tuple[SourceRegion, ...]:
    """Return every ``#include`` line and every semicolon-terminated statement line.

    Excludes blank lines, brace-only lines, comment-only lines, and any line naming
    the harness entry point — remasking the entry point's own signature would delete
    the contract the compiler and libFuzzer both depend on, so it is never a
    candidate region.
    """
    regions: list[SourceRegion] = []
    offset = 0
    for line_number, line in enumerate(source.split("\n"), start=1):
        char_start = offset
        char_end = offset + len(line)
        offset = char_end + 1  # account for the split '\n'
        stripped = line.strip()
        if _ENTRY_POINT in stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue
        is_include = stripped.startswith("#include")
        is_statement = stripped.endswith(";")
        if is_include or is_statement:
            regions.append(SourceRegion(line_number, stripped, char_start, char_end))
    return tuple(regions)


def region_at_line(regions: Sequence[SourceRegion], line_number: int) -> SourceRegion | None:
    """Return the region at a 1-based line number (e.g. from a compiler error)."""
    for region in regions:
        if region.line_number == line_number:
            return region
    return None
