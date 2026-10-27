"""External (compiler-reported) localization of a defect inside a generated harness body.

RQ1 (224 independent samples) found the model's own commit-time signals -- confidence,
entropy, margin -- do not point at the line clang rejects any better than chance. This
module takes the location from the compiler instead: parse clang's ``file:line:col:
error:`` diagnostics, map the first one onto a line of the model-written body (the fixed
scaffold around it is never a candidate), and turn that line into a contiguous token
window that ``remask.infill_window`` can mask while keeping both sides as context.

Pure, like the rest of this package: no compiler invocation (that lives in
``tools/harness_synthesis/_compile.py``) and no backend.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from dllm_agent.harness_synthesis.regions import SourceRegion
from dllm_agent.harness_synthesis.remask import TokenWindow, region_token_positions


@dataclass(frozen=True, slots=True)
class CompilerDiagnostic:
    """One clang ``error:`` line: 1-based ``line``/``column`` in the compiled file."""

    line: int
    column: int
    message: str


def parse_compiler_diagnostics(
    compile_error: str, *, file_name: str = "harness.cc"
) -> tuple[CompilerDiagnostic, ...]:
    """Every ``error:``/``fatal error:`` diagnostic reported against ``file_name``, in order.

    Diagnostics in other files (an OpenEXR header's ``note:`` or a template instantiation
    inside a system header) are skipped -- they cannot be mapped onto the harness.
    Warnings promoted by ``-Werror`` are still reported as ``error:`` by clang, so they are
    included.
    """
    pattern = re.compile(
        rf"(?:^|[\s/]){re.escape(file_name)}:(\d+):(\d+): (?:fatal )?error: (.*)$", re.MULTILINE
    )
    return tuple(
        CompilerDiagnostic(
            line=int(match.group(1)), column=int(match.group(2)), message=match.group(3).strip()
        )
        for match in pattern.finditer(compile_error)
    )


def body_line_regions(body: str) -> tuple[SourceRegion, ...]:
    """Every non-blank line of ``body`` as a region, offsets relative to ``body`` itself.

    Wider than ``regions.statement_regions`` on purpose: a compiler error can land on a
    line that is not a complete statement (an unclosed call, a bare ``return``), and the
    random-location null policy must draw from the same candidate set the compiler's
    location can fall in.
    """
    regions: list[SourceRegion] = []
    offset = 0
    for line_number, line in enumerate(body.split("\n"), start=1):
        char_start = offset
        char_end = offset + len(line)
        offset = char_end + 1  # account for the split '\n'
        if line.strip():
            regions.append(SourceRegion(line_number, line.strip(), char_start, char_end))
    return tuple(regions)


def locate_in_body(harness_line: int, *, prefix: str, body: str) -> SourceRegion | None:
    r"""Map a 1-based harness line onto the body line it falls on, or ``None`` if outside.

    The harness is ``prefix + body + suffix``; ``prefix`` must end with a newline (the
    fixed scaffold's does -- ``try {\n``), so the body's first line is harness line
    ``prefix.count("\n") + 1``. ``None`` means the error is in the prefix or suffix (or
    on a blank body line) -- not localizable to anything the model wrote.
    """
    body_line = harness_line - prefix.count("\n")
    for region in body_line_regions(body):
        if region.line_number == body_line:
            return region
    return None


def window_covering(
    region: SourceRegion,
    token_char_offsets: Sequence[tuple[int, int]],
    *,
    position_offset: int = 0,
) -> TokenWindow | None:
    """The smallest contiguous token window covering every token that overlaps ``region``.

    ``None`` if no token overlaps it (``region`` lies past the generated text).
    """
    positions = region_token_positions(region, token_char_offsets, position_offset=position_offset)
    if not positions:
        return None
    return TokenWindow(start=positions[0], length=positions[-1] - positions[0] + 1)
