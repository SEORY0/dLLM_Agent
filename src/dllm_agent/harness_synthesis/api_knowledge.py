"""OpenEXR API knowledge extracted from the pinned runtime image.

Promoted from ``tools/harness_synthesis_feasibility/fb_sim.py``'s ``STATIC_KB``/``hint_for``
prototype: the feasibility simulation found this KB is not an optional quality-of-life
addition but the difference between 0/8 and 8/8 compiling harnesses
(``docs/harness-synthesis-feasibility-20260920.md`` Result 3), so it is promoted from a
prototype hint to spec §7's module tree (this module is new relative to that spec — see
``docs/superpowers/plans/2026-09-29-trajectory-signal-harness-synthesis.md`` §4).

``UsageExample`` is never populated from model output: seeding the knowledge base with the
model's own guesses would let generation validate itself in a circle instead of grounding it
in something a human independently confirmed compiles.
"""

import difflib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from subprocess import CompletedProcess
from typing import override

type RuntimeCommand = Callable[[tuple[str, ...], float], CompletedProcess[bytes]]

_HEADER_NAME = re.compile(r"Imf.*\.h")
_MISSING_HEADER = re.compile(r"'([^']+\.h)' file not found")


@dataclass(slots=True)
class ApiKnowledgeError(Exception):
    """Raised when extracting API knowledge from the runtime image fails."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the extraction failure reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class UsageExample:
    """A hand-authored OpenEXR call pattern, independently confirmed to compile."""

    name: str
    headers: tuple[str, ...]
    snippet: str


@dataclass(frozen=True, slots=True)
class ApiKnowledgeBase:
    """Facts extracted from the pinned image that a generation prompt can rely on."""

    header_index: tuple[str, ...]
    namespace_note: str
    declarations: tuple[str, ...]


def extract_header_index(
    runner: RuntimeCommand, *, image: str, include_dir: str, timeout: float = 30.0
) -> tuple[str, ...]:
    """List every ``Imf*.h`` header the pinned image actually ships.

    A plain ``ls``, filtered here in Python rather than with a shell pipe inside the
    container command, so the container boundary stays a fixed argv (this repo's
    convention — see ``targets/openexr/deep_parse.py``) and the filter itself is
    unit-testable without Docker.
    """
    process = runner(("docker", "run", "--rm", image, "ls", include_dir), timeout)
    if process.returncode != 0:
        reason = f"header listing failed: {process.stderr.decode(errors='replace')}"
        raise ApiKnowledgeError(reason)
    names = process.stdout.decode().split()
    return tuple(sorted(name for name in names if _HEADER_NAME.fullmatch(name)))


def extract_text(runner: RuntimeCommand, *, image: str, path: str, timeout: float = 30.0) -> str:
    """Return one file's decoded text from inside the pinned image."""
    process = runner(("docker", "run", "--rm", image, "cat", path), timeout)
    if process.returncode != 0:
        reason = f"reading {path} failed: {process.stderr.decode(errors='replace')}"
        raise ApiKnowledgeError(reason)
    return process.stdout.decode()


def find_declaration(header_text: str, pattern: str) -> str | None:
    """Return the first regex match in header text, whitespace-normalized to one line."""
    match = re.search(pattern, header_text, re.DOTALL)
    if match is None:
        return None
    return re.sub(r"\s+", " ", match.group(0)).strip()


def render_prompt_text(kb: ApiKnowledgeBase, examples: Sequence[UsageExample] = ()) -> str:
    """Render the knowledge base as prompt text for the generation call."""
    lines = [
        "Reference facts about the installed OpenEXR (use only these):",
        "Relevant headers: " + ", ".join(kb.header_index),
        kb.namespace_note,
        *kb.declarations,
    ]
    lines.extend(_render_example(example) for example in examples)
    return "\n".join(lines)


def _render_example(example: UsageExample) -> str:
    header = f"Verified usage example ({example.name}, known to compile against this OpenEXR):"
    return f"{header}\n```cpp\n{example.snippet}```"


def suggest_existing_headers(
    missing_header: str, header_index: Sequence[str], *, cutoff: float = 0.5, limit: int = 3
) -> tuple[str, ...]:
    """Return the closest real header names to a hallucinated one, or ``()``."""
    basename = missing_header.rsplit("/", maxsplit=1)[-1]
    return tuple(difflib.get_close_matches(basename, header_index, n=limit, cutoff=cutoff))


def hint_for_missing_headers(compiler_error: str, header_index: Sequence[str]) -> str:
    """Turn every ``'X.h' file not found`` error into a rule-based repair hint."""
    hints: list[str] = []
    missing_headers: list[str] = _MISSING_HEADER.findall(compiler_error)
    for missing in missing_headers:
        near = suggest_existing_headers(missing, header_index)
        hint = f"'{missing}' does not exist."
        if near:
            hint += f" Existing headers with similar names: {', '.join(near)}."
        hints.append(hint)
    return " ".join(hints)
