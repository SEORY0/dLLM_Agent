"""Comment-aware lexical analysis of a model-written harness body.

The RQ4 pilot scored repair by "does it compile". A post-hoc read found many compiling
bodies had lost their inspection call: deleted, or still there as text inside a ``//``
comment. A plain substring search counts that comment as a call. This module masks
comments (and string/char literals) out before looking for code, so text inside a comment
is never treated as code.

Lexical only. It cannot tell a call in dead code (``if (false) { f.header(); }``) from a
live one, and it does not resolve types. ``functional_eval`` covers both using clang's
AST and execution coverage. Use this module where no compiler is available, such as the
retrospective audit of stored results, and for the line-level structure that the AST does
not keep (comment-only lines).
"""

import re
from dataclasses import dataclass
from typing import Literal

type LineKind = Literal["blank", "comment", "code"]
type RightContext = Literal["none", "comment-only", "code"]

_MEMBER_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\.|->)\s*([A-Za-z_]\w*)\s*\(")
_NAMESPACE_QUALIFIER = r"\b(?:OPENEXR_IMF_NAMESPACE::|Imf::|Imf_3_4::)?"


@dataclass(frozen=True, slots=True)
class MemberCall:
    """``receiver.method(`` (or ``->``), with the 1-based body line it starts on."""

    receiver: str
    method: str
    line_number: int


def _spans(code: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """``(comment spans, literal-body spans)`` as half-open offsets into ``code``."""
    comments: list[tuple[int, int]] = []
    literals: list[tuple[int, int]] = []
    index, length = 0, len(code)
    while index < length:
        pair = code[index : index + 2]
        char = code[index]
        if pair == "//":
            end = code.find("\n", index)
            end = length if end == -1 else end
            comments.append((index, end))
            index = end
        elif pair == "/*":
            end = code.find("*/", index + 2)
            end = length if end == -1 else end + 2
            comments.append((index, end))
            index = end
        elif char in "\"'":
            end = index + 1
            while end < length and code[end] not in (char, "\n"):
                end += 2 if code[end] == "\\" else 1
            end = min(end, length)
            literals.append((index + 1, end))
            index = end + 1
        else:
            index += 1
    return comments, literals


def _blank_out(code: str, spans: list[tuple[int, int]]) -> str:
    out = list(code)
    for start, end in spans:
        out[start:end] = ["\n" if c == "\n" else " " for c in code[start:end]]
    return "".join(out)


def mask_comments(code: str) -> str:
    """``code`` with every comment character and string/char-literal body set to a space.

    Newlines are kept and nothing is inserted or deleted, so offsets and line numbers
    in the result match ``code`` exactly. An unterminated ``/*`` masks to the end.
    """
    comments, literals = _spans(code)
    return _blank_out(code, comments + literals)


def comment_text(code: str) -> str:
    """The complement for comments: only comment characters survive, same offsets."""
    comments, _ = _spans(code)
    kept = [False] * len(code)
    for start, end in comments:
        kept[start:end] = [True] * (end - start)
    return "".join(c if keep or c == "\n" else " " for c, keep in zip(code, kept, strict=True))


def line_kinds(body: str) -> tuple[LineKind, ...]:
    """Each line of ``body``: blank, comment-only, or containing code."""
    masked = mask_comments(body).split("\n")
    kinds: list[LineKind] = []
    for original, code in zip(body.split("\n"), masked, strict=True):
        if not original.strip():
            kinds.append("blank")
        elif not code.strip():
            kinds.append("comment")
        else:
            kinds.append("code")
    return tuple(kinds)


def right_context(body: str, line_number: int) -> RightContext:
    """What follows 1-based ``line_number`` in ``body``, ignoring blank lines.

    The pilot's ``has_right_context`` was true whenever any non-blank line followed,
    so a trailing ``// Optionally inspect...`` comment counted. This separates those
    cases from executable code.
    """
    after = [kind for kind in line_kinds(body)[line_number:] if kind != "blank"]
    if not after:
        return "none"
    return "code" if "code" in after else "comment-only"


def _calls(text: str) -> tuple[MemberCall, ...]:
    return tuple(
        MemberCall(match.group(1), match.group(2), text.count("\n", 0, match.start()) + 1)
        for match in _MEMBER_CALL.finditer(text)
    )


def active_member_calls(body: str) -> tuple[MemberCall, ...]:
    """Member calls in code (never inside a comment or string literal)."""
    return _calls(mask_comments(body))


def commented_member_calls(body: str) -> tuple[MemberCall, ...]:
    """Member-call-shaped text that appears only inside comments."""
    return _calls(comment_text(body))


def constructed_object(body: str, class_name: str) -> str | None:
    """Name of the first variable declared (in code) with type ``class_name``, if any."""
    pattern = _NAMESPACE_QUALIFIER + re.escape(class_name) + r"\s+([A-Za-z_]\w*)\s*[({=]"
    match = re.search(pattern, mask_comments(body))
    return match.group(1) if match else None


_LEXICAL_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|::|->|\S")


def executable_statements(body: str) -> tuple[str, ...]:
    """Each code-bearing line, comments removed and whitespace collapsed (for comparison)."""
    return tuple(" ".join(line.split()) for line in mask_comments(body).split("\n") if line.strip())


def unaffected_statements_preserved(
    reference: str, candidate: str, *, defect_line: int
) -> tuple[int, int]:
    """``(kept, total)``: reference code lines other than ``defect_line`` still in ``candidate``.

    Multiset match on normalised text, order ignored. A statement the candidate kept but
    moved still counts. A statement it rewrote equivalently does not. That is a textual
    lower bound on behavioural preservation, and the evaluator's executed call chains
    cover the behavioural side.
    """
    masked = mask_comments(reference).split("\n")
    wanted = [
        " ".join(line.split())
        for number, line in enumerate(masked, start=1)
        if number != defect_line and line.strip()
    ]
    available = list(executable_statements(candidate))
    kept = 0
    for statement in wanted:
        if statement in available:
            available.remove(statement)
            kept += 1
    return kept, len(wanted)


def lexical_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over C++ lexical tokens of the code (comments ignored)."""
    left = [m.group(0) for m in _LEXICAL_TOKEN.finditer(mask_comments(a))]
    right = [m.group(0) for m in _LEXICAL_TOKEN.finditer(mask_comments(b))]
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (token != other))
            )
        previous = current
    return previous[-1]
