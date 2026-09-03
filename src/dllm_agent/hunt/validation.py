"""Fail-closed boundaries shared by the live-target hunt pipeline."""

import hashlib
import re
from pathlib import Path
from typing import Final

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.hunt.models import CandidateBank, HarnessCandidate

_INCLUDE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE
)
_FORBIDDEN_TOKENS = (
    "```",
    " system(",
    "std::system(",
    "popen(",
    "fork(",
    "execve(",
    "socket(",
    "connect(",
    "/proc/",
    "/etc/",
    "/home/",
    "asm(",
    "__asm__",
)


def validate_candidate_bank(bank: CandidateBank, prompt: bytes, path: Path) -> None:
    """Verify model-output hashes and monotonic observed step numbering."""
    if hashlib.sha256(prompt).hexdigest() != bank.prompt_sha256:
        detail = f"prompt SHA-256 does not match candidate bank: {path}"
        raise HuntSetupError(detail)
    candidate_ids: set[str] = set()
    for candidate in bank.candidates:
        if candidate.candidate_id in candidate_ids:
            detail = f"duplicate candidate id: {candidate.candidate_id}"
            raise HuntSetupError(detail)
        candidate_ids.add(candidate.candidate_id)
        source_hash = hashlib.sha256(candidate.source.encode()).hexdigest()
        if source_hash != candidate.source_sha256:
            detail = f"candidate source SHA-256 does not match: {candidate.candidate_id}"
            raise HuntSetupError(detail)
        expected_steps = tuple(range(1, len(candidate.steps) + 1))
        actual_steps = tuple(step.step_index for step in candidate.steps)
        if actual_steps != expected_steps:
            detail = f"non-monotonic trajectory: {candidate.candidate_id}"
            raise HuntSetupError(detail)
        if candidate.nfe < len(candidate.steps):
            detail = f"NFE is smaller than captured steps: {candidate.candidate_id}"
            raise HuntSetupError(detail)
        if any(step.committed_tokens > step.masked_tokens_before for step in candidate.steps):
            detail = f"invalid token commitment: {candidate.candidate_id}"
            raise HuntSetupError(detail)


def validate_harness_source(
    candidate: HarnessCandidate,
    *,
    allowed_includes: tuple[str, ...],
    required_symbols: tuple[str, ...],
) -> None:
    """Reject generated code outside the narrow libFuzzer harness contract."""
    normalized = f" {candidate.source}"
    for token in _FORBIDDEN_TOKENS:
        if token in normalized:
            detail = f"forbidden token {token.strip()!r}: {candidate.candidate_id}"
            raise HuntSetupError(detail)
    includes = tuple(match.group(1) for match in _INCLUDE_PATTERN.finditer(candidate.source))
    disallowed = tuple(header for header in includes if header not in allowed_includes)
    if disallowed:
        detail = f"disallowed include {disallowed[0]!r}: {candidate.candidate_id}"
        raise HuntSetupError(detail)
    missing = tuple(symbol for symbol in required_symbols if symbol not in candidate.source)
    if missing:
        detail = f"required symbol missing {missing[0]!r}: {candidate.candidate_id}"
        raise HuntSetupError(detail)
