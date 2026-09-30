"""Search-attempt logs are bounded, never fatal."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import re

from dllm_agent.experiments.openexr_one_day import cell_runner
from dllm_agent.experiments.openexr_one_day.cell_runner import _bounded_log


def test_log_within_the_cap_is_kept_verbatim() -> None:
    raw = b"INFO: Seed: 1\n" * 10

    assert _bounded_log(raw) == raw


def test_oversized_log_keeps_head_and_tail_and_reports_the_elision_exactly() -> None:
    # Given a log far over the cap: a distinctive head, a huge chatty middle, and a
    # sanitizer report + final stats at the very end (what classification needs)
    head = b"INFO: Seed: 1000001\n"
    tail = b"SUMMARY: UndefinedBehaviorSanitizer: x\nstat::number_of_executed_units: 9\n"
    raw = head + b"#1 NEW cov: 1\n" * 2_000_000 + tail
    assert len(raw) > cell_runner._MAX_LOG_BYTES

    # When
    bounded = _bounded_log(raw)

    # Then it fits the cap and keeps both ends
    assert len(bounded) <= cell_runner._MAX_LOG_BYTES
    assert bounded.startswith(head)
    assert bounded.endswith(tail)

    # and the recorded elision count is exactly what was dropped
    match = re.search(rb"\n\[dllm: (\d{12}) bytes of libFuzzer output elided\]\n", bounded)
    assert match is not None
    marker_len = len(match.group(0))
    assert int(match.group(1)) == len(raw) - (len(bounded) - marker_len)
