"""Region-confidence scoring, remask-target selection, and remask-sequence construction.

Deliberately has no dependency on a live generation backend: it consumes already-decoded
regions (``regions.py``) and already-recorded traces (``token_trace.py``) and produces plain
token-id sequences. The backend boundary that actually drives Dream's ``diffusion_generate``
lives in ``trajectory_hook.py`` instead, matching spec §7's module split.
"""

from collections.abc import Sequence
from random import Random

from dllm_agent.harness_synthesis.regions import SourceRegion, region_at_line
from dllm_agent.harness_synthesis.token_trace import TokenTrace, commit_confidence, position_traces


def region_token_positions(
    region: SourceRegion,
    token_char_offsets: Sequence[tuple[int, int]],
    *,
    position_offset: int = 0,
) -> tuple[int, ...]:
    """Map a source region's character span to absolute token positions.

    ``token_char_offsets[i]`` is generated token ``i``'s ``(char_start, char_end)`` span
    in the same decoded text ``region`` was detected from. ``position_offset`` shifts the
    result into the full sequence's position space (e.g. the prompt length, when
    ``TokenTrace.position`` counts from sequence start rather than generation start).
    """
    return tuple(
        position_offset + index
        for index, (start, end) in enumerate(token_char_offsets)
        if start < region.char_end and end > region.char_start
    )


def region_confidence(
    region: SourceRegion,
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    position_offset: int = 0,
) -> float | None:
    """Mean commit confidence over a region's token positions.

    Returns ``None`` if the region has no covering tokens or any of them never
    permanently committed (``token_trace.commit_confidence`` is ``None`` for that
    position) — an incomplete region cannot be scored.
    """
    positions = region_token_positions(region, token_char_offsets, position_offset=position_offset)
    if not positions:
        return None
    confidences: list[float] = []
    for position in positions:
        confidence = commit_confidence(position_traces(traces, position))
        if confidence is None:
            return None
        confidences.append(confidence)
    return sum(confidences) / len(confidences)


def select_lowest_confidence_region(
    regions: Sequence[SourceRegion],
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    position_offset: int = 0,
) -> SourceRegion | None:
    """Pick the region with the lowest mean commit confidence, or ``None`` if none score."""
    scored = [
        (region, confidence)
        for region in regions
        if (
            confidence := region_confidence(
                region, token_char_offsets, traces, position_offset=position_offset
            )
        )
        is not None
    ]
    if not scored:
        return None
    return min(scored, key=lambda pair: pair[1])[0]


def select_random_region(regions: Sequence[SourceRegion], rng: Random) -> SourceRegion | None:
    """Pick a uniformly random region — the RQ3 null-hypothesis policy."""
    if not regions:
        return None
    return rng.choice(list(regions))


def select_region_from_compiler_error(
    regions: Sequence[SourceRegion], error_line_numbers: Sequence[int]
) -> SourceRegion | None:
    """Pick the region at the first reported compiler-error line, if any region is there."""
    if not error_line_numbers:
        return None
    return region_at_line(regions, min(error_line_numbers))


def remask_from_position_to_end(
    token_ids: Sequence[int], start_position: int, *, mask_token_id: int
) -> tuple[int, ...]:
    """Remask every token from ``start_position`` through the sequence's end.

    Dream's ``diffusion_generate`` keeps a fixed token count per position, and a
    repaired statement is rarely the same token length as the one it replaces, so
    remasking only the flagged region cannot let the line grow or shrink. Remasking to
    the end lets everything after the flagged point re-flow, at the cost of also
    erasing already-good tail tokens — accepted for the MVP (spec §5's "one region
    only" scope decision; feasibility prototype `fb_sim.py` used the same policy).
    """
    return tuple(
        mask_token_id if index >= start_position else token_id
        for index, token_id in enumerate(token_ids)
    )
