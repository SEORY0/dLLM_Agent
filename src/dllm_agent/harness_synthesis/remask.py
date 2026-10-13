"""Region-confidence scoring, remask-target selection, and remask-sequence construction.

Deliberately has no dependency on a live generation backend: it consumes already-decoded
regions (``regions.py``) and already-recorded traces (``token_trace.py``) and produces plain
token-id sequences. The backend boundary that actually drives Dream's ``diffusion_generate``
lives in ``trajectory_hook.py`` instead, matching spec §7's module split.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random

from dllm_agent.harness_synthesis.regions import SourceRegion, region_at_line
from dllm_agent.harness_synthesis.token_trace import (
    TokenTrace,
    commit_confidence,
    commit_entropy,
    commit_margin,
    position_traces,
)

type _CommitFeature = Callable[[Sequence[TokenTrace]], float | None]


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


def _region_feature(
    region: SourceRegion,
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    feature: _CommitFeature,
    position_offset: int = 0,
) -> float | None:
    """Mean commit ``feature`` over a region's token positions.

    Returns ``None`` if the region has no covering tokens or any of them never
    permanently committed (``feature`` is ``None`` for that position) — an incomplete
    region cannot be scored.
    """
    positions = region_token_positions(region, token_char_offsets, position_offset=position_offset)
    if not positions:
        return None
    values: list[float] = []
    for position in positions:
        value = feature(position_traces(traces, position))
        if value is None:
            return None
        values.append(value)
    return sum(values) / len(values)


def region_confidence(
    region: SourceRegion,
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    position_offset: int = 0,
) -> float | None:
    """Mean commit confidence over a region's token positions (RQ1's primary signal)."""
    return _region_feature(
        region,
        token_char_offsets,
        traces,
        feature=commit_confidence,
        position_offset=position_offset,
    )


def region_entropy(
    region: SourceRegion,
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    position_offset: int = 0,
) -> float | None:
    """Mean commit entropy over a region's token positions (RQ1's option-(c) signal)."""
    return _region_feature(
        region, token_char_offsets, traces, feature=commit_entropy, position_offset=position_offset
    )


def region_margin(
    region: SourceRegion,
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    position_offset: int = 0,
) -> float | None:
    """Mean commit margin (top1-top2) over a region's token positions."""
    return _region_feature(
        region, token_char_offsets, traces, feature=commit_margin, position_offset=position_offset
    )


def _select_extreme_region(  # noqa: PLR0913 - internal helper shared by 3 public selectors.
    regions: Sequence[SourceRegion],
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    feature: _CommitFeature,
    pick: Callable[[list[tuple[SourceRegion, float]]], tuple[SourceRegion, float]],
    position_offset: int = 0,
) -> SourceRegion | None:
    scored = [
        (region, value)
        for region in regions
        if (
            value := _region_feature(
                region, token_char_offsets, traces, feature=feature, position_offset=position_offset
            )
        )
        is not None
    ]
    if not scored:
        return None
    return pick(scored)[0]


def select_lowest_confidence_region(
    regions: Sequence[SourceRegion],
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    position_offset: int = 0,
) -> SourceRegion | None:
    """Pick the region with the lowest mean commit confidence, or ``None`` if none score."""
    return _select_extreme_region(
        regions,
        token_char_offsets,
        traces,
        feature=commit_confidence,
        pick=lambda scored: min(scored, key=lambda pair: pair[1]),
        position_offset=position_offset,
    )


def select_highest_entropy_region(
    regions: Sequence[SourceRegion],
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    position_offset: int = 0,
) -> SourceRegion | None:
    """Pick the region with the highest mean commit entropy (most uncertain), or ``None``."""
    return _select_extreme_region(
        regions,
        token_char_offsets,
        traces,
        feature=commit_entropy,
        pick=lambda scored: max(scored, key=lambda pair: pair[1]),
        position_offset=position_offset,
    )


def select_lowest_margin_region(
    regions: Sequence[SourceRegion],
    token_char_offsets: Sequence[tuple[int, int]],
    traces: Sequence[TokenTrace],
    *,
    position_offset: int = 0,
) -> SourceRegion | None:
    """Pick the region with the lowest mean commit margin, or ``None`` if none score.

    Lowest margin means top1 barely beats top2 — the most uncertain region.
    """
    return _select_extreme_region(
        regions,
        token_char_offsets,
        traces,
        feature=commit_margin,
        pick=lambda scored: min(scored, key=lambda pair: pair[1]),
        position_offset=position_offset,
    )


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

    Known confound (RQ3-v3 pilot, 2026-09-30): because this masks *to the end*, a
    region nearer the end always costs less to regenerate than one nearer the start.
    Region-identity and remask cost are therefore entangled whenever selection can
    prefer one region over another based on its position — ``remask_window`` below
    exists to remove that entanglement by fixing the cost regardless of which
    candidate is chosen.
    """
    return tuple(
        mask_token_id if index >= start_position else token_id
        for index, token_id in enumerate(token_ids)
    )


@dataclass(frozen=True, slots=True)
class TokenWindow:
    """A fixed-length span of generated-token positions, used as a remask candidate.

    Unlike ``SourceRegion`` (a whole statement, whose length varies with what the
    model happened to write), every ``TokenWindow`` at a given ``length`` costs
    exactly the same number of tokens to remask no matter which one is picked —
    selection and remask cost are decoupled by construction.
    """

    start: int
    length: int

    @property
    def end(self) -> int:
        """The position one past this window's last covered token."""
        return self.start + self.length


def fixed_length_windows(
    token_count: int, window_length: int, *, position_offset: int = 0
) -> tuple[TokenWindow, ...]:
    """Partition ``token_count`` positions into non-overlapping, equal-length windows.

    A trailing partial window shorter than ``window_length`` is dropped rather than
    included short — every returned candidate has exactly ``window_length`` tokens, so
    ``remask_window`` always masks the same amount regardless of which one is chosen.
    """
    if window_length <= 0:
        return ()
    count = token_count // window_length
    return tuple(
        TokenWindow(start=position_offset + index * window_length, length=window_length)
        for index in range(count)
    )


def _window_feature(
    window: TokenWindow, traces: Sequence[TokenTrace], *, feature: _CommitFeature
) -> float | None:
    values: list[float] = []
    for position in range(window.start, window.end):
        value = feature(position_traces(traces, position))
        if value is None:
            return None
        values.append(value)
    return sum(values) / len(values)


def select_lowest_confidence_window(
    windows: Sequence[TokenWindow], traces: Sequence[TokenTrace]
) -> TokenWindow | None:
    """Pick the window with the lowest mean commit confidence, or ``None`` if none score."""
    scored = [
        (window, value)
        for window in windows
        if (value := _window_feature(window, traces, feature=commit_confidence)) is not None
    ]
    if not scored:
        return None
    return min(scored, key=lambda pair: pair[1])[0]


def select_random_window(windows: Sequence[TokenWindow], rng: Random) -> TokenWindow | None:
    """Pick a uniformly random window — the RQ3 null-hypothesis policy, window variant."""
    if not windows:
        return None
    return rng.choice(list(windows))


def remask_window(
    token_ids: Sequence[int], window: TokenWindow, *, mask_token_id: int
) -> tuple[int, ...]:
    """Remask only ``window``'s span, leaving every other position untouched.

    Every token before ``window.start`` and from ``window.end`` onward — including
    the fixed scaffold suffix, when this is applied to a harness's body tokens —
    keeps its already-committed id, so Dream's masked-diffusion process has nothing
    to revise there regardless of which window was chosen.
    """
    return tuple(
        mask_token_id if window.start <= index < window.end else token_id
        for index, token_id in enumerate(token_ids)
    )


def infill_window(
    token_ids: Sequence[int], window: TokenWindow, *, mask_length: int, mask_token_id: int
) -> tuple[int, ...]:
    """Replace ``window``'s span with ``mask_length`` masks, keeping both sides intact.

    Unlike ``remask_window``, the masked span need not keep the window's own length: a
    repaired line is rarely the same token count as the one it replaces, so the caller
    can give it slack (``mask_length > window.length``). Every token before
    ``window.start`` and from ``window.end`` onward keeps its committed id -- only its
    position shifts by ``mask_length - window.length`` -- so the regeneration sees the
    code on *both* sides of the span, which a left-to-right decoder repairing from
    ``window.start`` (``remask_from_position_to_end``) cannot. Slack positions the model
    does not need come back as pad/special tokens; stripping them is the caller's job.
    """
    if mask_length <= 0:
        msg = f"mask_length must be positive, got {mask_length}"
        raise ValueError(msg)
    return (
        *token_ids[: window.start],
        *([mask_token_id] * mask_length),
        *token_ids[window.end :],
    )
