"""Region confidence scoring, remask-target selection, and remask-sequence construction."""

from random import Random

import pytest

from dllm_agent.harness_synthesis.regions import SourceRegion
from dllm_agent.harness_synthesis.remask import (
    TokenWindow,
    fixed_length_windows,
    region_confidence,
    region_entropy,
    region_margin,
    region_token_positions,
    remask_from_position_to_end,
    remask_window,
    select_highest_entropy_region,
    select_lowest_confidence_region,
    select_lowest_confidence_window,
    select_lowest_margin_region,
    select_random_region,
    select_random_window,
    select_region_from_compiler_error,
)
from dllm_agent.harness_synthesis.token_trace import TokenTrace

_MASK = 999


def _region(line: int, text: str, start: int, end: int) -> SourceRegion:
    return SourceRegion(line_number=line, text=text, char_start=start, char_end=end)


def _committed(
    position: int, confidence: float, entropy: float | None = None, margin: float | None = None
) -> TokenTrace:
    return TokenTrace(
        step_index=0,
        position=position,
        token_id=1,
        masked=False,
        confidence=confidence,
        entropy=entropy,
        margin=margin,
    )


def _masked(position: int) -> TokenTrace:
    return TokenTrace(step_index=0, position=position, token_id=_MASK, masked=True)


def test_region_token_positions_selects_overlapping_tokens_only() -> None:
    # Given a region spanning chars [5, 15) and tokens at various spans
    region = _region(1, "text", start=5, end=15)
    offsets = [(0, 5), (5, 10), (10, 15), (15, 20)]

    positions = region_token_positions(region, offsets)

    # Then only the two tokens fully inside [5, 15) are selected
    assert positions == (1, 2)


def test_region_token_positions_applies_the_position_offset() -> None:
    region = _region(1, "text", start=0, end=5)
    offsets = [(0, 5)]

    positions = region_token_positions(region, offsets, position_offset=100)

    assert positions == (100,)


def test_region_confidence_averages_commit_confidence_over_covered_positions() -> None:
    region = _region(1, "text", start=0, end=10)
    offsets = [(0, 5), (5, 10)]
    traces = (_committed(0, 0.4), _committed(1, 0.8))

    assert region_confidence(region, offsets, traces) == pytest.approx(0.6)


def test_region_confidence_is_none_when_a_covered_position_never_committed() -> None:
    region = _region(1, "text", start=0, end=10)
    offsets = [(0, 5), (5, 10)]
    traces = (_committed(0, 0.4), _masked(1))

    assert region_confidence(region, offsets, traces) is None


def test_region_confidence_is_none_when_the_region_covers_no_tokens() -> None:
    region = _region(1, "text", start=100, end=110)
    offsets = [(0, 5)]

    assert region_confidence(region, offsets, ()) is None


def test_select_lowest_confidence_region_picks_the_minimum() -> None:
    high = _region(1, "high", start=0, end=5)
    low = _region(2, "low", start=5, end=10)
    offsets = [(0, 5), (5, 10)]
    traces = (_committed(0, 0.9), _committed(1, 0.2))

    selected = select_lowest_confidence_region([high, low], offsets, traces)

    assert selected is low


def test_select_lowest_confidence_region_skips_unscoreable_regions() -> None:
    scoreable = _region(1, "ok", start=0, end=5)
    unscoreable = _region(2, "gone", start=5, end=10)
    offsets = [(0, 5), (5, 10)]
    traces = (_committed(0, 0.5), _masked(1))

    selected = select_lowest_confidence_region([scoreable, unscoreable], offsets, traces)

    assert selected is scoreable


def test_select_lowest_confidence_region_returns_none_with_no_scoreable_region() -> None:
    region = _region(1, "gone", start=0, end=5)

    assert select_lowest_confidence_region([region], [(0, 5)], (_masked(0),)) is None


def test_region_entropy_averages_commit_entropy_over_covered_positions() -> None:
    region = _region(1, "text", start=0, end=10)
    offsets = [(0, 5), (5, 10)]
    traces = (
        _committed(0, confidence=0.5, entropy=0.4),
        _committed(1, confidence=0.5, entropy=0.8),
    )

    assert region_entropy(region, offsets, traces) == pytest.approx(0.6)


def test_select_highest_entropy_region_picks_the_maximum() -> None:
    low = _region(1, "low", start=0, end=5)
    high = _region(2, "high", start=5, end=10)
    offsets = [(0, 5), (5, 10)]
    traces = (
        _committed(0, confidence=0.5, entropy=0.2),
        _committed(1, confidence=0.5, entropy=0.9),
    )

    selected = select_highest_entropy_region([low, high], offsets, traces)

    assert selected is high


def test_select_highest_entropy_region_returns_none_with_no_scoreable_region() -> None:
    region = _region(1, "gone", start=0, end=5)

    assert select_highest_entropy_region([region], [(0, 5)], (_masked(0),)) is None


def test_region_margin_averages_commit_margin_over_covered_positions() -> None:
    region = _region(1, "text", start=0, end=10)
    offsets = [(0, 5), (5, 10)]
    traces = (
        _committed(0, confidence=0.5, margin=0.3),
        _committed(1, confidence=0.5, margin=0.7),
    )

    assert region_margin(region, offsets, traces) == pytest.approx(0.5)


def test_select_lowest_margin_region_picks_the_minimum() -> None:
    uncertain = _region(1, "uncertain", start=0, end=5)
    certain = _region(2, "certain", start=5, end=10)
    offsets = [(0, 5), (5, 10)]
    traces = (
        _committed(0, confidence=0.5, margin=0.05),
        _committed(1, confidence=0.5, margin=0.9),
    )

    selected = select_lowest_margin_region([uncertain, certain], offsets, traces)

    assert selected is uncertain


def test_select_lowest_margin_region_returns_none_with_no_scoreable_region() -> None:
    region = _region(1, "gone", start=0, end=5)

    assert select_lowest_margin_region([region], [(0, 5)], (_masked(0),)) is None


def test_select_random_region_returns_none_for_an_empty_list() -> None:
    assert select_random_region([], Random(0)) is None  # noqa: S311


def test_select_random_region_only_ever_returns_a_listed_region() -> None:
    regions = [_region(i, f"r{i}", i, i + 1) for i in range(5)]
    rng = Random(42)  # noqa: S311

    for _ in range(20):
        assert select_random_region(regions, rng) in regions


def test_select_region_from_compiler_error_uses_the_first_error_line() -> None:
    first = _region(3, "first", start=0, end=5)
    second = _region(7, "second", start=5, end=10)

    selected = select_region_from_compiler_error([first, second], error_line_numbers=[7, 3])

    assert selected is first


def test_select_region_from_compiler_error_returns_none_with_no_errors() -> None:
    region = _region(3, "only", start=0, end=5)

    assert select_region_from_compiler_error([region], error_line_numbers=[]) is None


def test_select_region_from_compiler_error_returns_none_when_line_has_no_region() -> None:
    region = _region(3, "only", start=0, end=5)

    assert select_region_from_compiler_error([region], error_line_numbers=[99]) is None


def test_remask_from_position_to_end_keeps_the_prefix_and_masks_the_rest() -> None:
    token_ids = (10, 11, 12, 13, 14)

    remasked = remask_from_position_to_end(token_ids, start_position=2, mask_token_id=_MASK)

    assert remasked == (10, 11, _MASK, _MASK, _MASK)


def test_remask_from_position_to_end_at_zero_masks_everything() -> None:
    token_ids = (10, 11, 12)

    remasked = remask_from_position_to_end(token_ids, start_position=0, mask_token_id=_MASK)

    assert remasked == (_MASK, _MASK, _MASK)


def test_remask_from_position_to_end_past_the_end_changes_nothing() -> None:
    token_ids = (10, 11, 12)

    remasked = remask_from_position_to_end(token_ids, start_position=10, mask_token_id=_MASK)

    assert remasked == token_ids


def test_fixed_length_windows_partitions_without_overlap() -> None:
    windows = fixed_length_windows(10, 3)

    assert windows == (
        TokenWindow(start=0, length=3),
        TokenWindow(start=3, length=3),
        TokenWindow(start=6, length=3),
    )


def test_fixed_length_windows_drops_a_short_trailing_remainder() -> None:
    windows = fixed_length_windows(7, 3)

    assert windows == (TokenWindow(start=0, length=3), TokenWindow(start=3, length=3))


def test_fixed_length_windows_applies_the_position_offset() -> None:
    windows = fixed_length_windows(3, 3, position_offset=100)

    assert windows == (TokenWindow(start=100, length=3),)


def test_fixed_length_windows_every_candidate_has_equal_length() -> None:
    windows = fixed_length_windows(20, 4)

    assert all(window.length == 4 for window in windows)


def test_token_window_end_is_start_plus_length() -> None:
    assert TokenWindow(start=5, length=3).end == 8


def test_select_lowest_confidence_window_picks_the_minimum() -> None:
    high = TokenWindow(start=0, length=2)
    low = TokenWindow(start=2, length=2)
    traces = (_committed(0, 0.9), _committed(1, 0.9), _committed(2, 0.1), _committed(3, 0.1))

    selected = select_lowest_confidence_window([high, low], traces)

    assert selected is low


def test_select_lowest_confidence_window_skips_unscoreable_windows() -> None:
    scoreable = TokenWindow(start=0, length=1)
    unscoreable = TokenWindow(start=1, length=1)
    traces = (_committed(0, 0.5), _masked(1))

    selected = select_lowest_confidence_window([scoreable, unscoreable], traces)

    assert selected is scoreable


def test_select_lowest_confidence_window_returns_none_with_no_scoreable_window() -> None:
    window = TokenWindow(start=0, length=1)

    assert select_lowest_confidence_window([window], (_masked(0),)) is None


def test_select_random_window_returns_none_for_an_empty_list() -> None:
    assert select_random_window([], Random(0)) is None  # noqa: S311


def test_select_random_window_only_ever_returns_a_listed_window() -> None:
    windows = [TokenWindow(start=i, length=1) for i in range(5)]
    rng = Random(42)  # noqa: S311

    for _ in range(20):
        assert select_random_window(windows, rng) in windows


def test_remask_window_masks_only_its_own_span() -> None:
    token_ids = (10, 11, 12, 13, 14)

    remasked = remask_window(token_ids, TokenWindow(start=1, length=2), mask_token_id=_MASK)

    assert remasked == (10, _MASK, _MASK, 13, 14)


def test_remask_window_preserves_tokens_after_the_window() -> None:
    token_ids = (10, 11, 12, 13, 14)

    remasked = remask_window(token_ids, TokenWindow(start=0, length=1), mask_token_id=_MASK)

    assert remasked[1:] == token_ids[1:]
