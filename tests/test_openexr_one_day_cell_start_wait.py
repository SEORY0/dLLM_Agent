"""A search cell must not launch before its scheduled start."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from dllm_agent.experiments.openexr_one_day.runtime import _wait_for_cell_start


def test_early_cell_waits_until_its_scheduled_start() -> None:
    # Given a cell whose start is 300s away (model prep finished early)
    slept: list[float] = []

    # When
    _wait_for_cell_start(1_300.0, now=lambda: 1_000.0, pause=slept.append)

    # Then it holds for exactly the remaining time instead of launching early
    assert slept == [300.0]


def test_late_or_on_time_cell_is_not_delayed() -> None:
    # Given cells whose start has already arrived (slow model prep)
    slept: list[float] = []

    # When
    _wait_for_cell_start(1_000.0, now=lambda: 1_000.0, pause=slept.append)
    _wait_for_cell_start(1_000.0, now=lambda: 1_250.0, pause=slept.append)

    # Then nothing sleeps; the search proceeds against the remaining window
    assert slept == []
