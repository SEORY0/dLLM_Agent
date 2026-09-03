"""Resident selective-revision probability margin tests."""

import pytest
from tools.selective_revision_worker import selected_vs_runner_up_margin


def test_selected_vs_runner_up_margin_uses_allowed_distribution() -> None:
    """Given nonuniform probabilities, the selected token is compared with its best peer."""
    # Given
    probabilities = (0.2, 0.7, 0.1)

    # When
    margin = selected_vs_runner_up_margin(probabilities, selected_index=1)

    # Then
    assert margin == pytest.approx(0.5)


def test_selected_vs_runner_up_margin_uses_zero_for_missing_alternative() -> None:
    """Given one allowed token, its runner-up probability is zero."""
    # Given
    probabilities = (1.0,)

    # When
    margin = selected_vs_runner_up_margin(probabilities, selected_index=0)

    # Then
    assert margin == pytest.approx(1.0)
