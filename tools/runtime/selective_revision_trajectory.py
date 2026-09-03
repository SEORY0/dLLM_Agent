"""Trajectory parsing helpers for selective-revision responses."""

from __future__ import annotations

from tools.runtime.selective_revision_json import (
    JsonValue,
    TrajectoryStep,
    WireProtocolError,
    integer,
    integers,
    require_keys,
)
from tools.runtime.selective_revision_wire import TRAJECTORY_STEP_KEYS


def trajectory_steps(item: dict[str, JsonValue]) -> tuple[TrajectoryStep, ...]:
    """Parse trajectory step evidence from a response object."""
    raw_steps = item.get("trajectory_steps")
    if not isinstance(raw_steps, list):
        raise WireProtocolError(reason="trajectory_steps must be a list")
    steps: list[TrajectoryStep] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            raise WireProtocolError(reason="trajectory_steps must contain objects")
        require_keys(raw_step, TRAJECTORY_STEP_KEYS)
        steps.append(
            TrajectoryStep(
                step_index=integer(raw_step, "step_index"),
                block_index=integer(raw_step, "block_index"),
                masked_before=integer(raw_step, "masked_before"),
                committed_positions=integers(raw_step, "committed_positions"),
                token_ids=integers(raw_step, "token_ids"),
                confidences=_numbers(raw_step, "confidences"),
                proposal_positions=integers(raw_step, "proposal_positions"),
                proposal_token_ids=integers(raw_step, "proposal_token_ids"),
                proposal_confidences=_numbers(raw_step, "proposal_confidences"),
                proposal_margins=_numbers(raw_step, "proposal_margins"),
                remaining_masked=integer(raw_step, "remaining_masked"),
            )
        )
    return tuple(steps)


def _numbers(item: dict[str, JsonValue], key: str) -> tuple[float, ...]:
    value = item.get(key)
    if not isinstance(value, list):
        raise WireProtocolError(reason=f"{key} must be a list")
    numbers: list[float] = []
    for entry in value:
        if not isinstance(entry, int | float) or isinstance(entry, bool):
            raise WireProtocolError(reason=f"{key} must contain only numbers")
        numbers.append(float(entry))
    return tuple(numbers)
