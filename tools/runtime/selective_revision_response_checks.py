"""Cross-row validation and hashing for proposal trajectory responses."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import NoReturn

from tools.runtime.selective_revision_json import (
    JsonValue,
    SamplerResponse,
    TrajectoryStep,
    WireProtocolError,
    canonical_json,
)


def validate_response_trajectory(response: SamplerResponse) -> None:
    """Validate complete per-forward proposal and commit evidence."""
    if response.nfe != response.forward_calls:
        _fail("revision worker returned mismatched nfe and forward_calls")
    if len(response.trajectory_steps) != response.nfe:
        _fail("revision worker returned mismatched trajectory length")
    if response.nfe > 0 and not response.forward_blocks:
        _fail("revision worker returned missing forward block evidence")
    if set(response.forward_blocks) != {step.block_index for step in response.trajectory_steps}:
        _fail("revision worker returned mismatched forward block evidence")
    remaining = set(response.remasked_positions)
    committed_positions: list[int] = []
    committed_tokens: list[int] = []
    for rows in _rows_by_step(response.trajectory_steps):
        step_commits = _validate_step(rows, remaining, response.token_ids)
        committed_positions.extend(position for position, _token_id in step_commits)
        committed_tokens.extend(token_id for _position, token_id in step_commits)
        remaining.difference_update(position for position, _token_id in step_commits)
    if len(committed_positions) != len(set(committed_positions)):
        _fail("revision worker trajectory duplicated a commit position")
    if set(committed_positions) != set(response.remasked_positions):
        _fail("revision worker trajectory commits do not match remasked positions")
    if Counter(committed_tokens) != Counter(response.selected_slot_token_ids):
        _fail("revision worker trajectory tokens do not match selected slot tokens")
    if response.trajectory_steps and response.trajectory_steps[-1].remaining_masked != 0:
        _fail("revision worker trajectory did not finish all masks")


def trajectory_sha256(response: SamplerResponse) -> str:
    """Hash all proposal and commit trajectory evidence canonically."""
    document: JsonValue = [
        {
            "block_index": step.block_index,
            "committed_positions": list(step.committed_positions),
            "confidences": list(step.confidences),
            "masked_before": step.masked_before,
            "proposal_confidences": list(step.proposal_confidences),
            "proposal_margins": list(step.proposal_margins),
            "proposal_positions": list(step.proposal_positions),
            "proposal_token_ids": list(step.proposal_token_ids),
            "remaining_masked": step.remaining_masked,
            "step_index": step.step_index,
            "token_ids": list(step.token_ids),
        }
        for step in response.trajectory_steps
    ]
    return sha256(canonical_json(document).encode()).hexdigest()


def _rows_by_step(steps: tuple[TrajectoryStep, ...]) -> tuple[tuple[TrajectoryStep, ...], ...]:
    grouped: dict[int, list[TrajectoryStep]] = {}
    for step in steps:
        _validate_row_lengths(step)
        grouped.setdefault(step.step_index, []).append(step)
    return tuple(tuple(grouped[step_index]) for step_index in sorted(grouped))


def _validate_step(
    rows: tuple[TrajectoryStep, ...],
    remaining: set[int],
    final_token_ids: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    masked_before = rows[0].masked_before
    remaining_masked = rows[0].remaining_masked
    if any(
        row.masked_before != masked_before or row.remaining_masked != remaining_masked
        for row in rows
    ):
        _fail("revision worker returned inconsistent step mask counts")
    if masked_before != len(remaining):
        _fail("revision worker returned impossible masked-before count")
    proposals: set[int] = set()
    commits: list[tuple[int, int]] = []
    for row in rows:
        if proposals.intersection(row.proposal_positions):
            _fail("revision worker duplicated a proposal position within one step")
        proposals.update(row.proposal_positions)
        commits.extend(_validate_row_commits(row, final_token_ids))
    if proposals != remaining:
        _fail("revision worker trajectory lost or duplicated a masked proposal")
    commit_positions = tuple(position for position, _token_id in commits)
    if len(commit_positions) != len(set(commit_positions)):
        _fail("revision worker duplicated a commit position within one step")
    if len(commits) != masked_before - remaining_masked:
        _fail("revision worker returned inconsistent committed count")
    return tuple(commits)


def _validate_row_commits(
    step: TrajectoryStep, final_token_ids: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    proposal_indexes = {position: index for index, position in enumerate(step.proposal_positions)}
    commits: list[tuple[int, int]] = []
    for position, token_id, confidence in zip(
        step.committed_positions, step.token_ids, step.confidences, strict=True
    ):
        proposal_index = proposal_indexes.get(position)
        if proposal_index is None:
            _fail("revision worker committed a position without a proposal")
        if step.proposal_token_ids[proposal_index] != token_id:
            _fail("revision worker committed a token different from its proposal")
        if step.proposal_confidences[proposal_index] != confidence:
            _fail("revision worker committed confidence differs from its proposal")
        if final_token_ids[position] != token_id:
            _fail("revision worker trajectory token does not match final state")
        commits.append((position, token_id))
    return tuple(commits)


def _validate_row_lengths(step: TrajectoryStep) -> None:
    commit_count = len(step.committed_positions)
    if len(step.token_ids) != commit_count or len(step.confidences) != commit_count:
        _fail("revision worker returned mismatched trajectory commit lengths")
    proposal_count = len(step.proposal_positions)
    if proposal_count == 0:
        _fail("revision worker returned empty trajectory proposal")
    if (
        len(step.proposal_token_ids) != proposal_count
        or len(step.proposal_confidences) != proposal_count
        or len(step.proposal_margins) != proposal_count
    ):
        _fail("revision worker returned mismatched trajectory proposal lengths")
    if len(set(step.proposal_positions)) != proposal_count:
        _fail("revision worker returned duplicate positions in one proposal row")
    if any(not 0.0 <= confidence <= 1.0 for confidence in step.proposal_confidences):
        _fail("revision worker returned invalid proposal confidence")
    if any(not -1.0 <= margin <= 1.0 for margin in step.proposal_margins):
        _fail("revision worker returned invalid proposal margin")
    if step.remaining_masked < 0 or step.remaining_masked >= step.masked_before:
        _fail("revision worker returned impossible remaining-mask count")


def _fail(reason: str) -> NoReturn:
    raise WireProtocolError(reason=reason)
