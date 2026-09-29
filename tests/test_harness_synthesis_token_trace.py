"""Per-position denoising trajectory records: permanence across remask cycles."""

import pytest

from dllm_agent.harness_synthesis.token_trace import (
    TokenTrace,
    commit_confidence,
    permanence_step,
    position_traces,
)


def _masked(step: int, position: int) -> TokenTrace:
    return TokenTrace(step_index=step, position=position, token_id=99, masked=True)


def _committed(
    step: int, position: int, token_id: int = 1, confidence: float = 0.9
) -> TokenTrace:
    return TokenTrace(
        step_index=step,
        position=position,
        token_id=token_id,
        masked=False,
        confidence=confidence,
        entropy=0.1,
        margin=0.5,
    )


def test_position_traces_filters_and_sorts_by_step() -> None:
    # Given a mixed-position, out-of-order stream
    stream = (
        _committed(2, position=0),
        _masked(0, position=1),
        _masked(1, position=0),
        _committed(1, position=1),
    )

    # When selecting one position's traces
    selected = position_traces(stream, position=0)

    # Then only that position remains, in ascending step order
    assert [t.step_index for t in selected] == [1, 2]
    assert all(t.position == 0 for t in selected)


def test_permanence_step_is_none_while_a_position_stays_masked() -> None:
    # Given a position never committed
    traces = (_masked(0, 0), _masked(1, 0))

    # Then it has no permanence step
    assert permanence_step(traces) is None


def test_permanence_step_is_none_for_an_empty_trace() -> None:
    assert permanence_step(()) is None


def test_permanence_step_is_the_single_commit_when_never_remasked() -> None:
    # Given a position that commits once and stays committed
    traces = (_masked(0, 0), _committed(1, 0), _committed(2, 0))

    # Then permanence is the step it first committed
    assert permanence_step(traces) == 1


def test_permanence_step_is_the_later_commit_after_a_remask_cycle() -> None:
    # Given a position that commits, gets remasked, then recommits
    # (the pattern slot_sampler_contracts.TrajectoryStep structurally cannot
    # produce, since GenericSlotSampler never revisits a committed position —
    # see docs/superpowers/specs/2026-09-15-trajectory-guided-harness-synthesis.md §4)
    traces = (_masked(0, 0), _committed(1, 0), _masked(2, 0), _committed(3, 0))

    # Then permanence is the LAST commit that stuck, not the first
    assert permanence_step(traces) == 3


def test_permanence_step_is_none_when_the_last_recorded_state_is_masked() -> None:
    # Given a position committed earlier but remasked with no later observation
    traces = (_masked(0, 0), _committed(1, 0), _masked(2, 0))

    # Then it has not permanently committed as of the last observation
    assert permanence_step(traces) is None


def test_commit_confidence_reads_the_confidence_at_the_permanence_step() -> None:
    # Given two commit attempts with different confidence
    traces = (
        _masked(0, 0),
        _committed(1, 0, confidence=0.4),
        _masked(2, 0),
        _committed(3, 0, confidence=0.95),
    )

    # Then the confidence returned is the one at the commit that stuck
    assert commit_confidence(traces) == pytest.approx(0.95)


def test_commit_confidence_is_none_when_there_is_no_permanence_step() -> None:
    traces = (_masked(0, 0), _committed(1, 0), _masked(2, 0))

    assert commit_confidence(traces) is None


def test_masked_trace_rejects_a_confidence_value() -> None:
    with pytest.raises(ValueError, match="masked"):
        TokenTrace(step_index=0, position=0, token_id=99, masked=True, confidence=0.5)


def test_committed_trace_requires_a_confidence_value() -> None:
    with pytest.raises(ValueError, match="commit confidence"):
        TokenTrace(step_index=0, position=0, token_id=1, masked=False)
