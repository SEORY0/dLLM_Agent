"""Dream commit-trajectory bookkeeping: assembling and cross-checking TokenTrace streams."""

import pytest

from dllm_agent.harness_synthesis.trajectory_hook import (
    CommitObservation,
    TrajectoryRecorder,
    TrajectoryRecorderError,
)

_MASK = 99


def _commit(position: int, token_id: int = 1, confidence: float = 0.9) -> CommitObservation:
    return CommitObservation(
        position=position, token_id=token_id, confidence=confidence, entropy=0.1, margin=0.5
    )


def test_the_seed_call_records_no_traces() -> None:
    recorder = TrajectoryRecorder(mask_token_id=_MASK)

    recorder.observe(None, [_MASK, _MASK])

    assert recorder.traces == ()


def test_a_step_with_no_commits_records_nothing() -> None:
    recorder = TrajectoryRecorder(mask_token_id=_MASK)
    recorder.observe(None, [_MASK, _MASK])

    recorder.observe(0, [_MASK, _MASK])

    assert recorder.traces == ()


def test_a_reported_commit_is_recorded_with_its_features() -> None:
    recorder = TrajectoryRecorder(mask_token_id=_MASK)
    recorder.observe(None, [_MASK, _MASK])

    recorder.observe(3, [1, _MASK], commits=[_commit(0, token_id=1, confidence=0.42)])

    assert len(recorder.traces) == 1
    trace = recorder.traces[0]
    assert trace.step_index == 3
    assert trace.position == 0
    assert trace.token_id == 1
    assert trace.masked is False
    assert trace.confidence == pytest.approx(0.42)


def test_simultaneous_commits_are_all_recorded() -> None:
    recorder = TrajectoryRecorder(mask_token_id=_MASK)
    recorder.observe(None, [_MASK, _MASK, _MASK])

    recorder.observe(1, [0, 2, _MASK], commits=[_commit(0), _commit(1)])

    assert sorted(trace.position for trace in recorder.traces) == [0, 1]


def test_already_committed_positions_accumulate_across_steps() -> None:
    recorder = TrajectoryRecorder(mask_token_id=_MASK)
    recorder.observe(None, [_MASK, _MASK])
    recorder.observe(0, [1, _MASK], commits=[_commit(0)])

    recorder.observe(1, [1, 2], commits=[_commit(1)])

    assert [trace.position for trace in recorder.traces] == [0, 1]
    assert [trace.step_index for trace in recorder.traces] == [0, 1]


def test_a_missing_commit_for_a_newly_committed_position_raises() -> None:
    recorder = TrajectoryRecorder(mask_token_id=_MASK)
    recorder.observe(None, [_MASK])

    with pytest.raises(TrajectoryRecorderError, match="newly committed"):
        recorder.observe(0, [1])  # token_ids says position 0 committed; no commit given


def test_a_stale_commit_for_a_still_masked_position_raises() -> None:
    recorder = TrajectoryRecorder(mask_token_id=_MASK)
    recorder.observe(None, [_MASK])

    with pytest.raises(TrajectoryRecorderError, match="newly committed"):
        recorder.observe(0, [_MASK], commits=[_commit(0)])  # position 0 is still masked


def test_a_step_before_the_seed_call_raises() -> None:
    recorder = TrajectoryRecorder(mask_token_id=_MASK)

    with pytest.raises(TrajectoryRecorderError, match=r"observe\(None"):
        recorder.observe(0, [_MASK])


def test_traces_is_an_immutable_snapshot() -> None:
    recorder = TrajectoryRecorder(mask_token_id=_MASK)
    recorder.observe(None, [_MASK])

    first = recorder.traces
    recorder.observe(0, [1], commits=[_commit(0)])

    assert first == ()
    assert len(recorder.traces) == 1
