"""Policy tests for deterministic feedback attribution and remasking."""

import pytest

from dllm_agent.revision.models import FieldAttribution, ProbeFeedback
from dllm_agent.revision.policy import REMASKABLE_FIELDS, attribute_feedback, decide_remask
from dllm_agent.targets.openexr.particle import FieldName

_ZERO_SHA = "0" * 64
_ONE_SHA = "1" * 64


def test_feedback_changes_selected_remask_field() -> None:
    """Given stalled feedback by field, when remask is decided, then fields differ."""
    chunk = decide_remask(_history(stalled_field="chunk_offset"), epoch=1)
    window = decide_remask(_history(stalled_field="width"), epoch=1)

    assert chunk.fields == (REMASKABLE_FIELDS[0],)
    assert window.fields == (REMASKABLE_FIELDS[0],)


def test_feedback_candidates_ignore_non_remaskable_fields() -> None:
    """Given strong non-effective feedback, when deciding, then field is remaskable."""
    chunk = _feedback(field="chunk_offset", sanitizer=True, new_edges=999)
    width = _feedback(field="width", new_edges=1)

    decision = decide_remask(
        (
            _attr("chunk_offset", chunk.feedback_sha256, reward_delta=999.0, trials=3),
            _attr("width", width.feedback_sha256, reward_delta=1.0, trials=3),
        ),
        epoch=2,
        latest_feedback=(chunk, width),
    )

    assert decision.fields == ("width",)
    assert decision.fields[0] in REMASKABLE_FIELDS


def test_nonpositive_feedback_does_not_force_same_field_reuse() -> None:
    """Given negative matched feedback, when deciding, then policy explores untried arm."""
    origin = _feedback(field="origin_x", new_edges=1)

    decision = decide_remask(
        (_attr("origin_x", origin.feedback_sha256, reward_delta=-1.0, trials=1),),
        epoch=3,
        latest_feedback=(origin,),
    )

    assert decision.fields == ("origin_y",)
    assert decision.reason == "ucb"


def test_positive_feedback_can_exploit_same_remaskable_field() -> None:
    """Given positive matched feedback, when deciding, then policy can reuse that field."""
    origin = _feedback(field="origin_x", new_edges=5)

    decision = decide_remask(
        (_attr("origin_x", origin.feedback_sha256, reward_delta=1.0, trials=1),),
        epoch=4,
        latest_feedback=(origin,),
    )

    assert decision.fields == ("origin_x",)
    assert decision.reason == "lexicographic_feedback"


def test_attribute_feedback_credits_controlled_one_field_delta() -> None:
    """Given one changed field, when child improves, then that field receives credit."""
    parent = _feedback(field="width", target_branches=0, new_edges=1)
    child = _feedback(field="width", target_branches=1, new_edges=1)

    attributions = attribute_feedback(parent, child, ("width",))

    assert attributions == (
        FieldAttribution(
            field="width",
            parent_feedback_sha256=parent.feedback_sha256,
            child_feedback_sha256=child.feedback_sha256,
            reward_delta=1.0,
            trials=1,
            rejected_count=child.rejected_count,
            dedup_count=child.dedup_count,
        ),
    )


def test_attribute_feedback_does_not_reward_higher_trajectory_uncertainty() -> None:
    """Given equal execution feedback, when only uncertainty rises, then reward is neutral."""
    parent = _feedback(field="width", trajectory_uncertainty=0.1)
    child = _feedback(field="width", trajectory_uncertainty=0.9)

    attributions = attribute_feedback(parent, child, ("width",))

    assert attributions[0].reward_delta == 0.0


def test_attribute_feedback_rejects_empty_changed_fields() -> None:
    """Given no controlled field, when attributing, then credit fails closed."""
    parent = _feedback(field="width")
    child = _feedback(field="width", new_edges=1)

    with pytest.raises(ValueError, match="exactly one"):
        _ = attribute_feedback(parent, child, ())


def test_attribute_feedback_rejects_multiple_changed_fields() -> None:
    """Given multiple changed fields, when attributing, then credit fails closed."""
    parent = _feedback(field="width")
    child = _feedback(field="height", new_edges=1)

    with pytest.raises(ValueError, match="exactly one"):
        _ = attribute_feedback(parent, child, ("width", "height"))


def test_reward_order_ignores_sanitizer_when_coverage_differs() -> None:
    """Given sanitizer-only signal, when deciding, then search reward uses coverage."""
    sanitizer = _feedback(field="height", sanitizer=True, new_edges=0)
    coverage = _feedback(field="width", sanitizer=False, new_edges=999)

    decision = decide_remask(
        (
            _attr("height", sanitizer.feedback_sha256, reward_delta=1.0, trials=3),
            _attr("width", coverage.feedback_sha256, reward_delta=999.0, trials=3),
        ),
        epoch=2,
        latest_feedback=(sanitizer, coverage),
    )

    assert decision.fields == ("width",)


def test_attribute_feedback_ignores_sanitizer_only_difference() -> None:
    """Given sanitizer is the only difference, when attributing, then reward is neutral."""
    parent = _feedback(field="width", sanitizer=False, new_edges=5)
    child = _feedback(field="width", sanitizer=True, new_edges=5)

    attribution = attribute_feedback(parent, child, ("width",))[0]

    assert attribution.reward_delta == 0.0


def test_feedback_ties_are_order_invariant_by_canonical_field_name() -> None:
    """Given equal feedback rewards, when input order changes, then field tie is stable."""
    width = _feedback(field="width", new_edges=7)
    height = _feedback(field="height", new_edges=7)
    history = (
        _attr("width", width.feedback_sha256, reward_delta=1.0, trials=3),
        _attr("height", height.feedback_sha256, reward_delta=1.0, trials=3),
    )

    forward = decide_remask(history, epoch=4, latest_feedback=(width, height))
    reverse = decide_remask(history, epoch=4, latest_feedback=(height, width))

    assert forward.fields == ("height",)
    assert reverse.fields == ("height",)


def test_unmatched_feedback_hash_falls_back_to_remaskable_ucb() -> None:
    """Given unmatched feedback, when deciding, then fallback remains remaskable."""
    unmatched = _feedback(field="width", sanitizer=True)

    decision = decide_remask(
        _history(stalled_field="chunk_offset"),
        epoch=5,
        latest_feedback=(unmatched,),
    )

    assert decision.fields == (REMASKABLE_FIELDS[0],)
    assert decision.reason == "ucb"


def test_ucb_uses_exact_total_trials_without_plus_one_boundary() -> None:
    """Given a tight UCB boundary, when deciding, then exact log(total) selects height."""
    decision = decide_remask(
        (
            _attr("origin_x", _ZERO_SHA, reward_delta=-1.0, trials=1),
            _attr("origin_y", _ZERO_SHA, reward_delta=-1.0, trials=1),
            _attr("width", _ZERO_SHA, reward_delta=0.0, trials=1),
            _attr("height", _ONE_SHA, reward_delta=1.0, trials=2),
            _attr("payload_pattern", _ZERO_SHA, reward_delta=-1.0, trials=1),
        ),
        epoch=6,
    )

    assert decision.fields == ("height",)


def test_ucb_ties_are_deterministic_by_field_name() -> None:
    """Given equal rewards and trials, when deciding, then field names break ties."""
    decision = decide_remask(
        (
            _attr("origin_x", _ZERO_SHA, reward_delta=0.0, trials=2),
            _attr("origin_y", _ZERO_SHA, reward_delta=0.0, trials=2),
            _attr("width", _ZERO_SHA, reward_delta=1.0, trials=2),
            _attr("height", _ONE_SHA, reward_delta=1.0, trials=2),
            _attr("payload_pattern", _ZERO_SHA, reward_delta=0.0, trials=2),
        ),
        epoch=3,
    )

    assert decision.fields == ("height",)


def test_decision_hash_binds_feedback_hashes_and_cost_accounting() -> None:
    """Given distinct attribution costs, when selected field matches, then hashes differ."""
    first = decide_remask(
        (_attr("width", _ZERO_SHA, reward_delta=1.0, trials=2),),
        epoch=7,
    )
    second = decide_remask(
        (_attr("width", _ONE_SHA, reward_delta=1.0, trials=2, cost=(1, 2)),),
        epoch=7,
    )

    assert first.fields == second.fields
    assert first.decision_sha256 != second.decision_sha256


def _history(stalled_field: FieldName) -> tuple[FieldAttribution, ...]:
    return (_attr(stalled_field, _ZERO_SHA, reward_delta=0.0, trials=1),)


def _attr(
    field: FieldName,
    child_feedback_sha256: str,
    reward_delta: float,
    trials: int,
    cost: tuple[int, int] = (0, 0),
) -> FieldAttribution:
    return FieldAttribution(
        field=field,
        parent_feedback_sha256=_ZERO_SHA,
        child_feedback_sha256=child_feedback_sha256,
        reward_delta=reward_delta,
        trials=trials,
        rejected_count=cost[0],
        dedup_count=cost[1],
    )


def _feedback(
    field: FieldName,
    *,
    sanitizer: bool = False,
    target_branches: int = 0,
    new_edges: int = 0,
    trajectory_uncertainty: float = 0.0,
) -> ProbeFeedback:
    return ProbeFeedback(
        particle_id=f"p-{field}",
        snapshot_sha256=_ZERO_SHA,
        build_status="succeeded",
        run_status="completed",
        valid_snapshot=True,
        sanitizer_observed=sanitizer,
        target_branches_hit=target_branches,
        new_edges=new_edges,
        rare_edges=0,
        value_profile_features=0,
        executions_per_second=10,
        trajectory_uncertainty=trajectory_uncertainty,
        corpus_artifact_sha256=(),
        wall_ms=20,
        build_log_sha256=_ZERO_SHA,
        run_log_sha256=_ZERO_SHA,
        build_argv_sha256=_ZERO_SHA,
        run_argv_sha256=_ZERO_SHA,
        rejected_count=0,
        dedup_count=0,
    )
