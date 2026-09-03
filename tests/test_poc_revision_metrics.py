"""Live PoC revision metric derivation tests."""

import pytest
from tools.poc_revision_experiment_json import JsonObject, JsonParseError
from tools.poc_revision_experiment_metrics import live_metrics_from_revision


def test_live_metrics_derive_probe_maxima_and_mean_execs() -> None:
    """Given completed feedback rows, when metrics derive, then coverage is not flattened."""
    report: JsonObject = {
        "proof": {"oracle_status": "skipped"},
        "probe_feedback": [
            {
                "run_status": "completed",
                "target_branches_hit": 1568,
                "new_edges": 1568,
                "executions_per_second": 1915,
            },
            {
                "run_status": "completed",
                "target_branches_hit": 1510,
                "new_edges": 1510,
                "executions_per_second": 2307,
            },
            {
                "run_status": "timed_out",
                "target_branches_hit": 9999,
                "new_edges": 9999,
                "executions_per_second": 9999,
            },
        ],
    }

    metrics = live_metrics_from_revision(report)

    assert metrics["target_branches"] == 1568
    assert metrics["new_edges"] == 1568
    assert metrics["execs_per_second"] == 2111.0
    assert metrics["proof_status"] == "skipped"
    assert metrics["validated_success"] is False


def test_live_metrics_keep_empty_feedback_unknown_execs() -> None:
    """Given no feedback rows, when metrics derive, then unknown execution rate is null."""
    metrics = live_metrics_from_revision(
        {"proof": {"oracle_status": "proved"}, "probe_feedback": []}
    )

    assert metrics["target_branches"] == 0
    assert metrics["new_edges"] == 0
    assert metrics["execs_per_second"] is None
    assert metrics["validated_success"] is True


def test_live_metrics_accept_online_proved_verdict_without_summary_proof() -> None:
    """Given online proof verdicts, when proof is null, then success is derived."""
    report: JsonObject = {
        "proof": None,
        "online_proof_verdicts": [
            {"oracle_status": "not_reproduced", "cause_id": None},
            {"oracle_status": "proved_duplicate_cause", "cause_id": "a" * 64},
            {"oracle_status": "proved_new_cause", "cause_id": "b" * 64},
        ],
        "runtime_accounting": {
            "recorded_nfe": 12,
            "recorded_forward_calls": 12,
            "recorded_gpu_ms": 34,
            "probe_invocations": 8,
            "probe_budget_seconds": 8,
            "exact_proof_invocations": 3,
            "proof_budget_seconds": 30,
        },
        "probe_feedback": [],
    }

    metrics = live_metrics_from_revision(report)

    assert metrics["validated_success"] is True
    assert metrics["proof_observed"] is True
    assert metrics["proof_status"] == "proved_duplicate_cause"
    assert metrics["unique_proved_causes"] == 2
    assert metrics["recorded_nfe"] == 12
    assert metrics["recorded_gpu_ms"] == 34
    assert metrics["exact_proof_invocations"] == 3


def test_live_metrics_treat_null_proof_without_online_verdict_as_not_observed() -> None:
    """Given no proof summary or verdict, when metrics derive, then proof is not observed."""
    report: JsonObject = {
        "proof": None,
        "online_proof_verdicts": [],
        "probe_feedback": [],
    }

    metrics = live_metrics_from_revision(report)

    assert metrics["validated_success"] is False
    assert metrics["proof_observed"] is False
    assert metrics["proof_status"] == "not_observed"


def test_live_metrics_reject_malformed_completed_feedback_types() -> None:
    """Given malformed completed feedback, when metrics derive, then parsing fails."""
    report: JsonObject = {
        "proof": {"oracle_status": "skipped"},
        "probe_feedback": [
            {
                "run_status": "completed",
                "target_branches_hit": "bad",
                "new_edges": 1,
                "executions_per_second": 10,
            }
        ],
    }

    with pytest.raises(JsonParseError, match="target_branches_hit"):
        _ = live_metrics_from_revision(report)
