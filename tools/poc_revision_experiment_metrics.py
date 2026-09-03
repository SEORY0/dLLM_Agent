# ruff: noqa: D100,D103

from __future__ import annotations

from typing import NoReturn

from tools.poc_revision_experiment_json import (
    JsonObject,
    JsonParseError,
    JsonValue,
    require_array,
    require_int,
    require_str,
)


def live_metrics_from_revision(revision: JsonObject) -> JsonObject:
    proof_status = _proof_status(revision)
    success = proof_status in {"proved", "proved_new_cause", "proved_duplicate_cause"}
    target_branches, new_edges, execs_per_second = _probe_metrics(revision)
    resources = _resource_metrics(revision)
    return {
        "validated_success": success,
        "target_branches": target_branches,
        "new_edges": new_edges,
        "execs_per_second": execs_per_second,
        "proof_status": proof_status,
        "proof_observed": success,
        "unique_proved_causes": _unique_proved_causes(revision),
        "proof_time_seconds": None,
        "proof_censored_seconds": None,
        **resources,
    }


def _proof_status(revision: JsonObject) -> str:
    for verdict in _online_verdicts(revision):
        status = require_str(verdict, "oracle_status")
        if status in {"proved_new_cause", "proved_duplicate_cause"}:
            return status
    value = revision["proof"]
    if value is None:
        return "not_observed"
    if isinstance(value, dict):
        return require_str(value, "oracle_status")
    return _fail("proof: expected object or null")


def _online_verdicts(revision: JsonObject) -> tuple[JsonObject, ...]:
    value = revision.get("online_proof_verdicts", [])
    if not isinstance(value, list):
        _fail("online_proof_verdicts: expected array")
    return tuple(_verdict(item) for item in value)


def _verdict(value: JsonValue) -> JsonObject:
    if isinstance(value, dict):
        return value
    return _fail("online_proof_verdicts: expected object")


def _unique_proved_causes(revision: JsonObject) -> int:
    causes: set[str] = set()
    for verdict in _online_verdicts(revision):
        cause_id = verdict.get("cause_id")
        if cause_id is None:
            continue
        if not isinstance(cause_id, str):
            _fail("online_proof_verdicts.cause_id: expected string or null")
        causes.add(cause_id)
    return len(causes)


def _resource_metrics(revision: JsonObject) -> JsonObject:
    value = revision.get("runtime_accounting")
    names = (
        "recorded_nfe",
        "recorded_forward_calls",
        "recorded_gpu_ms",
        "probe_invocations",
        "probe_budget_seconds",
        "exact_proof_invocations",
        "proof_budget_seconds",
    )
    if value is None:
        return dict.fromkeys(names, 0)
    if not isinstance(value, dict):
        return _fail("runtime_accounting: expected object or null")
    return {name: require_int(value, name) for name in names}


def _probe_metrics(revision: JsonObject) -> tuple[int, int, float | None]:
    target_branches: list[int] = []
    new_edges: list[int] = []
    execs: list[int] = []
    for item in require_array(revision, "probe_feedback"):
        feedback = _feedback(item)
        if require_str(feedback, "run_status") != "completed":
            continue
        target = _int_or_none(feedback, "target_branches_hit")
        if target is not None:
            target_branches.append(target)
        new_edges.append(require_int(feedback, "new_edges"))
        execs_value = _int_or_none(feedback, "executions_per_second")
        if execs_value is not None:
            execs.append(execs_value)
    execs_per_second = None if not execs else sum(execs) / len(execs)
    return (
        max(target_branches, default=0),
        max(new_edges, default=0),
        execs_per_second,
    )


def _feedback(value: JsonValue) -> JsonObject:
    if isinstance(value, dict):
        return value
    return _fail("probe_feedback: expected object")


def _int_or_none(payload: JsonObject, key: str) -> int | None:
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = f"{key}: expected integer or null"
    raise JsonParseError(message)


def _fail(message: str) -> NoReturn:
    raise JsonParseError(message)
