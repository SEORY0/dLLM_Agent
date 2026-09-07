"""Tests for cross-run persistence of the tri-modal field-outcome signal."""

from pathlib import Path

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.controller_models import RevisionResult, RevisionTrajectoryStep
from dllm_agent.revision.failure_registry import (
    CauseBucket,
    FailureRecord,
    FailureRegistryError,
    append,
    failure_records,
    load,
    seed_outcomes,
)
from dllm_agent.revision.models import RegistryMode, RevisionConfig
from dllm_agent.revision.pipeline import load_seed_records, persist_failure_records
from dllm_agent.revision.trajectory_policy import LineageFieldOutcome, choose_lineage_field
from dllm_agent.targets.openexr.particle import FIELD_ORDER, FieldName, SelectorSlots

_PIN_A = sha256_bytes(b"target-a")
_PIN_B = sha256_bytes(b"target-b")


def test_load_returns_empty_tuple_when_file_is_absent(tmp_path: Path) -> None:
    """Given no registry file, when loading, then no records are returned."""
    assert load(tmp_path / "missing.jsonl") == ()


def test_round_trip_append_and_load(tmp_path: Path) -> None:
    """Given one appended record, when loading, then the exact record round-trips."""
    path = tmp_path / "registry.jsonl"
    record = _record(target_pin_sha256=_PIN_A, field="width", cause_bucket="invalid", count=2)

    append(path, record)

    assert load(path) == (record,)


def test_append_accumulates_multiple_records_in_order(tmp_path: Path) -> None:
    """Given two appends, when loading, then both records are present in append order."""
    path = tmp_path / "registry.jsonl"
    first = _record(target_pin_sha256=_PIN_A, field="width", cause_bucket="invalid", count=1)
    second = _record(target_pin_sha256=_PIN_A, field="height", cause_bucket="duplicate", count=3)

    append(path, first)
    append(path, second)

    assert load(path) == (first, second)


def test_load_rejects_a_corrupted_registry_line(tmp_path: Path) -> None:
    """Given a malformed JSONL line, when loading, then the registry fails closed."""
    path = tmp_path / "registry.jsonl"
    _ = path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(FailureRegistryError, match="line 1"):
        _ = load(path)


def test_seed_outcomes_isolates_by_target_pin() -> None:
    """Given records for two target pins, when seeding one pin, then only its records seed."""
    own = _record(target_pin_sha256=_PIN_A, field="width", cause_bucket="invalid", count=2)
    other = _record(target_pin_sha256=_PIN_B, field="width", cause_bucket="invalid", count=9)

    outcomes = seed_outcomes((own, other), lineage_id="p0000", target_pin_sha256=_PIN_A)

    assert len(outcomes) == 1
    assert outcomes[0].lineage_id == "p0000"
    assert outcomes[0].invalid_count == 2


def test_seed_outcomes_maps_each_cause_bucket_to_its_own_outcome_field() -> None:
    """Given one record per cause bucket, when seeding, then each maps to its own counter."""
    records = (
        _record(target_pin_sha256=_PIN_A, field="width", cause_bucket="invalid", count=2),
        _record(target_pin_sha256=_PIN_A, field="width", cause_bucket="duplicate", count=3),
        _record(target_pin_sha256=_PIN_A, field="width", cause_bucket="proof_failure", count=4),
    )

    outcomes = seed_outcomes(records, lineage_id="p0000", target_pin_sha256=_PIN_A)

    counted = {
        (outcome.invalid_count, outcome.duplicate_count, outcome.proof_failure_count)
        for outcome in outcomes
    }
    assert counted == {(2, 0, 0), (0, 3, 0), (0, 0, 4)}


def test_failure_records_explodes_only_nonzero_cause_buckets() -> None:
    """Given an outcome with one zero bucket, when exploding, then only nonzero buckets persist."""
    outcome = LineageFieldOutcome(
        lineage_id="p0000",
        field="width",
        reward_delta=0.5,
        trials=3,
        invalid_count=2,
        duplicate_count=0,
        proof_failure_count=1,
    )

    records = failure_records(outcome, target_pin_sha256=_PIN_A, run_id="run-1")

    buckets = {record.cause_bucket: record.count for record in records}
    assert buckets == {"invalid": 2, "proof_failure": 1}


def test_revision_config_requires_a_path_in_warm_mode() -> None:
    """Given warm mode without a path, when constructing config, then it fails closed."""
    with pytest.raises(ValueError, match="warm registry mode requires"):
        _ = _config(registry_mode="warm", failure_registry_path=None)


def test_load_seed_records_stays_empty_in_cold_mode_even_with_an_existing_file(
    tmp_path: Path,
) -> None:
    """Given a populated registry file, when cold, then no records seed this run."""
    path = tmp_path / "registry.jsonl"
    append(path, _record(target_pin_sha256=_PIN_A, field="width", cause_bucket="invalid", count=5))
    config = _config(
        registry_mode="cold", failure_registry_path=str(path), target_pin_sha256=_PIN_A
    )

    assert load_seed_records(config) == ()


def test_load_seed_records_returns_persisted_records_in_warm_mode(tmp_path: Path) -> None:
    """Given a populated registry file, when warm, then its records are loaded."""
    path = tmp_path / "registry.jsonl"
    record = _record(target_pin_sha256=_PIN_A, field="width", cause_bucket="invalid", count=5)
    append(path, record)
    config = _config(
        registry_mode="warm", failure_registry_path=str(path), target_pin_sha256=_PIN_A
    )

    assert load_seed_records(config) == (record,)


def test_persist_failure_records_appends_regardless_of_registry_mode(tmp_path: Path) -> None:
    """Given a cold-mode run with a path configured, when persisting, then it still writes."""
    path = tmp_path / "registry.jsonl"
    config = _config(
        registry_mode="cold", failure_registry_path=str(path), target_pin_sha256=_PIN_A
    )
    outcomes = (
        LineageFieldOutcome(
            lineage_id="p0000", field="width", reward_delta=1.0, trials=2, invalid_count=1
        ),
    )

    persist_failure_records(config, outcomes, run_id="run-1")

    written = load(path)
    assert len(written) == 1
    assert written[0].field == "width"
    assert written[0].cause_bucket == "invalid"
    assert written[0].count == 1


def test_persist_failure_records_is_a_noop_without_a_configured_path() -> None:
    """Given no failure_registry_path, when persisting, then nothing is written or raised."""
    config = _config(registry_mode="cold", failure_registry_path=None, target_pin_sha256=_PIN_A)
    outcomes = (
        LineageFieldOutcome(
            lineage_id="p0000", field="width", reward_delta=1.0, trials=1, invalid_count=1
        ),
    )

    persist_failure_records(config, outcomes, run_id="run-1")  # must not raise


def test_seeded_outcomes_deterministically_steer_field_choice() -> None:
    """Given the same warm records seeded twice, when choosing, then the field choice matches."""
    result = _result(
        trajectory_steps=(
            _step(
                step_index=1,
                committed_positions=(0, 1, 3, 8),
                token_ids=(30, 31, 33, 38),
                confidences=(0.95, 0.95, 0.95, 0.95),
                remaining_masked=1,
            ),
            _step(
                step_index=3,
                committed_positions=(2,),
                token_ids=(32,),
                confidences=(0.15,),
                remaining_masked=0,
            ),
        ),
    )
    records = (
        _record(
            target_pin_sha256=_PIN_A,
            field="width",
            cause_bucket="invalid",
            count=5,
            reward_delta_sum=-4.0,
        ),
    )

    unseeded_choice = choose_lineage_field(result, outcomes=(), lineage_id="lineage-a")
    first_seed = seed_outcomes(records, lineage_id="lineage-a", target_pin_sha256=_PIN_A)
    second_seed = seed_outcomes(records, lineage_id="lineage-a", target_pin_sha256=_PIN_A)
    first_choice = choose_lineage_field(result, outcomes=first_seed, lineage_id="lineage-a")
    second_choice = choose_lineage_field(result, outcomes=second_seed, lineage_id="lineage-a")

    # By trajectory alone (no seeding) "width" commits latest/least confidently and wins.
    # The warm-seeded penalty on "width" must deterministically steer both seeded calls away
    # from it, to the same alternative field both times.
    assert unseeded_choice == "width"
    assert first_choice == second_choice
    assert first_choice != "width"


def _config(
    *,
    registry_mode: RegistryMode = "cold",
    failure_registry_path: str | None,
    target_pin_sha256: str | None = None,
) -> RevisionConfig:
    return RevisionConfig(
        schema_version="dllm-poc-revision-config/v1",
        particle_count=4,
        epochs=3,
        initial_steps=128,
        revision_steps=16,
        probe_seconds=3,
        fuzzer_seeds=(7, 11),
        proof_seconds=30,
        registry_mode=registry_mode,
        failure_registry_path=failure_registry_path,
        target_pin_sha256=target_pin_sha256,
    )


def _record(
    *,
    target_pin_sha256: str,
    field: FieldName,
    cause_bucket: CauseBucket,
    count: int,
    reward_delta_sum: float = 0.0,
) -> FailureRecord:
    return FailureRecord(
        target_pin_sha256=target_pin_sha256,
        field=field,
        cause_bucket=cause_bucket,
        count=count,
        reward_delta_sum=reward_delta_sum,
        trials=1,
        last_seen_run_id="run-0",
    )


def _result(*, trajectory_steps: tuple[RevisionTrajectoryStep, ...]) -> RevisionResult:
    selectors: SelectorSlots = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    token_ids = tuple(selector + 100 for selector in selectors)
    return RevisionResult(
        selectors=selectors,
        token_ids=token_ids,
        slot_positions=tuple(range(len(FIELD_ORDER))),
        selected_slot_token_ids=token_ids,
        token_state_sha256=sha256_bytes(b"token"),
        transfer_evidence_sha256=sha256_bytes(b"transfer"),
        rng_state_sha256=sha256_bytes(b"rng"),
        nfe=8,
        forward_calls=8,
        gpu_ms=4,
        cache_hit=False,
        trajectory_uncertainty=0.5,
        suggested_field="origin_x",
        trajectory_steps=trajectory_steps,
        trajectory_evidence_sha256=sha256_bytes(b"trajectory"),
    )


def _step(
    *,
    step_index: int,
    committed_positions: tuple[int, ...],
    token_ids: tuple[int, ...],
    confidences: tuple[float, ...],
    remaining_masked: int,
) -> RevisionTrajectoryStep:
    return RevisionTrajectoryStep(
        step_index=step_index,
        block_index=0,
        masked_before=len(committed_positions) + remaining_masked,
        committed_positions=committed_positions,
        token_ids=token_ids,
        confidences=confidences,
        proposal_positions=committed_positions,
        proposal_token_ids=token_ids,
        proposal_confidences=confidences,
        proposal_margins=tuple(0.0 for _ in committed_positions),
        remaining_masked=remaining_masked,
    )
