"""Schema tests for feedback-conditioned PoC revision models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.command import PocRevisionCommandConfig
from dllm_agent.revision.models import (
    PocRevisionReport,
    ProbeFeedback,
    RevisionConfig,
    RevisionEvent,
    RevisionEventKind,
    SnapshotEvidence,
)
from dllm_agent.targets.openexr.particle import FIELD_ORDER

_ZERO_SHA = "0" * 64
_ONE_SHA = "1" * 64
_TWO_SHA = "2" * 64
_ROOT = Path(__file__).parents[1]


def test_revision_config_rejects_budget_that_cannot_form_an_epoch() -> None:
    """Given one fuzzer seed, when config is parsed, then epoch budget is rejected."""
    with pytest.raises(ValidationError):
        _ = RevisionConfig(
            schema_version="dllm-poc-revision-config/v1",
            particle_count=4,
            epochs=3,
            initial_steps=128,
            revision_steps=16,
            probe_seconds=3,
            fuzzer_seeds=(7,),
            proof_seconds=30,
        )


def test_revision_config_preserves_declared_field_order() -> None:
    """Given a valid config, when parsed, then field order matches the materializer."""
    config = RevisionConfig(
        schema_version="dllm-poc-revision-config/v1",
        particle_count=4,
        epochs=3,
        initial_steps=128,
        revision_steps=16,
        probe_seconds=3,
        fuzzer_seeds=(7, 11),
        proof_seconds=30,
    )

    assert config.field_order == FIELD_ORDER


def test_revision_config_defaults_to_legacy_online_controls() -> None:
    """Given omitted online fields, when parsed, then legacy behavior remains default."""
    config = RevisionConfig(
        schema_version="dllm-poc-revision-config/v1",
        particle_count=4,
        epochs=3,
        initial_steps=128,
        revision_steps=16,
        probe_seconds=3,
        fuzzer_seeds=(7, 11),
        proof_seconds=30,
    )

    assert config.controller_mode == "legacy"
    assert config.archive_size >= config.particle_count
    assert config.plateau_epochs > 0
    assert config.proof_family_attempt_cap > 0


def test_revision_config_rejects_archive_too_small_for_initial_anchors() -> None:
    """Given protected initial lineages, when archive is smaller, then parsing rejects it."""
    with pytest.raises(ValidationError, match="protected initial lineages"):
        _ = RevisionConfig(
            schema_version="dllm-poc-revision-config/v1",
            particle_count=2,
            epochs=3,
            initial_steps=128,
            revision_steps=16,
            probe_seconds=3,
            fuzzer_seeds=(7, 11),
            proof_seconds=30,
            controller_mode="online-portfolio",
            archive_size=1,
        )


def test_historical_poc_revision_config_remains_legacy_compatible() -> None:
    """Given the historical config, when parsed, then online mode is not forced."""
    config = PocRevisionCommandConfig.model_validate_json(
        (_ROOT / "configs/openexr-v3.4.14-poc-revision.json").read_bytes()
    )

    assert config.revision.controller_mode == "legacy"
    assert config.revision.epochs == 1


def test_online_portfolio_config_opts_in_with_plateau_epochs() -> None:
    """Given dedicated online config, when parsed, then plateau rounds are exercisable."""
    config = PocRevisionCommandConfig.model_validate_json(
        (_ROOT / "configs/openexr-v3.4.14-online-portfolio.json").read_bytes()
    )

    assert config.revision.controller_mode == "online-portfolio"
    assert config.revision.epochs >= 3
    assert config.revision.archive_size >= config.revision.particle_count


def test_revision_event_rejects_unknown_kind() -> None:
    """Given an unknown ledger kind, when parsed, then the strict union rejects it."""
    event_payload = _event_payload("particle_started")
    event_payload["kind"] = "decision_timestamped"
    with pytest.raises(ValidationError):
        _ = RevisionEvent.model_validate(event_payload)


def test_poc_revision_report_binds_exact_raw_ledger_sha() -> None:
    """Given raw JSONL bytes, when a report is parsed, then its hash field is exact."""
    snapshot = SnapshotEvidence(
        particle_id="p0",
        snapshot_sha256=_ZERO_SHA,
        anchor_sha256=_ONE_SHA,
        field_map_sha256=_TWO_SHA,
        materializer_ms=3,
        valid=True,
        fault_field=None,
        byte_length=64,
    )
    feedback = ProbeFeedback(
        particle_id="p0",
        snapshot_sha256=_ZERO_SHA,
        build_status="succeeded",
        run_status="completed",
        valid_snapshot=True,
        sanitizer_observed=False,
        target_branches_hit=1,
        new_edges=2,
        rare_edges=3,
        value_profile_features=4,
        executions_per_second=100,
        trajectory_uncertainty=0.25,
        corpus_artifact_sha256=(_ONE_SHA,),
        wall_ms=90,
        build_log_sha256=_ZERO_SHA,
        run_log_sha256=_ONE_SHA,
        build_argv_sha256=_TWO_SHA,
        run_argv_sha256=_ZERO_SHA,
        rejected_count=1,
        dedup_count=2,
    )
    raw_ledger = b'{"sequence":1,"kind":"particle_started"}\n'

    report = PocRevisionReport(
        schema_version="dllm-poc-revision-report/v1",
        config_hash_sha256=_ZERO_SHA,
        snapshots=(snapshot,),
        probe_feedback=(feedback,),
        field_attributions=(),
        remask_decisions=(),
        revision_events_sha256=sha256_bytes(raw_ledger),
        accounting_rejected=1,
        accounting_dedup=2,
        proof=None,
    )

    assert report.revision_events_sha256 == sha256_bytes(raw_ledger)


def _event_payload(kind: RevisionEventKind) -> dict[str, str | int | bool | tuple[str, ...] | None]:
    return {
        "sequence": 1,
        "kind": kind,
        "particle_id": "p0",
        "snapshot_sha256": _ZERO_SHA,
        "parent_sha256": None,
        "feedback_sha256": None,
        "remask_sha256": None,
        "fields": (),
        "nfe": 0,
        "forward_calls": 0,
        "gpu_ms": 0,
        "validator_ms": 0,
        "fuzz_seconds": 0,
        "executions_per_second": None,
        "cache_hit": False,
        "dedup_hit": False,
        "artifact_sha256": (),
    }
