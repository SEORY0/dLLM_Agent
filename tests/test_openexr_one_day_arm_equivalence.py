"""Causal-control equivalence tests for the OpenEXR one-day revision arms."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.controller_models import (
    OnlineControllerRequiredError,
    RevisionRequest,
    RevisionRunRequest,
)
from dllm_agent.revision.models import RevisionConfig
from dllm_agent.revision.pipeline import run_poc_revision
from dllm_agent.targets.openexr.particle import (
    FIELD_ORDER,
    MATERIALIZED_MUTABLE_FIELDS,
)
from dllm_agent.targets.openexr.probe import ProbeResult
from tests.test_poc_revision_pipeline import RecordingProof, TrajectoryDivergenceSampler

if TYPE_CHECKING:
    from pathlib import Path

    from dllm_agent.targets.openexr.probe import ProbeRequest


class RequestRecordingProbe:
    """Probe fake that retains the exact scheduled requests."""

    def __init__(self) -> None:
        self.requests: list[ProbeRequest] = []
        self.snapshot_hashes: list[str] = []

    def run(self, request: ProbeRequest) -> ProbeResult:
        """Record one real probe request before producing deterministic feedback."""
        self.requests.append(request)
        self.snapshot_hashes.append(request.snapshot_sha256)
        crash_bytes = f"crash:{request.snapshot_sha256}".encode()
        return ProbeResult(
            snapshot_sha256=request.snapshot_sha256,
            cache_key_sha256=sha256_bytes(
                f"{request.snapshot_sha256}:{request.fuzzer_seed}".encode()
            ),
            cache_hit=False,
            run_status="crashed",
            sanitizer_observed=True,
            final_units=100,
            coverage_edges=7,
            value_profile_features=3,
            corpus_count=1,
            executions_per_second=10,
            run_log_sha256=sha256_bytes(f"log:{request.snapshot_sha256}".encode()),
            run_argv_sha256=sha256_bytes(f"argv:{request.fuzzer_seed}".encode()),
            crash_artifact_sha256=(sha256_bytes(crash_bytes),),
            crash_artifact_bytes=(crash_bytes,),
            sanitizer_class="heap-buffer-overflow",
            root_signature="matched root",
            provisional_family_key=sha256_bytes(b"matched-family"),
        )


@pytest.mark.parametrize("arm", ["final-only", "full-regenerate"])
def test_one_day_control_fails_closed_before_work_with_default_controller(
    tmp_path: Path,
    arm: str,
) -> None:
    """Given a default controller, a one-day control fails before spending work."""
    # Given
    sampler = TrajectoryDivergenceSampler()
    probe = RequestRecordingProbe()
    proof = RecordingProof("not_reproduced")
    artifact_dir = tmp_path / arm
    config = _default_config()

    # When
    with pytest.raises(OnlineControllerRequiredError) as caught:
        _ = run_poc_revision(
            config,
            _request(arm),
            sampler,
            probe,
            proof,
            artifact_dir,
        )

    # Then
    assert caught.value.arm == arm
    assert caught.value.controller_mode == "legacy"
    assert config.controller_mode == "legacy"
    assert sampler.initial_requests == []
    assert sampler.revision_requests == []
    assert probe.requests == []
    assert proof.requests == []
    assert not artifact_dir.exists()


def test_final_only_matches_selective_online_path_except_trajectory_view(
    tmp_path: Path,
) -> None:
    """Given matched arms, final-only hides trajectory features and preserves the path."""
    # Given
    selective_sampler = TrajectoryDivergenceSampler()
    final_only_sampler = TrajectoryDivergenceSampler()
    selective_probe = RequestRecordingProbe()
    final_only_probe = RequestRecordingProbe()
    selective_proof = RecordingProof("not_reproduced")
    final_only_proof = RecordingProof("not_reproduced")
    config = _online_config()

    # When
    selective_report = run_poc_revision(
        config,
        _request("selective-revision"),
        selective_sampler,
        selective_probe,
        selective_proof,
        tmp_path / "selective",
    )
    final_only_report = run_poc_revision(
        config,
        _request("final-only"),
        final_only_sampler,
        final_only_probe,
        final_only_proof,
        tmp_path / "final-only",
    )

    # Then
    selective_revision = selective_sampler.revision_requests[0]
    final_only_revision = final_only_sampler.revision_requests[0]
    assert selective_revision.changed_field == "width"
    assert final_only_revision.changed_field == "height"
    assert selective_revision.remask_fields == ("width",)
    assert final_only_revision.remask_fields == ("height",)
    assert selective_revision.parent_selectors == final_only_revision.parent_selectors
    assert selective_revision.parent_token_ids == final_only_revision.parent_token_ids
    assert selective_revision.rng_state_sha256 == final_only_revision.rng_state_sha256
    assert selective_revision.feedback_digest == final_only_revision.feedback_digest
    assert selective_revision.seed == final_only_revision.seed
    assert selective_revision.revision_steps == final_only_revision.revision_steps
    assert _probe_schedule(selective_probe) == _probe_schedule(final_only_probe)
    assert selective_probe.requests[0].snapshot == final_only_probe.requests[0].snapshot
    assert selective_report.probe_feedback[0] == final_only_report.probe_feedback[0]
    assert _proof_schedule(selective_proof) == _proof_schedule(final_only_proof)
    assert selective_proof.requests[0] == final_only_proof.requests[0]
    assert selective_report.runtime_accounting == final_only_report.runtime_accounting
    assert _event_kinds(tmp_path / "selective") == _event_kinds(tmp_path / "final-only")


def test_full_regenerate_matches_online_path_with_all_mutable_remask(
    tmp_path: Path,
) -> None:
    """Given matched arms, full regeneration changes only the revision remask scope."""
    # Given
    selective_sampler = TrajectoryDivergenceSampler()
    full_sampler = TrajectoryDivergenceSampler()
    selective_probe = RequestRecordingProbe()
    full_probe = RequestRecordingProbe()
    selective_proof = RecordingProof("not_reproduced")
    full_proof = RecordingProof("not_reproduced")
    config = _online_config()

    # When
    selective_report = run_poc_revision(
        config,
        _request("selective-revision"),
        selective_sampler,
        selective_probe,
        selective_proof,
        tmp_path / "selective",
    )
    full_report = run_poc_revision(
        config,
        _request("full-regenerate"),
        full_sampler,
        full_probe,
        full_proof,
        tmp_path / "full",
    )

    # Then
    selective_revision = selective_sampler.revision_requests[0]
    full_revision = full_sampler.revision_requests[0]
    mutable_bitmap = sum(1 << FIELD_ORDER.index(field) for field in MATERIALIZED_MUTABLE_FIELDS)
    all_slots_bitmap = (1 << len(FIELD_ORDER)) - 1
    assert full_revision.remask_fields == MATERIALIZED_MUTABLE_FIELDS
    assert full_revision.remask_bitmap == mutable_bitmap
    assert full_revision.frozen_bitmap == all_slots_bitmap ^ mutable_bitmap
    assert _with_selective_remask(selective_revision) == _with_selective_remask(full_revision)
    assert _probe_schedule(selective_probe) == _probe_schedule(full_probe)
    assert selective_probe.snapshot_hashes == full_probe.snapshot_hashes
    assert _proof_schedule(selective_proof) == _proof_schedule(full_proof)
    assert selective_proof.requests == full_proof.requests
    assert selective_report.runtime_accounting == full_report.runtime_accounting
    assert _event_kinds(tmp_path / "selective") == _event_kinds(tmp_path / "full")


def _request(arm: str) -> RevisionRunRequest:
    return RevisionRunRequest.model_validate(
        {
            "schema_version": "dllm-poc-revision-run/v1",
            "arm": arm,
            "particle_seed_base": 101,
            "proof_seed": 103,
        }
    )


def _online_config() -> RevisionConfig:
    return RevisionConfig(
        schema_version="dllm-poc-revision-config/v1",
        particle_count=1,
        epochs=1,
        initial_steps=8,
        revision_steps=2,
        probe_seconds=1,
        fuzzer_seeds=(7, 11),
        proof_seconds=5,
        controller_mode="online-portfolio",
        archive_size=8,
        plateau_epochs=1,
        proof_family_attempt_cap=2,
    )


def _default_config() -> RevisionConfig:
    return RevisionConfig.model_validate(_online_config().model_dump(exclude={"controller_mode"}))


def _probe_schedule(probe: RequestRecordingProbe) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((request.fuzzer_seed, request.duration_seconds) for request in probe.requests)
    )


def _proof_schedule(proof: RecordingProof) -> tuple[tuple[int, int], ...]:
    return tuple((request.duration_seconds, request.proof_seed) for request in proof.requests)


def _event_kinds(root: Path) -> tuple[str, ...]:
    return tuple(
        json.loads(line)["event"]["kind"]
        for line in (root / "revision-events.jsonl").read_text(encoding="utf-8").splitlines()
    )


def _with_selective_remask(request: RevisionRequest) -> RevisionRequest:
    bitmap = 1 << FIELD_ORDER.index("width")
    return request.model_copy(
        update={
            "remask_fields": ("width",),
            "remask_bitmap": bitmap,
            "frozen_bitmap": ((1 << len(FIELD_ORDER)) - 1) ^ bitmap,
        }
    )
