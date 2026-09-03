"""Strict schemas for feedback-conditioned PoC revision runs."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, computed_field, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Probability, Sha256
from dllm_agent.revision.proof_archive import OnlineProofStatus, OnlineProofVerdict
from dllm_agent.targets.openexr.particle import FIELD_ORDER, FieldName, SelectorSlots

type BuildStatus = Literal["succeeded", "failed", "timed_out", "skipped"]
type RunStatus = Literal["completed", "crashed", "timed_out", "skipped"]
type ControllerMode = Literal["legacy", "online-portfolio"]
type ArchiveDecision = Literal["promoted", "preserved", "dropped", "capped"]
type RevisionEventKind = Literal[
    "particle_started",
    "snapshot_materialized",
    "probe_completed",
    "field_attribution",
    "remask_decided",
    "revision_completed",
    "particle_promoted",
    "particle_dropped",
    "proof_completed",
    "proof_family_capped",
    "proof_missing_input",
]


class RevisionConfig(FrozenModel):
    """Versioned equal-budget revision controller configuration."""

    schema_version: Literal["dllm-poc-revision-config/v1"]
    particle_count: int = Field(gt=0)
    epochs: int = Field(gt=0)
    initial_steps: int = Field(gt=0)
    revision_steps: int = Field(gt=0)
    probe_seconds: int = Field(gt=0)
    fuzzer_seeds: tuple[int, ...] = Field(min_length=2)
    proof_seconds: int = Field(gt=0)
    field_order: tuple[FieldName, ...] = FIELD_ORDER
    controller_mode: ControllerMode = "legacy"
    archive_size: int = Field(default=32, gt=0)
    plateau_epochs: int = Field(default=1, gt=0)
    proof_family_attempt_cap: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def _require_repeatable_epoch_budget(self) -> Self:
        """Reject schedules that cannot run independent fixed-seed probes."""
        if any(seed < 0 for seed in self.fuzzer_seeds):
            msg = "fuzzer seeds must be nonnegative"
            raise ValueError(msg)
        if len(set(self.fuzzer_seeds)) != len(self.fuzzer_seeds):
            msg = "fuzzer seeds must be distinct"
            raise ValueError(msg)
        if self.field_order != FIELD_ORDER:
            msg = "field order must match OpenEXR particle fields"
            raise ValueError(msg)
        if self.archive_size < self.particle_count:
            msg = "archive must hold protected initial lineages"
            raise ValueError(msg)
        return self


class ParticleState(FrozenModel):
    """Controller-visible state for one particle trajectory."""

    particle_id: str = Field(min_length=1)
    selectors: SelectorSlots
    snapshot_sha256: Sha256 | None
    token_state_sha256: Sha256
    rng_state_sha256: Sha256
    nfe: int = Field(ge=0)
    forward_calls: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    dedup_count: int = Field(ge=0)


class SnapshotEvidence(FrozenModel):
    """Materialized snapshot hashes and validation accounting."""

    particle_id: str = Field(min_length=1)
    snapshot_sha256: Sha256
    anchor_sha256: Sha256
    field_map_sha256: Sha256
    materializer_ms: int = Field(ge=0)
    valid: bool
    fault_field: FieldName | None
    byte_length: int = Field(gt=0)


class SnapshotArtifact(FrozenModel):
    """Content-addressed persisted snapshot artifact."""

    snapshot_sha256: Sha256
    path: Path
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _artifact_matches_evidence(self) -> Self:
        try:
            payload = self.path.read_bytes()
        except OSError as error:
            msg = "snapshot artifact unavailable"
            raise ValueError(msg) from error
        if len(payload) != self.size_bytes:
            msg = "snapshot artifact length mismatch"
            raise ValueError(msg)
        if sha256_bytes(payload) != self.snapshot_sha256:
            msg = "snapshot artifact hash mismatch"
            raise ValueError(msg)
        return self


class ProbeFeedback(FrozenModel):
    """Normalized cheap-probe feedback bound to raw logs and argv hashes."""

    particle_id: str = Field(min_length=1)
    snapshot_sha256: Sha256
    build_status: BuildStatus
    run_status: RunStatus
    valid_snapshot: bool
    sanitizer_observed: bool
    target_branches_hit: int | None = Field(default=None, ge=0)
    new_edges: int = Field(ge=0)
    rare_edges: int = Field(ge=0)
    value_profile_features: int = Field(ge=0)
    executions_per_second: int | None = Field(default=None, ge=0)
    trajectory_uncertainty: Probability
    corpus_artifact_sha256: tuple[Sha256, ...]
    wall_ms: int = Field(ge=0)
    build_log_sha256: Sha256
    run_log_sha256: Sha256
    build_argv_sha256: Sha256
    run_argv_sha256: Sha256
    sanitizer_class: str | None = None
    root_signature: str | None = None
    provisional_family_key: Sha256 | None = None
    rejected_count: int = Field(ge=0)
    dedup_count: int = Field(ge=0)

    @computed_field
    @property
    def feedback_sha256(self) -> str:
        """Hash the exact normalized feedback fields used by policy decisions."""
        return sha256_bytes(self.model_dump_json(exclude={"feedback_sha256"}).encode())


class FieldAttribution(FrozenModel):
    """Controlled one-field reward delta attributed after a child probe."""

    field: FieldName
    parent_feedback_sha256: Sha256
    child_feedback_sha256: Sha256
    reward_delta: float
    trials: int = Field(gt=0)
    rejected_count: int = Field(ge=0)
    dedup_count: int = Field(ge=0)


class RemaskDecision(FrozenModel):
    """Deterministic next-field decision and accounting for one epoch."""

    sequence: int = Field(ge=0)
    epoch: int = Field(ge=0)
    fields: tuple[FieldName, ...] = Field(min_length=1)
    decision_sha256: Sha256
    reason: Literal["reward", "ucb", "lexicographic_feedback", "trajectory_bootstrap"]
    total_trials: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    dedup_count: int = Field(ge=0)


class RevisionEvent(FrozenModel):
    """Timestamp-free event row for the deterministic PoC revision ledger."""

    sequence: int = Field(ge=1)
    kind: RevisionEventKind
    particle_id: str = Field(min_length=1)
    snapshot_sha256: Sha256 | None
    parent_sha256: Sha256 | None
    feedback_sha256: Sha256 | None
    remask_sha256: Sha256 | None
    fields: tuple[FieldName, ...]
    nfe: int = Field(ge=0)
    forward_calls: int = Field(ge=0)
    gpu_ms: int = Field(ge=0)
    trajectory_evidence_sha256: Sha256 | None = None
    validator_ms: int = Field(ge=0)
    fuzz_seconds: int = Field(ge=0)
    executions_per_second: int | None = Field(default=None, ge=0)
    cache_hit: bool
    dedup_hit: bool
    artifact_sha256: tuple[Sha256, ...]
    online_proof_status: OnlineProofStatus | None = None
    cause_id: Sha256 | None = None
    proof_family_key: Sha256 | None = None
    archive_snapshot_sha256: tuple[Sha256, ...] = ()
    archive_decision: ArchiveDecision | None = None


class ProofSummary(FrozenModel):
    """Independent proof status summary without vulnerability claims."""

    oracle_status: Literal["proved", "not_reproduced", "crash_without_sanitizer", "skipped"]
    input_sha256: Sha256 | None
    proof_log_sha256: Sha256 | None


class RuntimeAccounting(FrozenModel):
    """Recorded model work and scheduled validation budget for one run."""

    recorded_nfe: int = Field(default=0, ge=0)
    recorded_forward_calls: int = Field(default=0, ge=0)
    recorded_gpu_ms: int = Field(default=0, ge=0)
    probe_invocations: int = Field(default=0, ge=0)
    probe_budget_seconds: int = Field(default=0, ge=0)
    exact_proof_invocations: int = Field(default=0, ge=0)
    proof_budget_seconds: int = Field(default=0, ge=0)


class PocRevisionReport(FrozenModel):
    """Versioned report for the new selective-revision PoC runtime."""

    schema_version: Literal["dllm-poc-revision-report/v1"]
    config_hash_sha256: Sha256
    snapshots: tuple[SnapshotEvidence, ...]
    snapshot_artifacts: tuple[SnapshotArtifact, ...] = ()
    probe_feedback: tuple[ProbeFeedback, ...]
    field_attributions: tuple[FieldAttribution, ...]
    remask_decisions: tuple[RemaskDecision, ...]
    revision_events_sha256: Sha256
    accounting_rejected: int = Field(ge=0)
    accounting_dedup: int = Field(ge=0)
    proof: ProofSummary | None
    online_proof_verdicts: tuple[OnlineProofVerdict, ...] = ()
    archive_snapshot_sha256: tuple[Sha256, ...] = ()
    runtime_accounting: RuntimeAccounting = RuntimeAccounting()
