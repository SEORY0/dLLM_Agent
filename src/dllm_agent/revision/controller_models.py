# pyright: reportUnnecessaryComparison=false
# noqa: SIZE_OK
"""Controller-local schemas and protocols for PoC revision orchestration."""

import json
from dataclasses import dataclass
from typing import Literal, Protocol, assert_never, override

from pydantic import Field

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Probability, Sha256
from dllm_agent.revision.models import (
    ArchiveDecision,
    ProbeFeedback,
    ProofSummary,
    RemaskDecision,
    RevisionConfig,
    RevisionEvent,
    RevisionEventKind,
    SnapshotEvidence,
)
from dllm_agent.revision.proof_archive import OnlineProofStatus
from dllm_agent.targets.openexr.particle import FieldName, MaterializedSnapshot, SelectorSlots
from dllm_agent.targets.openexr.probe import ProbeRequest, ProbeResult, provisional_probe_family_key
from dllm_agent.targets.openexr.proof_execution import OPENEXR_PROOF_IMAGE_DIGEST

type HistoricalExperimentArm = Literal[
    "libfuzzer-only",
    "structured-one-shot",
    "feedback-no-remask",
    "pseudo-feedback",
]
type OnlineControlVariant = Literal["final-only", "full-regenerate"]
type OnlineRevisionVariant = Literal["selective-revision"] | OnlineControlVariant
type ExperimentArm = HistoricalExperimentArm | OnlineRevisionVariant
type TrajectoryFeatureView = Literal["trajectory-aware", "final-only"]
type RevisionRemaskScope = Literal["selected-field", "all-mutable"]
type ProofOracleStatus = Literal["proved", "not_reproduced", "crash_without_sanitizer", "skipped"]


class OnlineRevisionPolicy(FrozenModel):
    """Typed treatment switch for online-matched revision variants."""

    variant: OnlineRevisionVariant
    trajectory_feature_view: TrajectoryFeatureView
    remask_scope: RevisionRemaskScope


def online_revision_policy(arm: ExperimentArm) -> OnlineRevisionPolicy | None:
    """Return the matched online treatment policy for a revision arm."""
    match arm:
        case "selective-revision":
            return OnlineRevisionPolicy(
                variant=arm,
                trajectory_feature_view="trajectory-aware",
                remask_scope="selected-field",
            )
        case "final-only":
            return OnlineRevisionPolicy(
                variant=arm,
                trajectory_feature_view="final-only",
                remask_scope="selected-field",
            )
        case "full-regenerate":
            return OnlineRevisionPolicy(
                variant=arm,
                trajectory_feature_view="trajectory-aware",
                remask_scope="all-mutable",
            )
        case "libfuzzer-only" | "structured-one-shot" | "feedback-no-remask" | "pseudo-feedback":
            return None
        case unreachable:
            assert_never(unreachable)


class RevisionRunRequest(FrozenModel):  # noqa: D101
    schema_version: Literal["dllm-poc-revision-run/v1"]
    arm: ExperimentArm
    pseudo_feedback_rotation_seed: int = Field(default=0, ge=0)
    particle_seed_base: int = Field(default=0, ge=0)
    proof_seed: int = Field(default=20260820, ge=0)


class InitialParticleRequest(FrozenModel):  # noqa: D101
    particle_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    steps: int = Field(gt=0)
    arm: ExperimentArm
    mode: Literal["initial"]


class RevisionRequest(FrozenModel):  # noqa: D101
    particle_id: str = Field(min_length=1)
    parent_token_ids: tuple[int, ...]
    parent_selectors: SelectorSlots
    slot_positions: tuple[int, ...]
    remask_fields: tuple[FieldName, ...] = Field(min_length=1)
    changed_field: FieldName
    remask_bitmap: int = Field(gt=0)
    frozen_bitmap: int = Field(ge=0)
    revision_steps: int = Field(gt=0)
    seed: int = Field(ge=0)
    rng_state_sha256: Sha256
    feedback_digest: Sha256
    mode: Literal["selective-revision/v1"]


class RevisionResult(FrozenModel):  # noqa: D101
    selectors: SelectorSlots
    token_ids: tuple[int, ...]
    slot_positions: tuple[int, ...]
    selected_slot_token_ids: tuple[int, ...]
    token_state_sha256: Sha256
    transfer_evidence_sha256: Sha256
    rng_state_sha256: Sha256
    nfe: int = Field(ge=0)
    forward_calls: int = Field(ge=0)
    gpu_ms: int = Field(ge=0)
    cache_hit: bool
    trajectory_uncertainty: Probability
    suggested_field: FieldName
    trajectory_steps: tuple["RevisionTrajectoryStep", ...] = ()
    trajectory_evidence_sha256: Sha256 = Field(default="0" * 64)


class RevisionTrajectoryStep(FrozenModel):  # noqa: D101
    step_index: int = Field(gt=0)
    block_index: int = Field(ge=0)
    masked_before: int = Field(gt=0)
    committed_positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    confidences: tuple[Probability, ...]
    proposal_positions: tuple[int, ...] = ()
    proposal_token_ids: tuple[int, ...] = ()
    proposal_confidences: tuple[Probability, ...] = ()
    proposal_margins: tuple[float, ...] = ()
    remaining_masked: int = Field(ge=0)


class ProofRequest(FrozenModel):  # noqa: D101
    snapshot: bytes = Field(min_length=1)
    snapshot_sha256: Sha256
    duration_seconds: int = Field(gt=0)
    proof_seed: int = Field(ge=0)
    # ponytail: this poc-revision pipeline still assumes the historical baked
    # digest rather than reading the probe's real image_id -- same class of bug
    # the one-day campaign's proof runner had before it was fixed to thread
    # config.search_image_digest through. Not fixed here to keep this PR's
    # blast radius to the one-day campaign; thread ProbeConfig.image_id
    # through RevisionConfig/_RunInputs when this pipeline needs a real digest.
    search_image_digest: str = Field(
        default=OPENEXR_PROOF_IMAGE_DIGEST, pattern=r"^sha256:[0-9a-f]{64}$"
    )


class RevisionSampler(Protocol):  # noqa: D101
    def initial(self, request: InitialParticleRequest) -> RevisionResult: ...  # noqa: D102

    def revise(self, request: RevisionRequest) -> RevisionResult: ...  # noqa: D102


class ProbeRunner(Protocol):  # noqa: D101
    def run(self, request: ProbeRequest) -> ProbeResult: ...  # noqa: D102


class ProofRunnerResult(Protocol):  # noqa: D101
    @property
    def oracle_status(self) -> ProofOracleStatus: ...  # noqa: D102
    @property
    def input_sha256(self) -> Sha256 | None: ...  # noqa: D102
    @property
    def proof_log_sha256(self) -> Sha256 | None: ...  # noqa: D102


class ProofRunner(Protocol):  # noqa: D101
    def run(self, request: ProofRequest) -> ProofRunnerResult: ...  # noqa: D102


@dataclass(frozen=True, slots=True)
class PseudoFeedbackParticleCountError(ValueError):  # noqa: D101
    particle_count: int

    @override
    def __str__(self) -> str:
        return f"pseudo-feedback requires at least two particles, got {self.particle_count}"


@dataclass(frozen=True, slots=True)
class OnlineControllerRequiredError(ValueError):
    """Raised when an online-only causal control is paired with legacy mode."""

    arm: OnlineControlVariant
    controller_mode: Literal["legacy"]

    @override
    def __str__(self) -> str:
        """Return the rejected control and controller mode."""
        return f"{self.arm} requires online-portfolio controller mode, got {self.controller_mode}"


@dataclass(frozen=True, slots=True)
class Particle:  # noqa: D101
    particle_id: str
    result: RevisionResult
    snapshot: MaterializedSnapshot


@dataclass(frozen=True, slots=True)
class CrashArtifact:  # noqa: D101
    sha256: Sha256
    content: bytes
    family_key: Sha256 | None = None


@dataclass(frozen=True, slots=True)
class ProbeRecord:  # noqa: D101
    particle: Particle
    feedback: ProbeFeedback
    cache_hit: bool
    crash_artifacts: tuple[CrashArtifact, ...] = ()


def feedback_from_results(  # noqa: D103
    config: RevisionConfig, particle: Particle, results: tuple[ProbeResult, ...], dedup_count: int
) -> ProbeFeedback:
    return ProbeFeedback(
        particle_id=particle.particle_id,
        snapshot_sha256=particle.snapshot.sha256,
        build_status="skipped",
        run_status=_aggregate_status(tuple(result.run_status for result in results)),
        valid_snapshot=True,
        sanitizer_observed=any(result.sanitizer_observed for result in results),
        target_branches_hit=max(result.coverage_edges or 0 for result in results),
        new_edges=max(result.coverage_edges or 0 for result in results),
        rare_edges=0,
        value_profile_features=max(result.value_profile_features or 0 for result in results),
        executions_per_second=min(result.executions_per_second or 0 for result in results),
        trajectory_uncertainty=particle.result.trajectory_uncertainty,
        corpus_artifact_sha256=tuple(
            artifact for result in results for artifact in result.crash_artifact_sha256
        ),
        wall_ms=config.probe_seconds * 1000,
        build_log_sha256=sha256_bytes(b""),
        run_log_sha256=_hash_tuple(tuple(result.run_log_sha256 for result in results)),
        build_argv_sha256=sha256_bytes(b"[]"),
        run_argv_sha256=_hash_tuple(tuple(result.run_argv_sha256 for result in results)),
        sanitizer_class=_first_present(tuple(result.sanitizer_class for result in results)),
        root_signature=_first_present(tuple(result.root_signature for result in results)),
        provisional_family_key=_probe_family_key(results),
        rejected_count=0,
        dedup_count=dedup_count,
    )


def crash_artifacts_from_results(results: tuple[ProbeResult, ...]) -> tuple[CrashArtifact, ...]:
    """Return hash-deduped, hash-validated crash artifact payloads."""
    by_hash: dict[Sha256, CrashArtifact] = {}
    for result in results:
        for artifact_hash, artifact_bytes in zip(
            result.crash_artifact_sha256,
            result.crash_artifact_bytes,
            strict=True,
        ):
            _ = by_hash.setdefault(
                artifact_hash,
                CrashArtifact(
                    sha256=artifact_hash,
                    content=artifact_bytes,
                    family_key=result.provisional_family_key,
                ),
            )
    return tuple(by_hash[key] for key in sorted(by_hash))


def _aggregate_status(statuses: tuple[str, ...]) -> Literal["completed", "crashed", "timed_out"]:
    if "crashed" in statuses:
        return "crashed"
    if "timed_out" in statuses:
        return "timed_out"
    return "completed"


def proof_if_needed(  # noqa: D103
    config: RevisionConfig,
    proof_runner: ProofRunner,
    proof_seed: int,
    feedback: list[ProbeFeedback],
    snapshot_bytes: dict[str, bytes],
) -> ProofSummary | None:
    sanitizer_hashes = {item.snapshot_sha256 for item in feedback if item.sanitizer_observed}
    if len(feedback) == 0:
        return None
    if len(sanitizer_hashes) == 0:
        return ProofSummary(oracle_status="skipped", input_sha256=None, proof_log_sha256=None)
    for snapshot_sha256 in sorted(snapshot_bytes):
        if snapshot_sha256 in sanitizer_hashes:
            # fmt: off
            proof = proof_runner.run(
                ProofRequest(snapshot=snapshot_bytes[snapshot_sha256], snapshot_sha256=snapshot_sha256, duration_seconds=config.proof_seconds, proof_seed=proof_seed)  # noqa: E501
            )
            return ProofSummary(oracle_status=proof.oracle_status, input_sha256=proof.input_sha256, proof_log_sha256=proof.proof_log_sha256)  # noqa: E501
            # fmt: on
    return None


def snapshot_evidence(particle: Particle) -> SnapshotEvidence:  # noqa: D103
    snapshot = particle.snapshot
    return SnapshotEvidence(
        particle_id=particle.particle_id,
        snapshot_sha256=snapshot.sha256,
        anchor_sha256=snapshot.anchor_sha256,
        field_map_sha256=_hash_tuple(
            tuple(f"{span.field}:{span.start}:{span.end}" for span in snapshot.field_spans)
        ),
        materializer_ms=0,
        valid=True,
        fault_field=snapshot.fault_field,
        byte_length=len(snapshot.content),
    )


def event(  # noqa: D103, PLR0913, PLR0917 - mirrors RevisionEvent fields.
    sequence: int,
    kind: RevisionEventKind,
    particle_id: str,
    particle: Particle | None = None,
    feedback: ProbeFeedback | None = None,
    remask: RemaskDecision | None = None,
    parent_sha256: str | None = None,
    snapshot_sha256: str | None = None,
    feedback_sha256: str | None = None,
    cache_hit: bool = False,
    dedup_hit: bool = False,
    fuzz_seconds: int = 0,
    validator_ms: int = 0,
    artifact_sha256: tuple[str, ...] = (),
    online_proof_status: OnlineProofStatus | None = None,
    cause_id: str | None = None,
    proof_family_key: str | None = None,
    archive_snapshot_sha256: tuple[str, ...] = (),
    archive_decision: ArchiveDecision | None = None,
) -> RevisionEvent:
    # fmt: off
    event_snapshot_sha256 = snapshot_sha256 if snapshot_sha256 is not None else None if particle is None else particle.snapshot.sha256  # noqa: E501
    event_feedback_sha256 = feedback_sha256 if feedback_sha256 is not None else None if feedback is None else feedback.feedback_sha256  # noqa: E501
    # fmt: on
    return RevisionEvent(
        sequence=sequence,
        kind=kind,
        particle_id=particle_id,
        snapshot_sha256=event_snapshot_sha256,
        parent_sha256=parent_sha256,
        feedback_sha256=event_feedback_sha256,
        remask_sha256=None if remask is None else remask.decision_sha256,
        fields=() if remask is None else remask.fields,
        nfe=0 if particle is None else particle.result.nfe,
        forward_calls=0 if particle is None else particle.result.forward_calls,
        gpu_ms=0 if particle is None else particle.result.gpu_ms,
        trajectory_evidence_sha256=None
        if particle is None
        else particle.result.trajectory_evidence_sha256,
        validator_ms=validator_ms,
        fuzz_seconds=fuzz_seconds,
        executions_per_second=None if feedback is None else feedback.executions_per_second,
        cache_hit=cache_hit,
        dedup_hit=dedup_hit,
        artifact_sha256=artifact_sha256,
        online_proof_status=online_proof_status,
        cause_id=cause_id,
        proof_family_key=proof_family_key,
        archive_snapshot_sha256=archive_snapshot_sha256,
        archive_decision=archive_decision,
    )


def _hash_tuple(values: tuple[str, ...]) -> str:
    return sha256_bytes(json.dumps(values, separators=(",", ":")).encode())


def _first_present(values: tuple[str | None, ...]) -> str | None:
    for value in values:
        if value is not None:
            return value
    return None


def _probe_family_key(results: tuple[ProbeResult, ...]) -> str | None:
    explicit = _first_present(tuple(result.provisional_family_key for result in results))
    if explicit is not None:
        return explicit
    sanitizer_class = _first_present(tuple(result.sanitizer_class for result in results))
    root_signature = _first_present(tuple(result.root_signature for result in results))
    if sanitizer_class is None or root_signature is None:
        return None
    return provisional_probe_family_key(sanitizer_class, root_signature)
