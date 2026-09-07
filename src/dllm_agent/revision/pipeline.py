# pyright: reportUnnecessaryComparison=false
# noqa: SIZE_OK
"""Deterministic feedback-conditioned PoC revision controller."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, Unpack, assert_never

import anyio

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.archive import (
    ArchiveCandidateCost,
    ArchiveCandidateKey,
    ArchiveSelection,
    ProofNovelty,
    ProofNoveltyKey,
    select_parent_child,
)
from dllm_agent.revision.artifacts import revision_events_bytes, revision_events_sha256
from dllm_agent.revision.controller_helpers import (
    attribute_records,
    decide_next,
    initial_result,
    materialized_particle,
    probe_particles,
    reject_impossible_pseudo_feedback,
    resample_unique,
    revise_unique,
)
from dllm_agent.revision.controller_models import (
    CrashArtifact,
    OnlineControllerRequiredError,
    OnlineRevisionPolicy,
    Particle,
    ProbeRecord,
    ProbeRunner,
    ProofRequest,
    ProofRunner,
    ProofRunnerResult,
    RevisionResult,
    RevisionRunRequest,
    RevisionSampler,
    event,
    online_revision_policy,
    proof_if_needed,
    snapshot_evidence,
)
from dllm_agent.revision.failure_registry import (
    FailureRecord,
    append_many,
    failure_records,
    load,
    records_sha256,
    seed_outcomes,
)
from dllm_agent.revision.models import (
    ArchiveDecision,
    FieldAttribution,
    PocRevisionReport,
    ProbeFeedback,
    ProofSummary,
    RemaskDecision,
    RevisionConfig,
    RevisionEvent,
    RevisionEventKind,
    RuntimeAccounting,
    SnapshotArtifact,
    SnapshotEvidence,
)
from dllm_agent.revision.policy import attribute_feedback
from dllm_agent.revision.portfolio import (
    LineageState,
    PortfolioPolicy,
    choose_backtrack_key,
    revision_due,
    update_lineage,
)
from dllm_agent.revision.proof_archive import (
    CauseArchive,
    CompletedProofStatus,
    OnlineProofStatus,
    OnlineProofVerdict,
    classify_online_proof,
)
from dllm_agent.revision.search_tree import select_lineage_frontier
from dllm_agent.revision.trajectory_policy import (
    LineageFieldOutcome,
    choose_lineage_field,
    field_trajectories,
)
from dllm_agent.targets.openexr.particle import FieldName


class _EventEvidence(TypedDict, total=False):
    particle: Particle | None
    feedback: ProbeFeedback | None
    remask: RemaskDecision | None
    parent_sha256: str | None
    snapshot_sha256: str | None
    feedback_sha256: str | None
    cache_hit: bool
    dedup_hit: bool
    fuzz_seconds: int
    validator_ms: int
    artifact_sha256: tuple[str, ...]
    online_proof_status: OnlineProofStatus | None
    cause_id: str | None
    proof_family_key: str | None
    archive_snapshot_sha256: tuple[str, ...]
    archive_decision: ArchiveDecision | None


@dataclass(frozen=True, slots=True)
class _RunInputs:
    config: RevisionConfig
    request: RevisionRunRequest
    sampler: RevisionSampler
    probe: ProbeRunner
    proof_runner: ProofRunner
    artifact_dir: Path


@dataclass(slots=True)  # noqa: MUTABLE_OK
class _RunState:
    events: list[RevisionEvent] = field(default_factory=list)
    snapshots: list[SnapshotEvidence] = field(default_factory=list)
    feedback: list[ProbeFeedback] = field(default_factory=list)
    attributions: list[FieldAttribution] = field(default_factory=list)
    decisions: list[RemaskDecision] = field(default_factory=list)
    snapshot_bytes: dict[str, bytes] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    rejected: int = 0
    dedup: int = 0
    online_proofs: list[OnlineProofVerdict] = field(default_factory=list)
    field_outcomes: list[LineageFieldOutcome] = field(default_factory=list)
    proof_archive: CauseArchive = field(default_factory=CauseArchive)
    proof_novelty: dict[ProofNoveltyKey, ProofNovelty] = field(default_factory=dict)
    proof_failure_counts: dict[ArchiveCandidateKey, int] = field(default_factory=dict)
    archive_costs: dict[ArchiveCandidateKey, ArchiveCandidateCost] = field(default_factory=dict)
    archive_snapshot_sha256: tuple[str, ...] = ()
    proof: ProofSummary | None = None
    proofed_snapshot_sha256: set[str] = field(default_factory=set)
    exact_proof_invocations: int = 0
    failure_registry_seed_sha256: str | None = None

    def append_event(
        self, kind: RevisionEventKind, particle_id: str, **evidence: Unpack[_EventEvidence]
    ) -> None:
        self.events.append(event(len(self.events) + 1, kind, particle_id, **evidence))

    def add_snapshot(self, particle: Particle) -> None:
        self.snapshots.append(snapshot_evidence(particle))
        self.snapshot_bytes[particle.snapshot.sha256] = particle.snapshot.content


@dataclass(frozen=True, slots=True)
class _CompletedProof:
    oracle_status: CompletedProofStatus
    input_sha256: str
    proof_log_sha256: str
    target_commit: str
    sanitizer_class: str | None
    root_signature: str | None


@dataclass(frozen=True, slots=True)
class _OnlineProofCandidate:
    record: ProbeRecord
    artifact: CrashArtifact
    family_key: str


def run_poc_revision(  # noqa: D103, PLR0913, PLR0917 - public interface fixed by Task 4.
    config: RevisionConfig,
    run_request: RevisionRunRequest,
    sampler: RevisionSampler,
    probe: ProbeRunner,
    proof_runner: ProofRunner,
    artifact_dir: Path,
) -> PocRevisionReport:
    inputs = _RunInputs(config, run_request, sampler, probe, proof_runner, artifact_dir)
    return anyio.run(_run_poc_revision_async, inputs)


async def _run_poc_revision_async(inputs: _RunInputs) -> PocRevisionReport:
    revision_policy = _online_policy(inputs)
    reject_impossible_pseudo_feedback(inputs.config, inputs.request)
    online_enabled = revision_policy is not None
    state = _RunState()
    particles = _initial_particles(inputs, state)
    latest = await _probe_and_record(inputs, state, particles)

    match inputs.request.arm:
        case "libfuzzer-only" | "structured-one-shot":
            pass
        case "feedback-no-remask":
            latest = await _run_no_remask(inputs, state, latest)
        case "selective-revision" | "final-only" | "full-regenerate":
            if revision_policy is not None:
                latest = await _run_online_selective(inputs, state, latest, revision_policy)
            else:
                latest = await _run_selective(inputs, state, latest)
        case "pseudo-feedback":
            latest = await _run_selective(inputs, state, latest)
        case unreachable:
            assert_never(unreachable)

    if not online_enabled:
        state.proof = proof_if_needed(
            inputs.config,
            inputs.proof_runner,
            inputs.request.proof_seed,
            state.feedback,
            state.snapshot_bytes,
        )
    if state.proof is not None and not online_enabled:
        state.append_event(
            "proof_completed",
            "proof",
            snapshot_sha256=state.proof.input_sha256,
            feedback_sha256=state.proof.proof_log_sha256,
            artifact_sha256=()
            if state.proof.proof_log_sha256 is None
            else (state.proof.proof_log_sha256,),
        )
    ledger_bytes = revision_events_bytes(tuple(state.events))
    await anyio.Path(inputs.artifact_dir).mkdir(parents=True, exist_ok=True)
    _ = await anyio.Path(inputs.artifact_dir / "revision-events.jsonl").write_bytes(ledger_bytes)
    snapshot_artifacts = await _write_snapshot_artifacts(inputs.artifact_dir, state)
    _ = latest
    config_hash_sha256 = sha256_bytes(inputs.config.model_dump_json().encode())
    registry_run_id = _registry_run_id(inputs.config, inputs.request)
    persist_failure_records(inputs.config, state.field_outcomes, run_id=registry_run_id)
    return PocRevisionReport(
        schema_version="dllm-poc-revision-report/v1",
        config_hash_sha256=config_hash_sha256,
        snapshots=tuple(state.snapshots),
        snapshot_artifacts=snapshot_artifacts,
        probe_feedback=tuple(state.feedback),
        field_attributions=tuple(state.attributions),
        remask_decisions=tuple(state.decisions),
        revision_events_sha256=revision_events_sha256(ledger_bytes),
        accounting_rejected=state.rejected,
        accounting_dedup=state.dedup,
        proof=state.proof,
        online_proof_verdicts=tuple(state.online_proofs),
        archive_snapshot_sha256=state.archive_snapshot_sha256,
        runtime_accounting=_runtime_accounting(inputs, state, online_enabled),
        failure_registry_seed_sha256=state.failure_registry_seed_sha256,
    )


async def _write_snapshot_artifacts(
    artifact_dir: Path, state: _RunState
) -> tuple[SnapshotArtifact, ...]:
    snapshot_dir = artifact_dir / "snapshots"
    await anyio.Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    artifacts: list[SnapshotArtifact] = []
    for evidence in state.snapshots:
        payload = state.snapshot_bytes[evidence.snapshot_sha256]
        path = snapshot_dir / f"{evidence.snapshot_sha256}.exr"
        temporary = snapshot_dir / f".{evidence.snapshot_sha256}.tmp"
        if path.exists():
            if path.read_bytes() != payload:
                msg = "snapshot artifact hash collision"
                raise ValueError(msg)
        else:
            _ = await anyio.Path(temporary).write_bytes(payload)
            _ = await anyio.Path(temporary).replace(path)
        artifacts.append(
            SnapshotArtifact(
                snapshot_sha256=evidence.snapshot_sha256,
                path=path,
                size_bytes=len(payload),
            )
        )
    return tuple(artifacts)


def _online_policy(inputs: _RunInputs) -> OnlineRevisionPolicy | None:
    policy = online_revision_policy(inputs.request.arm)
    match inputs.config.controller_mode:
        case "online-portfolio":
            return policy
        case "legacy":
            variant = None if policy is None else policy.variant
            match variant:
                case None | "selective-revision":
                    return None
                case "final-only":
                    raise OnlineControllerRequiredError(arm="final-only", controller_mode="legacy")
                case "full-regenerate":
                    raise OnlineControllerRequiredError(
                        arm="full-regenerate", controller_mode="legacy"
                    )
                case unreachable:
                    assert_never(unreachable)
        case unreachable:
            assert_never(unreachable)


def _initial_particles(inputs: _RunInputs, state: _RunState) -> list[Particle]:
    particles: list[Particle] = []
    for index in range(inputs.config.particle_count):
        particle_id = f"p{index:04d}"
        state.append_event("particle_started", particle_id)
        result = initial_result(inputs.config, inputs.request, inputs.sampler, particle_id, index)
        particle = None if result is None else materialized_particle(particle_id, result)
        if particle is None:
            state.rejected += 1
            continue
        state.add_snapshot(particle)
        # fmt: off
        state.append_event("snapshot_materialized", particle_id, particle=particle, cache_hit=particle.result.cache_hit)  # noqa: E501
        # fmt: on
        if particle.snapshot.sha256 in state.seen:
            state.dedup += 1
            continue
        state.seen.add(particle.snapshot.sha256)
        state.archive_costs[_particle_key(particle)] = _cost_from_result(particle.result)
        particles.append(particle)
    return particles


async def _run_no_remask(
    inputs: _RunInputs, state: _RunState, latest: list[ProbeRecord]
) -> list[ProbeRecord]:
    for epoch in range(inputs.config.epochs):
        resampled, duplicate_resampled, dedup, rejected = resample_unique(
            inputs.config, inputs.request, inputs.sampler, epoch, state.seen
        )
        state.dedup += dedup
        state.rejected += rejected
        _record_snapshots(state, resampled)
        _record_snapshots(state, duplicate_resampled, dedup_hit=True)
        records = await _probe_and_record(inputs, state, resampled)
        latest = records or latest
    return latest


async def _run_selective(
    inputs: _RunInputs, state: _RunState, latest: list[ProbeRecord]
) -> list[ProbeRecord]:
    for epoch in range(inputs.config.epochs):
        if len(latest) == 0:
            break
        remask = decide_next(epoch, latest, state.attributions)
        state.decisions.append(remask)
        state.append_event("remask_decided", "controller", remask=remask)
        revised, duplicate_revised, dedup, rejected = revise_unique(
            inputs.config, inputs.request, inputs.sampler, latest, remask, state.seen, epoch
        )
        state.dedup += dedup
        state.rejected += rejected
        parents = {
            record.particle.particle_id: record.particle.snapshot.sha256 for record in latest
        }
        _record_snapshots(state, revised, remask, parents)
        _record_snapshots(state, duplicate_revised, remask, parents, dedup_hit=True)
        records = await _probe_and_record(inputs, state, revised)
        new_attributions = attribute_records(inputs.request, latest, records, remask)
        state.attributions.extend(new_attributions)
        for attribution in new_attributions:
            state.append_event("field_attribution", attribution.field)
        latest = records
    return latest


async def _run_online_selective(
    inputs: _RunInputs,
    state: _RunState,
    latest: list[ProbeRecord],
    treatment_policy: OnlineRevisionPolicy,
) -> list[ProbeRecord]:
    portfolio_policy = PortfolioPolicy(
        plateau_epochs=inputs.config.plateau_epochs,
        proof_family_attempt_cap=inputs.config.proof_family_attempt_cap,
    )
    lineages = {
        record.particle.particle_id: LineageState(lineage_id=record.particle.particle_id)
        for record in latest
    }
    active = {record.particle.particle_id: record for record in latest}
    protected = frozenset(_record_key(record) for record in latest)
    archive = list(latest)
    state.archive_snapshot_sha256 = tuple(
        record.particle.snapshot.sha256 for record in sorted(latest, key=_record_key)
    )
    await _run_online_proofs(inputs, state, latest)
    for record in latest:
        lineages[record.particle.particle_id] = update_lineage(
            lineages[record.particle.particle_id],
            record.feedback,
            proof_family_id=record.feedback.provisional_family_key,
            proof_attempted=_feedback_attempted(state, record.feedback),
        )
    seed_records = load_seed_records(inputs.config)
    state.failure_registry_seed_sha256 = records_sha256(seed_records)

    for epoch in range(inputs.config.epochs):
        due = _online_revision_records(epoch, active, lineages, portfolio_policy)
        eligible = _revision_uncapped_records(inputs, state, due)
        if len(eligible) == 0:
            break
        revised: list[Particle] = []
        parents: dict[str, ProbeRecord] = {}
        remasks: dict[str, RemaskDecision] = {}
        for record in eligible:
            lineage_id = record.particle.particle_id
            parent_record = _backtrack_parent(record, archive, protected, inputs.config, epoch)
            controller_result = _controller_visible_result(
                parent_record.particle.result, treatment_policy
            )
            field_name = choose_lineage_field(
                controller_result,
                _seeded_outcomes(state, seed_records, inputs.config, lineage_id),
                lineage_id=lineage_id,
            )
            remask = _online_remask_decision(
                state, epoch, parent_record, field_name, controller_result
            )
            state.decisions.append(remask)
            state.append_event("remask_decided", lineage_id, remask=remask)
            children, duplicates, dedup, rejected = revise_unique(
                inputs.config,
                inputs.request,
                inputs.sampler,
                [parent_record],
                remask,
                state.seen,
                epoch,
            )
            state.dedup += dedup
            state.rejected += rejected
            _record_failed_revision_outcome(
                state,
                lineage_id=lineage_id,
                remask=remask,
                duplicate_count=dedup,
                invalid_count=rejected,
            )
            parent_sha = {lineage_id: parent_record.particle.snapshot.sha256}
            _record_snapshots(state, children, remask, parent_sha)
            _record_snapshots(state, duplicates, remask, parent_sha, dedup_hit=True)
            for child in children:
                revised.append(child)
                parents[child.particle_id] = parent_record
                remasks[child.particle_id] = remask
                parent_cost = state.archive_costs[_record_key(parent_record)]
                state.archive_costs[_particle_key(child)] = _add_costs(
                    parent_cost, _cost_from_result(child.result)
                )
        child_records = await _probe_and_record(inputs, state, revised)
        await _run_online_proofs(inputs, state, child_records)
        _record_online_attributions(state, parents, child_records, remasks)
        children_by_id = {record.particle.particle_id: record for record in child_records}
        selection = _select_online_archive(inputs, state, archive, child_records, protected)
        archive = [candidate.record for candidate in selection.active]
        selected_child_keys = frozenset(selection.active_keys)
        for child in child_records:
            parent = parents[child.particle.particle_id]
            _apply_archive_decision(state, active, parent, child, selected_child_keys)
        for parent in eligible:
            particle_id = parent.particle.particle_id
            observed = children_by_id.get(particle_id, parent)
            lineages[particle_id] = update_lineage(
                lineages[particle_id],
                observed.feedback,
                proof_family_id=observed.feedback.provisional_family_key,
                proof_attempted=_feedback_attempted(state, observed.feedback),
            )
        latest = [active[key] for key in sorted(active)]
    return latest


def _apply_archive_decision(
    state: _RunState,
    active: dict[str, ProbeRecord],
    parent: ProbeRecord,
    child: ProbeRecord,
    selected_child_keys: frozenset[ArchiveCandidateKey],
) -> None:
    if _record_key(child) in selected_child_keys:
        active[child.particle.particle_id] = child
        state.append_event(
            "particle_promoted",
            child.particle.particle_id,
            particle=child.particle,
            feedback=child.feedback,
            parent_sha256=parent.particle.snapshot.sha256,
            archive_decision="promoted",
            archive_snapshot_sha256=state.archive_snapshot_sha256,
        )
    else:
        active[parent.particle.particle_id] = parent
        state.append_event(
            "particle_dropped",
            child.particle.particle_id,
            particle=child.particle,
            feedback=child.feedback,
            parent_sha256=parent.particle.snapshot.sha256,
            archive_decision="dropped",
            archive_snapshot_sha256=state.archive_snapshot_sha256,
        )


def _backtrack_parent(
    record: ProbeRecord,
    archive: list[ProbeRecord],
    protected: frozenset[ArchiveCandidateKey],
    config: RevisionConfig,
    epoch: int,
) -> ProbeRecord:
    """Pick this epoch's expansion point for one lineage from its frontier.

    `frontier_width<=1` always returns `record` unchanged: `_lineage_frontier_keys`
    short-circuits to the single-element `(record's own key,)` without ever calling
    `select_lineage_frontier`, so `choose_backtrack_key` has nothing to pick but that
    same key -- today's single-active-scout choice, by construction.
    """
    frontier_keys = _lineage_frontier_keys(record, archive, protected, config.frontier_width)
    backtrack_key = choose_backtrack_key(record.particle.particle_id, frontier_keys, epoch)
    if backtrack_key == _record_key(record):
        return record
    return _archive_record(archive, backtrack_key, fallback=record)


def _lineage_frontier_keys(
    active_record: ProbeRecord,
    archive: list[ProbeRecord],
    protected: frozenset[ArchiveCandidateKey],
    frontier_width: int,
) -> tuple[ArchiveCandidateKey, ...]:
    """Return up to `frontier_width` candidate keys, always with the active record first."""
    active_key = _record_key(active_record)
    if frontier_width <= 1:
        return (active_key,)
    lineage_frontier = select_lineage_frontier(
        active_record.particle.particle_id,
        tuple(archive),
        (),
        frontier_width=frontier_width,
        protected_ids=protected,
    )
    others = tuple(node.key for node in lineage_frontier.nodes if node.key != active_key)
    return (active_key, *others[: frontier_width - 1])


def _archive_record(
    archive: list[ProbeRecord], key: ArchiveCandidateKey, *, fallback: ProbeRecord
) -> ProbeRecord:
    for candidate in archive:
        if _record_key(candidate) == key:
            return candidate
    return fallback


def load_seed_records(config: RevisionConfig) -> tuple[FailureRecord, ...]:
    """Load persisted failure records only in warm registry mode."""
    if config.registry_mode != "warm" or config.failure_registry_path is None:
        return ()
    return load(Path(config.failure_registry_path))


def _seeded_outcomes(
    state: _RunState,
    seed_records: tuple[FailureRecord, ...],
    config: RevisionConfig,
    lineage_id: str,
) -> tuple[LineageFieldOutcome, ...]:
    seeded = seed_outcomes(seed_records, lineage_id, config.target_pin_sha256 or "")
    return (*state.field_outcomes, *seeded)


def _registry_run_id(config: RevisionConfig, request: RevisionRunRequest) -> str:
    """Derive a persisted run identity from config + this run's own seeds.

    `config_hash_sha256` alone would collapse two run requests that share a
    config but differ in seeds (particle_seed_base/proof_seed/...) onto the
    same registry run_id. Folding in the run request keeps run_id a
    deterministic function of the actual run identity, matching how
    config_hash_sha256 itself is derived (no wall clock / random uuid).
    """
    return sha256_bytes((config.model_dump_json() + request.model_dump_json()).encode())


def persist_failure_records(
    config: RevisionConfig, outcomes: Sequence[LineageFieldOutcome], *, run_id: str
) -> None:
    """Append this run's observed outcomes for a later warm run, regardless of this run's mode.

    Batched into one `append_many` call rather than one `append` per record, so a run with
    many outcomes acquires the registry lock once instead of once per (outcome, cause bucket).
    """
    if config.failure_registry_path is None or config.target_pin_sha256 is None:
        return
    records = tuple(
        record
        for outcome in outcomes
        for record in failure_records(
            outcome, target_pin_sha256=config.target_pin_sha256, run_id=run_id
        )
    )
    append_many(Path(config.failure_registry_path), records)


def _controller_visible_result(
    result: RevisionResult, policy: OnlineRevisionPolicy
) -> RevisionResult:
    """Return the candidate state visible to the selected controller treatment."""
    match policy.trajectory_feature_view:
        case "trajectory-aware":
            return result
        case "final-only":
            return result.model_copy(update={"trajectory_steps": ()})
        case unreachable:
            assert_never(unreachable)


async def _run_online_proofs(
    inputs: _RunInputs, state: _RunState, records: list[ProbeRecord]
) -> None:
    _record_missing_online_proof_inputs(
        state,
        records,
        attempt_cap=inputs.config.proof_family_attempt_cap,
    )
    for candidate in online_proof_candidates(records, state.proofed_snapshot_sha256):
        record = candidate.record
        feedback = record.feedback
        family_key = candidate.family_key
        if not state.proof_archive.can_attempt_family(
            family_key=family_key,
            attempt_cap=inputs.config.proof_family_attempt_cap,
        ):
            state.append_event(
                "proof_family_capped",
                record.particle.particle_id,
                particle=record.particle,
                feedback=feedback,
                proof_family_key=family_key,
                archive_decision="capped",
            )
            continue
        await _persist_online_proof_input(inputs.artifact_dir, candidate.artifact)
        state.exact_proof_invocations += 1
        proof = inputs.proof_runner.run(
            ProofRequest(
                snapshot=candidate.artifact.content,
                snapshot_sha256=candidate.artifact.sha256,
                duration_seconds=inputs.config.proof_seconds,
                proof_seed=inputs.request.proof_seed,
            )
        )
        state.proofed_snapshot_sha256.add(candidate.artifact.sha256)
        state.proof_archive = state.proof_archive.record_attempt(family_key)
        summary = _proof_summary(proof)
        state.proof = summary
        completed = _completed_proof(proof, feedback, family_key)
        if completed is None:
            _increment_proof_failure(state, record)
            state.append_event(
                "proof_completed",
                record.particle.particle_id,
                snapshot_sha256=summary.input_sha256,
                feedback_sha256=summary.proof_log_sha256,
                artifact_sha256=()
                if summary.proof_log_sha256 is None
                else (summary.proof_log_sha256,),
                proof_family_key=family_key,
            )
            continue
        verdict = classify_online_proof(completed, state.proof_archive)
        verdict = verdict.model_copy(update={"family_key": family_key})
        if verdict.oracle_status in {"not_reproduced", "crash_without_sanitizer"}:
            _increment_proof_failure(state, record)
        if verdict.cause_id is not None:
            state.proof_archive = state.proof_archive.record_cause(verdict.cause_id)
        state.online_proofs.append(verdict)
        _record_proof_novelty(state, record, _proof_novelty(verdict))
        state.append_event(
            "proof_completed",
            record.particle.particle_id,
            snapshot_sha256=summary.input_sha256,
            feedback_sha256=summary.proof_log_sha256,
            artifact_sha256=() if summary.proof_log_sha256 is None else (summary.proof_log_sha256,),
            online_proof_status=verdict.oracle_status,
            cause_id=verdict.cause_id,
            proof_family_key=verdict.family_key,
        )


def online_proof_candidates(
    records: list[ProbeRecord], proofed_snapshot_sha256: set[str]
) -> tuple[_OnlineProofCandidate, ...]:
    """Return sanitizer crash artifacts whose exact bytes have not already been proved."""
    selected: list[_OnlineProofCandidate] = []
    seen = set(proofed_snapshot_sha256)
    for record in sorted(records, key=lambda item: item.particle.particle_id):
        feedback = record.feedback
        if not feedback.sanitizer_observed:
            continue
        for artifact in record.crash_artifacts:
            if artifact.sha256 in seen:
                continue
            selected.append(
                _OnlineProofCandidate(
                    record=record,
                    artifact=artifact,
                    family_key=artifact.family_key or _family_key(feedback),
                )
            )
            seen.add(artifact.sha256)
    return tuple(selected)


def _record_missing_online_proof_inputs(
    state: _RunState,
    records: list[ProbeRecord],
    *,
    attempt_cap: int,
) -> None:
    for record in sorted(records, key=lambda item: item.particle.particle_id):
        feedback = record.feedback
        if not feedback.sanitizer_observed or len(record.crash_artifacts) > 0:
            continue
        family_key = _family_key(feedback)
        if state.proof_archive.can_attempt_family(
            family_key=family_key,
            attempt_cap=attempt_cap,
        ):
            state.proof_archive = state.proof_archive.record_attempt(family_key)
            _increment_proof_failure(state, record)
            state.append_event(
                "proof_missing_input",
                record.particle.particle_id,
                particle=record.particle,
                feedback=feedback,
                proof_family_key=family_key,
            )
            continue
        state.append_event(
            "proof_family_capped",
            record.particle.particle_id,
            particle=record.particle,
            feedback=feedback,
            proof_family_key=family_key,
            archive_decision="capped",
        )


async def _persist_online_proof_input(root: Path, artifact: CrashArtifact) -> None:
    if sha256_bytes(artifact.content) != artifact.sha256:
        msg = "online proof input hash does not match artifact bytes"
        raise ValueError(msg)
    proof_inputs = anyio.Path(root / "proof-inputs")
    await proof_inputs.mkdir(parents=True, exist_ok=True)
    path = proof_inputs / f"{artifact.sha256}.bin"
    if await path.exists():
        if await path.read_bytes() != artifact.content:
            msg = "persisted proof input does not match hash-bound artifact"
            raise ValueError(msg)
        return
    temporary = proof_inputs / f".{artifact.sha256}.tmp"
    _ = await temporary.write_bytes(artifact.content)
    _ = await temporary.replace(path)


def _online_revision_records(
    epoch: int,
    active: dict[str, ProbeRecord],
    lineages: dict[str, LineageState],
    policy: PortfolioPolicy,
) -> list[ProbeRecord]:
    if epoch == 0:
        return [active[key] for key in sorted(active)]
    return [active[key] for key in sorted(active) if revision_due(lineages[key], policy)]


def _revision_uncapped_records(
    inputs: _RunInputs, state: _RunState, records: list[ProbeRecord]
) -> list[ProbeRecord]:
    eligible: list[ProbeRecord] = []
    for record in records:
        family_key = _family_key(record.feedback)
        if state.proof_archive.can_attempt_family(
            family_key=family_key,
            attempt_cap=inputs.config.proof_family_attempt_cap,
        ):
            eligible.append(record)
            continue
        state.append_event(
            "proof_family_capped",
            record.particle.particle_id,
            particle=record.particle,
            feedback=record.feedback,
            proof_family_key=family_key,
            archive_decision="capped",
        )
    return eligible


def _online_remask_decision(
    state: _RunState,
    epoch: int,
    record: ProbeRecord,
    field_name: FieldName,
    controller_result: RevisionResult,
) -> RemaskDecision:
    particle = record.particle
    particle_id = particle.particle_id
    reason = "trajectory_bootstrap" if epoch == 0 else "lexicographic_feedback"
    lineage_outcomes = tuple(
        outcome for outcome in state.field_outcomes if outcome.lineage_id == particle_id
    )
    payload = {
        "epoch": epoch,
        "field": field_name,
        "field_outcomes": [outcome.model_dump(mode="json") for outcome in lineage_outcomes],
        "particle_id": particle_id,
        "parent_feedback_sha256": record.feedback.feedback_sha256,
        "parent_snapshot_sha256": particle.snapshot.sha256,
        "parent_trajectory_evidence_sha256": particle.result.trajectory_evidence_sha256,
        "reason": reason,
        "sequence": len(state.decisions),
        "trajectory_features": [
            trajectory.model_dump(mode="json")
            for trajectory in field_trajectories(controller_result)
        ],
    }
    return RemaskDecision(
        sequence=len(state.decisions),
        epoch=epoch + 1,
        fields=(field_name,),
        decision_sha256=sha256_bytes(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ),
        reason=reason,
        total_trials=0,
        rejected_count=state.rejected,
        dedup_count=state.dedup,
    )


def _record_online_attributions(
    state: _RunState,
    parents: dict[str, ProbeRecord],
    children: list[ProbeRecord],
    remasks: dict[str, RemaskDecision],
) -> None:
    for child in children:
        parent = parents[child.particle.particle_id]
        remask = remasks[child.particle.particle_id]
        attributions = attribute_feedback(parent.feedback, child.feedback, remask.fields)
        state.attributions.extend(attributions)
        for attribution in attributions:
            state.field_outcomes.append(
                LineageFieldOutcome(
                    lineage_id=child.particle.particle_id,
                    field=attribution.field,
                    reward_delta=attribution.reward_delta,
                    trials=attribution.trials,
                    proof_failure_count=state.proof_failure_counts.get(_record_key(child), 0),
                )
            )
            state.append_event("field_attribution", attribution.field)


def _record_failed_revision_outcome(
    state: _RunState,
    *,
    lineage_id: str,
    remask: RemaskDecision,
    duplicate_count: int,
    invalid_count: int,
) -> None:
    if duplicate_count == 0 and invalid_count == 0:
        return
    state.field_outcomes.append(
        LineageFieldOutcome(
            lineage_id=lineage_id,
            field=remask.fields[0],
            reward_delta=0.0,
            trials=1,
            duplicate_count=duplicate_count,
            invalid_count=invalid_count,
        )
    )


def _increment_proof_failure(state: _RunState, record: ProbeRecord) -> None:
    key = _record_key(record)
    state.proof_failure_counts[key] = state.proof_failure_counts.get(key, 0) + 1


def _select_online_archive(
    inputs: _RunInputs,
    state: _RunState,
    archive: list[ProbeRecord],
    children: list[ProbeRecord],
    protected: frozenset[ArchiveCandidateKey],
) -> ArchiveSelection:
    selection = select_parent_child(
        tuple(archive),
        tuple(children),
        capacity=inputs.config.archive_size,
        protected_ids=protected,
        proof_novelty_by_key=state.proof_novelty,
        cost_by_key=state.archive_costs,
    )
    state.archive_snapshot_sha256 = tuple(
        candidate.snapshot_sha256 for candidate in selection.active
    )
    return selection


def _proof_novelty(verdict: OnlineProofVerdict) -> ProofNovelty:
    match verdict.oracle_status:
        case "proved_new_cause":
            return ProofNovelty.NEW_CAUSE
        case "proved_duplicate_cause":
            return ProofNovelty.DUPLICATE_CAUSE
        case "not_reproduced" | "crash_without_sanitizer":
            return ProofNovelty.NONE
        case unreachable:
            assert_never(unreachable)


def _record_proof_novelty(state: _RunState, record: ProbeRecord, novelty: ProofNovelty) -> None:
    key = _record_key(record)
    previous = state.proof_novelty.get(key, ProofNovelty.NONE)
    state.proof_novelty[key] = max(previous, novelty, key=_proof_novelty_rank)


def _proof_novelty_rank(novelty: ProofNovelty) -> int:
    return {
        ProofNovelty.NONE: 0,
        ProofNovelty.DUPLICATE_CAUSE: 0,
        ProofNovelty.NEW_CAUSE: 2,
    }[novelty]


def _record_key(record: ProbeRecord) -> ArchiveCandidateKey:
    return (record.particle.particle_id, record.particle.snapshot.sha256)


def _particle_key(particle: Particle) -> ArchiveCandidateKey:
    return (particle.particle_id, particle.snapshot.sha256)


def _cost_from_result(result: RevisionResult) -> ArchiveCandidateCost:
    return ArchiveCandidateCost(
        nfe=result.nfe,
        forward_calls=result.forward_calls,
        gpu_ms=result.gpu_ms,
    )


def _add_costs(left: ArchiveCandidateCost, right: ArchiveCandidateCost) -> ArchiveCandidateCost:
    return ArchiveCandidateCost(
        nfe=left.nfe + right.nfe,
        forward_calls=left.forward_calls + right.forward_calls,
        gpu_ms=left.gpu_ms + right.gpu_ms,
    )


def _family_key(feedback: ProbeFeedback) -> str:
    if feedback.provisional_family_key is not None:
        return feedback.provisional_family_key
    return sha256_bytes(f"probe-log-family:{feedback.run_log_sha256}".encode())


def _proof_summary(proof: ProofRunnerResult) -> ProofSummary:
    return ProofSummary(
        oracle_status=proof.oracle_status,
        input_sha256=proof.input_sha256,
        proof_log_sha256=proof.proof_log_sha256,
    )


def _completed_proof(
    proof: ProofRunnerResult, feedback: ProbeFeedback, family_key: str
) -> _CompletedProof | None:
    match proof.oracle_status:
        case "proved" | "not_reproduced" | "crash_without_sanitizer":
            if proof.input_sha256 is None or proof.proof_log_sha256 is None:
                return None
            return _CompletedProof(
                oracle_status=proof.oracle_status,
                input_sha256=proof.input_sha256,
                proof_log_sha256=proof.proof_log_sha256,
                target_commit=getattr(proof, "target_commit", ""),
                sanitizer_class=getattr(proof, "sanitizer_class", None),
                root_signature=getattr(proof, "root_signature", None),
            )
        case "skipped":
            _ = feedback
            _ = family_key
            return None
        case unreachable:
            assert_never(unreachable)


def _feedback_attempted(state: _RunState, feedback: ProbeFeedback) -> bool:
    return state.proof_archive.family_attempt_count(_family_key(feedback)) > 0


def _runtime_accounting(
    inputs: _RunInputs,
    state: _RunState,
    online_enabled: bool,
) -> RuntimeAccounting:
    model_events = tuple(
        event
        for event in state.events
        if event.kind in {"snapshot_materialized", "revision_completed"} and not event.cache_hit
    )
    legacy_proof_invocations = int(
        not online_enabled and state.proof is not None and state.proof.oracle_status != "skipped"
    )
    exact_proof_invocations = state.exact_proof_invocations + legacy_proof_invocations
    probe_invocations = len(state.feedback) * len(inputs.config.fuzzer_seeds)
    return RuntimeAccounting(
        recorded_nfe=sum(event.nfe for event in model_events),
        recorded_forward_calls=sum(event.forward_calls for event in model_events),
        recorded_gpu_ms=sum(event.gpu_ms for event in model_events),
        probe_invocations=probe_invocations,
        probe_budget_seconds=probe_invocations * inputs.config.probe_seconds,
        exact_proof_invocations=exact_proof_invocations,
        proof_budget_seconds=exact_proof_invocations * inputs.config.proof_seconds,
    )


def _record_snapshots(
    state: _RunState,
    particles: list[Particle],
    remask: RemaskDecision | None = None,
    parents: dict[str, str] | None = None,
    dedup_hit: bool = False,
) -> None:
    kind = "snapshot_materialized" if remask is None else "revision_completed"
    for particle in particles:
        if not dedup_hit:
            state.add_snapshot(particle)
            state.seen.add(particle.snapshot.sha256)
        state.append_event(
            kind,
            particle.particle_id,
            particle=particle,
            remask=remask,
            parent_sha256=None if parents is None else parents[particle.particle_id],
            cache_hit=particle.result.cache_hit,
            dedup_hit=dedup_hit,
        )


async def _probe_and_record(
    inputs: _RunInputs, state: _RunState, particles: list[Particle]
) -> list[ProbeRecord]:
    records = await probe_particles(inputs.config, inputs.probe, particles, state.dedup)
    for record in records:
        state.feedback.append(record.feedback)
        state.append_event(
            "probe_completed",
            record.particle.particle_id,
            particle=record.particle,
            feedback=record.feedback,
            cache_hit=record.cache_hit,
            fuzz_seconds=inputs.config.probe_seconds,
            validator_ms=record.feedback.wall_ms,
            artifact_sha256=(
                record.feedback.run_log_sha256,
                record.feedback.run_argv_sha256,
                *record.feedback.corpus_artifact_sha256,
            ),
        )
    return records
