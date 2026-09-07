# pyright: reportUnnecessaryComparison=false
# noqa: SIZE_OK
"""Controller tests for deterministic feedback-conditioned PoC revision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock
from time import sleep
from typing import TYPE_CHECKING, Literal, assert_never, override

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.controller_models import (
    CrashArtifact,
    InitialParticleRequest,
    Particle,
    ProbeRecord,
    ProofRequest,
    RevisionRequest,
    RevisionResult,
    RevisionRunRequest,
    RevisionTrajectoryStep,
)
from dllm_agent.revision.models import (
    PocRevisionReport,
    ProbeFeedback,
    ProofSummary,
    RevisionConfig,
)
from dllm_agent.revision.pipeline import online_proof_candidates, run_poc_revision
from dllm_agent.targets.openexr.particle import (
    FieldName,
    ParticleDecodeError,
    SelectorSlots,
    decode_particle,
    materialize_particle,
)
from dllm_agent.targets.openexr.probe import ProbeRequest, ProbeResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dllm_agent.core.models import Sha256
    from dllm_agent.revision.controller_models import ProofOracleStatus

_ZERO_SHA = "0" * 64
_ONE_SHA = "1" * 64
_TWO_SHA = "2" * 64


class DeterministicSampler:
    """Sampler double that records initial and selective revision requests."""

    def __init__(
        self,
        initial_selectors: SelectorSlots = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        revisions: dict[FieldName, SelectorSlots] | None = None,
        uncertainty_field: FieldName = "width",
    ) -> None:
        self.initial_requests: list[InitialParticleRequest] = []
        self.revision_requests: list[RevisionRequest] = []
        self._initial_selectors: SelectorSlots = initial_selectors
        self._revisions: dict[FieldName, SelectorSlots] = revisions or {}
        self._uncertainty_field: FieldName = uncertainty_field

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        """Return a deterministic initial particle."""
        self.initial_requests.append(request)
        return RevisionResult(
            selectors=self._initial_selectors,
            token_ids=self._initial_selectors,
            slot_positions=tuple(range(10)),
            selected_slot_token_ids=(),
            token_state_sha256=sha256_bytes(f"initial:{request.particle_id}".encode()),
            transfer_evidence_sha256=sha256_bytes(b"initial-transfer"),
            rng_state_sha256=sha256_bytes(f"rng:{request.seed}".encode()),
            nfe=request.steps,
            forward_calls=request.steps,
            gpu_ms=request.steps,
            cache_hit=False,
            trajectory_uncertainty=0.75,
            suggested_field=self._uncertainty_field,
        )

    def revise(self, request: RevisionRequest) -> RevisionResult:
        """Return the configured one-field child particle."""
        self.revision_requests.append(request)
        selectors = self._revisions[request.changed_field]
        return RevisionResult(
            selectors=selectors,
            token_ids=selectors,
            slot_positions=request.slot_positions,
            selected_slot_token_ids=(selectors[2],),
            token_state_sha256=sha256_bytes(f"revision:{request.changed_field}".encode()),
            transfer_evidence_sha256=sha256_bytes(f"transfer:{request.changed_field}".encode()),
            rng_state_sha256=request.rng_state_sha256,
            nfe=request.revision_steps,
            forward_calls=request.revision_steps,
            gpu_ms=1,
            cache_hit=False,
            trajectory_uncertainty=0.25,
            suggested_field=request.changed_field,
        )


class SequenceProbe:
    """Probe double that can return deterministic feedback in a chosen order."""

    def __init__(
        self,
        feedback_for: Callable[[str], _ProbeMetrics],
        order: Literal["normal", "reverse"] = "normal",
    ) -> None:
        self.snapshot_hashes: list[str] = []
        self._feedback_for: Callable[[str], _ProbeMetrics] = feedback_for
        self._order: Literal["normal", "reverse"] = order
        self._lock: Lock = Lock()
        self._call_count: int = 0

    def run(self, request: ProbeRequest) -> ProbeResult:
        """Record a probe and return deterministic libFuzzer-like metrics."""
        with self._lock:
            call_index = self._call_count
            self._call_count += 1
        if self._order == "reverse" and call_index == 0:
            sleep(0.02)
        self.snapshot_hashes.append(request.snapshot_sha256)
        metrics = self._feedback_for(request.snapshot_sha256)
        crash_artifacts = _crash_artifacts(metrics, request.snapshot_sha256)
        return ProbeResult(
            snapshot_sha256=request.snapshot_sha256,
            cache_key_sha256=sha256_bytes(
                f"{request.snapshot_sha256}:{request.fuzzer_seed}".encode()
            ),
            cache_hit=False,
            run_status="crashed" if metrics.sanitizer else "completed",
            sanitizer_observed=metrics.sanitizer,
            final_units=100,
            coverage_edges=metrics.new_edges,
            value_profile_features=metrics.value_profile_features,
            corpus_count=1,
            executions_per_second=metrics.execs,
            run_log_sha256=sha256_bytes(f"log:{request.snapshot_sha256}".encode()),
            run_argv_sha256=sha256_bytes(f"argv:{request.fuzzer_seed}".encode()),
            crash_artifact_sha256=tuple(sha256_bytes(item) for item in crash_artifacts),
            crash_artifact_bytes=crash_artifacts,
            sanitizer_class=metrics.sanitizer_class,
            root_signature=metrics.root_signature,
            provisional_family_key=metrics.provisional_family_key,
        )


class MultiFamilyCrashProbe:
    """Probe double that emits one crash artifact family per fuzzer seed."""

    def __init__(self, crash_snapshot_sha256: str) -> None:
        self.snapshot_hashes: list[str] = []
        self.family_keys: list[str] = []
        self._crash_snapshot_sha256: str = crash_snapshot_sha256

    def run(self, request: ProbeRequest) -> ProbeResult:
        """Return seed-specific crash artifacts for the configured snapshot."""
        self.snapshot_hashes.append(request.snapshot_sha256)
        if request.snapshot_sha256 != self._crash_snapshot_sha256:
            return ProbeResult(
                snapshot_sha256=request.snapshot_sha256,
                cache_key_sha256=sha256_bytes(
                    f"{request.snapshot_sha256}:{request.fuzzer_seed}".encode()
                ),
                cache_hit=False,
                run_status="completed",
                sanitizer_observed=False,
                final_units=100,
                coverage_edges=5,
                value_profile_features=1,
                corpus_count=1,
                executions_per_second=10,
                run_log_sha256=sha256_bytes(f"log:{request.snapshot_sha256}".encode()),
                run_argv_sha256=sha256_bytes(f"argv:{request.fuzzer_seed}".encode()),
                crash_artifact_sha256=(),
                crash_artifact_bytes=(),
            )
        crash_bytes = f"crash:{request.fuzzer_seed}".encode()
        family_key = sha256_bytes(f"family:{request.fuzzer_seed}".encode())
        self.family_keys.append(family_key)
        return ProbeResult(
            snapshot_sha256=request.snapshot_sha256,
            cache_key_sha256=sha256_bytes(
                f"{request.snapshot_sha256}:{request.fuzzer_seed}".encode()
            ),
            cache_hit=False,
            run_status="crashed",
            sanitizer_observed=True,
            final_units=100,
            coverage_edges=5,
            value_profile_features=1,
            corpus_count=1,
            executions_per_second=10,
            run_log_sha256=sha256_bytes(f"log:{request.snapshot_sha256}".encode()),
            run_argv_sha256=sha256_bytes(f"argv:{request.fuzzer_seed}".encode()),
            crash_artifact_sha256=(sha256_bytes(crash_bytes),),
            crash_artifact_bytes=(crash_bytes,),
            sanitizer_class="heap-buffer-overflow",
            root_signature=f"root:{request.fuzzer_seed}",
            provisional_family_key=family_key,
        )


type ProofOracle = Literal["proved", "not_reproduced", "crash_without_sanitizer", "skipped"]


class RecordingProof:
    """Independent proof runner double."""

    def __init__(self, oracle: ProofOracle) -> None:
        self.requests: list[ProofRequest] = []
        self._oracle: ProofOracle = oracle

    def run(self, request: ProofRequest) -> ProofSummary:
        """Record exact proof input and return the configured proof summary."""
        self.requests.append(request)
        return ProofSummary(
            oracle_status=self._oracle,
            input_sha256=request.snapshot_sha256,
            proof_log_sha256=_ONE_SHA,
        )


class RichProof:
    """Proof runner double with stable rich cause metadata."""

    def __init__(self, oracle: Literal["proved", "not_reproduced"]) -> None:
        self.requests: list[ProofRequest] = []
        self._oracle: Literal["proved", "not_reproduced"] = oracle

    def run(self, request: ProofRequest) -> RichProofView:
        """Record exact proof input and return a rich proof view."""
        self.requests.append(request)
        return RichProofView(
            oracle_status=self._oracle,
            input_sha256=request.snapshot_sha256,
            proof_log_sha256=sha256_bytes(f"proof:{request.snapshot_sha256}".encode()),
            target_commit="target-a",
            sanitizer_class="heap-buffer-overflow",
            root_signature="Imf_3_4::InputFile::readPixels 0xabc",
        )


class ScriptedRichProof:
    """Proof runner double that returns configured rich statuses in order."""

    def __init__(self, statuses: tuple[Literal["proved", "not_reproduced"], ...]) -> None:
        self.requests: list[ProofRequest] = []
        self._statuses: tuple[Literal["proved", "not_reproduced"], ...] = statuses

    def run(self, request: ProofRequest) -> RichProofView:
        """Record exact proof input and return the next configured status."""
        self.requests.append(request)
        status = self._statuses[len(self.requests) - 1]
        return RichProofView(
            oracle_status=status,
            input_sha256=request.snapshot_sha256,
            proof_log_sha256=sha256_bytes(f"proof:{request.snapshot_sha256}".encode()),
            target_commit="target-a",
            sanitizer_class="heap-buffer-overflow",
            root_signature="stable root",
        )


@dataclass(frozen=True, slots=True)
class RichProofView:
    """Target-like proof view with rich cause metadata."""

    oracle_status: Literal["proved", "not_reproduced"]
    input_sha256: Sha256
    proof_log_sha256: Sha256
    target_commit: str
    sanitizer_class: str | None
    root_signature: str | None


@dataclass(frozen=True, slots=True)
class StructuralProofView:
    """Target-like proof view without revision schema inheritance."""

    oracle_status: ProofOracleStatus
    input_sha256: Sha256 | None
    proof_log_sha256: Sha256 | None


class StructuralProof:
    """Proof runner double returning a structural view instead of ProofSummary."""

    def __init__(self) -> None:
        self.requests: list[ProofRequest] = []

    def run(self, request: ProofRequest) -> StructuralProofView:
        """Record exact proof input and return a target-owned structural view."""
        self.requests.append(request)
        return StructuralProofView(
            oracle_status="not_reproduced",
            input_sha256=request.snapshot_sha256,
            proof_log_sha256=_ONE_SHA,
        )


@dataclass(frozen=True, slots=True)
class _ProbeMetrics:
    new_edges: int = 0
    value_profile_features: int = 0
    execs: int = 10
    sanitizer: bool = False
    sanitizer_class: str | None = None
    root_signature: str | None = None
    provisional_family_key: str | None = None
    crash_artifact_bytes: tuple[bytes, ...] | None = None


def _crash_artifacts(metrics: _ProbeMetrics, snapshot_sha256: str) -> tuple[bytes, ...]:
    if metrics.crash_artifact_bytes is not None:
        return metrics.crash_artifact_bytes
    if not metrics.sanitizer:
        return ()
    return (f"crash:{snapshot_sha256}".encode(),)


def test_controller_remasks_attributed_slot_and_probes_new_snapshot(tmp_path: Path) -> None:
    """Given selective revision, when width is best, then only width is remasked."""
    # Given
    sampler = DeterministicSampler(
        revisions={"width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)},
        uncertainty_field="width",
    )
    probe = SequenceProbe(lambda snapshot: _ProbeMetrics(new_edges=1 + len(snapshot)))

    # When
    report = run_poc_revision(
        _config(epochs=1),
        _run_request("selective-revision"),
        sampler,
        probe,
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert report.remask_decisions[0].fields == ("width",)
    assert report.runtime_accounting.recorded_nfe == 10
    assert report.runtime_accounting.recorded_forward_calls == 10
    assert report.runtime_accounting.recorded_gpu_ms == 9
    assert report.runtime_accounting.probe_invocations == 4
    assert report.runtime_accounting.exact_proof_invocations == 0
    assert sampler.revision_requests[0].changed_field == "width"
    assert sampler.revision_requests[0].remask_bitmap != 0
    assert report.snapshots[1].snapshot_sha256 != report.snapshots[0].snapshot_sha256
    assert probe.snapshot_hashes[-1] == report.snapshots[1].snapshot_sha256


def test_controller_falls_back_from_non_remaskable_suggestion(tmp_path: Path) -> None:
    """Given fixed non-effective suggestion, then controller revises effective bytes."""
    # Given
    sampler = DeterministicSampler(
        revisions={"origin_x": (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)},
        uncertainty_field="chunk_offset",
    )

    # When
    report = run_poc_revision(
        _config(epochs=1),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=1)),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert report.remask_decisions[0].fields == ("origin_x",)
    assert sampler.revision_requests[0].changed_field == "origin_x"
    assert report.snapshots[1].snapshot_sha256 != report.snapshots[0].snapshot_sha256


def test_out_of_order_completion_keeps_byte_identical_ledger(tmp_path: Path) -> None:
    """Given reversed probe completion, when barriers sort, then ledgers match."""
    # Given
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    sampler = DeterministicSampler(
        revisions={"width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)},
        uncertainty_field="width",
    )

    # When
    first = run_poc_revision(
        _config(particle_count=2, epochs=1),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=3), order="normal"),
        RecordingProof("skipped"),
        first_dir,
    )
    second = run_poc_revision(
        _config(particle_count=2, epochs=1),
        _run_request("selective-revision"),
        DeterministicSampler(
            revisions={"width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)},
            uncertainty_field="width",
        ),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=3), order="reverse"),
        RecordingProof("skipped"),
        second_dir,
    )

    # Then
    assert first.revision_events_sha256 == second.revision_events_sha256
    assert (first_dir / "revision-events.jsonl").read_bytes() == (
        second_dir / "revision-events.jsonl"
    ).read_bytes()


def test_duplicate_snapshot_skips_probe_and_revision_work(tmp_path: Path) -> None:
    """Given duplicate materialization, when dedup sees it, then duplicate work is skipped."""
    # Given
    sampler = DeterministicSampler(revisions={"width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)})
    probe = SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=2))

    # When
    report = run_poc_revision(
        _config(particle_count=2, epochs=1),
        _run_request("selective-revision"),
        sampler,
        probe,
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(probe.snapshot_hashes) == 4
    assert len(set(probe.snapshot_hashes)) == 2
    assert len(sampler.revision_requests) == 1
    assert report.accounting_dedup >= 1


def test_same_batch_revision_duplicate_skips_duplicate_probe(tmp_path: Path) -> None:
    """Given two revised children collide, when batch dedup runs, then one probe is skipped."""
    # Given
    sampler = _particle_id_sampler()
    probe = SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=2))

    # When
    report = run_poc_revision(
        _config(particle_count=2, epochs=1),
        _run_request("selective-revision"),
        sampler,
        probe,
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(sampler.revision_requests) == 2
    assert len(probe.snapshot_hashes) == 6
    assert len(set(probe.snapshot_hashes)) == 3
    assert report.accounting_dedup == 1
    revision_events = [row for row in _ledger_rows(tmp_path) if row["kind"] == "revision_completed"]
    assert [row["dedup_hit"] for row in revision_events] == [False, True]
    assert [row["nfe"] for row in revision_events] == [2, 2]
    assert [row["forward_calls"] for row in revision_events] == [2, 2]
    assert [row["gpu_ms"] for row in revision_events] == [1, 1]


def test_same_batch_no_remask_duplicate_skips_duplicate_probe(tmp_path: Path) -> None:
    """Given no-remask resamples collide, when batch dedup runs, then one probe is skipped."""
    # Given
    sampler = NoRemaskCollisionSampler()
    probe = SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=2))

    # When
    report = run_poc_revision(
        _config(particle_count=2, epochs=1),
        _run_request("feedback-no-remask"),
        sampler,
        probe,
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert [request.steps for request in sampler.initial_requests] == [8, 8, 2, 2]
    assert len(probe.snapshot_hashes) == 6
    assert len(set(probe.snapshot_hashes)) == 3
    assert report.accounting_dedup == 1
    resample_events = [
        row for row in _ledger_rows(tmp_path) if row["kind"] == "snapshot_materialized"
    ][2:]
    assert [row["dedup_hit"] for row in resample_events] == [False, True]
    assert [row["nfe"] for row in resample_events] == [2, 2]
    assert [row["forward_calls"] for row in resample_events] == [2, 2]
    assert [row["gpu_ms"] for row in resample_events] == [2, 2]


def test_invalid_materialization_never_probes(tmp_path: Path) -> None:
    """Given invalid selectors, when materialization fails, then probe is not called."""
    # Given
    sampler = InvalidMaterializationSampler()
    probe = SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=1))

    # When
    report = run_poc_revision(
        _config(epochs=1),
        _run_request("selective-revision"),
        sampler,
        probe,
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert probe.snapshot_hashes == []
    assert report.probe_feedback == ()
    assert report.accounting_rejected == 1


def test_revision_materialization_rejection_is_accounted(tmp_path: Path) -> None:
    """Given invalid revision materialization, then rejection is counted."""
    # Given
    sampler = RevisionInvalidMaterializationSampler()

    # When
    report = run_poc_revision(
        _config(epochs=1),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=1)),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(sampler.revision_requests) == 1
    assert report.accounting_rejected == 1
    assert len(report.probe_feedback) == 1


def test_all_particle_failure_prevents_proof(tmp_path: Path) -> None:
    """Given all particles fail before probe, when run ends, then proof is absent."""
    # Given
    proof = RecordingProof("proved")

    # When
    report = run_poc_revision(
        _config(epochs=1),
        _run_request("selective-revision"),
        InvalidMaterializationSampler(),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(sanitizer=True)),
        proof,
        tmp_path,
    )

    # Then
    assert report.proof is None
    assert proof.requests == []


def test_pseudo_feedback_deranges_without_fixed_points_and_preserves_hash_multiset(
    tmp_path: Path,
) -> None:
    """Given pseudo feedback, when rotating, then no particle receives its own hash."""
    # Given
    proof = RecordingProof("skipped")

    # When
    report = run_poc_revision(
        _config(particle_count=3, epochs=1),
        _run_request("pseudo-feedback", pseudo_feedback_rotation_seed=1),
        _particle_id_unique_revision_sampler(),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=4)),
        proof,
        tmp_path,
    )

    # Then
    parent_hashes = tuple(attr.parent_feedback_sha256 for attr in report.field_attributions)
    child_hashes = tuple(attr.child_feedback_sha256 for attr in report.field_attributions)
    initial_hashes = tuple(feedback.feedback_sha256 for feedback in report.probe_feedback[:3])
    assert sorted(parent_hashes) == sorted(initial_hashes)
    assert all(parent != child for parent, child in zip(parent_hashes, child_hashes, strict=True))


def test_pseudo_feedback_single_survivor_skips_fixed_point_attribution(
    tmp_path: Path,
) -> None:
    """Given dedup leaves one parent, then pseudo feedback skips fixed-point attribution."""
    # Given
    sampler = _particle_id_sampler()

    # When
    report = run_poc_revision(
        _config(particle_count=2, epochs=3),
        _run_request("pseudo-feedback", pseudo_feedback_rotation_seed=0),
        sampler,
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=4)),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(report.field_attributions) == 1
    attribution = report.field_attributions[0]
    assert attribution.parent_feedback_sha256 != attribution.child_feedback_sha256


def test_pseudo_rotated_feedback_diverges_from_selective_next_remask(
    tmp_path: Path,
) -> None:
    """Given paired candidates, then real and pseudo histories choose different next fields."""
    # Given
    metrics = {
        _snapshot_sha256((0, 0, 0, 0, 0, 0, 0, 0, 0, 0)): 1,
        _snapshot_sha256((1, 0, 0, 0, 0, 0, 0, 0, 0, 0)): 10,
        _snapshot_sha256((2, 0, 0, 0, 0, 0, 0, 0, 0, 0)): 5,
        _snapshot_sha256((3, 0, 0, 0, 0, 0, 0, 0, 0, 0)): 5,
        _snapshot_sha256((2, 1, 0, 0, 0, 0, 0, 0, 0, 0)): 5,
    }

    def feedback(snapshot: str) -> _ProbeMetrics:
        return _ProbeMetrics(new_edges=metrics[snapshot])

    # When
    selective = run_poc_revision(
        _config(particle_count=2, epochs=3),
        _run_request("selective-revision"),
        RotatedFeedbackDivergenceSampler(),
        SequenceProbe(feedback),
        RecordingProof("skipped"),
        tmp_path / "selective",
    )
    pseudo = run_poc_revision(
        _config(particle_count=2, epochs=3),
        _run_request("pseudo-feedback", pseudo_feedback_rotation_seed=0),
        RotatedFeedbackDivergenceSampler(),
        SequenceProbe(feedback),
        RecordingProof("skipped"),
        tmp_path / "pseudo",
    )

    # Then
    selective_fields = tuple(decision.fields[0] for decision in selective.remask_decisions)
    pseudo_fields = tuple(decision.fields[0] for decision in pseudo.remask_decisions)
    assert selective_fields[:2] == ("origin_x", "origin_x")
    assert pseudo_fields[:2] == ("origin_x", "origin_y")


def test_pseudo_feedback_rejects_single_particle(tmp_path: Path) -> None:
    """Given one particle, when pseudo feedback is requested, then parsing rejects it."""
    # Given / When / Then
    with pytest.raises(ValueError, match="at least two particles"):
        _ = run_poc_revision(
            _config(particle_count=1),
            _run_request("pseudo-feedback"),
            DeterministicSampler(),
            SequenceProbe(lambda _snapshot: _ProbeMetrics()),
            RecordingProof("skipped"),
            tmp_path,
        )


def test_prepared_probe_mapping_uses_skipped_build_and_deterministic_wall(
    tmp_path: Path,
) -> None:
    """Given hot probe output, when normalized, then build and wall are deterministic."""
    # Given
    probe = SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=7))

    # When
    report = run_poc_revision(
        _config(probe_seconds=3),
        _run_request("structured-one-shot"),
        DeterministicSampler(),
        probe,
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    feedback = report.probe_feedback[0]
    assert probe.snapshot_hashes == [feedback.snapshot_sha256, feedback.snapshot_sha256]
    assert feedback.build_status == "skipped"
    assert feedback.wall_ms == 3000
    assert feedback.build_log_sha256 == sha256_bytes(b"")
    assert feedback.build_argv_sha256 == sha256_bytes(b"[]")


def test_probe_sanitizer_does_not_override_independent_failed_proof(tmp_path: Path) -> None:
    """Given probe sanitizer, when proof fails, then report keeps proof failure."""
    # Given
    proof = RecordingProof("not_reproduced")

    # When
    report = run_poc_revision(
        _config(),
        _run_request("structured-one-shot"),
        DeterministicSampler(),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(sanitizer=True)),
        proof,
        tmp_path,
    )

    # Then
    assert report.probe_feedback[-1].sanitizer_observed is True
    assert report.proof is not None
    assert report.proof.oracle_status == "not_reproduced"
    assert sha256_bytes(proof.requests[0].snapshot) == proof.requests[0].snapshot_sha256


def test_structural_proof_result_is_normalized_to_revision_summary(tmp_path: Path) -> None:
    """Given a target-like proof view, then the report stores canonical ProofSummary."""
    # Given
    proof = StructuralProof()

    # When
    report = run_poc_revision(
        _config(),
        _run_request("structured-one-shot"),
        DeterministicSampler(),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(sanitizer=True)),
        proof,
        tmp_path,
    )

    # Then
    assert isinstance(report.proof, ProofSummary)
    assert type(report.proof) is ProofSummary
    assert report.proof.input_sha256 == proof.requests[0].snapshot_sha256
    assert report.proof.proof_log_sha256 == _ONE_SHA


def test_run_request_seed_provenance_drives_model_and_proof_requests(tmp_path: Path) -> None:
    """Given explicit seed bases, then model and proof seams receive derived seeds."""
    # Given
    selective_sampler = _particle_id_unique_revision_sampler()
    no_remask_sampler = NoRemaskCollisionSampler()

    # When
    _ = run_poc_revision(
        _config(particle_count=2, epochs=1),
        _run_request("selective-revision", particle_seed_base=100, proof_seed=909),
        selective_sampler,
        SequenceProbe(lambda _snapshot: _ProbeMetrics(sanitizer=True)),
        RecordingProof("not_reproduced"),
        tmp_path / "selective",
    )
    proof = RecordingProof("not_reproduced")
    _ = run_poc_revision(
        _config(particle_count=2, epochs=1),
        _run_request("feedback-no-remask", particle_seed_base=100, proof_seed=909),
        no_remask_sampler,
        SequenceProbe(lambda _snapshot: _ProbeMetrics(sanitizer=True)),
        proof,
        tmp_path / "no-remask",
    )

    # Then
    assert [request.seed for request in selective_sampler.initial_requests] == [100, 101]
    assert [request.seed for request in selective_sampler.revision_requests] == [
        1_100_100,
        1_100_101,
    ]
    assert [request.seed for request in no_remask_sampler.initial_requests] == [
        100,
        101,
        1_200_100,
        1_200_101,
    ]
    assert proof.requests[0].proof_seed == 909


def test_ledger_probe_revision_and_proof_events_bind_report_evidence(tmp_path: Path) -> None:
    """Given sanitizer proof, when ledger is written, then event evidence matches the report."""
    # Given
    proof = RecordingProof("not_reproduced")

    # When
    report = run_poc_revision(
        _config(probe_seconds=3),
        _run_request("selective-revision"),
        DeterministicSampler(revisions={"width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)}),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=5, sanitizer=True)),
        proof,
        tmp_path,
    )

    # Then
    rows = _ledger_rows(tmp_path)
    probe_event = _event_row(rows, "probe_completed")
    revision_event = _event_row(rows, "revision_completed")
    proof_event = _event_row(rows, "proof_completed")
    feedback = report.probe_feedback[0]
    assert probe_event["feedback_sha256"] == feedback.feedback_sha256
    assert probe_event["fuzz_seconds"] == 3
    assert probe_event["validator_ms"] == 3000
    assert probe_event["artifact_sha256"] == [
        feedback.run_log_sha256,
        feedback.run_argv_sha256,
        *feedback.corpus_artifact_sha256,
    ]
    assert probe_event["dedup_hit"] is False
    assert revision_event["parent_sha256"] == report.snapshots[0].snapshot_sha256
    assert revision_event["remask_sha256"] == report.remask_decisions[0].decision_sha256
    assert proof_event["snapshot_sha256"] == proof.requests[0].snapshot_sha256
    proof_summary = report.proof
    assert proof_summary is not None
    assert proof_event["feedback_sha256"] == proof_summary.proof_log_sha256
    assert proof_event["artifact_sha256"] == [proof_summary.proof_log_sha256]


def test_materialization_events_propagate_sampler_cache_hit(tmp_path: Path) -> None:
    """Given cached sampler outputs, then materialization ledger rows preserve cache hits."""
    # Given
    selective_dir = tmp_path / "selective"
    no_remask_dir = tmp_path / "no-remask"

    # When
    _ = run_poc_revision(
        _config(epochs=1),
        _run_request("selective-revision"),
        CacheHitSampler(),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=1)),
        RecordingProof("skipped"),
        selective_dir,
    )
    _ = run_poc_revision(
        _config(epochs=1),
        _run_request("feedback-no-remask"),
        CacheHitSampler(),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=1)),
        RecordingProof("skipped"),
        no_remask_dir,
    )

    # Then
    selective_rows = _ledger_rows(selective_dir)
    initial_event = _event_row(selective_rows, "snapshot_materialized")
    revision_event = _event_row(selective_rows, "revision_completed")
    no_remask_events = [
        row for row in _ledger_rows(no_remask_dir) if row["kind"] == "snapshot_materialized"
    ]
    assert initial_event["cache_hit"] is True
    assert revision_event["cache_hit"] is True
    assert [row["cache_hit"] for row in no_remask_events] == [True, True]


def test_revision_run_request_does_not_extend_task2_report_schema() -> None:
    """Given arm metadata, when schemas are inspected, then Task 2 report is untouched."""
    # Given / When
    request = _run_request("selective-revision")

    # Then
    assert request.schema_version == "dllm-poc-revision-run/v1"
    assert "arm" not in PocRevisionReport.model_fields


@pytest.mark.parametrize("arm", ["final-only", "full-regenerate"])
def test_revision_run_request_accepts_one_day_revision_variants(arm: str) -> None:
    """Given a one-day control arm, when parsed, then it is a typed revision variant."""
    # Given
    payload = {
        "schema_version": "dllm-poc-revision-run/v1",
        "arm": arm,
        "particle_seed_base": 41,
        "proof_seed": 43,
    }

    # When
    request = RevisionRunRequest.model_validate(payload)

    # Then
    assert request.arm == arm


def test_online_proof_event_occurs_before_bootstrap_revision(tmp_path: Path) -> None:
    """Given sanitizer feedback, when online mode runs, then proof precedes revision."""
    # Given
    proof = RecordingProof("not_reproduced")

    # When
    report = run_poc_revision(
        _config(controller_mode="online-portfolio"),
        _run_request("selective-revision"),
        DeterministicSampler(
            revisions={
                "width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                "height": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            }
        ),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=5, sanitizer=True)),
        proof,
        tmp_path,
    )

    # Then
    rows = _ledger_rows(tmp_path)
    proof_sequence = _event_row(rows, "proof_completed")["sequence"]
    revision_sequence = _event_row(rows, "revision_completed")["sequence"]
    assert isinstance(proof_sequence, int)
    assert isinstance(revision_sequence, int)
    assert proof_sequence < revision_sequence
    assert proof.requests[0].snapshot_sha256 != report.probe_feedback[0].snapshot_sha256
    assert proof.requests[0].snapshot_sha256 in report.probe_feedback[0].corpus_artifact_sha256
    assert report.online_proof_verdicts[0].oracle_status == "not_reproduced"
    assert report.remask_decisions[0].reason == "trajectory_bootstrap"


def test_online_proof_replays_exact_crash_artifact_bytes(tmp_path: Path) -> None:
    """Given crash artifact bytes, when online proof runs, then it replays that input."""
    # Given
    crash_bytes = b"mutated-crash-input"
    crash_sha = sha256_bytes(crash_bytes)
    proof = RecordingProof("not_reproduced")

    # When
    report = run_poc_revision(
        _config(controller_mode="online-portfolio"),
        _run_request("selective-revision"),
        DeterministicSampler(
            revisions={
                "width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                "height": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            }
        ),
        SequenceProbe(
            lambda _snapshot: _ProbeMetrics(
                new_edges=5,
                sanitizer=True,
                crash_artifact_bytes=(crash_bytes,),
            )
        ),
        proof,
        tmp_path,
    )

    # Then
    assert proof.requests[0].snapshot == crash_bytes
    assert proof.requests[0].snapshot_sha256 == crash_sha
    assert report.probe_feedback[0].snapshot_sha256 != crash_sha
    assert report.online_proof_verdicts[0].input_sha256 == crash_sha
    assert (tmp_path / "proof-inputs" / f"{crash_sha}.bin").read_bytes() == crash_bytes


def test_online_sanitizer_without_crash_bytes_does_not_proof_seed(
    tmp_path: Path,
) -> None:
    """Given sanitizer but no exact crash bytes, then online proof records missing input."""
    # Given
    proof = RecordingProof("not_reproduced")
    sampler = DeterministicSampler(
        revisions={
            "width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
            "height": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        }
    )

    # When
    report = run_poc_revision(
        _config(
            controller_mode="online-portfolio",
            proof_family_attempt_cap=1,
        ),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(
            lambda _snapshot: _ProbeMetrics(
                new_edges=5,
                sanitizer=True,
                crash_artifact_bytes=(),
            )
        ),
        proof,
        tmp_path,
    )

    # Then
    assert proof.requests == []
    assert sampler.revision_requests == []
    assert report.proof is None
    assert report.online_proof_verdicts == ()
    assert _event_row(_ledger_rows(tmp_path), "proof_missing_input")["proof_family_key"] is not None


def test_online_existing_proof_input_is_verified_before_reuse(tmp_path: Path) -> None:
    """Given a tampered persisted input, when proving, then reuse fails closed."""
    # Given
    crash_bytes = b"exact-crash-input"
    crash_sha = sha256_bytes(crash_bytes)
    proof_inputs = tmp_path / "proof-inputs"
    proof_inputs.mkdir()
    _ = (proof_inputs / f"{crash_sha}.bin").write_bytes(b"tampered")
    proof = RecordingProof("not_reproduced")

    # When / Then
    with pytest.raises(ValueError, match="persisted proof input"):
        _ = run_poc_revision(
            _config(controller_mode="online-portfolio"),
            _run_request("selective-revision"),
            DeterministicSampler(
                revisions={
                    "width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                    "height": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
                }
            ),
            SequenceProbe(
                lambda _snapshot: _ProbeMetrics(
                    new_edges=5,
                    sanitizer=True,
                    crash_artifact_bytes=(crash_bytes,),
                )
            ),
            proof,
            tmp_path,
        )
    assert proof.requests == []


def test_online_multi_artifact_families_and_monotonic_new_cause(
    tmp_path: Path,
) -> None:
    """Given later failed artifact proof, then prior new-cause novelty is preserved."""
    # Given
    child_selectors: SelectorSlots = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    child_sha = _snapshot_sha256(child_selectors)
    sampler = DeterministicSampler(
        revisions={
            "origin_x": child_selectors,
            "width": child_selectors,
            "height": child_selectors,
        },
        uncertainty_field="width",
    )
    probe = MultiFamilyCrashProbe(child_sha)
    proof = ScriptedRichProof(("proved", "not_reproduced"))

    # When
    report = run_poc_revision(
        _config(
            archive_size=2,
            controller_mode="online-portfolio",
            epochs=2,
            plateau_epochs=1,
        ),
        _run_request("selective-revision"),
        sampler,
        probe,
        proof,
        tmp_path,
    )

    # Then
    proof_rows = [
        row
        for row in _ledger_rows(tmp_path)
        if row["kind"] == "proof_completed" and row["online_proof_status"] is not None
    ]
    assert {row["proof_family_key"] for row in proof_rows} == set(probe.family_keys)
    assert [verdict.oracle_status for verdict in report.online_proof_verdicts] == [
        "proved_new_cause",
        "not_reproduced",
    ]
    assert len(sampler.revision_requests) == 2
    assert sampler.revision_requests[1].parent_selectors == child_selectors


def test_online_worse_child_keeps_parent_as_next_revision_parent(tmp_path: Path) -> None:
    """Given an inferior child, when plateau revises again, then parent remains active."""
    # Given
    initial_selectors: SelectorSlots = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    child_selectors: SelectorSlots = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    initial_hash = _snapshot_sha256(initial_selectors)
    sampler = DeterministicSampler(
        initial_selectors=initial_selectors,
        revisions={
            "origin_x": child_selectors,
            "width": child_selectors,
            "height": child_selectors,
        },
        uncertainty_field="width",
    )

    # When
    _ = run_poc_revision(
        _config(controller_mode="online-portfolio", epochs=2, plateau_epochs=1),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(
            lambda snapshot: _ProbeMetrics(new_edges=10 if snapshot == initial_hash else 1)
        ),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(sampler.revision_requests) == 2
    assert sampler.revision_requests[1].parent_selectors == initial_selectors


def test_online_equal_search_child_is_not_promoted_for_lower_revision_nfe(tmp_path: Path) -> None:
    """Given equal search metrics, when child costs less marginal NFE, then parent stays."""
    # Given
    initial_selectors: SelectorSlots = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    child_selectors: SelectorSlots = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    sampler = DeterministicSampler(
        initial_selectors=initial_selectors,
        revisions={"width": child_selectors, "height": child_selectors},
        uncertainty_field="width",
    )

    # When
    _ = run_poc_revision(
        _config(controller_mode="online-portfolio", epochs=2, plateau_epochs=1),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=5)),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(sampler.revision_requests) == 2
    assert sampler.revision_requests[1].parent_selectors == initial_selectors


def test_online_duplicate_cause_equal_search_child_is_not_promoted(tmp_path: Path) -> None:
    """Given duplicate-only proof, when search is equal, then child is not active parent."""
    # Given
    sampler = DuplicateCauseSampler()
    p0_initial = _snapshot_sha256((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    p1_child = _snapshot_sha256((1, 0, 0, 1, 0, 0, 0, 0, 0, 0))

    def metrics(snapshot: str) -> _ProbeMetrics:
        return _ProbeMetrics(
            new_edges=5,
            sanitizer=snapshot in {p0_initial, p1_child},
            sanitizer_class="heap-buffer-overflow" if snapshot in {p0_initial, p1_child} else None,
            root_signature="same root" if snapshot in {p0_initial, p1_child} else None,
        )

    # When
    report = run_poc_revision(
        _config(
            particle_count=2,
            controller_mode="online-portfolio",
            epochs=2,
            plateau_epochs=1,
            proof_family_attempt_cap=3,
        ),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(metrics),
        RichProof("proved"),
        tmp_path,
    )

    # Then
    assert "proved_duplicate_cause" in {
        verdict.oracle_status for verdict in report.online_proof_verdicts
    }
    p1_requests = [
        request for request in sampler.revision_requests if request.particle_id == "p0001"
    ]
    assert len(p1_requests) == 2
    assert p1_requests[1].parent_selectors == (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_online_only_plateaued_lineage_revises_after_bootstrap(tmp_path: Path) -> None:
    """Given mixed child outcomes, when next epoch starts, then only stalled lineage revises."""
    # Given
    sampler = TwoLineageOutcomeSampler()
    p0_child = _snapshot_sha256((0, 0, 1, 0, 0, 0, 0, 0, 0, 0))

    # When
    _ = run_poc_revision(
        _config(
            particle_count=2,
            epochs=2,
            controller_mode="online-portfolio",
            plateau_epochs=1,
        ),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(lambda snapshot: _ProbeMetrics(new_edges=20 if snapshot == p0_child else 5)),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert [request.particle_id for request in sampler.revision_requests] == [
        "p0000",
        "p0001",
        "p0001",
    ]


def test_online_ineligible_lineage_is_not_aged_by_stale_feedback(tmp_path: Path) -> None:
    """Given one improving lineage, when another stalls, then stale feedback does not age it."""
    # Given
    sampler = TwoLineageOutcomeSampler()
    p0_child = _snapshot_sha256((0, 0, 1, 0, 0, 0, 0, 0, 0, 0))

    # When
    _ = run_poc_revision(
        _config(
            particle_count=2,
            epochs=3,
            controller_mode="online-portfolio",
            plateau_epochs=1,
        ),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(lambda snapshot: _ProbeMetrics(new_edges=20 if snapshot == p0_child else 5)),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert [request.particle_id for request in sampler.revision_requests] == [
        "p0000",
        "p0001",
        "p0001",
        "p0001",
    ]


def test_online_bootstrap_chooses_lineage_specific_fields(tmp_path: Path) -> None:
    """Given different trajectories, when bootstrapping, then lineages choose different fields."""
    # Given
    sampler = TrajectoryDivergenceSampler()

    # When
    _ = run_poc_revision(
        _config(particle_count=2, controller_mode="online-portfolio"),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=5)),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert [request.changed_field for request in sampler.revision_requests] == [
        "width",
        "height",
    ]


def test_online_duplicate_root_cause_is_new_once_then_duplicate(tmp_path: Path) -> None:
    """Given repeated rich cause, when two proofs complete, then novelty is paid once."""
    # Given
    proof = RichProof("proved")

    # When
    report = run_poc_revision(
        _config(controller_mode="online-portfolio", proof_family_attempt_cap=3),
        _run_request("selective-revision"),
        DeterministicSampler(
            revisions={
                "width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                "height": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            }
        ),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=5, sanitizer=True)),
        proof,
        tmp_path,
    )

    # Then
    assert [verdict.oracle_status for verdict in report.online_proof_verdicts] == [
        "proved_new_cause",
        "proved_duplicate_cause",
    ]
    assert len(proof.requests) == 2


def test_online_family_cap_suppresses_repeat_proof(tmp_path: Path) -> None:
    """Given capped provisional family, when it repeats, then proof is not re-run."""
    # Given
    proof = RecordingProof("not_reproduced")

    # When
    report = run_poc_revision(
        _config(controller_mode="online-portfolio", proof_family_attempt_cap=1),
        _run_request("selective-revision"),
        DeterministicSampler(
            revisions={
                "width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                "height": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            }
        ),
        SequenceProbe(
            lambda _snapshot: _ProbeMetrics(
                new_edges=5,
                sanitizer=True,
                sanitizer_class="heap-buffer-overflow",
                root_signature="same root",
            )
        ),
        proof,
        tmp_path,
    )

    # Then
    assert len(proof.requests) == 1
    assert len(report.online_proof_verdicts) == 1
    cap_events = [row for row in _ledger_rows(tmp_path) if row["kind"] == "proof_family_capped"]
    assert len(cap_events) == 1


def test_online_skipped_proof_result_consumes_probe_family_cap(tmp_path: Path) -> None:
    """Given skipped proof result, when family repeats, then next proof is capped."""
    # Given
    proof = RecordingProof("skipped")

    # When
    report = run_poc_revision(
        _config(controller_mode="online-portfolio", proof_family_attempt_cap=1),
        _run_request("selective-revision"),
        DeterministicSampler(
            revisions={
                "width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                "height": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            }
        ),
        SequenceProbe(
            lambda _snapshot: _ProbeMetrics(
                new_edges=5,
                sanitizer=True,
                sanitizer_class="heap-buffer-overflow",
                root_signature="same root",
            )
        ),
        proof,
        tmp_path,
    )

    # Then
    assert len(proof.requests) == 1
    assert report.online_proof_verdicts == ()
    cap_events = [row for row in _ledger_rows(tmp_path) if row["kind"] == "proof_family_capped"]
    assert len(cap_events) == 1


def test_online_family_cap_blocks_epoch_zero_revision_bootstrap(tmp_path: Path) -> None:
    """Given a capped shared family, when epoch zero starts, then revision is gated."""
    # Given
    proof = RecordingProof("not_reproduced")
    sampler = _particle_id_unique_revision_sampler()

    # When
    report = run_poc_revision(
        _config(
            particle_count=2,
            controller_mode="online-portfolio",
            proof_family_attempt_cap=1,
        ),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(
            lambda _snapshot: _ProbeMetrics(
                new_edges=5,
                sanitizer=True,
                sanitizer_class="heap-buffer-overflow",
                root_signature="same root",
                provisional_family_key=sha256_bytes(b"shared-family"),
            )
        ),
        proof,
        tmp_path,
    )

    # Then
    assert len(proof.requests) == 1
    assert sampler.revision_requests == []
    assert len(_cap_events(tmp_path)) == 3
    assert report.archive_snapshot_sha256 == tuple(
        snapshot.snapshot_sha256 for snapshot in report.snapshots
    )


def test_online_rich_proof_consumes_probe_family_cap(tmp_path: Path) -> None:
    """Given rich proof with capped probe family, when family repeats, then proof is skipped."""
    # Given
    proof = RichProof("proved")

    # When
    report = run_poc_revision(
        _config(controller_mode="online-portfolio", proof_family_attempt_cap=1),
        _run_request("selective-revision"),
        DeterministicSampler(
            revisions={
                "width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                "height": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            }
        ),
        SequenceProbe(
            lambda _snapshot: _ProbeMetrics(
                new_edges=5,
                sanitizer=True,
                sanitizer_class="heap-buffer-overflow",
                root_signature="same root",
            )
        ),
        proof,
        tmp_path,
    )

    # Then
    assert len(proof.requests) == 1
    assert [verdict.oracle_status for verdict in report.online_proof_verdicts] == [
        "proved_new_cause"
    ]
    cap_events = [row for row in _ledger_rows(tmp_path) if row["kind"] == "proof_family_capped"]
    assert len(cap_events) == 1


def test_online_archive_dropped_child_is_not_scheduled_next(tmp_path: Path) -> None:
    """Given protected parent capacity, when superior child drops, then parent stays active."""
    # Given
    child_selectors: SelectorSlots = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    sampler = DeterministicSampler(revisions={"width": child_selectors, "height": child_selectors})
    initial_selectors: SelectorSlots = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # When
    _ = run_poc_revision(
        _config(
            archive_size=1,
            controller_mode="online-portfolio",
            epochs=2,
            plateau_epochs=1,
        ),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=10)),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(sampler.revision_requests) == 2
    assert sampler.revision_requests[1].parent_selectors == initial_selectors
    drop_rows = [row for row in _ledger_rows(tmp_path) if row["kind"] == "particle_dropped"]
    assert len(drop_rows) == 1


def test_online_archive_retained_incomparable_child_is_scheduled_next(
    tmp_path: Path,
) -> None:
    """Given retained tradeoff child, when next epoch starts, then child is active."""
    # Given
    child_selectors: SelectorSlots = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    sampler = DeterministicSampler(
        revisions={
            "origin_x": child_selectors,
            "width": child_selectors,
            "height": child_selectors,
        }
    )
    parent_sha = _snapshot_sha256((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    child_sha = _snapshot_sha256(child_selectors)

    def metrics(snapshot: str) -> _ProbeMetrics:
        if snapshot == parent_sha:
            return _ProbeMetrics(new_edges=10, value_profile_features=1, execs=100)
        if snapshot == child_sha:
            return _ProbeMetrics(new_edges=5, value_profile_features=5, execs=50)
        return _ProbeMetrics(new_edges=5, value_profile_features=5, execs=50)

    # When
    report = run_poc_revision(
        _config(
            archive_size=2,
            controller_mode="online-portfolio",
            epochs=2,
            plateau_epochs=1,
        ),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(metrics),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(sampler.revision_requests) == 2
    assert sampler.revision_requests[1].parent_selectors == child_selectors
    assert parent_sha in report.archive_snapshot_sha256
    assert child_sha in report.archive_snapshot_sha256


def test_online_frontier_width_one_matches_default_single_scout_pipeline(
    tmp_path: Path,
) -> None:
    """Given frontier_width=1 explicitly, when the same scenario runs, then the second
    revision still expands the promoted child -- identical to the implicit default."""
    # Given
    parent_selectors: SelectorSlots = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    child_selectors: SelectorSlots = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    sampler = DeterministicSampler(
        revisions={
            "origin_x": child_selectors,
            "width": child_selectors,
            "height": child_selectors,
        }
    )
    parent_sha = _snapshot_sha256(parent_selectors)
    child_sha = _snapshot_sha256(child_selectors)

    def metrics(snapshot: str) -> _ProbeMetrics:
        if snapshot == parent_sha:
            return _ProbeMetrics(new_edges=10, value_profile_features=1, execs=100)
        if snapshot == child_sha:
            return _ProbeMetrics(new_edges=5, value_profile_features=5, execs=50)
        return _ProbeMetrics(new_edges=5, value_profile_features=5, execs=50)

    # When
    report = run_poc_revision(
        _config(
            archive_size=2,
            controller_mode="online-portfolio",
            epochs=2,
            plateau_epochs=1,
            frontier_width=1,
        ),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(metrics),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(sampler.revision_requests) == 2
    assert sampler.revision_requests[1].parent_selectors == child_selectors
    assert parent_sha in report.archive_snapshot_sha256
    assert child_sha in report.archive_snapshot_sha256


def test_online_frontier_width_backtracks_to_archived_alternate_snapshot(
    tmp_path: Path,
) -> None:
    """Given frontier_width=2 on the same retained-tradeoff scenario, when the second
    revision comes due, then it backtracks to the archived parent instead of the
    promoted child -- the search-tree mechanism this width unlocks."""
    # Given
    parent_selectors: SelectorSlots = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    child_selectors: SelectorSlots = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    sampler = DeterministicSampler(
        revisions={
            "origin_x": child_selectors,
            "width": child_selectors,
            "height": child_selectors,
        }
    )
    parent_sha = _snapshot_sha256(parent_selectors)
    child_sha = _snapshot_sha256(child_selectors)

    def metrics(snapshot: str) -> _ProbeMetrics:
        if snapshot == parent_sha:
            return _ProbeMetrics(new_edges=10, value_profile_features=1, execs=100)
        if snapshot == child_sha:
            return _ProbeMetrics(new_edges=5, value_profile_features=5, execs=50)
        return _ProbeMetrics(new_edges=5, value_profile_features=5, execs=50)

    # When
    report = run_poc_revision(
        _config(
            archive_size=2,
            controller_mode="online-portfolio",
            epochs=2,
            plateau_epochs=1,
            frontier_width=2,
        ),
        _run_request("selective-revision"),
        sampler,
        SequenceProbe(metrics),
        RecordingProof("skipped"),
        tmp_path,
    )

    # Then
    assert len(sampler.revision_requests) == 2
    assert sampler.revision_requests[1].parent_selectors == parent_selectors
    assert parent_sha in report.archive_snapshot_sha256
    assert child_sha in report.archive_snapshot_sha256


def test_online_proof_dedupes_exact_snapshot_before_runner() -> None:
    """Given duplicate proof records, when proofs are selected, then exact artifact is once."""
    # Given
    first = _proof_record("p0000", (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    second = _proof_record("p0001", (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    # When
    selected = online_proof_candidates([second, first], set())

    # Then
    assert tuple(item.record for item in selected) == (first,)


def test_online_config_non_selective_arm_keeps_legacy_end_proof(tmp_path: Path) -> None:
    """Given online config on no-remask arm, when sanitizer appears, then proof is at end."""
    # Given
    proof = RecordingProof("not_reproduced")

    # When
    report = run_poc_revision(
        _config(particle_count=2, controller_mode="online-portfolio"),
        _run_request("feedback-no-remask"),
        NoRemaskCollisionSampler(),
        SequenceProbe(lambda _snapshot: _ProbeMetrics(new_edges=5, sanitizer=True)),
        proof,
        tmp_path,
    )

    # Then
    assert report.proof is not None
    rows = _ledger_rows(tmp_path)
    assert rows[-1]["kind"] == "proof_completed"
    assert len(proof.requests) == 1


def _config(  # noqa: PLR0913
    *,
    particle_count: int = 1,
    epochs: int = 1,
    probe_seconds: int = 1,
    controller_mode: Literal["legacy", "online-portfolio"] = "legacy",
    plateau_epochs: int = 1,
    proof_family_attempt_cap: int = 2,
    archive_size: int | None = None,
    frontier_width: int = 1,
) -> RevisionConfig:
    return RevisionConfig(
        schema_version="dllm-poc-revision-config/v1",
        particle_count=particle_count,
        epochs=epochs,
        initial_steps=8,
        revision_steps=2,
        probe_seconds=probe_seconds,
        fuzzer_seeds=(7, 11),
        proof_seconds=5,
        controller_mode=controller_mode,
        archive_size=max(8, particle_count) if archive_size is None else archive_size,
        plateau_epochs=plateau_epochs,
        proof_family_attempt_cap=proof_family_attempt_cap,
        frontier_width=frontier_width,
    )


def _run_request(
    arm: Literal[
        "libfuzzer-only",
        "structured-one-shot",
        "feedback-no-remask",
        "selective-revision",
        "pseudo-feedback",
    ],
    *,
    pseudo_feedback_rotation_seed: int = 0,
    particle_seed_base: int = 0,
    proof_seed: int = 20260820,
) -> RevisionRunRequest:
    return RevisionRunRequest(
        schema_version="dllm-poc-revision-run/v1",
        arm=arm,
        pseudo_feedback_rotation_seed=pseudo_feedback_rotation_seed,
        particle_seed_base=particle_seed_base,
        proof_seed=proof_seed,
    )


def _particle_id_sampler() -> DeterministicSampler:
    class ParticleIdSampler(DeterministicSampler):
        @override
        def initial(self, request: InitialParticleRequest) -> RevisionResult:
            self.initial_requests.append(request)
            selector = int(request.particle_id.removeprefix("p"))
            return RevisionResult(
                selectors=(selector, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                token_ids=(selector, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                slot_positions=tuple(range(10)),
                selected_slot_token_ids=(),
                token_state_sha256=sha256_bytes(f"initial:{request.particle_id}".encode()),
                transfer_evidence_sha256=sha256_bytes(b"initial-transfer"),
                rng_state_sha256=sha256_bytes(f"rng:{request.seed}".encode()),
                nfe=request.steps,
                forward_calls=request.steps,
                gpu_ms=0,
                cache_hit=False,
                trajectory_uncertainty=0.75,
                suggested_field="width",
            )

    return ParticleIdSampler(
        revisions={
            "origin_x": (1, 0, 1, 0, 0, 0, 0, 0, 0, 0),
            "width": (1, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        }
    )


def _particle_id_unique_revision_sampler() -> DeterministicSampler:
    class ParticleIdUniqueRevisionSampler(DeterministicSampler):
        @override
        def initial(self, request: InitialParticleRequest) -> RevisionResult:
            self.initial_requests.append(request)
            selector = int(request.particle_id.removeprefix("p"))
            selectors: SelectorSlots = (selector, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            return RevisionResult(
                selectors=selectors,
                token_ids=selectors,
                slot_positions=tuple(range(10)),
                selected_slot_token_ids=(),
                token_state_sha256=sha256_bytes(f"initial:{request.particle_id}".encode()),
                transfer_evidence_sha256=sha256_bytes(b"initial-transfer"),
                rng_state_sha256=sha256_bytes(f"rng:{request.seed}".encode()),
                nfe=request.steps,
                forward_calls=request.steps,
                gpu_ms=0,
                cache_hit=False,
                trajectory_uncertainty=0.75,
                suggested_field="width",
            )

        @override
        def revise(self, request: RevisionRequest) -> RevisionResult:
            self.revision_requests.append(request)
            selector = int(request.particle_id.removeprefix("p"))
            selectors: SelectorSlots = (selector, 0, 1, 0, 0, 0, 0, 0, 0, 0)
            return RevisionResult(
                selectors=selectors,
                token_ids=selectors,
                slot_positions=request.slot_positions,
                selected_slot_token_ids=(selectors[2],),
                token_state_sha256=sha256_bytes(f"revision:{request.particle_id}".encode()),
                transfer_evidence_sha256=sha256_bytes(f"transfer:{request.particle_id}".encode()),
                rng_state_sha256=request.rng_state_sha256,
                nfe=request.revision_steps,
                forward_calls=request.revision_steps,
                gpu_ms=1,
                cache_hit=False,
                trajectory_uncertainty=0.25,
                suggested_field=request.changed_field,
            )

    return ParticleIdUniqueRevisionSampler()


class InvalidMaterializationSampler(DeterministicSampler):
    """Sampler double that hits the controller's narrow materialization failure path."""

    @override
    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        self.initial_requests.append(request)
        raise ParticleDecodeError(reason="controlled invalid materialization")


class RevisionInvalidMaterializationSampler(DeterministicSampler):
    """Sampler double that fails during the revision materialization path."""

    @override
    def revise(self, request: RevisionRequest) -> RevisionResult:
        self.revision_requests.append(request)
        raise ParticleDecodeError(reason="controlled invalid revision")


class NoRemaskCollisionSampler(DeterministicSampler):
    """Sampler double with unique initial particles and colliding no-remask resamples."""

    @override
    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        self.initial_requests.append(request)
        selector = int(request.particle_id.removeprefix("p")) if request.steps == 8 else 0
        selectors: SelectorSlots = (
            selector,
            0,
            0 if request.steps == 8 else 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        return RevisionResult(
            selectors=selectors,
            token_ids=selectors,
            slot_positions=tuple(range(10)),
            selected_slot_token_ids=(),
            token_state_sha256=sha256_bytes(
                f"initial:{request.particle_id}:{request.steps}".encode()
            ),
            transfer_evidence_sha256=sha256_bytes(b"initial-transfer"),
            rng_state_sha256=sha256_bytes(f"rng:{request.seed}".encode()),
            nfe=request.steps,
            forward_calls=request.steps,
            gpu_ms=request.steps,
            cache_hit=False,
            trajectory_uncertainty=0.75,
            suggested_field="width",
        )


class RotatedFeedbackDivergenceSampler(DeterministicSampler):
    """Sampler double with paired first-epoch candidates and distinct second fields."""

    @override
    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        self.initial_requests.append(request)
        selector = int(request.particle_id.removeprefix("p"))
        selectors: SelectorSlots = (selector, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return _result(
            request.particle_id,
            selectors,
            request.steps,
            request.steps,
            request.steps,
            "origin_x",
            request.seed,
        )

    @override
    def revise(self, request: RevisionRequest) -> RevisionResult:
        self.revision_requests.append(request)
        origin_x_calls = sum(item.changed_field == "origin_x" for item in self.revision_requests)
        match request.changed_field:
            case "origin_x" if origin_x_calls <= 2:
                selectors: SelectorSlots = (2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            case "origin_x":
                selectors = (3, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            case "origin_y":
                selectors = (2, 1, 0, 0, 0, 0, 0, 0, 0, 0)
            case (
                "width"
                | "height"
                | "compression"
                | "pixel_type"
                | "line_order"
                | "x_sampling"
                | "payload_pattern"
                | "chunk_offset"
            ):
                selectors = (2, 0, 1, 0, 0, 0, 0, 0, 0, 0)
            case unreachable:
                assert_never(unreachable)
        return _result(
            request.particle_id,
            selectors,
            request.revision_steps,
            request.revision_steps,
            1,
            request.changed_field,
            request.seed,
            request.slot_positions,
        )


class CacheHitSampler(DeterministicSampler):
    """Sampler double that marks initial, revision, and resample results cached."""

    @override
    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        self.initial_requests.append(request)
        selectors: SelectorSlots = (
            0,
            0,
            0 if request.steps == 8 else 1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        return RevisionResult(
            selectors=selectors,
            token_ids=selectors,
            slot_positions=tuple(range(10)),
            selected_slot_token_ids=(),
            token_state_sha256=sha256_bytes(
                f"cache-hit-initial:{request.particle_id}:{request.steps}".encode()
            ),
            transfer_evidence_sha256=sha256_bytes(b"cache-hit-initial-transfer"),
            rng_state_sha256=sha256_bytes(f"cache-hit-rng:{request.seed}".encode()),
            nfe=request.steps,
            forward_calls=request.steps,
            gpu_ms=3,
            cache_hit=True,
            trajectory_uncertainty=0.75,
            suggested_field="width",
        )

    @override
    def revise(self, request: RevisionRequest) -> RevisionResult:
        self.revision_requests.append(request)
        selectors: SelectorSlots = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        return RevisionResult(
            selectors=selectors,
            token_ids=selectors,
            slot_positions=request.slot_positions,
            selected_slot_token_ids=(selectors[2],),
            token_state_sha256=sha256_bytes(b"cache-hit-revision"),
            transfer_evidence_sha256=sha256_bytes(b"cache-hit-revision-transfer"),
            rng_state_sha256=request.rng_state_sha256,
            nfe=request.revision_steps,
            forward_calls=request.revision_steps,
            gpu_ms=2,
            cache_hit=True,
            trajectory_uncertainty=0.25,
            suggested_field=request.changed_field,
        )


class TwoLineageOutcomeSampler(DeterministicSampler):
    """Sampler double with two lineages whose first children diverge in reward."""

    @override
    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        self.initial_requests.append(request)
        selector = int(request.particle_id.removeprefix("p"))
        selectors: SelectorSlots = (selector, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return _result(
            request.particle_id,
            selectors,
            request.steps,
            request.steps,
            request.steps,
            "width",
            request.seed,
        )

    @override
    def revise(self, request: RevisionRequest) -> RevisionResult:
        self.revision_requests.append(request)
        selector = int(request.particle_id.removeprefix("p"))
        width_selector = 1 if selector == 0 else 0
        height_selector = 1 if selector == 1 else 0
        selectors: SelectorSlots = (
            selector,
            0,
            width_selector,
            height_selector,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        return _result(
            request.particle_id,
            selectors,
            request.revision_steps,
            request.revision_steps,
            1,
            request.changed_field,
            request.seed,
            request.slot_positions,
        )


class DuplicateCauseSampler(DeterministicSampler):
    """Sampler double with p0001 child used for duplicate-cause promotion checks."""

    @override
    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        self.initial_requests.append(request)
        selector = int(request.particle_id.removeprefix("p"))
        selectors: SelectorSlots = (selector, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return _result(
            request.particle_id,
            selectors,
            request.steps,
            request.steps,
            request.steps,
            "height",
            request.seed,
        )

    @override
    def revise(self, request: RevisionRequest) -> RevisionResult:
        self.revision_requests.append(request)
        selector = int(request.particle_id.removeprefix("p"))
        selectors: SelectorSlots = (selector, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        return _result(
            request.particle_id,
            selectors,
            request.revision_steps,
            request.revision_steps,
            1,
            request.changed_field,
            request.seed,
            request.slot_positions,
        )


class TrajectoryDivergenceSampler(DeterministicSampler):
    """Sampler double whose lineages have different late-commit fields."""

    @override
    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        self.initial_requests.append(request)
        selector = int(request.particle_id.removeprefix("p"))
        selectors: SelectorSlots = (selector, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        late_position = 2 if request.particle_id == "p0000" else 3
        early_positions = (0, 1, 3, 8) if late_position == 2 else (0, 1, 2, 8)
        return _trajectory_result(
            particle_id=request.particle_id,
            selectors=selectors,
            steps=request.steps,
            seed=request.seed,
            late_position=late_position,
            early_positions=early_positions,
        )

    @override
    def revise(self, request: RevisionRequest) -> RevisionResult:
        self.revision_requests.append(request)
        if request.changed_field == "width":
            selectors: SelectorSlots = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        else:
            selectors = (1, 0, 0, 1, 0, 0, 0, 0, 0, 0)
        return _result(
            request.particle_id,
            selectors,
            request.revision_steps,
            request.revision_steps,
            1,
            request.changed_field,
            request.seed,
            request.slot_positions,
        )


def _ledger_rows(root: Path) -> tuple[dict[str, str | int | bool | tuple[str, ...] | None], ...]:
    return tuple(
        json.loads(line)["event"]
        for line in (root / "revision-events.jsonl").read_text(encoding="utf-8").splitlines()
    )


def _cap_events(
    root: Path,
) -> tuple[dict[str, str | int | bool | tuple[str, ...] | None], ...]:
    return tuple(row for row in _ledger_rows(root) if row["kind"] == "proof_family_capped")


def _event_row(
    rows: tuple[dict[str, str | int | bool | tuple[str, ...] | None], ...],
    kind: str,
) -> dict[str, str | int | bool | tuple[str, ...] | None]:
    for row in rows:
        if row["kind"] == kind:
            return row
    raise AssertionError(kind)


def _snapshot_sha256(selectors: SelectorSlots) -> str:
    return materialize_particle(decode_particle(selectors, "validity-preserving")).sha256


def _result(  # noqa: PLR0913, PLR0917
    particle_id: str,
    selectors: SelectorSlots,
    nfe: int,
    forward_calls: int,
    gpu_ms: int,
    suggested_field: FieldName,
    seed: int,
    slot_positions: tuple[int, ...] | None = None,
) -> RevisionResult:
    slots = tuple(range(10)) if slot_positions is None else slot_positions
    return RevisionResult(
        selectors=selectors,
        token_ids=selectors,
        slot_positions=slots,
        selected_slot_token_ids=(),
        token_state_sha256=sha256_bytes(f"result:{particle_id}:{seed}".encode()),
        transfer_evidence_sha256=sha256_bytes(f"transfer:{particle_id}:{seed}".encode()),
        rng_state_sha256=sha256_bytes(f"rng:{seed}".encode()),
        nfe=nfe,
        forward_calls=forward_calls,
        gpu_ms=gpu_ms,
        cache_hit=False,
        trajectory_uncertainty=0.5,
        suggested_field=suggested_field,
    )


def _proof_record(particle_id: str, selectors: SelectorSlots) -> ProbeRecord:
    snapshot = materialize_particle(decode_particle(selectors, "validity-preserving"))
    crash_bytes = b"same-crash"
    crash_sha = sha256_bytes(crash_bytes)
    result = _result(
        particle_id,
        selectors,
        nfe=1,
        forward_calls=1,
        gpu_ms=1,
        suggested_field="width",
        seed=1,
    )
    return ProbeRecord(
        particle=Particle(particle_id=particle_id, result=result, snapshot=snapshot),
        feedback=ProbeFeedback(
            particle_id=particle_id,
            snapshot_sha256=snapshot.sha256,
            build_status="skipped",
            run_status="crashed",
            valid_snapshot=True,
            sanitizer_observed=True,
            target_branches_hit=1,
            new_edges=1,
            rare_edges=0,
            value_profile_features=0,
            executions_per_second=10,
            trajectory_uncertainty=0.1,
            corpus_artifact_sha256=(crash_sha,),
            wall_ms=1,
            build_log_sha256=_ZERO_SHA,
            run_log_sha256=sha256_bytes(b"same-log"),
            build_argv_sha256=_ZERO_SHA,
            run_argv_sha256=_ZERO_SHA,
            sanitizer_class="heap-buffer-overflow",
            root_signature="same root",
            provisional_family_key=sha256_bytes(b"same-family"),
            rejected_count=0,
            dedup_count=0,
        ),
        cache_hit=False,
        crash_artifacts=(CrashArtifact(sha256=crash_sha, content=crash_bytes),),
    )


def _trajectory_result(  # noqa: PLR0913
    *,
    particle_id: str,
    selectors: SelectorSlots,
    steps: int,
    seed: int,
    late_position: int,
    early_positions: tuple[int, ...],
) -> RevisionResult:
    return RevisionResult(
        selectors=selectors,
        token_ids=selectors,
        slot_positions=tuple(range(10)),
        selected_slot_token_ids=selectors,
        token_state_sha256=sha256_bytes(f"trajectory:{particle_id}:{seed}".encode()),
        transfer_evidence_sha256=sha256_bytes(f"trajectory-transfer:{particle_id}".encode()),
        rng_state_sha256=sha256_bytes(f"trajectory-rng:{seed}".encode()),
        nfe=steps,
        forward_calls=steps,
        gpu_ms=steps,
        cache_hit=False,
        trajectory_uncertainty=0.5,
        suggested_field="origin_x",
        trajectory_steps=(
            RevisionTrajectoryStep(
                step_index=1,
                block_index=0,
                masked_before=len(early_positions) + 1,
                committed_positions=early_positions,
                token_ids=tuple(range(100, 100 + len(early_positions))),
                confidences=tuple(0.95 for _ in early_positions),
                proposal_positions=early_positions,
                proposal_token_ids=tuple(range(100, 100 + len(early_positions))),
                proposal_confidences=tuple(0.95 for _ in early_positions),
                proposal_margins=tuple(0.9 for _ in early_positions),
                remaining_masked=1,
            ),
            RevisionTrajectoryStep(
                step_index=3,
                block_index=0,
                masked_before=1,
                committed_positions=(late_position,),
                token_ids=(200 + late_position,),
                confidences=(0.15,),
                proposal_positions=(late_position,),
                proposal_token_ids=(200 + late_position,),
                proposal_confidences=(0.15,),
                proposal_margins=(0.05,),
                remaining_masked=0,
            ),
        ),
        trajectory_evidence_sha256=sha256_bytes(f"trajectory-evidence:{particle_id}".encode()),
    )
