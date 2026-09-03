"""Candidate-local outcome accounting for online selective revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision import pipeline
from dllm_agent.revision.controller_models import (
    InitialParticleRequest,
    RevisionResult,
    RevisionRunRequest,
)
from dllm_agent.revision.models import RevisionConfig
from dllm_agent.revision.trajectory_policy import LineageFieldOutcome
from dllm_agent.targets.openexr.probe import ProbeRequest, ProbeResult
from tests.test_poc_revision_pipeline import (
    DeterministicSampler,
    RecordingProof,
    RevisionInvalidMaterializationSampler,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Literal

    import pytest

    from dllm_agent.targets.openexr.particle import FieldName, SelectorSlots


def test_duplicate_revision_is_attributed_only_to_its_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deduplicated child must become one local field outcome before the next choice."""
    captured = _capture_outcomes(monkeypatch)
    initial: SelectorSlots = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    _ = pipeline.run_poc_revision(
        config(epochs=2),
        run_request(),
        DeterministicSampler(initial_selectors=initial, revisions={"width": initial}),
        LocalProbe(lambda _snapshot: ProbeMetrics(new_edges=5)),
        RecordingProof("skipped"),
        tmp_path,
    )

    assert captured[0] == ()
    assert captured[1] == (
        LineageFieldOutcome(
            lineage_id="p0000",
            field="width",
            reward_delta=0.0,
            trials=1,
            duplicate_count=1,
        ),
    )


def test_failed_exact_proof_is_attributed_to_the_child_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child-only failed proof must affect that revision field, not global history."""
    captured = _capture_outcomes(monkeypatch)

    _ = pipeline.run_poc_revision(
        config(
            epochs=2,
            proof_family_attempt_cap=4,
        ),
        run_request(),
        DeterministicSampler(
            revisions={
                "width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
            }
        ),
        LocalProbe(
            lambda snapshot: ProbeMetrics(
                new_edges=5,
                sanitizer=True,
                crash_artifact_bytes=(f"crash:{snapshot}".encode(),),
            )
        ),
        RecordingProof("not_reproduced"),
        tmp_path,
    )

    assert captured[0] == ()
    assert captured[1][0].lineage_id == "p0000"
    assert captured[1][0].field == "width"
    assert captured[1][0].duplicate_count == 0
    assert captured[1][0].proof_failure_count == 1


def test_invalid_revision_is_attributed_only_to_its_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected materialization must become one local field outcome."""
    captured = _capture_outcomes(monkeypatch)

    _ = pipeline.run_poc_revision(
        config(epochs=2),
        run_request(),
        RevisionInvalidMaterializationSampler(),
        LocalProbe(lambda _snapshot: ProbeMetrics(new_edges=5)),
        RecordingProof("skipped"),
        tmp_path,
    )

    assert captured[0] == ()
    assert captured[1] == (
        LineageFieldOutcome(
            lineage_id="p0000",
            field="width",
            reward_delta=0.0,
            trials=1,
            invalid_count=1,
        ),
    )


def test_remask_decision_hash_binds_parent_trajectory_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same chosen field with a different dLLM trace must produce a new decision hash."""
    _ = _capture_outcomes(monkeypatch)
    first = pipeline.run_poc_revision(
        config(epochs=1),
        run_request(),
        TrajectoryEvidenceSampler("1" * 64),
        LocalProbe(lambda _snapshot: ProbeMetrics(new_edges=5)),
        RecordingProof("skipped"),
        tmp_path / "first",
    )
    second = pipeline.run_poc_revision(
        config(epochs=1),
        run_request(),
        TrajectoryEvidenceSampler("2" * 64),
        LocalProbe(lambda _snapshot: ProbeMetrics(new_edges=5)),
        RecordingProof("skipped"),
        tmp_path / "second",
    )

    assert first.remask_decisions[0].fields == second.remask_decisions[0].fields
    assert first.remask_decisions[0].decision_sha256 != second.remask_decisions[0].decision_sha256


def _capture_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[LineageFieldOutcome, ...]]:
    captured: list[tuple[LineageFieldOutcome, ...]] = []

    def choose(
        _result: RevisionResult,
        outcomes: tuple[LineageFieldOutcome, ...],
        lineage_id: str | None = None,
    ) -> FieldName:
        _ = lineage_id
        captured.append(outcomes)
        return "width"

    monkeypatch.setattr(pipeline, "choose_lineage_field", choose)
    return captured


@dataclass(frozen=True, slots=True)
class ProbeMetrics:
    """Local structural probe metrics consumed by the shared probe double."""

    new_edges: int = 0
    value_profile_features: int = 0
    execs: int = 10
    sanitizer: bool = False
    sanitizer_class: str | None = None
    root_signature: str | None = None
    provisional_family_key: str | None = None
    crash_artifact_bytes: tuple[bytes, ...] | None = None


class LocalProbe:
    """Minimal typed probe double for candidate-local outcome integration."""

    def __init__(self, feedback_for: Callable[[str], ProbeMetrics]) -> None:
        self._feedback_for: Callable[[str], ProbeMetrics] = feedback_for

    def run(self, request: ProbeRequest) -> ProbeResult:
        """Return deterministic coverage and optional exact crash bytes."""
        metrics = self._feedback_for(request.snapshot_sha256)
        crash_artifacts = metrics.crash_artifact_bytes
        if crash_artifacts is None:
            crash_artifacts = (
                (f"crash:{request.snapshot_sha256}".encode(),) if metrics.sanitizer else ()
            )
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


class TrajectoryEvidenceSampler(DeterministicSampler):
    """Sampler double that varies only the parent trajectory evidence digest."""

    def __init__(self, trajectory_evidence_sha256: str) -> None:
        super().__init__(revisions={"width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)})
        self._trajectory_evidence_sha256: str = trajectory_evidence_sha256

    @override
    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        result = super().initial(request)
        return result.model_copy(
            update={"trajectory_evidence_sha256": self._trajectory_evidence_sha256}
        )


def config(
    *,
    epochs: int,
    proof_family_attempt_cap: int = 2,
) -> RevisionConfig:
    """Return the smallest online-portfolio config needed by this boundary suite."""
    return RevisionConfig(
        schema_version="dllm-poc-revision-config/v1",
        particle_count=1,
        epochs=epochs,
        initial_steps=8,
        revision_steps=2,
        probe_seconds=1,
        fuzzer_seeds=(7, 11),
        proof_seconds=5,
        controller_mode="online-portfolio",
        archive_size=8,
        plateau_epochs=1,
        proof_family_attempt_cap=proof_family_attempt_cap,
    )


def run_request() -> RevisionRunRequest:
    """Return a deterministic selective-revision request."""
    arm: Literal["selective-revision"] = "selective-revision"
    return RevisionRunRequest(
        schema_version="dllm-poc-revision-run/v1",
        arm=arm,
        particle_seed_base=0,
        proof_seed=20260820,
    )
