"""Task 5 fix4A production model-execution regressions.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
    build_campaign_manifest,
    build_cell_requests,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.model_executor import (
    DeferredProofAdapter,
    ModelPreparationError,
    ProductionModelArmExecutor,
)
from dllm_agent.experiments.openexr_one_day.post_search import PostSearchOutcome
from dllm_agent.experiments.openexr_one_day.runtime import ProductionRuntimeFactory
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from dllm_agent.revision.controller_models import (
    InitialParticleRequest,
    ProbeRunner,
    ProofRequest,
    RevisionRequest,
    RevisionResult,
    RevisionRunRequest,
    RevisionSampler,
)
from dllm_agent.revision.models import (
    PocRevisionReport,
    ProofSummary,
    RevisionConfig,
    RuntimeAccounting,
    SnapshotArtifact,
    SnapshotEvidence,
)
from dllm_agent.revision.pipeline import run_poc_revision
from dllm_agent.targets.openexr.probe import ProbeRequest, ProbeResult
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakeResidentPool,
    FakeRevisionDelegate,
    _fake_result,
)
from tests.test_openexr_one_day_coordinator import _passing_gates, _qualified_config
from tests.test_openexr_one_day_fix2 import SpyRevisionWorkerFactory

if TYPE_CHECKING:
    from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, CellRunRequest
    from dllm_agent.experiments.openexr_one_day.models import ModelArm
    from dllm_agent.experiments.openexr_one_day.runtime import RevisionWorkerFactory

_MODEL_BYTES = b"\x76\x2f\x31\x01actual-exr-bytes-\xff"


def test_task2_report_persists_content_addressed_snapshot_artifacts(tmp_path: Path) -> None:
    report = run_poc_revision(
        RevisionConfig(
            schema_version="dllm-poc-revision-config/v1",
            particle_count=2,
            epochs=1,
            initial_steps=5,
            revision_steps=5,
            probe_seconds=1,
            fuzzer_seeds=(100, 101),
            proof_seconds=1,
            controller_mode="legacy",
        ),
        RevisionRunRequest(
            schema_version="dllm-poc-revision-run/v1",
            arm="structured-one-shot",
            particle_seed_base=100,
            proof_seed=1000,
        ),
        PipelineSampler(),
        RecordingProbe(),
        SkippedProof(),
        tmp_path / "task2",
    )

    assert len(report.snapshot_artifacts) == len(report.snapshots)
    for snapshot, artifact in zip(report.snapshots, report.snapshot_artifacts, strict=True):
        payload = artifact.path.read_bytes()
        assert artifact.snapshot_sha256 == snapshot.snapshot_sha256
        assert artifact.size_bytes == snapshot.byte_length
        assert sha256_bytes(payload) == snapshot.snapshot_sha256


def test_production_model_executor_copies_actual_task2_snapshot_bytes(data_root: Path) -> None:
    config = _qualified_config(data_root)
    request = _model_request(config, data_root)
    runner = SnapshotReportRunner(_MODEL_BYTES)
    executor = ProductionModelArmExecutor(run_revision=runner.run, probe=RecordingProbe())

    result = executor.prepare_corpus(request)

    assert runner.configs[0].initial_steps == config.candidate_steps
    assert runner.configs[0].revision_steps == config.candidate_steps
    assert result.written_snapshot_count == 1
    assert result.written_paths[0].read_bytes() == _MODEL_BYTES
    assert result.written_paths[0].name == f"model-{sha256_bytes(_MODEL_BYTES)}.exr"


def test_production_runtime_factory_injects_real_openexr_probe(data_root: Path) -> None:
    graph = ProductionRuntimeFactory(
        runtime_python=Path("/usr/bin/python3"),
        source_root=Path("/data/dllm-agent/source/openexr"),
        cache_dir=Path("/data/dllm-agent/cache"),
        prompt_path=Path("prompts/openexr-system.md"),
        worker_factory=cast(
            "RevisionWorkerFactory",
            cast("object", SpyRevisionWorkerFactory()),
        ),
    ).build(_qualified_config(data_root))

    with graph:
        assert graph.model_executor.probe.__class__.__name__ == "PreparedProbe"


def test_model_prep_failure_marks_only_that_cell_treatment_failed(data_root: Path) -> None:
    config = _qualified_config(data_root)
    failing = "wave-00-rep-00-selective-revision"

    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        model_executor=FailingModelExecutor(failing_cell_id=failing),
        cell_runner=FakeCellRunner(),
        post_search=NoopPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    failed = next(cell for cell in outcome.cells if cell.cell_id == failing)
    assert failed.terminal_reason == "treatment-failed"
    assert failed.coverage_frozen_through_seconds == 10_800
    assert outcome.launched_cell_count == 50
    assert sum(cell.terminal_reason == "treatment-failed" for cell in outcome.cells) == 1
    assert (failed.artifact_root / "model-prep-failure.json").is_file()


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fix4")
    return Path("/data/dllm-agent") / f"task5-fix4-{sha256_bytes(str(root).encode())[:12]}"


@dataclass(frozen=True, slots=True)
class PipelineSampler:
    calls: list[int] = field(default_factory=list)

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        del request
        _ = self.calls.append(len(self.calls))
        return _fake_result(len(self.calls), len(self.calls))

    def revise(self, request: RevisionRequest) -> RevisionResult:
        del request
        return _fake_result(len(self.calls) + 1, len(self.calls) + 1)


@dataclass(frozen=True, slots=True)
class RecordingProbe:
    requests: list[ProbeRequest] = field(default_factory=list)

    def run(self, request: ProbeRequest) -> ProbeResult:
        _ = self.requests.append(request)
        return ProbeResult(
            snapshot_sha256=request.snapshot_sha256,
            cache_key_sha256=sha256_bytes(
                f"{request.snapshot_sha256}:{request.fuzzer_seed}".encode()
            ),
            cache_hit=False,
            run_status="completed",
            sanitizer_observed=False,
            final_units=1,
            coverage_edges=1,
            value_profile_features=1,
            corpus_count=1,
            executions_per_second=1,
            run_log_sha256="1" * 64,
            run_argv_sha256="2" * 64,
            crash_artifact_sha256=(),
            crash_artifact_bytes=(),
        )


@dataclass(frozen=True, slots=True)
class SkippedProof:
    def run(self, request: ProofRequest) -> ProofSummary:
        return ProofSummary(
            oracle_status="skipped",
            input_sha256=request.snapshot_sha256,
            proof_log_sha256=None,
        )


@dataclass(frozen=True, slots=True)
class SnapshotReportRunner:
    payload: bytes
    configs: list[RevisionConfig] = field(default_factory=list)

    def run(  # noqa: PLR0913
        self,
        *,
        config: RevisionConfig,
        arm: ModelArm,
        sampler: RevisionSampler,
        run_request: RevisionRunRequest,
        probe: ProbeRunner,
        proof_runner: DeferredProofAdapter,
        artifact_dir: Path,
    ) -> PocRevisionReport:
        del arm, sampler, run_request, probe, proof_runner
        _ = self.configs.append(config)
        artifact_dir.mkdir(parents=True)
        digest = sha256_bytes(self.payload)
        path = artifact_dir / "snapshots" / f"{digest}.exr"
        path.parent.mkdir()
        _ = path.write_bytes(self.payload)
        return PocRevisionReport(
            schema_version="dllm-poc-revision-report/v1",
            config_hash_sha256="a" * 64,
            snapshots=(
                SnapshotEvidence(
                    particle_id="p0000",
                    snapshot_sha256=digest,
                    anchor_sha256="b" * 64,
                    field_map_sha256="c" * 64,
                    materializer_ms=1,
                    valid=True,
                    fault_field=None,
                    byte_length=len(self.payload),
                ),
            ),
            snapshot_artifacts=(
                SnapshotArtifact(
                    snapshot_sha256=digest,
                    path=path,
                    size_bytes=len(self.payload),
                ),
            ),
            probe_feedback=(),
            field_attributions=(),
            remask_decisions=(),
            revision_events_sha256="d" * 64,
            accounting_rejected=0,
            accounting_dedup=0,
            proof=None,
            runtime_accounting=RuntimeAccounting(
                recorded_nfe=3,
                recorded_forward_calls=4,
                recorded_gpu_ms=5,
            ),
        )


@dataclass(frozen=True, slots=True)
class FailingModelExecutor:
    failing_cell_id: str

    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        if request.cell.cell_id == self.failing_cell_id:
            reason = "unit-test failure"
            raise ModelPreparationError(request.cell.cell_id, reason)
        path = request.cell.artifact_root / "corpus" / "model.exr"
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(b"model")
        return ModelArmExecutionResult(
            written_snapshot_count=1,
            written_paths=(path,),
            nfe=1,
            forward_calls=1,
            gpu_ms=1,
            bank_ready_elapsed_seconds=0.1,
        )


@dataclass(frozen=True, slots=True)
class NoopPostSearch:
    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        del request, result
        return PostSearchOutcome(
            coverage_replay_count=0,
            proof_replay_count=0,
            coverage_series_sha256="0" * 64,
            proof_result_sha256="0" * 64,
            proof_wall_seconds=0.0,
            proof_cpu_seconds=0.0,
        )

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        del request
        return PostSearchOutcome(
            coverage_replay_count=0,
            proof_replay_count=0,
            coverage_series_sha256="0" * 64,
            proof_result_sha256="0" * 64,
            proof_wall_seconds=0.0,
            proof_cpu_seconds=0.0,
        )


def _model_request(config: CampaignConfig, root: Path) -> ModelArmExecutionRequest:
    del root
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    manifest = build_campaign_manifest(config, schedule)
    request = build_cell_requests(
        config=config,
        manifest=manifest,
        schedule=schedule,
        bank_by_repetition=dict.fromkeys(range(10), "a" * 64),
        resource_map=config.resource_map,
    )[0]
    return ModelArmExecutionRequest(
        config=config,
        cell=next(
            cell
            for cell in coordinate_campaign(
                config,
                resident_pool=FakeResidentPool(),
                model_executor=None,
                cell_runner=FakeCellRunner(),
                post_search=NoopPostSearch(),
                gate_runner=FakeGateRunner(_passing_gates()),
            ).cells
            if cell.cell_id == request.cell_id
        ),
        arm="selective-revision",
        revision_delegate=FakeRevisionDelegate(),
    )
