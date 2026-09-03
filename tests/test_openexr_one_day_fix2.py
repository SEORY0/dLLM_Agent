"""Task 5 fix2 reviewer reproductions.

# noqa: SIZE_OK
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING, Final, Literal, Protocol, Self, cast, override

import pytest
from tools.run_openexr_one_day import qualify_verified

from dllm_agent.experiments.openexr_one_day.c0_source import (
    C0SourceError,
    ExactC0Source,
    load_exact_c0_source,
)
from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, SearchAccounting
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignManifest,
    CampaignOutcome,
    CoordinatorError,
    ModelArmExecutionRequest,
    PostSearchRunner,
    build_campaign_manifest,
    build_cell_requests,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CoverageReplayer,
    CoverageSeries,
    CoverageStepResult,
    TargetCoverageCounts,
)
from dllm_agent.experiments.openexr_one_day.model_executor import (
    DeferredProofAdapter,
    ProductionModelArmExecutor,
)
from dllm_agent.experiments.openexr_one_day.models import GpuIndex
from dllm_agent.experiments.openexr_one_day.post_search import (
    PostSearchOutcome,
    ProductionPostSearch,
)
from dllm_agent.experiments.openexr_one_day.proof_batch import (
    ExactProofRequestModel,
    ExactProofRunner,
    ProofBatchResult,
    ProofClock,
    ProofOutcome,
)
from dllm_agent.experiments.openexr_one_day.qualification import (
    ProductionGateRunner,
    QualificationEvidence,
)
from dllm_agent.experiments.openexr_one_day.recovery import resume_campaign
from dllm_agent.experiments.openexr_one_day.runtime import (
    ProductionRuntimeFactory,
    ProductionTelemetryProbe,
    RevisionWorkerFactory,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from dllm_agent.targets.openexr.particle import decode_particle, materialize_particle
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakePostSearch,
    FakeResidentPool,
    FakeRevisionDelegate,
    fake_bank_sampler,
)
from tests.test_openexr_one_day_fix1 import _config_json, _delegates, _passing_gates

if TYPE_CHECKING:
    from dllm_agent.experiments.multitarget.ledger import JsonValue
    from dllm_agent.experiments.openexr_one_day.cell_runner import CellRunRequest
    from dllm_agent.experiments.openexr_one_day.coverage_series import (
        CoverageSeriesRequest,
        CoverageStepRequest,
    )
    from dllm_agent.experiments.openexr_one_day.models import ModelArm
    from dllm_agent.experiments.openexr_one_day.proof_batch import (
        ProofBatchRequest,
    )
    from dllm_agent.revision.controller_models import RevisionSampler
    from dllm_agent.runtime.revision_process import WorkerCpuAffinity

_C0_SNAPSHOT = materialize_particle(
    decode_particle((0, 0, 0, 0, 0, 0, 0, 0, 0, 0), "validity-preserving")
)
_C0_SHA: Final = _C0_SNAPSHOT.sha256


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    case = tmp_path_factory.mktemp("fix2")
    suffix = sha256(str(case).encode()).hexdigest()[:12]
    path = Path("/data/dllm-agent") / f"task5-fix2-{suffix}"
    if path.exists():
        message = f"test data root already exists: {path}"
        raise AssertionError(message)
    return path


def test_exact_c0_loader_rejects_hash_derived_bytes_and_accepts_materializer_fixture(
    tmp_path: Path,
) -> None:
    # Given
    valid = tmp_path / "c0.hex"
    _ = valid.write_text(_C0_SNAPSHOT.content.hex(), encoding="ascii")
    fake = tmp_path / "fake.hex"
    _ = fake.write_text(_C0_SHA.encode().hex(), encoding="ascii")

    # When
    loaded = load_exact_c0_source(ExactC0Source(path=valid, sha256=_C0_SHA))

    # Then
    assert loaded.content == _C0_SNAPSHOT.content
    assert loaded.sha256 == _C0_SHA
    with pytest.raises(C0SourceError, match="C0 sha256 mismatch"):
        _ = load_exact_c0_source(ExactC0Source(path=fake, sha256=_C0_SHA))


def test_exact_c0_source_rejects_symlink_ancestor(tmp_path: Path) -> None:
    # Given
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    source = real_dir / "c0.hex"
    _ = source.write_text(_C0_SNAPSHOT.content.hex(), encoding="ascii")
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    # When / Then
    with pytest.raises(ValueError, match="symlink"):
        _ = load_exact_c0_source(ExactC0Source(path=link_dir / "c0.hex", sha256=_C0_SHA))


def test_production_telemetry_routes_requested_gpu_and_pid_to_real_readers() -> None:
    # Given
    proc = ScriptedProcReader()
    runner = RecordingCommandRunner(
        stdout=b"24576 MiB\n",
        process=CompletedProcess(args=("nvidia-smi",), returncode=0, stdout=b"24576 MiB\n"),
    )
    probe = ProductionTelemetryProbe(proc_reader=proc, command_runner=runner)

    # When
    snapshot = probe.capture(GpuIndex(3), 4242)

    # Then
    assert proc.meminfo_reads == 1
    assert proc.status_pids == [4242]
    assert runner.argvs == [
        (
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            "3",
        )
    ]
    assert snapshot.gpu_free_bytes == 24_576 * 1024 * 1024
    assert snapshot.process_rss_bytes == 1234 * 1024


def test_production_gate_rejects_missing_hash_bound_qualification_evidence(
    data_root: Path,
    tmp_path: Path,
) -> None:
    # Given
    config = _qualified_config(data_root / "gate")
    evidence_path = tmp_path / "qualification.json"
    evidence = QualificationEvidence.for_manifest(
        manifest_sha256="0" * 64,
        config_sha256=config.config_sha256,
        facts={"hashes_match": True},
    )
    _ = evidence_path.write_text(
        json.dumps(evidence.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    manifest = _manifest(config)

    # When / Then
    with pytest.raises(CoordinatorError, match="qualification evidence"):
        _ = ProductionGateRunner(evidence_path=evidence_path).evaluate(manifest)


def test_production_post_search_calls_task4_coverage_and_proof_with_cell_result(
    data_root: Path,
) -> None:
    # Given
    request = _cell_request(data_root)
    result = _cell_result(request)
    coverage = RecordingCoverageBuilder()
    proof = RecordingProofBuilder()
    post_search = ProductionPostSearch(
        coverage_builder=coverage,
        proof_builder=proof,
        coverage_replayer=RecordingCoverageReplayer(),
        proof_runner=RecordingProofRunner(),
        clock=ManualClock(),
    )

    # When
    outcome = post_search.coverage_and_proof(request, result)

    # Then
    assert coverage.requests[0].corpus_index.corpus_root == request.corpus_root
    assert proof.requests[0].search_cpu_seconds == result.search_accounting.cpu_seconds
    assert outcome.coverage_replay_count == 13
    assert outcome.proof_replay_count == 2


def test_model_arm_executor_calls_task2_with_frozen_adapter_and_no_exact_proof(
    data_root: Path,
) -> None:
    # Given
    bank = fake_bank_sampler()
    config = _qualified_config(data_root)
    campaign = _campaign_outcome(config)
    cell = next(item for item in campaign.cells if item.arm == "structured-one-shot")
    runner = SpyPocRevisionRunner()
    executor = ProductionModelArmExecutor(legacy_run_revision=runner.run)

    # When
    result = executor.prepare_corpus(
        ModelArmExecutionRequest(
            config=config,
            cell=cell,
            bank_sampler=bank,
            revision_delegate=FakeRevisionDelegate(),
            arm="structured-one-shot",
        )
    )

    # Then
    assert runner.arms == ["structured-one-shot"]
    assert isinstance(runner.proof_adapters[0], DeferredProofAdapter)
    assert runner.proof_adapters[0].exact_replay_count == 0
    assert result.written_snapshot_count >= 1
    assert all(path.read_bytes() for path in result.written_paths)


def test_runtime_factory_reuses_entered_resident_clients_for_bank_and_arm_delegates(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    worker_factory = SpyRevisionWorkerFactory()
    factory = ProductionRuntimeFactory(
        runtime_python=Path("/usr/bin/python3"),
        source_root=Path("/data/dllm-agent/source/openexr"),
        cache_dir=Path("/data/dllm-agent/cache"),
        prompt_path=Path("prompts/openexr-system.md"),
        worker_factory=cast("RevisionWorkerFactory", cast("object", worker_factory)),
    )

    # When
    graph = factory.build(config)

    # Then
    pool_workers = tuple(binding.worker for binding in graph.resident_pool.bindings)
    assert graph.bank_sampler in pool_workers
    assert set(graph.revision_delegates.values()) <= set(pool_workers)
    assert len(pool_workers) == 8
    assert tuple(affinity.cpus for affinity in worker_factory.affinities) == (
        (0, 20),
        (1, 21),
        (2, 22),
        (3, 23),
        (10, 30),
        (11, 31),
        (12, 32),
        (13, 33),
    )


def test_cli_stale_detached_digest_fails_before_root_write_and_runtime_construction(
    tmp_path: Path,
) -> None:
    # Given
    config_path = tmp_path / "stale.json"
    root = tmp_path / "qualification"
    payload = _config_json("/data/dllm-agent/fix2-stale")
    payload["config_sha256"] = "1234" * 16
    _ = config_path.write_text(json.dumps(payload), encoding="utf-8")

    # When / Then
    with pytest.raises(CoordinatorError, match="config digest"):
        _ = qualify_verified(config_path, root)
    assert not root.exists()


def test_raw_reports_fail_closed_without_real_post_search_accounting(data_root: Path) -> None:
    # Given
    config = _qualified_config(data_root)

    # When / Then
    with pytest.raises(CoordinatorError, match="post-search accounting"):
        _ = coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(),
            post_search=cast("PostSearchRunner", cast("object", MissingAccountingPostSearch())),
            gate_runner=FakeGateRunner(_passing_gates()),
        )


def test_expired_resume_preserves_existing_treatment_failed_payload(data_root: Path) -> None:
    # Given
    config = _qualified_config(data_root)
    failed = "wave-00-rep-00-selective-revision"
    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(treatment_failures={failed: "controller-crash"}),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # When
    outcome = resume_campaign(
        config.artifact_root,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        now_monotonic_seconds=200_000.0,
    )

    # Then
    treated = next(cell for cell in outcome.cells if cell.cell_id == failed)
    assert treated.terminal_reason == "treatment-failed"
    assert treated.coverage_frozen_through_seconds == 10_800


class ProcReader(Protocol):
    def meminfo(self) -> str:
        """Read host meminfo."""
        ...

    def status(self, pid: int) -> str:
        """Read process status."""
        ...


@dataclass(frozen=True, slots=True)
class ScriptedProcReader:
    """Scripted `/proc` reader for telemetry tests."""

    meminfo_reads: int = 0
    status_pids: list[int] = field(default_factory=list)

    def meminfo(self) -> str:
        """Return host memory and swap evidence."""
        object.__setattr__(self, "meminfo_reads", self.meminfo_reads + 1)
        return "MemAvailable: 33554432 kB\nSwapFree: 16777216 kB\nSwapTotal: 16777216 kB\n"

    def status(self, pid: int) -> str:
        """Return process RSS evidence."""
        self.status_pids.append(pid)
        return "Name:\tpython\nVmRSS:\t1234 kB\n"


@dataclass(frozen=True, slots=True)
class RecordingCommandRunner:
    """Command runner that records argv without shell execution."""

    stdout: bytes
    process: CompletedProcess[bytes]
    argvs: list[tuple[str, ...]] = field(default_factory=list)

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[bytes]:
        """Record argv and return the scripted process."""
        self.argvs.append(argv)
        return self.process


@dataclass(frozen=True, slots=True)
class ManualClock:
    """Deterministic post-search clock."""

    now: float = 20_000.0
    cpu: float = 100.0

    def monotonic(self) -> float:
        """Return monotonic time."""
        return self.now

    def process_time(self) -> float:
        """Return process CPU time."""
        return self.cpu


@dataclass(frozen=True, slots=True)
class RecordingCoverageReplayer:
    """Coverage replayer spy."""

    requests: list[CoverageStepRequest] = field(default_factory=list)

    def replay(self, request: CoverageStepRequest) -> CoverageStepResult:
        """Record replay requests."""
        self.requests.append(request)
        return CoverageStepResult(
            counts=TargetCoverageCounts(
                target_file_count=1,
                branches_covered=1,
                branches_total=2,
                regions_covered=1,
                regions_total=2,
                lines_covered=1,
                lines_total=2,
            ),
            run_argv_sha256=("1" * 64,),
            profraw_sha256=("2" * 64,),
            merge_argv_sha256="3" * 64,
            profdata_sha256="4" * 64,
            report_argv_sha256="5" * 64,
            report_sha256="6" * 64,
        )


@dataclass(frozen=True, slots=True)
class RecordingCoverageBuilder:
    """Spy around Task4 coverage series builder."""

    requests: list[CoverageSeriesRequest] = field(default_factory=list)

    def build(self, request: CoverageSeriesRequest, replayer: CoverageReplayer) -> CoverageSeries:
        """Record coverage request and return a minimal series."""
        del replayer
        self.requests.append(request)
        return CoverageSeries.model_validate(
            {
                "corpus_index": (),
                "points": tuple(_coverage_point(index) for index in range(13)),
            }
        )


@dataclass(frozen=True, slots=True)
class RecordingProofRunner:
    """Proof runner spy."""

    requests: list[ProofBatchRequest] = field(default_factory=list)

    def run(self, request: ExactProofRequestModel) -> ProofOutcome:
        """Return a typed proof outcome if direct proof is unexpectedly called."""
        del request
        return FakeProofOutcome()


@dataclass(frozen=True, slots=True)
class RecordingProofBuilder:
    """Spy around Task4 deferred proof batch."""

    requests: list[ProofBatchRequest] = field(default_factory=list)

    def run(
        self, request: ProofBatchRequest, runner: ExactProofRunner, clock: ProofClock
    ) -> ProofBatchResult:
        """Record proof request and return typed proof accounting."""
        del runner, clock
        self.requests.append(request)
        return ProofBatchResult.model_validate(
            {
                "verdicts": (),
                "cause_index": {"causes": ()},
                "search_accounting": {"cpu_seconds": request.search_cpu_seconds},
                "proof_accounting": {
                    "replay_count": 2,
                    "wall_seconds": 3.0,
                    "cpu_seconds": 4.0,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class SpyPocRevisionRunner:
    """Spy for Task2 revision pipeline calls."""

    arms: list[str] = field(default_factory=list)
    proof_adapters: list[DeferredProofAdapter] = field(default_factory=list)

    def run(
        self,
        *,
        arm: ModelArm,
        sampler: RevisionSampler,
        proof_runner: DeferredProofAdapter,
        artifact_dir: Path,
    ) -> tuple[bytes, ...]:
        """Record one model-arm execution and return exact snapshots."""
        del sampler, artifact_dir
        self.arms.append(arm)
        self.proof_adapters.append(proof_runner)
        return (_C0_SNAPSHOT.content,)


@dataclass(frozen=True, slots=True)
class SpyRevisionWorkerFactory:
    """Factory that creates distinguishable resident worker clients."""

    affinities: list[WorkerCpuAffinity] = field(default_factory=list)

    def create(self, *, device: str, cpu_affinity: WorkerCpuAffinity) -> SpyResidentRevisionWorker:
        """Return one fake revision worker for a CUDA device."""
        self.affinities.append(cpu_affinity)
        return SpyResidentRevisionWorker(device=device)


@dataclass(frozen=True, slots=True)
class SpyResidentRevisionWorker(FakeRevisionDelegate):
    """Resident worker plus revision-sampler fake."""

    device: str = "cuda:0"
    cache_dir: Path = Path("/data/dllm-agent/cache")
    read_only_cache: bool = True
    ready: None = None

    @override
    def __hash__(self) -> int:
        """Hash by identity so reuse assertions can compare sets."""
        return id(self)

    @property
    def process_id_if_started(self) -> int | None:
        """Return a stable fake process ID."""
        return 20_000 + int(self.device.rsplit(":", maxsplit=1)[1])

    @property
    def last_process_id(self) -> int | None:
        """Return the last fake process ID."""
        return self.process_id_if_started

    @property
    def is_alive(self) -> bool:
        """Return whether the fake is alive."""
        return True

    def __enter__(self) -> Self:
        """Enter the fake worker."""
        return self

    def close(self) -> None:
        """Close the fake worker."""


@dataclass(frozen=True, slots=True)
class FakeProofOutcome:
    """Minimal proof outcome for protocol compliance."""

    oracle_status: Literal["not_reproduced"] = "not_reproduced"
    input_sha256: str = "0" * 64
    proof_log_sha256: str = "0" * 64
    proof_log: bytes = b""
    sanitizer_class: None = None
    root_signature: None = None
    run_argv_sha256: tuple[str, ...] = ("0" * 64, "1" * 64)
    run_argv_bytes: tuple[bytes, ...] = (b"argv-0", b"argv-1")
    result_bytes: bytes = b"result"


@dataclass(frozen=True, slots=True)
class RecordingPostSearch:
    """Post-search runner returning real accounting."""

    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        """Return accounting consumed by coordinator reports."""
        del request, result
        return _post_accounting()

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        """Return accounting for treatment-failed coordinator cells."""
        del request
        return _post_accounting()

    def coverage(self, request: CellRunRequest) -> None:
        """Compatibility method must not be used by fix2 coordinator."""
        del request

    def proof(self, request: CellRunRequest) -> None:
        """Compatibility method must not be used by fix2 coordinator."""
        del request


@dataclass(frozen=True, slots=True)
class MissingAccountingPostSearch:
    """Post-search fake that exposes the old no-accounting API only."""

    def coverage(self, request: CellRunRequest) -> None:
        """Old coverage method."""
        del request

    def proof(self, request: CellRunRequest) -> None:
        """Old proof method."""
        del request


def _qualified_config(data_root: Path) -> CampaignConfig:
    return CampaignConfig.model_validate(
        _config_json(str(data_root)) | {"qualification_state": "qualified", "c0_sha256": _C0_SHA}
    )


def _cell_request(root: Path) -> CellRunRequest:
    config = _qualified_config(root / "campaign")
    outcome = _campaign_outcome(config)
    cell = outcome.cells[0]

    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    manifest = build_campaign_manifest(config, schedule)
    return build_cell_requests(
        config=config,
        manifest=manifest,
        schedule=schedule,
        bank_by_repetition=dict.fromkeys(range(10), _C0_SHA),
        resource_map=config.resource_map,
    )[cell.repetition * 5]


def _cell_result(request: CellRunRequest) -> CellResult:
    return CellResult(
        cell_id=request.cell_id,
        status="completed",
        started_monotonic_seconds=request.start_monotonic_seconds,
        finished_monotonic_seconds=request.deadline_monotonic_seconds,
        coverage_freeze_elapsed_seconds=None,
        attempt_count=1,
        manifest_sha256="a" * 64,
        event_stream_sha256="b" * 64,
        corpus_index_stream_sha256="c" * 64,
        crash_stream_sha256="0" * 64,
        crash_occurrence_count=0,
        corpus_index_scheduled_seconds=tuple(range(0, 10_801, 900)),
        search_accounting=SearchAccounting(cpu_seconds=11.0, wall_seconds=10_800.0),
    )


def _manifest(config: CampaignConfig) -> CampaignManifest:
    return build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )


def _campaign_outcome(config: CampaignConfig) -> CampaignOutcome:
    return coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=RecordingPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )


def _post_accounting() -> PostSearchOutcome:
    return PostSearchOutcome(
        coverage_replay_count=1,
        proof_replay_count=1,
        coverage_series_sha256="7" * 64,
        proof_result_sha256="8" * 64,
        proof_wall_seconds=1.0,
        proof_cpu_seconds=1.0,
    )


def _coverage_point(index: int) -> dict[str, JsonValue]:
    digest = f"{index + 1:064x}"[-64:]
    return {
        "scheduled_elapsed_seconds": index,
        "observation_kind": "observed",
        "source_elapsed_seconds": float(index),
        "corpus_set_sha256": digest,
        "replayed_input_sha256": (digest,),
        "branches_covered": 1,
        "branches_total": 2,
        "regions_covered": 1,
        "regions_total": 2,
        "lines_covered": 1,
        "lines_total": 2,
        "profdata_sha256": digest,
        "report_sha256": digest,
    }
