"""Deterministic Task 5 fakes for coordinator, recovery, and CLI tests.

# noqa: SIZE_OK
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Never, Self, override

from tools.selective_revision_payloads import sha256_token_ids

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import (
    EMPTY_STREAM_SHA256,
    append_event,
    validate_ledger,
)
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    CellEvent,
    CellManifest,
    CellResult,
    SearchAccounting,
    validate_cell_evidence,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    SCHEDULE_SECONDS,
    CorpusIndexRequest,
    CoverageStepRequest,
    CoverageStepResult,
    TargetCoverageCounts,
    freeze_corpus_index,
    load_corpus_index,
    snapshot_corpus,
)
from dllm_agent.experiments.openexr_one_day.evidence_backing import persist_backing_bytes
from dllm_agent.experiments.openexr_one_day.failure_taxonomy import ExogenousCampaignError
from dllm_agent.experiments.openexr_one_day.models import GpuIndex
from dllm_agent.experiments.openexr_one_day.post_search import (
    HashOnlyDeepParseReplayer,
    PostSearchOutcome,
    ProductionPostSearch,
)
from dllm_agent.experiments.openexr_one_day.proof_batch import (
    CauseIndex,
    ExactProofRunner,
    ProofAccounting,
    ProofBatchRequest,
    ProofBatchResult,
    ProofClock,
    SearchAccountingRecord,
)
from dllm_agent.revision.controller_models import RevisionResult

if TYPE_CHECKING:
    from types import TracebackType

    from dllm_agent.experiments.openexr_one_day.cell_runner import CellRunRequest
    from dllm_agent.experiments.openexr_one_day.coordinator import (
        CampaignManifest,
        StartGateEvidence,
    )
    from dllm_agent.experiments.openexr_one_day.models import CellSchedule
    from dllm_agent.revision.controller_models import InitialParticleRequest, RevisionRequest


@dataclass(frozen=True, slots=True)
class FakeResidentWorker:
    """Minimal resident worker fake with stable process identity."""

    gpu_index: GpuIndex
    process_id: int
    events: list[str] = field(default_factory=list)
    device: str = field(init=False)
    cache_dir: Path = field(default=Path("/data/dllm-agent/cache"), init=False)
    read_only_cache: bool = True
    ready: None = None
    enter_count: int = 0
    close_count: int = 0
    initial_requests: list[InitialParticleRequest] = field(default_factory=list)
    revision_requests: list[RevisionRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Bind the CUDA device string after the GPU index is available."""
        object.__setattr__(self, "device", f"cuda:{int(self.gpu_index)}")

    @property
    def process_id_if_started(self) -> int | None:
        """Expose the stable process ID after pool entry."""
        return self.process_id

    @property
    def last_process_id(self) -> int | None:
        """Expose the stable process ID after close."""
        return self.process_id

    @property
    def is_alive(self) -> bool:
        """Return whether the fake is still resident."""
        return True

    def __enter__(self) -> Self:
        """Record pool entry."""
        object.__setattr__(self, "enter_count", self.enter_count + 1)
        return self

    def close(self) -> None:
        """Record pool close."""
        object.__setattr__(self, "close_count", self.close_count + 1)

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        """Generate a deterministic parent only after the worker is entered."""
        if self.enter_count <= 0:
            message = "initial called before worker entry"
            raise AssertionError(message)
        self.events.append(f"bank:{int(self.gpu_index)}:{request.particle_id}")
        self.initial_requests.append(request)
        variant = int(request.particle_id.rsplit("-", maxsplit=1)[1])
        return _fake_result(variant, len(self.initial_requests))

    def revise(self, request: RevisionRequest) -> RevisionResult:
        """Record a private revision request through the entered worker."""
        if self.enter_count <= 0:
            message = "revise called before worker entry"
            raise AssertionError(message)
        self.revision_requests.append(request)
        return _fake_result(len(self.revision_requests), len(self.revision_requests))


@dataclass(frozen=True, slots=True)
class FakeResidentPool:
    """Eight-worker pool fake that is entered once and reused for all waves."""

    events: list[str] = field(default_factory=list)
    workers: dict[int, FakeResidentWorker] = field(default_factory=dict)
    enter_count: int = 0
    exit_count: int = 0
    worker_requests: list[int] = field(default_factory=list)

    def __enter__(self) -> Self:
        """Enter every worker exactly once."""
        if not self.workers:
            object.__setattr__(
                self,
                "workers",
                {
                    gpu_index: FakeResidentWorker(
                        GpuIndex(gpu_index), 10_000 + gpu_index, events=self.events
                    )
                    for gpu_index in range(8)
                },
            )
        object.__setattr__(self, "enter_count", self.enter_count + 1)
        self.events.append("pool-enter")
        for worker in self.workers.values():
            _ = worker.__enter__()
        return self

    def __post_init__(self) -> None:
        """Create workers with access to the shared event log."""
        if not self.workers:
            object.__setattr__(
                self,
                "workers",
                {
                    gpu_index: FakeResidentWorker(
                        GpuIndex(gpu_index), 10_000 + gpu_index, events=self.events
                    )
                    for gpu_index in range(8)
                },
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close every worker exactly once."""
        del exc_type, exc, tb
        object.__setattr__(self, "exit_count", self.exit_count + 1)
        for worker in self.workers.values():
            worker.close()

    def worker_for(self, cell: CellSchedule) -> FakeResidentWorker:
        """Return the worker bound to a scheduled model cell GPU."""
        gpu_index = cell.gpu_index
        if gpu_index is None:
            raise AssertionError
        self.worker_requests.append(int(gpu_index))
        return self.workers[int(gpu_index)]

    def bank_worker_for(self, cell: CellSchedule) -> FakeResidentWorker:
        """Return the bank worker without counting model-arm adapter binding."""
        gpu_index = cell.gpu_index
        if gpu_index is None:
            raise AssertionError
        return self.workers[int(gpu_index)]


@dataclass(frozen=True, slots=True)
class FakeRevisionDelegate:
    """Private per-arm delegate used after the shared frozen bank."""

    initial_requests: list[InitialParticleRequest] = field(default_factory=list)
    revision_requests: list[RevisionRequest] = field(default_factory=list)

    @override
    def __hash__(self) -> int:
        """Hash by identity so runtime composition can compare worker reuse."""
        return id(self)

    def initial(self, request: InitialParticleRequest) -> Never:
        """Initial generation is owned by the shared bank sampler."""
        self.initial_requests.append(request)
        raise AssertionError

    def revise(self, request: RevisionRequest) -> Never:
        """Record private revision requests."""
        self.revision_requests.append(request)
        raise AssertionError


@dataclass(frozen=True, slots=True)
class FakeCellRunner:
    """Search-cell fake with injectable interruption and failure outcomes."""

    interrupt_after: int | None = None
    treatment_failures: dict[str, str] = field(default_factory=dict)
    exogenous_failures: dict[str, str] = field(default_factory=dict)
    requests: list[CellRunRequest] = field(default_factory=list)

    def wave_monotonic(self, scheduled_start: float) -> float:
        """Anchor deterministic fake work at each fixed wave start."""
        return scheduled_start

    def run(self, request: CellRunRequest) -> CellResult:
        """Record a Task 4 request and return a deterministic terminal result."""
        if self.interrupt_after is not None and len(self.requests) >= self.interrupt_after:
            raise KeyboardInterrupt
        self.requests.append(request)
        if request.cell_id in self.treatment_failures:
            return _persist_cell_evidence(
                request, "failed", request.start_monotonic_seconds + 120.0
            )
        if request.cell_id in self.exogenous_failures:
            raise ExogenousCampaignError(
                cause="gpu-xid", evidence=self.exogenous_failures[request.cell_id]
            )
        return _persist_cell_evidence(request, "completed", request.deadline_monotonic_seconds)


@dataclass(frozen=True, slots=True)
class FakePostSearch:
    """Coverage/proof fake proving the phases are outside search launches."""

    coverage_calls: list[str] = field(default_factory=list)
    proof_calls: list[str] = field(default_factory=list)

    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        """Record Task4 post-search accounting after search completion."""
        self.coverage_calls.append(request.cell_id)
        self.proof_calls.append(request.cell_id)
        return ProductionPostSearch(
            coverage_replayer=_FakeCoverageReplayer(request.artifact_root),
            proof_builder=_FakeProofBuilder(),
            clock=_FakeProofClock(),
            deep_parse_replayer=HashOnlyDeepParseReplayer(),
        ).coverage_and_proof(request, result)

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        """Record coverage/deep-parse accounting for treatment failures."""
        self.coverage_calls.append(request.cell_id)
        return ProductionPostSearch(
            coverage_replayer=_FakeCoverageReplayer(request.artifact_root),
            proof_builder=_FakeProofBuilder(),
            clock=_FakeProofClock(),
            deep_parse_replayer=HashOnlyDeepParseReplayer(),
        ).treatment_coverage(request)


@dataclass(frozen=True, slots=True)
class _FakeCoverageReplayer:
    artifact_root: Path

    def replay(self, request: CoverageStepRequest) -> CoverageStepResult:
        scheduled = request.scheduled_elapsed_seconds
        prefix = f"post-search/coverage-evidence/t{scheduled:05d}"
        payloads = {
            "run-argv": f"run:{scheduled}".encode(),
            "profraw": f"profraw:{scheduled}".encode(),
            "merge-argv": f"merge:{scheduled}".encode(),
            "profdata": f"profdata:{scheduled}".encode(),
            "report-argv": f"report-argv:{scheduled}".encode(),
            "report": f"report:{scheduled}".encode(),
        }
        run_argv = persist_backing_bytes(
            self.artifact_root, f"{prefix}/run-argv-0000.bin", payloads["run-argv"]
        )
        profraw = persist_backing_bytes(
            self.artifact_root, f"{prefix}/input-0000.profraw", payloads["profraw"]
        )
        merge_argv = persist_backing_bytes(
            self.artifact_root, f"{prefix}/merge-argv.bin", payloads["merge-argv"]
        )
        profdata = persist_backing_bytes(
            self.artifact_root, f"{prefix}/profile.profdata", payloads["profdata"]
        )
        report_argv = persist_backing_bytes(
            self.artifact_root, f"{prefix}/report-argv.bin", payloads["report-argv"]
        )
        report = persist_backing_bytes(
            self.artifact_root, f"{prefix}/report.json", payloads["report"]
        )
        return CoverageStepResult(
            counts=TargetCoverageCounts(
                target_file_count=1,
                branches_covered=1,
                branches_total=1,
                regions_covered=1,
                regions_total=1,
                lines_covered=1,
                lines_total=1,
            ),
            run_argv_sha256=(run_argv.sha256,),
            profraw_sha256=(profraw.sha256,),
            merge_argv_sha256=merge_argv.sha256,
            profdata_sha256=profdata.sha256,
            report_argv_sha256=report_argv.sha256,
            report_sha256=report.sha256,
            run_argv_evidence=(run_argv,),
            profraw_evidence=(profraw,),
            merge_argv_evidence=merge_argv,
            profdata_evidence=profdata,
            report_argv_evidence=report_argv,
            report_evidence=report,
        )


@dataclass(frozen=True, slots=True)
class _FakeProofBuilder:
    def run(
        self,
        request: ProofBatchRequest,
        runner: ExactProofRunner,
        clock: ProofClock,
    ) -> ProofBatchResult:
        del runner, clock
        return ProofBatchResult(
            verdicts=(),
            cause_index=CauseIndex(causes=()),
            search_accounting=SearchAccountingRecord(cpu_seconds=request.search_cpu_seconds),
            proof_accounting=ProofAccounting(
                replay_count=0,
                wall_seconds=1.0,
                cpu_seconds=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class _FakeProofClock:
    def monotonic(self) -> float:
        return 1.0

    def process_time(self) -> float:
        return 1.0


@dataclass(frozen=True, slots=True)
class FakeGateRunner:
    """Start-gate fake returning caller-supplied evidence."""

    evidence: StartGateEvidence
    call_count: int = 0

    def evaluate(self, manifest: CampaignManifest) -> StartGateEvidence:
        """Return the configured evidence for the manifest under test."""
        del manifest
        object.__setattr__(self, "call_count", self.call_count + 1)
        return self.evidence


@dataclass(frozen=True, slots=True)
class MultiRepetitionInitialSampler:
    """Initial sampler whose unique parent identity resets for each repetition."""

    initial_requests: list[InitialParticleRequest] = field(default_factory=list)

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        """Return a deterministic parent keyed by the request's bank offset."""
        self.initial_requests.append(request)
        variant = int(request.particle_id.rsplit("-", maxsplit=1)[1])
        return _fake_result(variant, len(self.initial_requests))

    def revise(self, request: RevisionRequest) -> Never:
        """Revision is delegated privately per arm, not through the bank sampler."""
        del request
        raise AssertionError


def fake_bank_sampler() -> MultiRepetitionInitialSampler:
    """Return a Task 3-compatible deterministic initial sampler."""
    return MultiRepetitionInitialSampler()


def _cell_result(
    request: CellRunRequest,
    status: Literal["completed", "failed", "deadline", "restart_limit"],
    finished: float,
) -> CellResult:
    return CellResult(
        cell_id=request.cell_id,
        status=status,
        started_monotonic_seconds=request.start_monotonic_seconds,
        finished_monotonic_seconds=finished,
        coverage_freeze_elapsed_seconds=(
            None
            if finished >= request.deadline_monotonic_seconds
            else finished - request.start_monotonic_seconds
        ),
        attempt_count=1,
        manifest_sha256="a" * 64,
        event_stream_sha256="b" * 64,
        corpus_index_stream_sha256="c" * 64,
        crash_stream_sha256="0" * 64,
        crash_occurrence_count=0,
        corpus_index_scheduled_seconds=tuple(range(0, 10_801, 900)),
        search_accounting=SearchAccounting(
            cpu_seconds=11.0,
            wall_seconds=finished - request.start_monotonic_seconds,
        ),
    )


def _persist_cell_evidence(
    request: CellRunRequest,
    status: Literal["completed", "failed", "deadline", "restart_limit"],
    finished: float,
) -> CellResult:
    if (request.artifact_root / "final-result.json").exists():
        return validate_cell_evidence(request)
    request.artifact_root.mkdir(parents=True, exist_ok=True)
    request.corpus_root.mkdir(parents=True, exist_ok=True)
    request.crash_root.mkdir(parents=True, exist_ok=True)
    request.log_root.mkdir(parents=True, exist_ok=True)
    if not any(request.corpus_root.iterdir()):
        message = "fake cell runner requires launched corpus seed evidence"
        raise AssertionError(message)
    manifest = CellManifest(request=request)
    manifest_payload = canonical_json_bytes(manifest.model_dump(mode="json"))
    _ = (request.artifact_root / "run-manifest.json").write_bytes(manifest_payload)
    manifest_sha256 = sha256_bytes(manifest_payload)
    index_request = CorpusIndexRequest(
        artifact_root=request.artifact_root,
        corpus_root=request.corpus_root,
        store_root=request.artifact_root / "corpus-inputs",
        index_path=request.artifact_root / "corpus-index.jsonl",
    )
    rows = load_corpus_index(index_request)
    if status == "completed":
        for scheduled in SCHEDULE_SECONDS[len(rows) :]:
            _ = snapshot_corpus(index_request, scheduled, float(scheduled))
    else:
        if not rows:
            _ = snapshot_corpus(index_request, 0, 0.0)
        _ = freeze_corpus_index(
            index_request, freeze_elapsed_seconds=finished - request.start_monotonic_seconds
        )
    log = f"{request.cell_id}:{status}".encode()
    _ = (request.log_root / "attempt-000000.log").write_bytes(log)
    outcome: Literal["completed", "target_crash", "failed", "deadline"]
    if status == "completed":
        outcome = "completed"
    elif status == "deadline":
        outcome = "deadline"
    else:
        outcome = "failed"
    event = CellEvent(
        sequence=0,
        attempt_index=0,
        outcome=outcome,
        return_code=0 if status == "completed" else 1,
        observed_monotonic_seconds=finished,
        remaining_seconds_at_launch=request.deadline_monotonic_seconds
        - request.start_monotonic_seconds,
        cpu_seconds=11.0,
        log_sha256=sha256_bytes(log),
        artifact_sha256=(),
        artifacts=(),
        manifest_sha256=manifest_sha256,
    )
    _ = append_event(
        request.artifact_root / "events.jsonl",
        timestamp_utc=f"monotonic+{finished:016.6f}",
        event_type="search_attempt",
        payload=event.model_dump(mode="json"),
    )
    result = _cell_result(request, status, finished).model_copy(
        update={
            "manifest_sha256": manifest_sha256,
            "event_stream_sha256": _fake_ledger_hash(request.artifact_root / "events.jsonl"),
            "corpus_index_stream_sha256": _fake_ledger_hash(
                request.artifact_root / "corpus-index.jsonl"
            ),
            "crash_stream_sha256": EMPTY_STREAM_SHA256,
        }
    )
    _ = (request.artifact_root / "final-result.json").write_bytes(
        canonical_json_bytes(result.model_dump(mode="json"))
    )
    return validate_cell_evidence(request)


def _fake_ledger_hash(path: Path) -> str:
    return validate_ledger(path).final_stream_sha256


def _fake_result(variant: int, call_index: int) -> RevisionResult:
    digits = tuple((variant // (4**index)) % 4 for index in range(5))
    selectors = (digits[0], digits[1], digits[2], digits[3], 0, 0, 0, 0, digits[4], 0)
    token_ids = tuple(selector + 10 for selector in selectors)
    return RevisionResult(
        selectors=selectors,
        token_ids=token_ids,
        slot_positions=tuple(range(10)),
        selected_slot_token_ids=token_ids,
        token_state_sha256=sha256_token_ids(token_ids),
        transfer_evidence_sha256=sha256_bytes(f"transfer:{call_index}".encode()),
        rng_state_sha256=sha256_bytes(f"rng:{call_index}".encode()),
        nfe=1,
        forward_calls=1,
        gpu_ms=10,
        cache_hit=False,
        trajectory_uncertainty=0.5,
        suggested_field="width",
        trajectory_evidence_sha256=sha256_bytes(b"[]"),
    )
