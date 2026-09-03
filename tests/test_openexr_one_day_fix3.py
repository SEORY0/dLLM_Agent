"""Task 5 fix3 production graph regressions.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import TypeAdapter
from tools.run_openexr_one_day import _production_factory

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, SearchAccounting
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
    ResourceMap,
    build_campaign_manifest,
    build_cell_requests,
    campaign_mutable_root,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CoverageReplayer,
    CoverageSeries,
    CoverageSeriesRequest,
)
from dllm_agent.experiments.openexr_one_day.post_search import (
    PostSearchOutcome,
    ProductionPostSearch,
)
from dllm_agent.experiments.openexr_one_day.proof_batch import (
    ExactProofRunner,
    ProofBatchRequest,
    ProofBatchResult,
    ProofClock,
)
from dllm_agent.experiments.openexr_one_day.qualification import (
    QualificationEvidence,
    QualificationFact,
)
from dllm_agent.experiments.openexr_one_day.recovery import append_campaign_event, resume_campaign
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.openexr_one_day_task5_fakes import FakeCellRunner, FakeGateRunner, FakeResidentPool
from tests.test_openexr_one_day_coordinator import (
    _config_json,
    _passing_gates,
    _qualified_config,
)
from tests.test_openexr_one_day_fix2 import (
    ManualClock,
    RecordingCoverageReplayer,
    RecordingProofRunner,
)

if TYPE_CHECKING:
    from dllm_agent.experiments.openexr_one_day.cell_runner import CellRunRequest

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    case = tmp_path_factory.mktemp("case")
    suffix = sha256(str(case).encode()).hexdigest()[:12]
    path = Path("/data/dllm-agent") / f"task5-fix3-{suffix}"
    if path.exists():
        raise AssertionError(path)
    return path


def test_coordinate_campaign_enters_workers_before_bank_model_prep_and_search(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    events: list[str] = []
    pool = FakeResidentPool(events=events)
    model_executor = RecordingModelExecutor(events=events)
    cell_runner = OrderedCellRunner(events=events)

    outcome = coordinate_campaign(
        config,
        resident_pool=pool,
        model_executor=model_executor,
        cell_runner=cell_runner,
        post_search=RecordingPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    assert outcome.launched_cell_count == 50
    assert events[0] == "pool-enter"
    first_bank = next(index for index, event in enumerate(events) if event.startswith("bank:"))
    first_model = next(index for index, event in enumerate(events) if event.startswith("model:"))
    first_search = next(index for index, event in enumerate(events) if event.startswith("search:"))
    assert first_bank > 0
    assert first_bank < first_model < first_search
    assert len(model_executor.requests) == 40
    assert {request.bank_sampler_id for request in model_executor.requests} <= {
        id(worker) for worker in pool.workers.values()
    }
    assert {request.revision_delegate_id for request in model_executor.requests} <= {
        id(worker) for worker in pool.workers.values()
    }
    assert set(pool.worker_requests) == set(range(8))


def test_coordinate_campaign_passes_real_cell_results_to_post_search(data_root: Path) -> None:
    config = _qualified_config(data_root)
    post_search = RecordingPostSearch()

    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        model_executor=RecordingModelExecutor(),
        cell_runner=DistinctResultCellRunner(),
        post_search=post_search,
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    assert post_search.result_stream_hashes
    assert set(post_search.result_stream_hashes) == {"d" * 64}
    assert set(post_search.result_cpu_seconds) == {123.0}


def test_production_post_search_uses_configured_coverage_and_proof_provenance(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    request = _first_cell_request(data_root)
    result = _result(request, "b" * 64, 11.0)
    coverage = RecordingCoverageBuilder()
    proof = RecordingProofBuilder()
    post_search = ProductionPostSearch(
        config=config,
        coverage_builder=coverage,
        proof_builder=proof,
        coverage_replayer=RecordingCoverageReplayer(),
        proof_runner=RecordingProofRunner(),
        clock=ManualClock(),
    )

    _ = post_search.coverage_and_proof(request, result)

    coverage_request = coverage.requests[0]
    proof_request = proof.requests[0]
    assert coverage_request.image_digest == config.coverage_image_digest
    assert coverage_request.binary_sha256 == config.coverage_binary_sha256
    assert coverage_request.build_manifest_sha256 == config.coverage_build_manifest_sha256
    assert coverage_request.corpus_index.store_root == request.artifact_root / "corpus-inputs"
    assert proof_request.duration_seconds == config.proof_duration_seconds
    assert proof_request.proof_seed != request.fuzzer_seed
    assert proof_request.search_cpu_seconds == result.search_accounting.cpu_seconds


def test_qualification_evidence_requires_typed_backing_artifact_hash(data_root: Path) -> None:
    artifact = data_root / "evidence" / "hashes.json"
    artifact.parent.mkdir(parents=True)
    _ = artifact.write_text("measured", encoding="utf-8")
    digest = sha256_bytes(artifact.read_bytes())

    evidence = QualificationEvidence.for_manifest(
        manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        records=(
            QualificationFact(
                name="hashes_match",
                manifest_sha256="a" * 64,
                config_sha256="b" * 64,
                passed=True,
                measurement_kind="digest-check",
                measurements={"description": "target/prompt/config digests matched"},
                artifact_path=artifact,
                artifact_sha256=digest,
                artifact_size_bytes=len("measured"),
            ),
        ),
    )

    assert evidence.records[0].artifact_path == artifact
    with pytest.raises(ValueError, match="artifact hash"):
        _ = QualificationFact(
            name="hashes_match",
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            passed=True,
            measurement_kind="digest-check",
            measurements={"description": "target/prompt/config digests matched"},
            artifact_path=artifact,
            artifact_sha256="0" * 64,
            artifact_size_bytes=len("measured"),
        )


def test_cli_production_factory_uses_configured_runtime_paths(data_root: Path) -> None:
    payload = _config_json(str(data_root)) | {
        "qualification_state": "qualified",
        "runtime_python": "/opt/dllm/runtime/bin/python",
        "model_cache_dir": "/data/dllm-agent/cache/from-config",
        "prompt_path": "/data/dllm-agent/prompts/openexr-system.md",
        "qualification_evidence_path": "/data/dllm-agent/qualification/evidence.json",
    }
    config = CampaignConfig.model_validate(payload)

    factory = _production_factory(config)

    assert factory.runtime_python == Path("/opt/dllm/runtime/bin/python")
    assert factory.cache_dir == Path("/data/dllm-agent/cache/from-config")
    assert factory.prompt_path == Path("/data/dllm-agent/prompts/openexr-system.md")
    assert factory.qualification_evidence_path == Path(
        "/data/dllm-agent/qualification/evidence.json"
    )


def test_expired_partial_resume_preserves_terminal_treatment_cells(data_root: Path) -> None:
    config = _qualified_config(data_root)
    failed = "wave-00-rep-00-selective-revision"
    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        model_executor=RecordingModelExecutor(),
        cell_runner=FakeCellRunner(treatment_failures={failed: "controller-crash"}),
        post_search=RecordingPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )
    failed_cell = next(cell for cell in _.cells if cell.cell_id == failed)
    (config.artifact_root / "campaign-events.jsonl").unlink()
    for derived in ("raw-arm-reports.jsonl", "campaign-outcome.json"):
        path = config.artifact_root / derived
        if path.exists():
            path.unlink()
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    manifest = build_campaign_manifest(config, schedule)
    append_campaign_event(
        config.artifact_root,
        "manifest",
        {
            "manifest_sha256": manifest.manifest_sha256 or "",
            "config_sha256": config.config_sha256,
            "resource_map_sha256": manifest.resource_map_sha256,
        },
    )
    for request, cell in zip(
        build_cell_requests(
            config=config,
            manifest=manifest,
            schedule=schedule,
            bank_by_repetition=dict.fromkeys(range(10), failed_cell.bank_sha256),
            resource_map=config.resource_map,
        ),
        schedule.cells,
        strict=True,
    ):
        request_payload = request.model_dump(mode="json")
        append_campaign_event(
            config.artifact_root,
            "launch",
            {
                "cell_id": request.cell_id,
                "fuzzer_seed": int(cell.fuzzer_seed),
                "deadline_monotonic_seconds": float(cell.deadline_monotonic_seconds),
                "artifact_root": str(request.artifact_root),
                "config_sha256": config.config_sha256,
                "resource_map_sha256": manifest.resource_map_sha256,
                "manifest_sha256": manifest.manifest_sha256 or "",
                "mutable_root": str(campaign_mutable_root(config, cell)),
                "bank_sha256": request.bank_sha256,
                "request": request_payload,
                "request_sha256": sha256_bytes(canonical_json_bytes(request_payload)),
                "build_sha256": sha256_bytes(
                    canonical_json_bytes(request.build.model_dump(mode="json"))
                ),
                "corpus_seed_manifest_sha256": sha256_bytes(
                    (request.artifact_root / "seed-manifest.json").read_bytes()
                ),
            },
        )
    append_campaign_event(config.artifact_root, "terminal", failed_cell.model_dump(mode="json"))

    outcome = resume_campaign(
        config.artifact_root,
        resident_pool=FakeResidentPool(),
        model_executor=RecordingModelExecutor(),
        cell_runner=FakeCellRunner(),
        post_search=RecordingPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        now_monotonic_seconds=200_000.0,
    )

    preserved = next(cell for cell in outcome.cells if cell.cell_id == failed)
    assert preserved.terminal_reason == "treatment-failed"
    assert preserved.coverage_frozen_through_seconds == 10_800


def test_resource_map_validates_explicit_gpu_identity_records() -> None:
    payload = _resource_map_with_gpu_records()

    resource_map = ResourceMap.model_validate(payload)

    assert tuple(record.index for record in resource_map.gpu_records) == tuple(range(8))
    assert resource_map.gpu_records[0].uuid.startswith("GPU-")
    gpu_records = cast("list[dict[str, JsonValue]]", payload["gpu_records"])
    gpu_records[0]["index"] = 1
    with pytest.raises(ValueError, match="GPU identity"):
        _ = ResourceMap.model_validate(payload)


def test_raw_reports_include_complete_model_search_proof_and_hash_chain(data_root: Path) -> None:
    config = _qualified_config(data_root)
    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        model_executor=RecordingModelExecutor(),
        cell_runner=DistinctResultCellRunner(),
        post_search=RecordingPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    payloads = [
        _json_line(line)
        for line in (config.artifact_root / "raw-arm-reports.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(payloads) == 50
    assert {
        "schema_version",
        "model_nfe",
        "model_forward_calls",
        "model_gpu_ms",
        "proof_wall_seconds",
        "proof_cpu_seconds",
        "previous_row_sha256",
        "row_sha256",
    } <= set(payloads[0])
    assert payloads[0]["previous_row_sha256"] == "0" * 64
    assert payloads[1]["previous_row_sha256"] == payloads[0]["row_sha256"]


@dataclass(frozen=True, slots=True)
class RecordingModelExecutor:
    events: list[str] = field(default_factory=list)
    requests: list[ModelArmExecutionRequest] = field(default_factory=list)

    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        self.events.append(f"model:{request.cell.cell_id}")
        self.requests.append(request)
        return ModelArmExecutionResult(
            written_snapshot_count=0,
            written_paths=(),
            nfe=7,
            forward_calls=11,
            gpu_ms=13,
            bank_ready_elapsed_seconds=0.0,
        )


@dataclass(frozen=True, slots=True)
class OrderedCellRunner:
    events: list[str]

    def run(self, request: CellRunRequest) -> CellResult:
        self.events.append(f"search:{request.cell_id}")
        return _result(request, "b" * 64, 11.0)


@dataclass(frozen=True, slots=True)
class DistinctResultCellRunner:
    def run(self, request: CellRunRequest) -> CellResult:
        return _result(request, "d" * 64, 123.0)


@dataclass(frozen=True, slots=True)
class RecordingPostSearch:
    result_stream_hashes: list[str] = field(default_factory=list)
    result_cpu_seconds: list[float] = field(default_factory=list)

    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        del request
        self.result_stream_hashes.append(result.event_stream_sha256)
        self.result_cpu_seconds.append(result.search_accounting.cpu_seconds)
        digest = sha256_bytes(canonical_json_bytes({"cell_id": result.cell_id}))
        return PostSearchOutcome(
            coverage_replay_count=1,
            proof_replay_count=1,
            coverage_series_sha256=digest,
            proof_result_sha256=digest,
            proof_wall_seconds=1.0,
            proof_cpu_seconds=1.0,
        )

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        digest = sha256_bytes(canonical_json_bytes({"cell_id": request.cell_id}))
        return PostSearchOutcome(
            coverage_replay_count=1,
            proof_replay_count=0,
            coverage_series_sha256=digest,
            proof_result_sha256=digest,
            proof_wall_seconds=0.0,
            proof_cpu_seconds=0.0,
        )


@dataclass(frozen=True, slots=True)
class RecordingCoverageBuilder:
    requests: list[CoverageSeriesRequest] = field(default_factory=list)

    def build(self, request: CoverageSeriesRequest, replayer: CoverageReplayer) -> CoverageSeries:
        del replayer
        self.requests.append(request)
        return CoverageSeries.model_validate({"corpus_index": (), "points": ()})


@dataclass(frozen=True, slots=True)
class RecordingProofBuilder:
    requests: list[ProofBatchRequest] = field(default_factory=list)

    def run(
        self, request: ProofBatchRequest, runner: ExactProofRunner, clock: ProofClock
    ) -> ProofBatchResult:
        del runner, clock
        self.requests.append(request)
        return ProofBatchResult.model_validate(
            {
                "verdicts": (),
                "cause_index": {"causes": ()},
                "search_accounting": {"cpu_seconds": request.search_cpu_seconds},
                "proof_accounting": {
                    "replay_count": 0,
                    "wall_seconds": 1.0,
                    "cpu_seconds": 1.0,
                },
            }
        )


def _first_cell_request(root: Path) -> CellRunRequest:
    config = _qualified_config(root / "campaign")
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    manifest = build_campaign_manifest(config, schedule)
    return build_cell_requests(
        config=config,
        manifest=manifest,
        schedule=schedule,
        bank_by_repetition=dict.fromkeys(range(10), "a" * 64),
        resource_map=config.resource_map,
    )[0]


def _resource_map_with_gpu_records() -> dict[str, JsonValue]:
    payload = _config_json("/data/dllm-agent/gpu-records")["resource_map"]
    if not isinstance(payload, dict):
        raise TypeError
    payload["gpu_records"] = [
        {
            "index": index,
            "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
            "numa_node": 0 if index < 4 else 1,
        }
        for index in range(8)
    ]
    return payload


def _json_line(line: str) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(line)


def _result(request: CellRunRequest, event_hash: str, cpu_seconds: float) -> CellResult:
    return CellResult(
        cell_id=request.cell_id,
        status="completed",
        started_monotonic_seconds=request.start_monotonic_seconds,
        finished_monotonic_seconds=request.deadline_monotonic_seconds,
        coverage_freeze_elapsed_seconds=None,
        attempt_count=1,
        manifest_sha256="a" * 64,
        event_stream_sha256=event_hash,
        corpus_index_stream_sha256="c" * 64,
        crash_stream_sha256="0" * 64,
        crash_occurrence_count=0,
        corpus_index_scheduled_seconds=tuple(range(0, 10_801, 900)),
        search_accounting=SearchAccounting(cpu_seconds=cpu_seconds, wall_seconds=10_800.0),
    )
