"""Task 6 campaign fixture built through the real Task 5 producer boundary. # noqa: SIZE_OK"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, cast

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
    StartGateEvidence,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CoveragePoint,
    CoverageReplayer,
    CoverageSeries,
    CoverageSeriesRequest,
    CoverageStepRequest,
    CoverageStepResult,
    load_corpus_index,
)
from dllm_agent.experiments.openexr_one_day.evidence_backing import persist_backing_bytes
from dllm_agent.experiments.openexr_one_day.models import ARMS, MODEL_ARMS, Arm
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
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakeResidentPool,
    FakeRevisionDelegate,
    fake_bank_sampler,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dllm_agent.experiments.multitarget.ledger import JsonValue
    from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, CellRunRequest


@dataclass(frozen=True, slots=True)
class _UnusedCoverageReplayer:
    def replay(self, request: CoverageStepRequest) -> CoverageStepResult:
        del request
        raise AssertionError


@dataclass(frozen=True, slots=True)
class _LiteralCoverageBuilder:
    arm: Arm
    repetition: int

    def build(self, request: CoverageSeriesRequest, replayer: CoverageReplayer) -> CoverageSeries:
        del replayer
        rows = load_corpus_index(request.corpus_index)
        baseline = 100 + self.repetition
        gain = {
            "selective-revision": 6,
            "final-only": 2,
            "full-regenerate": 4,
            "structured-one-shot": 1,
            "libfuzzer-only": 1,
        }[self.arm]
        points: list[CoveragePoint] = []
        for row in rows:
            prefix = (
                f"coverage/t{row.scheduled_elapsed_seconds:05d}-{self.repetition:02d}-{self.arm}"
            )
            run_argv = persist_backing_bytes(
                request.corpus_index.artifact_root,
                f"{prefix}/run-argv-0000.json",
                (
                    f'["fixture-coverage-run","{self.repetition}","{self.arm}",'
                    f'"{row.scheduled_elapsed_seconds}"]'
                ).encode(),
            )
            profraw = persist_backing_bytes(
                request.corpus_index.artifact_root,
                f"{prefix}/input-0000.profraw",
                (f"profraw:{self.repetition}:{self.arm}:{row.scheduled_elapsed_seconds}").encode(),
            )
            merge_argv = persist_backing_bytes(
                request.corpus_index.artifact_root,
                f"{prefix}/merge-argv.json",
                (
                    f'["fixture-coverage-merge","{self.repetition}","{self.arm}",'
                    f'"{row.scheduled_elapsed_seconds}"]'
                ).encode(),
            )
            profdata = persist_backing_bytes(
                request.corpus_index.artifact_root,
                f"{prefix}/profile.profdata",
                (f"profdata:{self.repetition}:{self.arm}:{row.scheduled_elapsed_seconds}").encode(),
            )
            report_argv = persist_backing_bytes(
                request.corpus_index.artifact_root,
                f"{prefix}/report-argv.json",
                (
                    f'["fixture-coverage-report","{self.repetition}","{self.arm}",'
                    f'"{row.scheduled_elapsed_seconds}"]'
                ).encode(),
            )
            report = persist_backing_bytes(
                request.corpus_index.artifact_root,
                f"{prefix}/report.json",
                (f"report:{self.repetition}:{self.arm}:{row.scheduled_elapsed_seconds}").encode(),
            )
            points.append(
                CoveragePoint(
                    scheduled_elapsed_seconds=row.scheduled_elapsed_seconds,
                    observation_kind=row.observation_kind,
                    source_elapsed_seconds=row.source_elapsed_seconds,
                    corpus_set_sha256=row.cumulative_set_sha256,
                    replayed_input_sha256=tuple(item.sha256 for item in row.new_inputs),
                    branches_covered=(
                        baseline if row.scheduled_elapsed_seconds == 0 else baseline + gain
                    ),
                    branches_total=1000,
                    regions_covered=(
                        baseline if row.scheduled_elapsed_seconds == 0 else baseline + gain
                    ),
                    regions_total=1000,
                    lines_covered=(
                        baseline if row.scheduled_elapsed_seconds == 0 else baseline + gain
                    ),
                    lines_total=1000,
                    run_argv_sha256=(run_argv.sha256,),
                    run_argv_evidence=(run_argv,),
                    profraw_sha256=(profraw.sha256,),
                    profraw_evidence=(profraw,),
                    merge_argv_sha256=merge_argv.sha256,
                    merge_argv_evidence=merge_argv,
                    profdata_sha256=profdata.sha256,
                    profdata_evidence=profdata,
                    report_argv_sha256=report_argv.sha256,
                    report_argv_evidence=report_argv,
                    report_sha256=report.sha256,
                    report_evidence=report,
                )
            )
        return CoverageSeries(corpus_index=rows, points=tuple(points))


@dataclass(frozen=True, slots=True)
class _EmptyProofBuilder:
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
class _ProofClock:
    def monotonic(self) -> float:
        return 1.0

    def process_time(self) -> float:
        return 1.0


@dataclass(frozen=True, slots=True)
class _Task6PostSearch:
    config: CampaignConfig

    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        return self._runner(request).coverage_and_proof(request, result)

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        return self._runner(request).treatment_coverage(request)

    def _runner(self, request: CellRunRequest) -> ProductionPostSearch:
        arm = cast("Arm", next(arm for arm in ARMS if request.cell_id.endswith(arm)))
        repetition = int(request.cell_id.split("-")[3])
        return ProductionPostSearch(
            config=self.config,
            coverage_replayer=_UnusedCoverageReplayer(),
            coverage_builder=_LiteralCoverageBuilder(arm, repetition),
            proof_builder=_EmptyProofBuilder(),
            clock=_ProofClock(),
            deep_parse_replayer=HashOnlyDeepParseReplayer(),
        )


@dataclass(frozen=True, slots=True)
class _Task6ModelExecutor:
    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        content = f"task6:{request.cell.cell_id}".encode()
        digest = sha256_bytes(content)
        path = request.cell.artifact_root / "corpus" / f"model-{digest}.exr"
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(content)
        return ModelArmExecutionResult(
            written_snapshot_count=1,
            written_paths=(path,),
            nfe=1000,
            forward_calls=100,
            gpu_ms=10_000,
            bank_ready_elapsed_seconds=450.0,
        )


def build_task6_campaign(root: Path) -> CampaignConfig:
    """Create one complete, canonical 50-cell campaign using Task 5."""
    config = CampaignConfig.model_validate(_config_payload(root))
    _ = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates={arm: FakeRevisionDelegate() for arm in MODEL_ARMS},
        cell_runner=FakeCellRunner(),
        post_search=_Task6PostSearch(config),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=_Task6ModelExecutor(),
    )
    return config


def _config_payload(root: Path) -> dict[str, JsonValue]:
    return {
        "schema_version": "dllm-openexr-one-day-config/v1",
        "config_id": "task6-fixture",
        "qualification_state": "qualified",
        "artifact_root": root.as_posix(),
        "model_cache_dir": (root / "cache").as_posix(),
        "main_start_monotonic_seconds": monotonic(),
        "target_commit": "777d231a179de1711d2a942810d9559216ab3f4c",
        "target_harness_sha256": "1" * 64,
        "prompt_sha256": "2" * 64,
        "model_revision": "10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
        "remote_code_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "search_image_digest": "sha256:" + "5" * 64,
        "search_binary_sha256": "6" * 64,
        "sanitizers": ["address", "undefined"],
        "coverage_image_digest": "sha256:" + "7" * 64,
        "coverage_binary_sha256": "8" * 64,
        "coverage_build_manifest_sha256": "9" * 64,
        "c0_source_path": "fixtures/openexr/c0-validity-preserving-zero.hex",
        "c0_sha256": "ae9d361e47cc503dc91b827f4956fdf8ca16ffb3804ecdfcf83dd627f8a64f96",
        "dictionary_sha256": "9419f2003be8f41d286360ea3dec3f8fa967244111eecec0fd292db0ae6fed93",
        "max_input_length": 1_048_576,
        "target_timeout_seconds": 10,
        "rss_limit_mb": 2_048,
        "candidate_steps": 8,
        "proof_duration_seconds": 30,
        "resource_map": _resource_map(),
    }


def _resource_map() -> dict[str, JsonValue]:
    return {
        "model_cores": [
            {"core_id": index, "sibling_core_id": index + 20}
            for index in (0, 1, 2, 3, 10, 11, 12, 13)
        ],
        "search_cores": [
            {"core_id": index, "sibling_core_id": index + 20}
            for index in (4, 5, 6, 7, 8, 9, 14, 15, 16, 17)
        ],
        "proof_cores": [{"core_id": 18, "sibling_core_id": 38}],
        "monitor_cores": [{"core_id": 19, "sibling_core_id": 39}],
        "gpu_numa_nodes": [0, 0, 0, 0, 1, 1, 1, 1],
        "gpu_records": [
            {
                "index": index,
                "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
                "numa_node": 0 if index < 4 else 1,
            }
            for index in range(8)
        ],
        "core_numa_nodes": {
            str(index): 0 if index in {*range(10), *range(20, 30)} else 1 for index in range(40)
        },
    }


def _passing_gates() -> StartGateEvidence:
    return StartGateEvidence(
        hashes_match=True,
        banks_complete=True,
        read_only_cache=True,
        external_coverage_check=True,
        control_arm_equivalence=True,
        resource_map_valid=True,
        disk_space=True,
        vram=True,
        ram=True,
        swap_stable=True,
        reserved_core_load=True,
        smoke_passed=True,
    )
