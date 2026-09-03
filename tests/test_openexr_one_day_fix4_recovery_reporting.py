"""Task 5 fix4C post-search, raw-report, and CLI resume regressions.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from tools.run_openexr_one_day import _load_config_from_root, _production_factory

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    CellResult,
    CellRunRequest,
    SearchAccounting,
    SearchBuildProvenance,
)
from dllm_agent.experiments.openexr_one_day.config_digest import config_payload_sha256
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CorpusIndexRecord,
    CorpusIndexRequest,
    CoveragePoint,
    CoverageSeries,
    freeze_corpus_index,
    load_corpus_index,
    snapshot_corpus,
)
from dllm_agent.experiments.openexr_one_day.evidence_backing import persist_backing_bytes
from dllm_agent.experiments.openexr_one_day.post_search import (
    CoverageSeriesEnvelope,
    HashOnlyDeepParseReplayer,
    PostSearchEvidenceError,
    ProductionPostSearch,
    load_post_search_outcome,
)
from dllm_agent.experiments.openexr_one_day.proof_batch import (
    CauseIndex,
    ProofAccounting,
    ProofBatchResult,
    SearchAccountingRecord,
)
from tests.test_openexr_one_day_fix1 import _config_json

if TYPE_CHECKING:
    from dllm_agent.experiments.openexr_one_day.coverage_series import (
        CoverageReplayer,
        CoverageSeriesRequest,
        CoverageStepRequest,
        CoverageStepResult,
    )
    from dllm_agent.experiments.openexr_one_day.proof_batch import (
        ExactProofRunner,
        ProofBatchRequest,
        ProofClock,
    )


def test_production_post_search_persists_full_artifacts_and_reuses_second_call(
    data_root: Path,
) -> None:
    request = _cell_request(data_root)
    _seed_complete_corpus_index(request)
    result = _cell_result(request)
    coverage = CountingCoverageBuilder()
    proof = CountingProofBuilder()
    post_search = ProductionPostSearch(
        coverage_replayer=NoopCoverageReplayer(),
        coverage_builder=coverage,
        proof_builder=proof,
        clock=StaticProofClock(),
        deep_parse_replayer=HashOnlyDeepParseReplayer(),
    )

    first = post_search.coverage_and_proof(request, result)
    if first.coverage_series_path is None:
        raise AssertionError
    envelope = CoverageSeriesEnvelope.model_validate_json(first.coverage_series_path.read_bytes())
    raw_report = canonical_json_bytes(envelope.coverage_series.model_dump(mode="json"))
    second = post_search.coverage_and_proof(request, result)

    assert first == second
    assert coverage.calls == 1
    assert proof.calls == 1
    assert first.coverage_series_sha256 == sha256_bytes(raw_report)
    assert first.proof_result_path is not None
    assert first.proof_result_path.is_file()


def test_post_search_reload_rejects_tampered_coverage_backing(data_root: Path) -> None:
    request = _cell_request(data_root)
    _seed_complete_corpus_index(request)
    result = _cell_result(request)
    post_search = ProductionPostSearch(
        coverage_replayer=NoopCoverageReplayer(),
        coverage_builder=CountingCoverageBuilder(),
        proof_builder=CountingProofBuilder(),
        clock=StaticProofClock(),
        deep_parse_replayer=HashOnlyDeepParseReplayer(),
    )
    first = post_search.coverage_and_proof(request, result)
    if first.coverage_series_path is None:
        raise AssertionError
    envelope = CoverageSeriesEnvelope.model_validate_json(first.coverage_series_path.read_bytes())
    evidence = envelope.coverage_series.points[0].report_evidence
    if evidence is None:
        raise AssertionError
    _ = (request.artifact_root / evidence.relative_path).write_bytes(b"tampered")

    with pytest.raises(PostSearchEvidenceError, match="coverage backing"):
        _ = load_post_search_outcome(request, result)


def test_cli_resume_factory_uses_digest_verified_root_config_paths(data_root: Path) -> None:
    root = data_root / "resume-root"
    root.mkdir(parents=True)
    qualification_path = root / "qualification/from-root.json"
    payload = _config_json(str(root)) | {
        "qualification_state": "qualified",
        "runtime_python": "/opt/dllm/runtime/bin/python",
        "model_cache_dir": "/data/dllm-agent/cache/from-root",
        "prompt_path": "/data/dllm-agent/prompts/from-root.md",
        "qualification_evidence_path": str(qualification_path),
    }
    payload["config_sha256"] = config_payload_sha256(payload)
    _ = (root / "campaign-manifest.json").write_text(
        json.dumps({"config": payload}, sort_keys=True),
        encoding="utf-8",
    )

    config = _load_config_from_root(root)
    factory = _production_factory(config)

    assert factory.runtime_python == Path("/opt/dllm/runtime/bin/python")
    assert factory.cache_dir == Path("/data/dllm-agent/cache/from-root")
    assert factory.prompt_path == Path("/data/dllm-agent/prompts/from-root.md")
    assert factory.qualification_evidence_path == qualification_path


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fix4c")
    return Path("/data/dllm-agent") / f"task5-fix4c-{sha256(str(root).encode()).hexdigest()[:12]}"


@dataclass(frozen=True, slots=True)
class CountingCoverageBuilder:
    calls: int = 0

    def build(self, request: CoverageSeriesRequest, replayer: CoverageReplayer) -> CoverageSeries:
        del replayer
        object.__setattr__(self, "calls", self.calls + 1)
        rows = load_corpus_index(request.corpus_index)
        return CoverageSeries(
            corpus_index=rows,
            points=tuple(_coverage_point(request, row) for row in rows),
        )


@dataclass(frozen=True, slots=True)
class CountingProofBuilder:
    calls: int = 0

    def run(
        self,
        request: ProofBatchRequest,
        runner: ExactProofRunner,
        clock: ProofClock,
    ) -> ProofBatchResult:
        del request, runner, clock
        object.__setattr__(self, "calls", self.calls + 1)
        return ProofBatchResult(
            verdicts=(),
            cause_index=CauseIndex(causes=()),
            search_accounting=SearchAccountingRecord(cpu_seconds=11.0),
            proof_accounting=ProofAccounting(
                replay_count=0,
                wall_seconds=1.0,
                cpu_seconds=0.5,
            ),
        )


@dataclass(frozen=True, slots=True)
class NoopCoverageReplayer:
    def replay(self, request: CoverageStepRequest) -> CoverageStepResult:
        raise AssertionError(request)


@dataclass(frozen=True, slots=True)
class StaticProofClock:
    def monotonic(self) -> float:
        return 100.0

    def process_time(self) -> float:
        return 10.0


def _cell_request(data_root: Path) -> CellRunRequest:
    root = data_root / "cells" / "wave-00" / "rep-00" / "libfuzzer-only"
    dictionary_path = data_root / "openexr.dict"
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    _ = dictionary_path.write_bytes(b"openexr-dictionary\n")
    return CellRunRequest(
        cell_id="wave-00-rep-00-libfuzzer-only",
        artifact_root=root,
        corpus_root=root / "corpus",
        crash_root=root / "crashes",
        log_root=root / "logs",
        target_commit="777d231a179de1711d2a942810d9559216ab3f4c",
        harness_sha256="1" * 64,
        config_sha256="4" * 64,
        bank_sha256="5" * 64,
        start_monotonic_seconds=0.0,
        deadline_monotonic_seconds=10_800.0,
        fuzzer_seed=100,
        max_input_length=1_048_576,
        target_timeout_seconds=10,
        rss_limit_mb=2_048,
        cpu_index=8,
        dictionary_path=dictionary_path,
        dictionary_sha256=sha256_bytes(dictionary_path.read_bytes()),
        build=SearchBuildProvenance(
            image_digest="sha256:" + "5" * 64,
            binary_sha256="6" * 64,
            sanitizers=("address", "undefined"),
        ),
    )


def _coverage_point(request: CoverageSeriesRequest, row: CorpusIndexRecord) -> CoveragePoint:
    root = request.corpus_index.artifact_root
    scheduled = row.scheduled_elapsed_seconds
    run_argv = json.dumps(("run", scheduled), separators=(",", ":")).encode()
    profraw = f"profraw:{scheduled}".encode()
    merge_argv = json.dumps(("merge", scheduled), separators=(",", ":")).encode()
    profdata = f"profdata:{scheduled}".encode()
    report_argv = json.dumps(("report", scheduled), separators=(",", ":")).encode()
    report = f"report:{scheduled}".encode()
    return CoveragePoint(
        scheduled_elapsed_seconds=scheduled,
        observation_kind=row.observation_kind,
        source_elapsed_seconds=row.source_elapsed_seconds,
        corpus_set_sha256=row.cumulative_set_sha256,
        replayed_input_sha256=tuple(item.sha256 for item in row.new_inputs),
        branches_covered=1,
        branches_total=2,
        regions_covered=1,
        regions_total=2,
        lines_covered=1,
        lines_total=2,
        run_argv_sha256=(sha256_bytes(run_argv),),
        run_argv_evidence=(
            persist_backing_bytes(root, f"post-search/test-coverage/{scheduled}/run.bin", run_argv),
        ),
        profraw_sha256=(sha256_bytes(profraw),),
        profraw_evidence=(
            persist_backing_bytes(
                root,
                f"post-search/test-coverage/{scheduled}/input.profraw",
                profraw,
            ),
        ),
        merge_argv_sha256=sha256_bytes(merge_argv),
        merge_argv_evidence=persist_backing_bytes(
            root,
            f"post-search/test-coverage/{scheduled}/merge.bin",
            merge_argv,
        ),
        profdata_sha256=sha256_bytes(profdata),
        profdata_evidence=persist_backing_bytes(
            root,
            f"post-search/test-coverage/{scheduled}/profile.profdata",
            profdata,
        ),
        report_argv_sha256=sha256_bytes(report_argv),
        report_argv_evidence=persist_backing_bytes(
            root,
            f"post-search/test-coverage/{scheduled}/report.bin",
            report_argv,
        ),
        report_sha256=sha256_bytes(report),
        report_evidence=persist_backing_bytes(
            root,
            f"post-search/test-coverage/{scheduled}/report.json",
            report,
        ),
    )


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


def _seed_complete_corpus_index(request: CellRunRequest) -> None:
    request.corpus_root.mkdir(parents=True, exist_ok=True)
    _ = (request.corpus_root / "seed").write_bytes(b"seed")
    index_request = CorpusIndexRequest(
        artifact_root=request.artifact_root,
        corpus_root=request.corpus_root,
        store_root=request.artifact_root / "corpus-inputs",
        index_path=request.artifact_root / "corpus-index.jsonl",
    )
    _ = snapshot_corpus(index_request, 0, 0.0)
    _ = freeze_corpus_index(index_request, freeze_elapsed_seconds=1.0)
