"""Recovery CLI contracts for completed interrupted multitarget campaigns."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
from pydantic import TypeAdapter
from tools import resume_multitarget_24h as resume
from tools.run_multitarget_24h import PROOF_IMAGES, SEARCH_IMAGES

from dllm_agent.experiments.multitarget.artifacts import write_json_artifact
from dllm_agent.experiments.multitarget.budget import ExperimentSpec
from dllm_agent.experiments.multitarget.campaign_execution import FinishedCell
from dllm_agent.experiments.multitarget.container_monitor import (
    CanarySeries,
    ContainerLaunch,
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.corpus import StagedCorpus
from dllm_agent.experiments.multitarget.docker_runtime import ContainerRequest, runtime_profile
from dllm_agent.experiments.multitarget.ledger import ArtifactEvidence, validate_ledger
from dllm_agent.experiments.multitarget.proof import (
    AsanSignature,
    ProofReplayResult,
    ProofRunEvidence,
)
from dllm_agent.experiments.multitarget.proof_batch import (
    ProofBatchItem,
    ProofBatchRecord,
    ProofBatchSummary,
    ProofCpuPair,
    ProofReplayRequestRecord,
)
from dllm_agent.experiments.multitarget.schedule import (
    CampaignSchedule,
    CellSchedule,
    build_campaign_schedule,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dllm_agent.experiments.multitarget.budget import Target

type JsonObject = dict[str, object]
type FakeProofStatus = Literal["proved", "not_reproduced", "crash_without_sanitizer", "failed"]

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(dict[str, object])


def test_parses_default_and_explicit_proof_cpu_pairs(tmp_path: Path) -> None:
    """Given CLI args, when parsed, then proof CPU pairs are deterministic."""
    # Given / When
    default_args = resume.parse_arguments(("--experiment-root", str(tmp_path / "run")))
    explicit_args = resume.parse_arguments(
        (
            "--experiment-root",
            str(tmp_path / "run"),
            "--proof-cpu-pairs",
            "2-3,8-9",
        )
    )

    # Then
    assert default_args.proof_cpu_pairs == tuple(
        ProofCpuPair(first=cpu, second=cpu + 1) for cpu in range(15, 39, 2)
    )
    assert explicit_args.proof_cpu_pairs == (
        ProofCpuPair(first=2, second=3),
        ProofCpuPair(first=8, second=9),
    )


def test_parse_rejects_malformed_proof_cpu_pairs_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given malformed CPU pairs, when parsed, then argparse reports usage."""
    # Given / When
    with pytest.raises(SystemExit) as error:
        _ = resume.parse_arguments(
            (
                "--experiment-root",
                str(tmp_path / "run"),
                "--proof-cpu-pairs",
                "not-a-pair",
            )
        )

    # Then
    assert error.value.code == 2
    assert "proof CPU pairs must use N-M integer syntax" in capsys.readouterr().err


def test_runs_partial_recovery_once_and_persists_stable_batch_index(tmp_path: Path) -> None:
    """Given no batch index, when recovering, then proof runs and one index event is recorded."""
    # Given
    repo_root = tmp_path / "repo"
    expected_source_hashes = _write_source_hash_fixture(repo_root)
    docs_root = repo_root / "docs"
    root, schedule, finished = _experiment_fixture(tmp_path)
    calls: list[tuple[Path, tuple[FinishedCell, ...], tuple[ProofCpuPair, ...]]] = []
    summary = _proof_summary(root, finished[:2], executed=1, skipped=1)

    def proof_batch(
        *,
        experiment_root: Path,
        finished_cells: tuple[FinishedCell, ...],
        proof_images: Mapping[Target, str],
        cpu_pairs: tuple[ProofCpuPair, ...],
    ) -> ProofBatchSummary:
        calls.append((experiment_root, finished_cells, cpu_pairs))
        assert proof_images == PROOF_IMAGES
        return summary

    # When
    code = resume.run_recovery(
        resume.ResumeArguments(root, (ProofCpuPair(2, 3), ProofCpuPair(4, 5))),
        repo_root=repo_root,
        schedule=schedule,
        finished_cells=finished,
        proof_batch=proof_batch,
    )

    # Then
    assert code == 0
    assert calls == [(root.resolve(), finished, (ProofCpuPair(2, 3), ProofCpuPair(4, 5)))]
    index = _json(root / "proof/batch-index.json")
    assert index["schema_version"] == 1
    assert index["source_hashes"] == expected_source_hashes
    assert index["status_counts"] == {"proved": 1, "not_reproduced": 1}
    assert index["by_target"] == {"libpng": {"proved": 1, "not_reproduced": 1}}
    assert index["proofed_inputs"] == [
        {"sha256": _sha256(b"alpha"), "target": "libpng"},
        {"sha256": _sha256(b"beta"), "target": "libpng"},
    ]
    assert index["proved_inputs"] == [{"sha256": _sha256(b"alpha"), "target": "libpng"}]
    assert index["proof_images"] == PROOF_IMAGES
    assert index["cpu_pairs"] == [{"first": 2, "second": 3}, {"first": 4, "second": 5}]
    assert index["operational"] == {
        "executed_count": 1,
        "recorded_replay_count": 2,
        "replay_count": 2,
        "scheduled_count": 2,
        "skipped_count": 1,
        "wall_seconds": 1.25,
    }
    assert _event_types(root) == [
        "recovery-proof-started",
        "recovery-proof-batch-indexed",
        "markdown-report-written",
        "repository-report-copied",
    ]
    assert validate_ledger(root / "events.jsonl").record_count == 4
    assert (root / "results/report.md").is_file()
    assert (docs_root / "multitarget-24h-results-20260822.md").is_file()
    assert root.with_name(f"{root.name}.evidence-tree-sha256.txt").is_file()


def test_rerun_with_existing_outputs_skips_proof_and_does_not_append_or_overwrite(
    tmp_path: Path,
) -> None:
    """Given completed recovery outputs, when rerun, then no duplicate writes occur."""
    # Given
    repo_root = tmp_path / "repo"
    root, schedule, finished = _experiment_fixture(tmp_path)
    summary = _full_proof_summary(root, finished)
    args = resume.ResumeArguments(root, (ProofCpuPair(2, 3),))

    _ = resume.run_recovery(
        args,
        repo_root=repo_root,
        schedule=schedule,
        finished_cells=finished,
        proof_batch=_summary_runner(summary),
    )
    ledger_before = (root / "events.jsonl").read_bytes()
    index_before = (root / "proof/batch-index.json").read_bytes()
    report_before = (root / "results/report.md").read_bytes()
    sidecar_before = root.with_name(f"{root.name}.evidence-tree-sha256.txt").read_bytes()

    def forbidden_proof_batch(**_kwargs: object) -> ProofBatchSummary:
        message = "proof batch must not rerun"
        raise AssertionError(message)

    # When
    code = resume.run_recovery(
        args,
        repo_root=repo_root,
        schedule=schedule,
        finished_cells=finished,
        proof_batch=forbidden_proof_batch,
    )

    # Then
    assert code == 0
    assert (root / "events.jsonl").read_bytes() == ledger_before
    assert (root / "proof/batch-index.json").read_bytes() == index_before
    assert (root / "results/report.md").read_bytes() == report_before
    assert root.with_name(f"{root.name}.evidence-tree-sha256.txt").read_bytes() == sidecar_before


@pytest.mark.parametrize(
    "case_name",
    [
        "schedule",
        "source_hashes",
        "proof_images",
        "cpu_pairs",
        "crash_set",
        "result_hash",
        "status_counts",
        "by_target",
        "result_count",
    ],
)
def test_existing_stale_index_fails_closed_before_report(
    tmp_path: Path,
    case_name: str,
) -> None:
    """Given a stale canonical index, when rerun, then proof is not skipped."""
    # Given
    repo_root = tmp_path / "repo"
    source_hashes = _write_source_hash_fixture(repo_root)
    root, schedule, finished = _experiment_fixture(tmp_path)
    cpu_pairs = (ProofCpuPair(2, 3), ProofCpuPair(4, 5))
    index = resume.build_batch_index(
        experiment_root=root,
        schedule=schedule,
        summary=_full_proof_summary(root, finished),
        cpu_pairs=cpu_pairs,
        source_hashes=source_hashes,
    )
    _mutate_index(index, case_name)
    _ = write_json_artifact(
        experiment_root=root,
        relative_path=Path("proof/batch-index.json"),
        value=index,
    )

    def forbidden_proof_batch(**_kwargs: object) -> ProofBatchSummary:
        message = "proof batch must not run for a stale existing index"
        raise AssertionError(message)

    # When / Then
    with pytest.raises(resume.ResumeRecoveryError, match="proof batch index"):
        _ = resume.run_recovery(
            resume.ResumeArguments(root, cpu_pairs),
            repo_root=repo_root,
            schedule=schedule,
            finished_cells=finished,
            proof_batch=forbidden_proof_batch,
        )
    assert not (root / "results/report.md").exists()
    assert not (repo_root / "docs/multitarget-24h-results-20260822.md").exists()


def test_repository_report_copy_event_records_docs_copy_evidence(tmp_path: Path) -> None:
    """Given report copy outside root, when recorded, then payload binds docs evidence."""
    # Given
    repo_root = tmp_path / "repo"
    _ = _write_source_hash_fixture(repo_root)
    root, schedule, finished = _experiment_fixture(tmp_path)

    # When
    _ = resume.run_recovery(
        resume.ResumeArguments(root, (ProofCpuPair(2, 3),)),
        repo_root=repo_root,
        schedule=schedule,
        finished_cells=finished,
        proof_batch=_summary_runner(_full_proof_summary(root, finished)),
    )

    # Then
    event = _event_payload(root, "repository-report-copied")
    docs_path = repo_root / "docs/multitarget-24h-results-20260822.md"
    report_path = root / "results/report.md"
    docs_bytes = docs_path.read_bytes()
    assert _json_object(event["artifact"])["path"] == "results/report.md"
    assert event["repository_copy_evidence"] == {
        "path": "docs/multitarget-24h-results-20260822.md",
        "sha256": _sha256(docs_bytes),
        "size_bytes": len(docs_bytes),
    }
    assert docs_bytes == report_path.read_bytes()


def test_existing_noncanonical_index_fails_closed_before_proof(tmp_path: Path) -> None:
    """Given a malformed index, when recovering, then proof is not rerun over it."""
    # Given
    root, schedule, finished = _experiment_fixture(tmp_path)
    (root / "proof").mkdir()
    _ = (root / "proof/batch-index.json").write_text(
        '{\n  "schema_version": 1\n}',
        encoding="utf-8",
    )

    def forbidden_proof_batch(**_kwargs: object) -> ProofBatchSummary:
        message = "proof batch must not run with an invalid existing index"
        raise AssertionError(message)

    # When / Then
    with pytest.raises(resume.ResumeRecoveryError, match="canonical"):
        _ = resume.run_recovery(
            resume.ResumeArguments(root, (ProofCpuPair(2, 3),)),
            repo_root=tmp_path / "repo",
            schedule=schedule,
            finished_cells=finished,
            proof_batch=forbidden_proof_batch,
        )


def test_report_states_descriptive_limits_and_effect_directions(tmp_path: Path) -> None:
    """Given finished cells and proofs, when rendered, then claims stay descriptive."""
    # Given
    root, schedule, finished = _experiment_fixture(tmp_path)
    index = resume.build_batch_index(
        experiment_root=root,
        schedule=schedule,
        summary=_proof_summary(root, finished[:3], executed=3, skipped=0),
        cpu_pairs=(ProofCpuPair(2, 3),),
        source_hashes={"tools/resume_multitarget_24h.py": "a" * 64},
    )

    # When
    report = resume.render_report(schedule=schedule, finished_cells=finished, batch_index=index)

    # Then
    assert "# dLLM-native multi-target 24-hour recovery report" in report
    assert "15/15 finished cells" in report
    assert "n=1 descriptive comparison" in report
    assert "fatal canary trigger is benchmark evidence, not a vulnerability" in report
    assert "zero-day" not in report.lower()
    assert "selective vs pseudo" in report
    assert "geometric mean direction" in report
    assert "| libpng | selective-revision |" in report
    assert "| libpng | pseudo-feedback |" in report
    assert "OOM state" in report
    assert "Independently proved in this cell" in report


def test_index_records_proofed_and_proved_inputs_separately(tmp_path: Path) -> None:
    """Given mixed proof statuses, when indexed, then proved hashes are target-qualified."""
    # Given
    root, schedule, finished = _experiment_fixture(tmp_path)
    summary = _proof_summary_for(
        root,
        (
            (finished[0], b"alpha", "proved", False),
            (finished[1], b"beta", "not_reproduced", False),
            (finished[2], b"gamma", "crash_without_sanitizer", False),
            (finished[3], b"delta", "failed", False),
        ),
    )

    # When
    index = resume.build_batch_index(
        experiment_root=root,
        schedule=schedule,
        summary=summary,
        cpu_pairs=(ProofCpuPair(2, 3),),
        source_hashes={"tools/resume_multitarget_24h.py": "a" * 64},
    )
    report = resume.render_report(schedule=schedule, finished_cells=finished, batch_index=index)

    # Then
    assert "validated_input_sha256" not in index
    assert index["proofed_inputs"] == _target_hashes(
        ("libpng", _sha256(b"alpha")),
        ("libpng", _sha256(b"beta")),
        ("libpng", _sha256(b"gamma")),
        ("libpng", _sha256(b"delta")),
    )
    assert index["proved_inputs"] == [{"sha256": _sha256(b"alpha"), "target": "libpng"}]
    assert index["status_counts"] == {
        "crash_without_sanitizer": 1,
        "failed": 1,
        "not_reproduced": 1,
        "proved": 1,
    }
    assert index["operational"] == {
        "executed_count": 4,
        "recorded_replay_count": 4,
        "replay_count": 8,
        "scheduled_count": 4,
        "skipped_count": 0,
        "wall_seconds": 1.25,
    }
    assert f"libpng/{_sha256(b'alpha')}" in report
    assert _sha256(b"beta") not in report
    assert "not_reproduced: 1" in report
    assert "crash_without_sanitizer: 1" in report
    assert "failed: 1" in report


def test_report_counts_only_crash_candidates_and_credits_all_finding_cells(
    tmp_path: Path,
) -> None:
    """Given duplicate crashes and slow units, when reported, then counts stay crash-only."""
    # Given
    root, schedule, finished = _experiment_fixture(tmp_path)
    shared = _sha256(b"shared-crash")
    finished = _replace_cell(
        finished,
        "libpng",
        "libfuzzer-only",
        (
            _finding(root, "crash-libpng-libfuzzer-shared", b"shared-crash"),
            _finding(root, "slow-unit-libpng-libfuzzer", b"slow"),
        ),
    )
    finished = _replace_cell(
        finished,
        "libpng",
        "selective-revision",
        (_finding(root, "crash-libpng-selective-shared", b"shared-crash"),),
    )
    index = resume.build_batch_index(
        experiment_root=root,
        schedule=schedule,
        summary=_proof_summary_for(
            root,
            ((_find_cell(finished, "libpng", "libfuzzer-only"), b"shared-crash", "proved", False),),
        ),
        cpu_pairs=(ProofCpuPair(2, 3),),
        source_hashes={"tools/resume_multitarget_24h.py": "a" * 64},
    )

    # When
    report = resume.render_report(schedule=schedule, finished_cells=finished, batch_index=index)

    # Then
    assert "Unique crash candidates: 14." in report
    assert _row_field(report, "libpng", "libfuzzer-only", "Unique crash candidates") == "1"
    assert (
        _row_field(report, "libpng", "libfuzzer-only", "Independently proved in this cell") == "1"
    )
    assert (
        _row_field(report, "libpng", "selective-revision", "Independently proved in this cell")
        == "1"
    )
    assert {"target": "libpng", "sha256": shared} in _json_list(index["proved_inputs"])


def test_report_uses_logical_trigger_times_and_cpm_effects(tmp_path: Path) -> None:
    """Given feedback-arm probe time, when reported, then first trigger is end-to-end."""
    # Given
    root, schedule, finished = _experiment_fixture(tmp_path)
    index = resume.build_batch_index(
        experiment_root=root,
        schedule=schedule,
        summary=_proof_summary(root, finished[:1], executed=1, skipped=0),
        cpu_pairs=(ProofCpuPair(2, 3),),
        source_hashes={"tools/resume_multitarget_24h.py": "a" * 64},
    )

    # When
    report = resume.render_report(schedule=schedule, finished_cells=finished, batch_index=index)

    # Then
    assert "First trigger time (logical search seconds)" in report
    assert _row_field(report, "libpng", "selective-revision", "First trigger time") == "983"
    assert "coverage-per-million-executions" in report
    assert "geometric mean coverage-per-million-executions direction" in report


def _experiment_fixture(tmp_path: Path) -> tuple[Path, CampaignSchedule, tuple[FinishedCell, ...]]:
    root = tmp_path / "experiment"
    root.mkdir()
    schedule = build_campaign_schedule(
        ExperimentSpec(),
        available_cpus=40,
        image_ids=SEARCH_IMAGES,
    )
    finished = tuple(_finished_cell(root, cell, index) for index, cell in enumerate(schedule.cells))
    return root, schedule, finished


def _write_source_hash_fixture(repo_root: Path) -> dict[str, str]:
    files = {
        "tools/resume_multitarget_24h.py": b"resume",
        "src/dllm_agent/experiments/multitarget/campaign_recovery.py": b"recovery",
        "src/dllm_agent/experiments/multitarget/proof.py": b"proof",
        "src/dllm_agent/experiments/multitarget/proof_batch.py": b"proof-batch",
    }
    for relative, content in files.items():
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(content)
    return {relative: _sha256(content) for relative, content in sorted(files.items())}


def _finished_cell(root: Path, cell: CellSchedule, index: int) -> FinishedCell:
    work_dir = root / "cells" / cell.target / cell.arm / f"rep-{cell.repetition}"
    shared_dir = work_dir / "shared"
    corpus_dir = work_dir / "corpus"
    shared_dir.mkdir(parents=True)
    corpus_dir.mkdir()
    content = f"{cell.target}-{cell.arm}-{index}".encode()
    finding_path = shared_dir / "findings" / f"crash-{index:02d}"
    finding_path.parent.mkdir()
    _ = finding_path.write_bytes(content)
    log_path = work_dir / "main.log"
    log_text = f"#1 cov: {100 + index} ft: {200 + index} exec/s: {300 + index}\n"
    _ = log_path.write_text(log_text, encoding="utf-8")
    return FinishedCell(
        cell=cell,
        work_dir=work_dir,
        staged_corpus=StagedCorpus(
            corpus_dir=corpus_dir,
            base_file_count=1,
            candidate_file_count=16,
            files=(),
            tree_sha256="0" * 64,
        ),
        request=ContainerRequest(
            name=f"container-{index}",
            profile=runtime_profile(cell.target),
            image_id=cell.image_id,
            shared_dir=shared_dir,
            corpus_dir=corpus_dir,
            duration_seconds=86_400,
            fuzzer_seed=cell.seeds.main_fuzzer,
            cpu_index=cell.cpu_index,
            poll_seconds=5,
        ),
        launch=ContainerLaunch(container_id=f"container-{index}"),
        result=ContainerRunResult(
            container_id=f"container-{index}",
            exit_code=0,
            docker_status="exited",
            oom_killed=False,
            timed_out=False,
            process_error=None,
            logs_text=log_text,
            log_sha256=hashlib.sha256(log_text.encode()).hexdigest(),
            metrics=FuzzerMetrics(
                coverage_edges=100 + index,
                value_profile_features=200 + index,
                executions_per_second=300 + index,
                executed_units=10_000 + index,
            ),
            canaries={
                f"CANARY_{index % 4}": CanarySeries(
                    reached=((10, 1),),
                    triggered=((20 + index, 1),),
                )
            },
            findings=(
                ArtifactEvidence(
                    path=str(finding_path),
                    sha256=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                ),
            ),
        ),
        log_path=log_path,
    )


def _proof_summary(
    root: Path,
    owners: tuple[FinishedCell, ...],
    *,
    executed: int,
    skipped: int,
) -> ProofBatchSummary:
    statuses = ("proved", "not_reproduced", "proved")
    contents = (b"alpha", b"beta", b"gamma")
    items = tuple(
        _proof_item(
            root,
            owners[index],
            contents[index],
            statuses[index],
            skipped=index >= executed,
        )
        for index in range(executed + skipped)
    )
    return ProofBatchSummary(
        scheduled_count=len(items),
        executed_count=executed,
        skipped_count=skipped,
        replay_count=executed * 2,
        wall_seconds=1.25,
        results=items,
        result_artifact_paths=tuple(item.artifact.path for item in items),
    )


def _proof_summary_for(
    root: Path,
    specs: tuple[tuple[FinishedCell, bytes, FakeProofStatus, bool], ...],
) -> ProofBatchSummary:
    items = tuple(
        _proof_item(root, owner, content, status, skipped=skipped)
        for owner, content, status, skipped in specs
    )
    return ProofBatchSummary(
        scheduled_count=len(items),
        executed_count=sum(1 for item in items if not item.skipped),
        skipped_count=sum(1 for item in items if item.skipped),
        replay_count=sum(2 for item in items if not item.skipped),
        wall_seconds=1.25,
        results=items,
        result_artifact_paths=tuple(item.artifact.path for item in items),
    )


def _full_proof_summary(root: Path, finished: tuple[FinishedCell, ...]) -> ProofBatchSummary:
    specs: tuple[tuple[FinishedCell, bytes, FakeProofStatus, bool], ...] = tuple(
        (item, Path(item.result.findings[0].path).read_bytes(), "proved", False)
        for item in finished
    )
    return _proof_summary_for(root, specs)


def _summary_runner(summary: ProofBatchSummary) -> resume.ProofBatchRunner:
    def proof_batch(
        *,
        experiment_root: Path,
        finished_cells: tuple[FinishedCell, ...],
        proof_images: Mapping[Target, str],
        cpu_pairs: tuple[ProofCpuPair, ...],
    ) -> ProofBatchSummary:
        _ = (experiment_root, finished_cells, proof_images, cpu_pairs)
        return summary

    return proof_batch


def _proof_item(
    root: Path,
    owner: FinishedCell,
    content: bytes,
    status: FakeProofStatus,
    *,
    skipped: bool,
) -> ProofBatchItem:
    digest = _sha256(content)
    result_path = Path("proof-results") / owner.cell.target / f"{digest}.json"
    request = ProofReplayRequestRecord(
        profile=runtime_profile(owner.cell.target),
        image_id=PROOF_IMAGES[owner.cell.target],
        input_path=f"proof-inputs/{owner.cell.target}/{digest}",
        input_sha256=digest,
        first_seed=1,
        second_seed=2,
        first_container_name=f"proof-{digest[:8]}-a",
        second_container_name=f"proof-{digest[:8]}-b",
        first_cpu=2,
        second_cpu=3,
        timeout_seconds=60,
    )
    proof_input = ArtifactEvidence(
        path=request.input_path,
        sha256=digest,
        size_bytes=len(content),
    )
    record = ProofBatchRecord(
        key=f"{owner.cell.target}/{digest}",
        target=owner.cell.target,
        input_sha256=digest,
        owner_cell=owner.cell.to_manifest(),
        source_finding_path=owner.result.findings[0].path,
        proof_input=proof_input,
        request=request,
        result=ProofReplayResult(
            status=status,
            profile=runtime_profile(owner.cell.target),
            image_id=PROOF_IMAGES[owner.cell.target],
            input_sha256=digest,
            runs=(
                ProofRunEvidence(
                    container_name=request.first_container_name,
                    proof_seed=1,
                    return_code=1,
                    log=b"asan",
                    log_sha256=_sha256(b"asan"),
                    argv_sha256="b" * 64,
                    signature=AsanSignature("heap-buffer-overflow", "target_frame"),
                    error=None,
                ),
            ),
            signature=AsanSignature("heap-buffer-overflow", "target_frame"),
            error=None,
        ),
    )
    artifact = write_json_artifact(experiment_root=root, relative_path=result_path, value=record)
    return ProofBatchItem(
        key=record.key,
        record=replace(record, result=record.result),
        artifact=artifact,
        skipped=skipped,
    )


def _finding(root: Path, stem: str, content: bytes) -> ArtifactEvidence:
    path = root / "test-findings" / stem
    path.parent.mkdir(exist_ok=True)
    _ = path.write_bytes(content)
    return ArtifactEvidence(path=str(path), sha256=_sha256(content), size_bytes=len(content))


def _target_hashes(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"sha256": digest, "target": target} for target, digest in sorted(pairs)]


def _replace_cell(
    finished: tuple[FinishedCell, ...],
    target: str,
    arm: str,
    findings: tuple[ArtifactEvidence, ...],
) -> tuple[FinishedCell, ...]:
    return tuple(
        replace(item, result=replace(item.result, findings=findings))
        if item.cell.target == target and item.cell.arm == arm
        else item
        for item in finished
    )


def _find_cell(finished: tuple[FinishedCell, ...], target: str, arm: str) -> FinishedCell:
    for item in finished:
        if item.cell.target == target and item.cell.arm == arm:
            return item
    message = f"missing cell: {target}/{arm}"
    raise AssertionError(message)


def _row_field(report: str, target: str, arm: str, field: str) -> str:
    header_line = next(line for line in report.splitlines() if line.startswith("| Target |"))
    headers = [part.strip() for part in header_line.split("|")[1:-1]]
    field_index = headers.index(field)
    for line in report.splitlines():
        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) == len(headers) and parts[0] == target and parts[1] == arm:
            return parts[field_index]
    message = f"missing report row: {target}/{arm}"
    raise AssertionError(message)


def _json(path: Path) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_json(path.read_bytes())


def _json_object(value: object) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def _json_list(value: object) -> list[JsonObject]:
    return TypeAdapter(list[dict[str, object]]).validate_python(value)


def _mutate_index(index: JsonObject, case_name: str) -> None:
    if case_name == "schedule":
        index["schedule_manifest_sha256"] = "0" * 64
    elif case_name == "source_hashes":
        source_hashes = _json_object(index["source_hashes"])
        source_hashes["tools/resume_multitarget_24h.py"] = "0" * 64
        index["source_hashes"] = source_hashes
    elif case_name == "proof_images":
        proof_images = _json_object(index["proof_images"])
        proof_images["libpng"] = "sha256:" + ("0" * 64)
        index["proof_images"] = proof_images
    elif case_name == "cpu_pairs":
        index["cpu_pairs"] = [{"first": 8, "second": 9}]
    elif case_name == "crash_set":
        index["proofed_inputs"] = _json_list(index["proofed_inputs"])[:-1]
    elif case_name == "result_hash":
        artifacts = _json_list(index["result_artifacts"])
        first = _json_object(artifacts[0])
        first["sha256"] = "0" * 64
        artifacts[0] = first
        index["result_artifacts"] = artifacts
    elif case_name == "status_counts":
        counts = _json_object(index["status_counts"])
        counts["proved"] = 999
        index["status_counts"] = counts
    elif case_name == "by_target":
        by_target = _json_object(index["by_target"])
        libpng = _json_object(by_target["libpng"])
        libpng["proved"] = 999
        by_target["libpng"] = libpng
        index["by_target"] = by_target
    elif case_name == "result_count":
        index["results"] = _json_list(index["results"])[:-1]
    else:
        message = f"unknown stale index case: {case_name}"
        raise AssertionError(message)


def _event_types(root: Path) -> list[str]:
    return [
        str(_json_line(line)["event_type"])
        for line in (root / "events.jsonl").read_text().splitlines()
    ]


def _json_line(line: str) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_json(line)


def _event_payload(root: Path, event_type: str) -> JsonObject:
    for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        record = _json_line(line)
        if record.get("event_type") == event_type:
            return _json_object(record["payload"])
    message = f"missing event: {event_type}"
    raise AssertionError(message)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
